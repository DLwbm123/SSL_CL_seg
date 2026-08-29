"""Read immutable training manifests without ever consulting diagnostics data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import read_csv

_FORBIDDEN_UNLABELED_FIELDS = {"label", "hidden_label", "diagnostic_path", "label_h5_relpath"}


@dataclass(frozen=True)
class SliceRecord:
    row: dict[str, str]
    slice_index: int | None


def _safe_relative_path(value: str, *, field: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe {field}: {value!r}")
    return path


def _site_from_row(row: dict[str, str]) -> str:
    return row.get("site_or_vendor") or row.get("site") or row.get("vendor") or ""


def load_training_records(
    data_root: Path,
    *,
    seed: int,
    dataset: str | None = None,
    sites: Iterable[str] | None = None,
    roles: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    """Read only a training manifest and enforce label visibility rules."""

    manifest = Path(data_root) / "manifests" / "training" / f"lcrseg_v1_seed{seed}.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"training manifest is missing: {manifest}")
    requested_sites = set(sites) if sites is not None else None
    requested_roles = set(roles) if roles is not None else None
    selected: list[dict[str, str]] = []
    for raw_row in read_csv(manifest):
        row = {key: value or "" for key, value in raw_row.items()}
        if dataset is not None and row.get("dataset") != dataset:
            continue
        if requested_sites is not None and _site_from_row(row) not in requested_sites:
            continue
        role = row.get("primary_20pct_split", "")
        if requested_roles is not None and role not in requested_roles:
            continue
        image_rel = _safe_relative_path(row.get("image_h5_relpath", ""), field="image_h5_relpath")
        if not (Path(data_root) / "h5" / "v1" / image_rel).is_file():
            raise FileNotFoundError(f"missing image HDF5 for {row.get('case_id')}: {image_rel}")
        if role == "train_unlabeled":
            if row.get("label_h5_relpath", ""):
                raise ValueError(f"hidden label leaked into training manifest for {row.get('case_id')}")
        else:
            label_rel = row.get("label_h5_relpath", "")
            if label_rel:
                safe_label = _safe_relative_path(label_rel, field="label_h5_relpath")
                if not (Path(data_root) / "h5" / "v1" / safe_label).is_file():
                    raise FileNotFoundError(f"missing label HDF5 for {row.get('case_id')}: {safe_label}")
        selected.append(row)
    if not selected:
        raise ValueError("no records matched the requested training view")
    return selected


class _H5SliceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        data_root: Path,
        records: list[dict[str, str]],
        *,
        require_label: bool,
        transform: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.h5_root = self.data_root / "h5" / "v1"
        self.require_label = require_label
        self.transform = transform
        self.records = records
        self.samples = self._expand_samples(records)

    def _expand_samples(self, records: list[dict[str, str]]) -> list[SliceRecord]:
        samples: list[SliceRecord] = []
        for row in records:
            path = self.h5_root / _safe_relative_path(row["image_h5_relpath"], field="image_h5_relpath")
            with h5py.File(path, "r") as handle:
                image = handle.get("image")
                if image is None:
                    raise ValueError(f"missing image dataset: {path}")
                shape = tuple(image.shape)
            if row.get("dataset") == "fundus":
                if len(shape) != 3 or shape[0] != 3:
                    raise ValueError(f"fundus image must be [3,H,W], got {shape} for {row.get('case_id')}")
                samples.append(SliceRecord(row=row, slice_index=None))
                continue
            if len(shape) != 3:
                raise ValueError(f"MRI image must be [Z,H,W], got {shape} for {row.get('case_id')}")
            samples.extend(SliceRecord(row=row, slice_index=index) for index in range(shape[0]))
        if not samples:
            raise ValueError("selected records expand to zero 2D samples")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _to_image_tensor(array: np.ndarray, dataset: str) -> torch.Tensor:
        image = np.asarray(array, dtype=np.float32)
        if dataset == "fundus":
            image = image / 255.0
        elif image.ndim == 2:
            image = image[None, ...]
        if image.ndim != 3:
            raise ValueError(f"image must be [C,H,W], got {image.shape}")
        if not np.isfinite(image).all():
            raise ValueError("image contains non-finite values")
        return torch.from_numpy(np.ascontiguousarray(image))

    @staticmethod
    def _to_label_tensor(array: np.ndarray) -> torch.Tensor:
        label = np.asarray(array, dtype=np.int64)
        if label.ndim != 2:
            raise ValueError(f"label must be [H,W], got {label.shape}")
        return torch.from_numpy(np.ascontiguousarray(label))

    def _read_image(self, sample: SliceRecord) -> torch.Tensor:
        path = self.h5_root / _safe_relative_path(sample.row["image_h5_relpath"], field="image_h5_relpath")
        with h5py.File(path, "r") as handle:
            image = handle["image"]
            array = image[...] if sample.slice_index is None else image[sample.slice_index]
        return self._to_image_tensor(array, sample.row.get("dataset", ""))

    def _read_label(self, sample: SliceRecord) -> torch.Tensor:
        label_rel = sample.row.get("label_h5_relpath", "")
        if not label_rel:
            raise RuntimeError(f"attempted to access a hidden label for {sample.row.get('case_id')}")
        path = self.h5_root / _safe_relative_path(label_rel, field="label_h5_relpath")
        with h5py.File(path, "r") as handle:
            label = handle["label"]
            array = label[...] if sample.slice_index is None else label[sample.slice_index]
        return self._to_label_tensor(array)

    @staticmethod
    def _metadata(sample: SliceRecord) -> dict[str, Any]:
        row = sample.row
        return {
            "case_id": row["case_id"],
            "patient_id": row.get("patient_id") or row["case_id"],
            "site": _site_from_row(row),
            "dataset": row.get("dataset", ""),
            "slice_index": sample.slice_index,
        }


class H5LabeledDataset(_H5SliceDataset):
    """Visible-GT dataset for labelled training, validation, or test records."""

    def __init__(
        self,
        data_root: Path,
        *,
        seed: int,
        dataset: str,
        sites: Iterable[str] | None = None,
        roles: Iterable[str] = ("train_labeled",),
        transform: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        records = load_training_records(data_root, seed=seed, dataset=dataset, sites=sites, roles=roles)
        if any(not row.get("label_h5_relpath", "") for row in records):
            raise ValueError("labeled dataset received a record without a visible label path")
        super().__init__(data_root, records, require_label=True, transform=transform)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self._read_image(sample)
        label = self._read_label(sample)
        valid_mask = torch.ones((1, *label.shape), dtype=torch.bool)
        if self.transform is not None:
            transformed = self.transform(image=image, label=label, valid_mask=valid_mask)
            image, label, valid_mask = transformed["image"], transformed["label"], transformed["valid_mask"]
        return {"image": image, "label": label, "valid_mask": valid_mask, **self._metadata(sample)}


class H5UnlabeledDataset(_H5SliceDataset):
    """Image-only training dataset that cannot expose a label path or tensor."""

    def __init__(
        self,
        data_root: Path,
        *,
        seed: int,
        dataset: str,
        sites: Iterable[str] | None = None,
        transform: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        records = load_training_records(
            data_root,
            seed=seed,
            dataset=dataset,
            sites=sites,
            roles=("train_unlabeled",),
        )
        for row in records:
            if any(row.get(field, "") for field in _FORBIDDEN_UNLABELED_FIELDS if field != "label_h5_relpath"):
                raise ValueError(f"forbidden unlabeled metadata in {row.get('case_id')}")
            if row.get("label_h5_relpath", ""):
                raise ValueError(f"hidden label leaked for {row.get('case_id')}")
        super().__init__(data_root, records, require_label=False, transform=transform)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self._read_image(sample)
        if self.transform is None:
            valid_mask = torch.ones((1, *image.shape[-2:]), dtype=torch.bool)
            transformed = {
                "weak_image": image,
                "strong_image": image.clone(),
                "strong_valid_mask": valid_mask,
                "geometry_record": {"hflip": False, "vflip": False},
            }
        else:
            transformed = self.transform(image=image)
        prohibited = _FORBIDDEN_UNLABELED_FIELDS.intersection(transformed)
        if prohibited:
            raise RuntimeError(f"unlabeled transform returned prohibited fields: {sorted(prohibited)}")
        return {**transformed, **self._metadata(sample)}
