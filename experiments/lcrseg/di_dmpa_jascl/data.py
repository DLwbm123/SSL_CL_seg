from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .manifest import ManifestRecord


@dataclass(frozen=True)
class SliceRef:
    record: ManifestRecord
    slice_index: int | None


def _translation(tensor: torch.Tensor, dx: int, dy: int, fill: float) -> torch.Tensor:
    output = torch.full_like(tensor, fill)
    height, width = tensor.shape[-2:]
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    dst_x0, dst_x1 = max(0, dx), min(width, width + dx)
    dst_y0, dst_y1 = max(0, dy), min(height, height + dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        output[..., dst_y0:dst_y1, dst_x0:dst_x1] = tensor[..., src_y0:src_y1, src_x0:src_x1]
    return output


class UpstreamPairedTransform:
    """JASCL's labeled resize, horizontal flip, and at-most-two-pixel translation."""

    def __init__(self, output_hw: tuple[int, int], *, augment: bool) -> None:
        self.output_hw = tuple(int(value) for value in output_hw)
        self.augment = bool(augment)

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.shape[-2:] != self.output_hw:
            image = F.interpolate(image.unsqueeze(0), self.output_hw, mode="bilinear", align_corners=False).squeeze(0)
            label = F.interpolate(label[None, None].float(), self.output_hw, mode="nearest").squeeze(0).squeeze(0).long()
        if self.augment:
            if random.random() < 0.5:
                image = torch.flip(image, (-1,))
                label = torch.flip(label, (-1,))
            dx, dy = random.randint(-2, 2), random.randint(-2, 2)
            image = _translation(image, dx, dy, 0.0)
            label = _translation(label, dx, dy, 255)
        return image.contiguous(), label.contiguous()


class LCRSegH5Dataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        records: Iterable[ManifestRecord],
        *,
        require_label: bool,
        output_hw: tuple[int, int],
        augment: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.h5_root = self.data_root / "h5" / "v1"
        self.records = tuple(records)
        self.require_label = bool(require_label)
        self.transform = UpstreamPairedTransform(output_hw, augment=augment) if require_label else None
        if not self.records:
            raise ValueError("dataset requires at least one record")
        if not require_label and any(record.label_h5_relpath is not None for record in self.records):
            raise RuntimeError("unlabeled dataset received a visible label path")
        self.samples = self._expand()

    def _expand(self) -> tuple[SliceRef, ...]:
        samples: list[SliceRef] = []
        for record in self.records:
            with h5py.File(self.h5_root / record.image_h5_relpath, "r") as handle:
                shape = tuple(handle["image"].shape)
            if record.dataset == "fundus":
                if len(shape) != 3 or shape[0] != 3:
                    raise ValueError(f"invalid Fundus image shape: {shape}")
                samples.append(SliceRef(record, None))
            else:
                if len(shape) != 3:
                    raise ValueError(f"invalid MRI image shape: {shape}")
                samples.extend(SliceRef(record, index) for index in range(shape[0]))
        return tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with h5py.File(self.h5_root / sample.record.image_h5_relpath, "r") as handle:
            dataset = handle["image"]
            image = dataset[...] if sample.slice_index is None else dataset[sample.slice_index]
        image_array = np.asarray(image, dtype=np.float32)
        if sample.record.dataset == "fundus":
            image_array = image_array / 255.0
        else:
            image_array = image_array[None]
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_array))
        item = {
            "image": image_tensor,
            "case_id": sample.record.case_id,
            "domain": sample.record.domain,
            "role": sample.record.role,
            "slice_index": sample.slice_index,
        }
        if self.require_label:
            if sample.record.label_h5_relpath is None:
                raise RuntimeError("attempted hidden GT access")
            with h5py.File(self.h5_root / sample.record.label_h5_relpath, "r") as handle:
                dataset = handle["label"]
                label = dataset[...] if sample.slice_index is None else dataset[sample.slice_index]
            label_tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(label, dtype=np.int64)))
            item["image"], item["label"] = self.transform(image_tensor, label_tensor)
        return item


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def batch_indices(
    size: int,
    batch_size: int,
    *,
    shuffle: bool,
    seed_parts: tuple[object, ...],
    start_batch: int = 0,
) -> Iterator[tuple[int, list[int]]]:
    if size <= 0 or batch_size <= 0:
        raise ValueError("size and batch_size must be positive")
    if shuffle:
        generator = torch.Generator().manual_seed(stable_seed(*seed_parts))
        indices = torch.randperm(size, generator=generator).tolist()
    else:
        indices = list(range(size))
    batches = [indices[offset : offset + batch_size] for offset in range(0, size, batch_size)]
    for batch_index in range(start_batch, len(batches)):
        yield batch_index, batches[batch_index]


def collate(dataset: Dataset, indices: Iterable[int], *, require_label: bool) -> dict:
    items = [dataset[index] for index in indices]
    batch = {
        "image": torch.stack([item["image"] for item in items]),
        "case_id": [item["case_id"] for item in items],
        "domain": [item["domain"] for item in items],
        "role": [item["role"] for item in items],
    }
    if require_label:
        batch["label"] = torch.stack([item["label"] for item in items]).long()
    elif any("label" in item for item in items):
        raise RuntimeError("hidden GT tensor entered an unlabeled batch")
    return batch
