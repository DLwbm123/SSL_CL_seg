"""Training-manifest-only HDF5 dataset used by smoke tests and later baselines."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from torch.utils.data import Dataset

from .common import read_csv


class H5ManifestDataset(Dataset[dict[str, Any]]):
    """Read a training manifest without consulting any diagnostics manifest."""

    def __init__(self, manifest_path: Path, h5_root: Path, *, dataset: str | None = None, max_records: int | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.h5_root = Path(h5_root)
        rows = read_csv(self.manifest_path)
        if dataset is not None:
            rows = [row for row in rows if row.get("dataset") == dataset]
        if not rows:
            raise ValueError(f"no rows selected from training manifest {self.manifest_path}")
        selected = self._smoke_selection(rows, max_records)
        for row in selected:
            image_rel = row.get("image_h5_relpath", "")
            if not image_rel or Path(image_rel).is_absolute() or ".." in Path(image_rel).parts:
                raise ValueError(f"unsafe image HDF5 path for {row.get('case_id')}")
            unlabeled = row.get("primary_20pct_split") == "train_unlabeled"
            if unlabeled and row.get("label_h5_relpath"):
                raise ValueError(f"hidden label leaked for {row.get('case_id')}")
        self.rows = selected

    @staticmethod
    def _smoke_selection(rows: list[dict[str, str]], max_records: int | None) -> list[dict[str, str]]:
        if max_records is None or len(rows) <= max_records:
            return rows
        # Include both the deterministic leading records and one unlabeled row
        # whenever it exists, so the no-label path is exercised in each smoke run.
        selected = rows[:max_records]
        if not any(row.get("primary_20pct_split") == "train_unlabeled" for row in selected):
            unlabeled = next((row for row in rows if row.get("primary_20pct_split") == "train_unlabeled"), None)
            if unlabeled is not None:
                selected[-1] = unlabeled
        return selected

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self.h5_root / row["image_h5_relpath"]
        with h5py.File(image_path, "r") as handle:
            image = np.asarray(handle["image"])
        label_path = row.get("label_h5_relpath", "")
        label: np.ndarray | None = None
        if label_path:
            with h5py.File(self.h5_root / label_path, "r") as handle:
                label = np.asarray(handle["label"])
        return {
            "image": image,
            "label": label,
            "has_label": label is not None,
            "case_id": row["case_id"],
            "dataset": row["dataset"],
            "primary_20pct_split": row["primary_20pct_split"],
        }
