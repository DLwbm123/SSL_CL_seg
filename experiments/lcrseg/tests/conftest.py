from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np


def make_synthetic_root(root: Path, *, records: int = 4) -> Path:
    data_root = root / "data"
    (data_root / "h5" / "v1" / "images" / "fundus" / "SITE").mkdir(parents=True)
    (data_root / "h5" / "v1" / "labels" / "fundus" / "SITE").mkdir(parents=True)
    (data_root / "manifests" / "training").mkdir(parents=True)
    (data_root / "splits").mkdir(parents=True)
    (data_root / "checksums").mkdir(parents=True)
    (data_root / "h5" / "v1" / "FROZEN").write_text("frozen\n")
    fields = [
        "case_id",
        "patient_id",
        "dataset",
        "primary_20pct_split",
        "site_or_vendor",
        "image_h5_relpath",
        "label_h5_relpath",
    ]
    rows = []
    for index in range(records):
        case_id = f"case{index}"
        image_rel = f"images/fundus/SITE/{case_id}.h5"
        label_rel = f"labels/fundus/SITE/{case_id}.h5"
        image = np.zeros((3, 32, 32), dtype=np.uint8)
        image[:, 8:24, 8:24] = index + 1
        label = np.zeros((32, 32), dtype=np.uint8)
        label[8:16, 8:16] = 1
        label[16:24, 16:24] = 2
        with h5py.File(data_root / "h5" / "v1" / image_rel, "w") as handle:
            handle.create_dataset("image", data=image)
        with h5py.File(data_root / "h5" / "v1" / label_rel, "w") as handle:
            handle.create_dataset("label", data=label)
        role = "train_labeled" if index < 2 else "train_unlabeled"
        rows.append(
            {
                "case_id": case_id,
                "patient_id": case_id,
                "dataset": "fundus",
                "primary_20pct_split": role,
                "site_or_vendor": "SITE",
                "image_h5_relpath": image_rel,
                "label_h5_relpath": label_rel if role == "train_labeled" else "",
            }
        )
    with (data_root / "manifests" / "training" / "lcrseg_v1_seed0.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (data_root / "splits" / "fundus_seed0.json").write_text("{}\n")
    return data_root
