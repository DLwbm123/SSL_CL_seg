from __future__ import annotations

import csv

import pytest

from lcrseg.data import H5UnlabeledDataset, WeakStrongTransform, collate_unlabeled

from .conftest import make_synthetic_root


def test_unlabeled_batch_has_no_label(tmp_path) -> None:
    root = make_synthetic_root(tmp_path)
    dataset = H5UnlabeledDataset(
        root,
        seed=0,
        dataset="fundus",
        transform=WeakStrongTransform(flip_probability=0.0, cutout_probability=0.0),
    )
    batch = collate_unlabeled([dataset[0]])
    assert not hasattr(batch, "label")
    assert "label" not in dataset[0]


def test_unlabeled_manifest_label_path_is_rejected(tmp_path) -> None:
    root = make_synthetic_root(tmp_path)
    manifest = root / "manifests" / "training" / "lcrseg_v1_seed0.csv"
    rows = list(csv.DictReader(manifest.open()))
    rows[2]["label_h5_relpath"] = "labels/fundus/SITE/case2.h5"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="hidden label"):
        H5UnlabeledDataset(root, seed=0, dataset="fundus")
