#!/usr/bin/env python3
"""Validate formal training-only HDF5 loaders against frozen LCR-Seg inputs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.data import H5LabeledDataset, H5UnlabeledDataset, LabeledTransform, WeakStrongTransform, collate_labeled, collate_unlabeled  # noqa: E402

_DATASET_CLASSES = {"fundus": {0, 1, 2}, "prostate": {0, 1}, "mnms": {0, 1, 2, 3}}


def _readonly(path: Path) -> bool:
    return not bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _iter_limited(loader: DataLoader[Any], limit: int):
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        yield batch


def _verify_manifest(data_root: Path, seed: int) -> dict[str, Any]:
    manifest = data_root / "manifests" / "training" / f"lcrseg_v1_seed{seed}.csv"
    rows = list(csv.DictReader(manifest.open()))
    errors: list[str] = []
    patient_roles: dict[tuple[str, str], set[str]] = {}
    auxiliary_rows = 0
    for row in rows:
        if row.get("primary_20pct_split") == "train_unlabeled" and row.get("label_h5_relpath"):
            errors.append(f"hidden-label path for {row.get('case_id')}")
        if row.get("dataset") == "mnms":
            patient_roles.setdefault((row.get("patient_id", ""), row.get("dataset", "")), set()).add(row.get("primary_20pct_split", ""))
            if row.get("cohort") == "auxiliary25":
                auxiliary_rows += 1
                if row.get("primary_20pct_split") != "train_unlabeled" or row.get("evaluation_eligible", "").lower() != "false":
                    errors.append(f"invalid auxiliary25 role for {row.get('case_id')}")
    for (patient_id, _), roles in patient_roles.items():
        if len(roles) != 1:
            errors.append(f"M&Ms patient split leakage for {patient_id}: {sorted(roles)}")
    return {"rows": len(rows), "auxiliary_rows": auxiliary_rows, "errors": errors}


def _validate_dataset(data_root: Path, seed: int, dataset: str, workers: int, max_batches: int) -> dict[str, Any]:
    labeled = H5LabeledDataset(data_root, seed=seed, dataset=dataset, transform=LabeledTransform(flip_probability=0.0))
    unlabeled = H5UnlabeledDataset(
        data_root,
        seed=seed,
        dataset=dataset,
        transform=WeakStrongTransform(flip_probability=0.0, cutout_probability=0.0),
    )
    kwargs = {"batch_size": 1, "shuffle": False, "num_workers": workers, "persistent_workers": workers > 0}
    labeled_loader = DataLoader(labeled, collate_fn=collate_labeled, **kwargs)
    unlabeled_loader = DataLoader(unlabeled, collate_fn=collate_unlabeled, **kwargs)
    report: dict[str, Any] = {"dataset": dataset, "workers": workers, "labeled_samples": len(labeled), "unlabeled_samples": len(unlabeled)}
    labels_seen: set[int] = set()
    labeled_batches = 0
    for batch in _iter_limited(labeled_loader, min(max_batches, len(labeled_loader))):
        labeled_batches += 1
        labels_seen.update(int(value) for value in torch.unique(batch.label))
        if batch.image.shape[-2:] != batch.label.shape[-2:]:
            raise RuntimeError(f"labeled image/label mismatch for {dataset}")
        if batch.label.dtype != torch.long:
            raise RuntimeError(f"labeled dtype is not long for {dataset}")
    unlabeled_batches = 0
    for batch in _iter_limited(unlabeled_loader, min(max_batches, len(unlabeled_loader))):
        unlabeled_batches += 1
        if hasattr(batch, "label"):
            raise RuntimeError(f"hidden label is exposed by {dataset} unlabeled loader")
        if batch.weak_image.shape != batch.strong_image.shape:
            raise RuntimeError(f"weak/strong geometry mismatch for {dataset}")
        if batch.strong_valid_mask.shape[-2:] != batch.weak_image.shape[-2:]:
            raise RuntimeError(f"cutout valid-mask mismatch for {dataset}")
    if not labels_seen.issubset(_DATASET_CLASSES[dataset]):
        raise RuntimeError(f"unexpected labels for {dataset}: {sorted(labels_seen)}")
    report.update({"labeled_batches": labeled_batches, "unlabeled_batches": unlabeled_batches, "labels_seen": sorted(labels_seen)})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("LCRSEG_DATA_ROOT", "/home/jiangsuiyang/SSL_CL")))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("LCRSEG_RUN_ROOT", "/home/jiangsuiyang/SSL_CL/runs")))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--require-readonly", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    frozen = (root / "h5" / "v1", root / "manifests", root / "splits", root / "checksums")
    frozen_modes = {str(path.relative_to(root)): _readonly(path) for path in frozen}
    if args.require_readonly and not all(frozen_modes.values()):
        raise RuntimeError(f"frozen inputs are still writable: {frozen_modes}")
    manifest_report = _verify_manifest(root, args.seed)
    if manifest_report["errors"]:
        raise RuntimeError(f"manifest contract errors: {manifest_report['errors']}")
    reports = []
    for workers in (0, 4):
        for dataset in _DATASET_CLASSES:
            reports.append(_validate_dataset(root, args.seed, dataset, workers, args.max_batches))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "manifest": manifest_report,
        "frozen_modes": frozen_modes,
        "runs": reports,
    }
    output = args.run_root.resolve() / "m0" / f"loader_validation_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite loader report: {output}")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
