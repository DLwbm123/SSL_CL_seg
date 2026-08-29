#!/usr/bin/env python3
"""Exercise the isolated training HDF5 manifests with 0 and 4 workers."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.common import DATA_ROOT, utc_now, write_json  # noqa: E402
from lcrseg.h5_dataset import H5ManifestDataset  # noqa: E402


def _collate_one(batch: list[dict]) -> dict:
    # Batch size one intentionally accommodates the three frozen image shapes.
    return batch[0]


def _run(manifest: Path, h5_root: Path, *, workers: int, max_per_dataset: int) -> dict:
    result: dict[str, object] = {"num_workers": workers, "datasets": {}}
    for dataset in ("prostate", "mnms", "fundus"):
        source = H5ManifestDataset(manifest, h5_root, dataset=dataset, max_records=max_per_dataset)
        loader = DataLoader(source, batch_size=1, shuffle=False, num_workers=workers, collate_fn=_collate_one, persistent_workers=False)
        seen: list[str] = []
        labels = Counter()
        shapes = Counter()
        for batch in loader:
            image = batch["image"]
            if not torch.isfinite(torch.as_tensor(image, dtype=torch.float32)).all():
                raise RuntimeError(f"{dataset}: non-finite image in {batch['case_id']}")
            if batch["primary_20pct_split"] == "train_unlabeled":
                if batch["has_label"] or batch["label"] is not None:
                    raise RuntimeError(f"{dataset}: train_unlabeled label was opened for {batch['case_id']}")
                labels["hidden"] += 1
            else:
                if not batch["has_label"] or batch["label"] is None:
                    raise RuntimeError(f"{dataset}: expected label absent for {batch['case_id']}")
                labels["visible"] += 1
            seen.append(batch["case_id"])
            shapes[str(tuple(image.shape))] += 1
        result["datasets"][dataset] = {
            "records": len(seen),
            "unique_case_ids": len(set(seen)),
            "label_access": dict(labels),
            "image_shapes": dict(shapes),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-per-dataset", type=int, default=16)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 4])
    args = parser.parse_args()
    manifest = args.root / "manifests" / "training" / f"lcrseg_v1_seed{args.seed}.csv"
    h5_root = args.root / "h5" / "v1"
    if args.max_per_dataset < 2:
        raise SystemExit("--max-per-dataset must be at least 2")
    runs = [_run(manifest, h5_root, workers=workers, max_per_dataset=args.max_per_dataset) for workers in args.workers]
    payload = {
        "generated_at": utc_now(),
        "manifest": manifest.relative_to(args.root).as_posix(),
        "max_per_dataset": args.max_per_dataset,
        "runs": runs,
        "valid": True,
    }
    write_json(args.root / "reports" / "validation" / "dataloader_smoke.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
