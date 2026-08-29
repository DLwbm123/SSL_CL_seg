#!/usr/bin/env python3
"""Characterize the frozen Fundus seed splits without creating or changing any split."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.v0_4 import RUN_NAMES, load_frozen_method, spherical_kmeans, stable_seed
from lcrseg.common import read_csv, sha256_path, write_csv, write_json


def _boundary_length(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return int(np.sum(mask & ~eroded))


def _records(root: Path, seed: int) -> list[dict[str, str]]:
    rows = read_csv(root / "manifests" / "training" / f"lcrseg_v1_seed{seed}.csv")
    selected = [row for row in rows if row.get("dataset") == "fundus"]
    if not selected:
        raise RuntimeError(f"no frozen Fundus rows for seed {seed}")
    return selected


def _h5(root: Path, relative: str, key: str) -> np.ndarray:
    path = root / "h5" / "v1" / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[key])


def _sample(features: list[np.ndarray], maximum: int, seed: int) -> np.ndarray:
    value = np.concatenate(features, axis=0)
    if len(value) <= maximum:
        return value
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(value), size=maximum, replace=False))
    return value[chosen]


@torch.no_grad()
def _feature_modes(
    *, root: Path, seed: int, labeled: list[dict[str, str]], device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint = root / "runs" / RUN_NAMES[(seed, "R0")] / "checkpoint_final_site0_REFUGE.pt"
    method, _ = load_frozen_method(checkpoint, device)
    features_by_site_class: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for row in labeled:
        image = _h5(root, row["image_h5_relpath"], "image").astype(np.float32) / 255.0
        label = _h5(root, row["label_h5_relpath"], "label").astype(np.int64)
        output = method.model(torch.from_numpy(image)[None].to(device))
        feature = F.normalize(output.relation_features.float(), p=2, dim=1, eps=1.0e-8)[0]
        grid_label = F.interpolate(
            torch.from_numpy(label)[None, None].float(), size=feature.shape[-2:], mode="nearest"
        )[0, 0].long()
        flat_feature = feature.permute(1, 2, 0).cpu().numpy()
        flat_label = grid_label.cpu().numpy()
        for class_id in (1, 2):
            selected = flat_feature[flat_label == class_id]
            if len(selected):
                features_by_site_class[(row["site_or_vendor"], class_id)].append(selected)
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for class_id in (1, 2):
        site_features: dict[str, np.ndarray] = {}
        for site in ("REFUGE", "RIM_ONE_r3", "Drishti_GS"):
            chunks = features_by_site_class[(site, class_id)]
            if not chunks:
                raise RuntimeError(f"no labeled features for seed {seed}, site {site}, class {class_id}")
            site_features[site] = _sample(
                chunks, 200_000, stable_seed("v0.4-split-feature", seed, site, class_id)
            )
        combined = np.concatenate(list(site_features.values()), axis=0)
        labels, centers, objective = spherical_kmeans(
            combined,
            k=4,
            seed=stable_seed("v0.4-split-mode", seed, class_id),
            restarts=5,
        )
        offset = 0
        for site, value in site_features.items():
            site_labels = labels[offset : offset + len(value)]
            offset += len(value)
            counts = np.bincount(site_labels, minlength=4)
            for cluster, count in enumerate(counts):
                rows.append(
                    {
                        "record_type": "labeled_feature_mode",
                        "seed": seed,
                        "site": site,
                        "class": class_id,
                        "cluster": cluster,
                        "count": int(count),
                        "occupancy": float(count / max(1, len(value))),
                        "k": 4,
                    }
                )
        details[str(class_id)] = {
            "objective": objective,
            "centers": centers,
            "site_sample_counts": {site: len(value) for site, value in site_features.items()},
        }
    anchor = method.current_anchor_bank.anchors[:, 0].detach().cpu().numpy().astype(np.float32)
    return rows, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "anchor_vectors": anchor,
        "feature_modes": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "split_characterization.csv"
    json_path = output_dir / "split_characterization.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("refusing to overwrite completed V0.4 split characterization")
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "protocol_id": "lcrseg_v0_4_failure_audit",
        "status": "complete",
        "split_mutation": False,
        "seeds": {},
    }
    for seed in (0, 1, 2):
        records = _records(root, seed)
        labeled = [row for row in records if row.get("primary_20pct_split") == "train_labeled"]
        split_sha = sha256_path(root / "splits" / f"fundus_seed{seed}.json")
        manifest_sha = sha256_path(root / "manifests" / "training" / f"lcrseg_v1_seed{seed}.csv")
        for row in records:
            all_rows.append(
                {
                    "record_type": "patient_composition",
                    "seed": seed,
                    "site": row["site_or_vendor"],
                    "case_id": row["case_id"],
                    "patient_id": row["patient_id"],
                    "split": row["split"],
                    "primary_20pct_split": row["primary_20pct_split"],
                }
            )
        case_ids: list[str] = []
        for row in labeled:
            image = _h5(root, row["image_h5_relpath"], "image").astype(np.float32) / 255.0
            label = _h5(root, row["label_h5_relpath"], "label").astype(np.int64)
            disc = label > 0
            rim = label == 1
            cup = label == 2
            case_ids.append(row["case_id"])
            all_rows.append(
                {
                    "record_type": "labeled_case",
                    "seed": seed,
                    "site": row["site_or_vendor"],
                    "case_id": row["case_id"],
                    "patient_id": row["patient_id"],
                    "split": row["split"],
                    "primary_20pct_split": row["primary_20pct_split"],
                    "foreground_area_fraction": float(disc.mean()),
                    "rim_area_fraction": float(rim.mean()),
                    "cup_area_fraction": float(cup.mean()),
                    "cup_disc_ratio": float(cup.sum() / max(1, disc.sum())),
                    "rim_boundary_length": _boundary_length(rim),
                    "cup_boundary_length": _boundary_length(cup),
                    "intensity_mean_r": float(image[0].mean()),
                    "intensity_mean_g": float(image[1].mean()),
                    "intensity_mean_b": float(image[2].mean()),
                    "intensity_std_r": float(image[0].std()),
                    "intensity_std_g": float(image[1].std()),
                    "intensity_std_b": float(image[2].std()),
                }
            )
        mode_rows, anchor = _feature_modes(
            root=root, seed=seed, labeled=labeled, device=torch.device(args.device)
        )
        all_rows.extend(mode_rows)
        summary["seeds"][str(seed)] = {
            "split_sha256": split_sha,
            "manifest_sha256": manifest_sha,
            "labeled_case_ids": sorted(case_ids),
            "labeled_case_count": len(labeled),
            "patient_composition": dict(Counter(
                f"{row['site_or_vendor']}:{row['split']}:{row['primary_20pct_split']}" for row in records
            )),
            "initial_anchor": anchor,
        }
    write_csv(csv_path, all_rows)
    write_json(json_path, summary)
    print(json.dumps({"status": "complete", "rows": len(all_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
