#!/usr/bin/env python3
"""Compile the preregistered V0.4 precision/coverage audit from frozen feature shards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.v0_4 import jensen_shannon
from lcrseg.common import sha256_path, write_csv, write_json


def _safe_mean(value: np.ndarray) -> float:
    return float(np.mean(value)) if value.size else float("nan")


def _quantile(value: np.ndarray, q: float) -> float:
    return float(np.quantile(value, q)) if value.size else float("nan")


def _region_masks(distance: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(distance), dtype=bool),
        "boundary": np.abs(distance) <= 3.0,
        "interior": distance > 3.0,
        "exterior_near_boundary": (distance >= -3.0) & (distance < 0.0),
        "exterior_far": distance < -3.0,
    }


def _distribution_js(candidate: np.ndarray, admitted: np.ndarray, groups: np.ndarray) -> float:
    values = np.unique(groups[candidate])
    if not len(values):
        return float("nan")
    candidate_counts = np.asarray([np.sum(candidate & (groups == value)) for value in values], dtype=np.float64)
    admitted_counts = np.asarray([np.sum(admitted & (groups == value)) for value in values], dtype=np.float64)
    return jensen_shannon(candidate_counts, admitted_counts)


def _quantile_groups(value: np.ndarray, bins: int, *, positive_only: bool = False) -> np.ndarray:
    selected = value[value > 0] if positive_only else value
    if not selected.size:
        return np.zeros(len(value), dtype=np.int16)
    edges = np.unique(np.quantile(selected, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    groups = np.digitize(value, edges, right=True).astype(np.int16)
    if positive_only:
        groups[value <= 0] = -1
    return groups


def _load_metadata(path: Path) -> dict[str, Any]:
    companion = path.with_suffix(".json")
    if not companion.is_file():
        raise FileNotFoundError(companion)
    metadata = json.loads(companion.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("hidden_gt_usage") != "post_hoc_only":
        raise RuntimeError(f"invalid V0.4 feature metadata: {companion}")
    if metadata.get("output_sha256") != sha256_path(path):
        raise RuntimeError(f"feature shard checksum mismatch: {path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=54)
    args = parser.parse_args()
    feature_paths = sorted(args.features_dir.resolve().glob("seed*_max200000.npz"))
    if len(feature_paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} complete shards, found {len(feature_paths)}")
    output_dir = args.output_dir.resolve()
    outputs = [
        output_dir / "precision_coverage.csv",
        output_dir / "boundary_coverage.csv",
        output_dir / "precision_coverage_summary.json",
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite completed V0.4 precision/coverage outputs")

    rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    shard_hashes: dict[str, str] = {}
    for path in feature_paths:
        metadata = _load_metadata(path)
        shard_hashes[path.name] = str(metadata["output_sha256"])
        with np.load(path) as shard:
            predicted = shard["predicted_class"].astype(np.int64)
            for class_id in (1, 2):
                class_mask = predicted == class_id
                if not class_mask.any():
                    raise RuntimeError(f"missing sampled foreground class {class_id}: {path}")
                correct = shard["correct"].astype(bool)[class_mask]
                admitted = shard["admitted"].astype(bool)[class_mask]
                deferred = shard["deferred"].astype(bool)[class_mask]
                distance = shard["boundary_distance"].astype(np.float64)[class_mask]
                component = shard["component_size"].astype(np.float64)[class_mask]
                learnability = shard["learnability"].astype(np.float64)[class_mask]
                anchor_margin = shard["anchor_margin"].astype(np.float64)[class_mask]
                logit_margin = shard["logit_margin"].astype(np.float64)[class_mask]
                spatial = shard["spatial_weight"].astype(np.float64)[class_mask]
                patient = shard["patient_id"][class_mask].astype(str)
                case = shard["case_id"][class_mask].astype(str)
                candidate = np.ones(len(correct), dtype=bool)
                base = {
                    "seed": int(metadata["seed"]),
                    "variant": str(metadata["variant"]),
                    "trained_through_site": str(metadata["trained_through_site"]),
                    "trained_through_site_index": int(metadata["trained_through_site_index"]),
                    "evaluation_site": str(metadata["evaluation_site"]),
                    "class": class_id,
                }
                regions = _region_masks(distance)
                states = {"candidate": candidate, "admitted": admitted, "deferred": deferred}
                for region_name, region_mask in regions.items():
                    correct_denominator = int(np.sum(region_mask & correct))
                    candidate_patient_count = len(np.unique(patient[region_mask]))
                    candidate_case_count = len(np.unique(case[region_mask]))
                    for state_name, state_mask in states.items():
                        selected = region_mask & state_mask
                        count = int(selected.sum())
                        rows.append(
                            {
                                **base,
                                "region": region_name,
                                "state": state_name,
                                "count": count,
                                "correct_count": int(np.sum(selected & correct)),
                                "pseudo_label_accuracy": _safe_mean(correct[selected]),
                                "precision": _safe_mean(correct[selected]),
                                "correct_candidate_recall": (
                                    float(np.sum(selected & correct) / correct_denominator)
                                    if correct_denominator
                                    else float("nan")
                                ),
                                "coverage_within_candidate": float(count / max(1, int(region_mask.sum()))),
                                "patient_coverage": float(len(np.unique(patient[selected])) / max(1, candidate_patient_count)),
                                "image_coverage": float(len(np.unique(case[selected])) / max(1, candidate_case_count)),
                                "boundary_distance_mean": _safe_mean(distance[selected]),
                                "boundary_distance_q25": _quantile(distance[selected], 0.25),
                                "boundary_distance_q50": _quantile(distance[selected], 0.50),
                                "boundary_distance_q75": _quantile(distance[selected], 0.75),
                                "component_size_mean": _safe_mean(component[selected]),
                                "component_size_q50": _quantile(component[selected], 0.50),
                                "learnability_mean": _safe_mean(learnability[selected]),
                                "logit_margin_mean": _safe_mean(logit_margin[selected]),
                                "anchor_margin_mean": _safe_mean(anchor_margin[selected]),
                                "spatial_consistency_mean": _safe_mean(spatial[selected]),
                            }
                        )
                boundary_rate = _safe_mean(admitted[regions["boundary"]])
                interior_rate = _safe_mean(admitted[regions["interior"]])
                region_group = np.full(len(distance), 3, dtype=np.int16)
                region_group[regions["boundary"]] = 0
                region_group[regions["interior"]] = 1
                region_group[regions["exterior_near_boundary"]] = 2
                component_group = _quantile_groups(component, 4, positive_only=True)
                learnability_group = _quantile_groups(learnability, 10)
                boundary_rows.append(
                    {
                        **base,
                        "boundary_candidate_count": int(regions["boundary"].sum()),
                        "interior_candidate_count": int(regions["interior"].sum()),
                        "boundary_admission_rate": boundary_rate,
                        "interior_admission_rate": interior_rate,
                        "boundary_coverage_ratio": (
                            float(boundary_rate / interior_rate)
                            if np.isfinite(interior_rate) and interior_rate > 0
                            else float("nan")
                        ),
                        "jsd_boundary_interior": _distribution_js(candidate, admitted, region_group),
                        "jsd_component_size_quartile": _distribution_js(candidate, admitted, component_group),
                        "jsd_patient": _distribution_js(candidate, admitted, patient),
                        "jsd_learnability_decile": _distribution_js(candidate, admitted, learnability_group),
                    }
                )
    write_csv(output_dir / "precision_coverage.csv", rows)
    write_csv(output_dir / "boundary_coverage.csv", boundary_rows)
    failed_seed_rows = [row for row in boundary_rows if int(row["seed"]) in (1, 2) and row["variant"] == "R1"]
    summary = {
        "protocol_id": "lcrseg_v0_4_failure_audit",
        "status": "complete",
        "hidden_gt_usage": "post_hoc_only",
        "source_shards": len(feature_paths),
        "source_sha256": shard_hashes,
        "group_rows": len(rows),
        "boundary_rows": len(boundary_rows),
        "gate_raw": {
            "failed_seed_boundary_ratio_below_0_80": [
                row for row in failed_seed_rows
                if np.isfinite(row["boundary_coverage_ratio"]) and float(row["boundary_coverage_ratio"]) < 0.80
            ],
            "failed_seed_group_jsd_at_least_0_10": [
                row for row in failed_seed_rows
                if max(
                    float(row[name]) for name in (
                        "jsd_boundary_interior",
                        "jsd_component_size_quartile",
                        "jsd_patient",
                        "jsd_learnability_decile",
                    ) if np.isfinite(row[name])
                ) >= 0.10
            ],
        },
        "definitions": {
            "boundary": "abs(signed processed-pixel distance) <= 3",
            "interior": "signed processed-pixel distance > 3",
            "exterior_near_boundary": "-3 <= signed processed-pixel distance < 0",
            "precision": "pseudo-label correctness among pixels in the state",
            "correct_candidate_recall": "correct pixels retained by the state divided by correct candidate pixels in the same group",
            "jsd_log_base": "natural",
        },
    }
    write_json(output_dir / "precision_coverage_summary.json", summary)
    print(json.dumps({"status": "complete", "rows": len(rows), "boundary_rows": len(boundary_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
