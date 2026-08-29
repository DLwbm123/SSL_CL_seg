#!/usr/bin/env python3
"""Compile deterministic spherical feature-mode coverage for the V0.4 audit."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.v0_4 import SITE_ORDER, jensen_shannon, spherical_kmeans, stable_seed
from lcrseg.common import sha256_path, write_csv, write_json


def _metadata(path: Path) -> dict[str, Any]:
    companion = path.with_suffix(".json")
    metadata = json.loads(companion.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("hidden_gt_usage") != "post_hoc_only":
        raise RuntimeError(f"invalid feature metadata: {companion}")
    if metadata.get("output_sha256") != sha256_path(path):
        raise RuntimeError(f"feature shard checksum mismatch: {path}")
    return metadata


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=54)
    parser.add_argument("--focus-variant", default="R1")
    args = parser.parse_args()
    paths = sorted(args.features_dir.resolve().glob("seed*_max200000.npz"))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} feature shards, found {len(paths)}")
    output_dir = args.output_dir.resolve()
    outputs = [
        output_dir / "mode_coverage_k2.csv",
        output_dir / "mode_coverage_k4.csv",
        output_dir / "mode_coverage_summary.json",
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite completed V0.4 mode-coverage outputs")

    grouped: dict[tuple[int, str, int, str, int], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    shard_hashes: dict[str, str] = {}
    for path in paths:
        metadata = _metadata(path)
        shard_hashes[path.name] = str(metadata["output_sha256"])
        for class_id in (1, 2):
            key = (
                int(metadata["seed"]),
                str(metadata["variant"]),
                int(metadata["trained_through_site_index"]),
                str(metadata["trained_through_site"]),
                class_id,
            )
            grouped[key].append((path, metadata))
    expected_groups = (args.expected_shards // len(SITE_ORDER)) * 2
    if len(grouped) != expected_groups:
        raise RuntimeError(f"expected {expected_groups} seed/variant/checkpoint/class groups, found {len(grouped)}")

    rows_by_k: dict[int, list[dict[str, Any]]] = {2: [], 4: []}
    group_summaries: list[dict[str, Any]] = []
    for key in sorted(grouped):
        seed, variant, site_index, trained_site, class_id = key
        sources = sorted(grouped[key], key=lambda item: SITE_ORDER.index(str(item[1]["evaluation_site"])))
        if [str(item[1]["evaluation_site"]) for item in sources] != list(SITE_ORDER):
            raise RuntimeError(f"mode group does not contain each evaluation site exactly once: {key}")
        feature_parts: list[np.ndarray] = []
        admitted_parts: list[np.ndarray] = []
        correct_parts: list[np.ndarray] = []
        boundary_parts: list[np.ndarray] = []
        site_parts: list[np.ndarray] = []
        anchor: np.ndarray | None = None
        for path, metadata in sources:
            with np.load(path) as shard:
                mask = shard["predicted_class"].astype(np.int64) == class_id
                feature_parts.append(shard["features"][mask].astype(np.float32))
                admitted_parts.append(shard["admitted"][mask].astype(bool))
                correct_parts.append(shard["correct"][mask].astype(bool))
                boundary_parts.append((np.abs(shard["boundary_distance"][mask].astype(np.float32)) <= 3.0))
                site_parts.append(np.full(int(mask.sum()), str(metadata["evaluation_site"])))
                shard_anchor = shard["anchors"].astype(np.float32)[class_id]
                if anchor is None:
                    anchor = shard_anchor
                elif not np.allclose(anchor, shard_anchor, atol=1.0e-6, rtol=1.0e-6):
                    raise RuntimeError(f"anchor mismatch across evaluation-site shards: {key}")
        features = np.concatenate(feature_parts, axis=0)
        admitted = np.concatenate(admitted_parts)
        correct = np.concatenate(correct_parts)
        boundary = np.concatenate(boundary_parts)
        site_origin = np.concatenate(site_parts)
        if anchor is None or not len(features):
            raise RuntimeError(f"empty feature-mode group: {key}")
        anchor = anchor / max(1.0e-12, float(np.linalg.norm(anchor)))
        for k in (2, 4):
            labels, centers, objective = spherical_kmeans(
                features,
                k=k,
                seed=stable_seed("v0.4-mode", seed, variant, site_index, class_id, k),
                restarts=5,
            )
            counts = np.bincount(labels, minlength=k)
            admitted_counts = np.bincount(labels[admitted], minlength=k)
            deferred_counts = counts - admitted_counts
            admission_rates = admitted_counts / np.maximum(1, counts)
            smallest = int(np.argmin(counts))
            largest = int(np.argmax(counts))
            overall_rate = float(admitted.mean())
            mode_jsd = jensen_shannon(counts.astype(np.float64), admitted_counts.astype(np.float64))
            smallest_rate = float(admission_rates[smallest])
            largest_rate = float(admission_rates[largest])
            smallest_ratio = float(smallest_rate / overall_rate) if overall_rate > 0 else float("nan")
            collapse_jsd = bool(np.isfinite(mode_jsd) and mode_jsd >= 0.10)
            collapse_gap = bool(smallest_rate < largest_rate - 0.20)
            collapse_retention = bool(smallest_rate < 0.50 * overall_rate)
            group_summary = {
                "seed": seed,
                "variant": variant,
                "trained_through_site_index": site_index,
                "trained_through_site": trained_site,
                "class": class_id,
                "k": k,
                "candidate_count": int(len(features)),
                "overall_admission_rate": overall_rate,
                "objective": objective,
                "mode_jsd": mode_jsd,
                "smallest_cluster": smallest,
                "largest_cluster": largest,
                "smallest_cluster_admission_rate": smallest_rate,
                "largest_cluster_admission_rate": largest_rate,
                "smallest_cluster_retention_ratio": smallest_ratio,
                "criterion_jsd": collapse_jsd,
                "criterion_admission_gap": collapse_gap,
                "criterion_retention": collapse_retention,
                "mode_coverage_problem": bool(collapse_jsd or collapse_gap or collapse_retention),
            }
            group_summaries.append(group_summary)
            for cluster in range(k):
                selected = labels == cluster
                selected_admitted = selected & admitted
                selected_deferred = selected & ~admitted
                site_composition = {
                    site: _safe_rate(int(np.sum(selected & (site_origin == site))), int(selected.sum()))
                    for site in SITE_ORDER
                }
                rows_by_k[k].append(
                    {
                        **group_summary,
                        "cluster": cluster,
                        "cluster_occupancy": int(selected.sum()),
                        "candidate_proportion": _safe_rate(int(selected.sum()), len(features)),
                        "admitted_proportion": _safe_rate(int(selected_admitted.sum()), int(admitted.sum())),
                        "deferred_proportion": _safe_rate(int(selected_deferred.sum()), int((~admitted).sum())),
                        "cluster_accuracy": float(correct[selected].mean()) if selected.any() else float("nan"),
                        "cluster_admission_rate": float(admitted[selected].mean()) if selected.any() else float("nan"),
                        "smallest_cluster_retention": smallest_rate,
                        "cluster_to_anchor_cosine": float(np.dot(centers[cluster], anchor)),
                        "site_origin_composition": site_composition,
                        "boundary_composition": float(boundary[selected].mean()) if selected.any() else float("nan"),
                    }
                )
    write_csv(output_dir / "mode_coverage_k2.csv", rows_by_k[2])
    write_csv(output_dir / "mode_coverage_k4.csv", rows_by_k[4])
    focus_variant = str(args.focus_variant).upper()
    failed = [row for row in group_summaries if row["variant"] == focus_variant and int(row["seed"]) in (1, 2)]
    seed_support = {
        str(seed): bool(any(row["mode_coverage_problem"] for row in group_summaries if row["variant"] == focus_variant and int(row["seed"]) == seed))
        for seed in (0, 1, 2)
    }
    summary = {
        "protocol_id": "lcrseg_v0_4_failure_audit",
        "status": "complete",
        "hidden_gt_usage": "post_hoc_only",
        "clustering": {"algorithm": "spherical_kmeans", "k": [2, 4], "restarts": 5, "device": "cpu"},
        "source_shards": len(paths),
        "source_sha256": shard_hashes,
        "focus_variant": focus_variant,
        "groups": group_summaries,
        "seed_support": seed_support,
        "support_count": int(sum(seed_support.values())),
        "mode_coverage_supported_2_of_3": bool(sum(seed_support.values()) >= 2),
        "failed_seed_problem_groups": failed,
        "definitions": {
            "smallest_cluster": "cluster with minimum candidate occupancy",
            "smallest_cluster_retention": "admission rate of the smallest candidate cluster",
            "mode_jsd": "natural-log Jensen-Shannon divergence of admitted and candidate cluster distributions",
        },
    }
    write_json(output_dir / "mode_coverage_summary.json", summary)
    print(json.dumps({"status": "complete", "groups": len(group_summaries), "seed_support": seed_support}, sort_keys=True))


if __name__ == "__main__":
    main()
