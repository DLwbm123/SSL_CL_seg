#!/usr/bin/env python3
"""Compile the three immutable ASPR feasibility shards and apply gates A-D."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.common import read_csv, sha256_path, write_csv, write_json, write_text  # noqa: E402


FOREGROUND_IDS = (1, 2)
FILENAMES = (
    "memory_selection_quality.csv",
    "prototype_drift.csv",
    "transport_quality.csv",
    "site_mode_utility.csv",
)


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _finite(values: list[float]) -> bool:
    return bool(values) and all(math.isfinite(value) for value in values)


def _quantile(values: list[float], probability: float) -> float:
    if not _finite(values):
        raise RuntimeError("gate input is empty or non-finite")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _merge(input_root: Path) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    combined = {filename: [] for filename in FILENAMES}
    seed_summaries: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        seed_dir = input_root / f"seed{seed}"
        summary_path = seed_dir / "feasibility_seed_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "ASPR_FEASIBILITY_SEED_AUDIT_COMPLETE" or summary.get("hidden_gt_usage") != "post_hoc_only":
            raise ValueError(f"invalid seed feasibility summary: {summary_path}")
        seed_summaries.append(summary)
        for filename in FILENAMES:
            path = seed_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            combined[filename].extend(read_csv(path))
    return combined, seed_summaries


def _gate_a(rows: list[dict[str, str]]) -> dict[str, Any]:
    aggregate = [row for row in rows if row["site_id"] == "ALL"]
    pairs = [row for row in rows if row["site_id"] != "ALL"]
    precision_counts: dict[str, int] = {}
    coverage_counts: dict[str, int] = {}
    for class_id in FOREGROUND_IDS:
        selected = [row for row in aggregate if int(row["class_id"]) == class_id]
        precision_counts[str(class_id)] = sum(_float(row, "selected_precision") >= 0.90 for row in selected)
        coverage_counts[str(class_id)] = sum(_float(row, "selected_foreground_coverage") >= 0.02 for row in selected)
    deltas = [_float(row, "delta_lu") for row in pairs]
    metrics = {
        "precision_seed_pass_count_by_class": precision_counts,
        "coverage_seed_pass_count_by_class": coverage_counts,
        "median_delta_lu": float(np.median(deltas)),
        "fraction_delta_lu_nonnegative": float(np.mean(np.asarray(deltas) >= 0.0)),
        "p10_delta_lu": _quantile(deltas, 0.10),
        "site_class_seed_pairs": len(deltas),
    }
    checks = {
        "precision_ge_0_90_in_at_least_2_of_3_by_class": all(value >= 2 for value in precision_counts.values()),
        "coverage_ge_0_02_in_at_least_2_of_3_by_class": all(value >= 2 for value in coverage_counts.values()),
        "median_delta_lu_ge_0_005": metrics["median_delta_lu"] >= 0.005,
        "fraction_delta_lu_nonnegative_ge_0_70": metrics["fraction_delta_lu_nonnegative"] >= 0.70,
        "p10_delta_lu_ge_minus_0_010": metrics["p10_delta_lu"] >= -0.010,
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _gate_b(rows: list[dict[str, str]]) -> dict[str, Any]:
    proposed = [row for row in rows if row["memory_source"] == "combined"]
    degradation = [_float(row, "cosine_degradation") for row in proposed]
    metrics = {
        "median_static_cosine_degradation": float(np.median(degradation)),
        "fraction_degradation_ge_0_050": float(np.mean(np.asarray(degradation) >= 0.050)),
        "pairs": len(degradation),
        "memory_source": "combined_labeled_plus_reliable_unlabeled",
    }
    checks = {
        "median_degradation_ge_0_030_or_fraction_ge_0_050_at_least_0_50": metrics[
            "median_static_cosine_degradation"
        ]
        >= 0.030
        or metrics["fraction_degradation_ge_0_050"] >= 0.50
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _gate_c(rows: list[dict[str, str]]) -> dict[str, Any]:
    proposed = [row for row in rows if row["memory_source"] == "combined"]
    delta = [_float(row, "shrinkage_minus_static") for row in proposed]
    shrink = [_float(row, "shrinkage_oracle_cosine") for row in proposed]
    full = [_float(row, "full_shift_oracle_cosine") for row in proposed]
    metrics = {
        "median_shrinkage_minus_static": float(np.median(delta)),
        "fraction_shrinkage_gt_static": float(np.mean(np.asarray(delta) > 0.0)),
        "p10_shrinkage_minus_static": _quantile(delta, 0.10),
        "median_shrinkage_oracle_cosine": float(np.median(shrink)),
        "median_full_shift_oracle_cosine": float(np.median(full)),
        "pairs": len(delta),
        "memory_source": "combined_labeled_plus_reliable_unlabeled",
    }
    checks = {
        "median_shrinkage_minus_static_ge_0_020": metrics["median_shrinkage_minus_static"] >= 0.020,
        "fraction_shrinkage_gt_static_ge_0_70": metrics["fraction_shrinkage_gt_static"] >= 0.70,
        "p10_delta_ge_minus_0_020": metrics["p10_shrinkage_minus_static"] >= -0.020,
        "shrinkage_median_ge_full_shift_median_minus_0_010": metrics["median_shrinkage_oracle_cosine"]
        >= metrics["median_full_shift_oracle_cosine"] - 0.010,
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _gate_d(rows: list[dict[str, str]]) -> dict[str, Any]:
    proposed = [row for row in rows if row["memory_source"] == "combined"]
    reductions = [_float(row, "nearest_distance_reduction") for row in proposed]
    accuracy_pairs = {
        (int(row["seed"]), _float(row, "global_ncm_accuracy_all_foreground"), _float(row, "max_over_site_ncm_accuracy_all_foreground"))
        for row in proposed
    }
    own_counts: dict[str, int] = {}
    for class_id in FOREGROUND_IDS:
        selected = [row for row in proposed if int(row["class_id"]) == class_id]
        own_counts[str(class_id)] = sum(_float(row, "own_site_top_mode_rate") >= 0.55 for row in selected)
    occupancy_values = [int(row["site0_occupancy"]) for row in proposed] + [int(row["site1_occupancy"]) for row in proposed]
    metrics = {
        "median_nearest_distance_reduction": float(np.median(reductions)),
        "minimum_max_site_minus_global_accuracy": min(maximum - global_value for _, global_value, maximum in accuracy_pairs),
        "own_site_seed_pass_count_by_class": own_counts,
        "minimum_historical_site_occupancy": min(occupancy_values),
        "seed_class_rows": len(proposed),
        "memory_source": "combined_labeled_plus_reliable_unlabeled",
    }
    checks = {
        "median_nearest_distance_reduction_ge_0_10": metrics["median_nearest_distance_reduction"] >= 0.10,
        "max_over_site_accuracy_ge_global_minus_0_005": metrics["minimum_max_site_minus_global_accuracy"] >= -0.005,
        "own_site_rate_ge_0_55_in_at_least_2_of_3_by_class": all(value >= 2 for value in own_counts.values()),
        "all_historical_prototypes_nonzero_occupancy": metrics["minimum_historical_site_occupancy"] > 0,
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ASPR-Seg V0.1 feasibility audit",
        "",
        f"**Status:** `{report['status']}`  ",
        "**Hidden GT:** `post_hoc_only`  ",
        "**Optimizer steps:** `0`",
        "",
        "## Gates",
        "",
        "| Gate | Result | Key metrics |",
        "|---|---|---|",
    ]
    labels = {
        "A_unlabeled_memory": "Reliable unlabeled memory",
        "B_prototype_drift": "Static prototype drift",
        "C_transport": "Evidence-adaptive transport",
        "D_site_mode": "Site-mode utility",
    }
    for key, label in labels.items():
        gate = report["gates"][key]
        lines.append(f"| {label} | {'PASS' if gate['passed'] else 'FAIL'} | `{json.dumps(gate['metrics'], sort_keys=True)}` |")
    lines.extend(
        [
            "",
            "## Protocol consequence",
            "",
            report["protocol_consequence"],
            "",
            "All three seeds were reconstructed independently from frozen R0 checkpoints. Hidden diagnostic labels were read only by the audit script after reconstruction; they were not present in memory builders, calibrators, or training loaders.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--status-dir", type=Path, required=True)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    analysis_dir = args.analysis_dir.resolve()
    status_dir = args.status_dir.resolve()
    canonical = {filename: analysis_dir / filename for filename in FILENAMES}
    json_path = status_dir / "ASPR_FEASIBILITY_AUDIT.json"
    markdown_path = status_dir / "ASPR_FEASIBILITY_AUDIT.md"
    if any(path.exists() for path in (*canonical.values(), json_path, markdown_path)):
        raise FileExistsError("refusing to overwrite canonical ASPR feasibility artifacts")
    combined, seed_summaries = _merge(input_root)
    engineering_checks = {
        "three_seed_summaries": len(seed_summaries) == 3,
        "hidden_gt_post_hoc_only": all(summary["hidden_gt_usage"] == "post_hoc_only" for summary in seed_summaries),
        "optimizer_steps_zero": all(int(summary["optimizer_steps"]) == 0 for summary in seed_summaries),
        "memory_rows_complete": len(combined["memory_selection_quality.csv"]) == 24,
        "drift_rows_complete": len(combined["prototype_drift.csv"]) == 36,
        "transport_rows_complete": len(combined["transport_quality.csv"]) == 36,
        "site_mode_rows_complete": len(combined["site_mode_utility.csv"]) == 12,
    }
    if not all(engineering_checks.values()):
        status = "HARD_STOP_ASPR_FEASIBILITY_ENGINEERING"
        gates = {}
    else:
        gates = {
            "A_unlabeled_memory": _gate_a(combined["memory_selection_quality.csv"]),
            "B_prototype_drift": _gate_b(combined["prototype_drift.csv"]),
            "C_transport": _gate_c(combined["transport_quality.csv"]),
            "D_site_mode": _gate_d(combined["site_mode_utility.csv"]),
        }
        if not gates["A_unlabeled_memory"]["passed"]:
            status = "ASPR_UNLABELED_MEMORY_NOT_SUPPORTED"
        elif not gates["B_prototype_drift"]["passed"]:
            status = "ASPR_PROTOTYPE_DRIFT_NOT_SUPPORTED"
        elif not gates["C_transport"]["passed"]:
            status = "ASPR_TRANSPORT_NOT_SUPPORTED"
        elif not gates["D_site_mode"]["passed"]:
            status = "ASPR_SITE_MODE_NOT_SUPPORTED"
        else:
            status = "ASPR_FEASIBILITY_SUPPORTED"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in combined.items():
        write_csv(canonical[filename], rows)
    consequence = (
        "Part B method implementation and training are authorized by the preregistered feasibility gate."
        if status == "ASPR_FEASIBILITY_SUPPORTED"
        else "Protocol hard stop: do not implement or train ASPR V0.1. A new protocol is required to change the proposed method line."
    )
    report = {
        "protocol_id": "asprseg_v0_1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "hidden_gt_usage": "post_hoc_only",
        "optimizer_steps": 0,
        "engineering_checks": engineering_checks,
        "gates": gates,
        "protocol_consequence": consequence,
        "seed_summaries": [
            {"seed": summary["seed"], "path": str(input_root / f"seed{summary['seed']}" / "feasibility_seed_summary.json")}
            for summary in seed_summaries
        ],
        "artifacts": {
            filename: {"path": str(path), "sha256": sha256_path(path)} for filename, path in canonical.items()
        },
    }
    write_json(json_path, report)
    write_text(markdown_path, _markdown(report))
    print(json.dumps({"status": status, "json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0 if status == "ASPR_FEASIBILITY_SUPPORTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
