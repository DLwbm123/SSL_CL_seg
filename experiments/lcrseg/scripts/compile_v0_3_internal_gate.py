#!/usr/bin/env python3
"""Compile frozen V0.3 R0/R1 multi-seed artifacts and evaluate the internal gate."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.v0_3 import METRICS, aggregate_paired_seed_metrics
from lcrseg.common import write_csv, write_json, write_text


RUN_NAMES = {
    (0, "R0"): "fundus_seed0_lcrseg_uniform_relation_kd_full200e",
    (0, "R1"): "fundus_seed0_lcrseg_v0_2a_r1_progressive_uniform_full200e",
    (1, "R0"): "fundus_seed1_lcrseg_v0_3_r0_legacy_uniform_full200e",
    (1, "R1"): "fundus_seed1_lcrseg_v0_3_r1_progressive_uniform_full200e",
    (2, "R0"): "fundus_seed2_lcrseg_v0_3_r0_legacy_uniform_full200e",
    (2, "R1"): "fundus_seed2_lcrseg_v0_3_r1_progressive_uniform_full200e",
}
CLASS_NAMES = {1: "optic_disc_rim", 2: "optic_cup"}


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _summary(run_dir: Path, seed: int, variant: str) -> dict[str, Any]:
    path = run_dir / "run_summary.json"
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete" or int(payload.get("completed_global_steps", -1)) != 13400:
        raise RuntimeError(f"incomplete formal run: {run_dir}")
    identity_ok = payload.get("variant_id") == variant
    if seed == 0 and variant == "R0":
        identity_ok = payload.get("method") == "lcrseg_v0_1" and payload.get("variant_id") is None
    if int(payload.get("seed", -1)) != seed or not identity_ok:
        raise RuntimeError(f"formal run identity mismatch: {run_dir}")
    return payload


def _coverage(run_dir: Path) -> dict[str, Any]:
    source = run_dir / "admission_coverage.csv"
    if not source.is_file():
        source = run_dir / "branch_coverage.csv"
    grouped = defaultdict(lambda: [0.0, 0.0, 0.0])
    for row in _csv(source):
        valid = float(row["valid_count"])
        if valid <= 0:
            continue
        key = (row["site"], int(row["site_index"]), int(row["epoch"]), int(row["predicted_class"]))
        grouped[key][0] += valid
        grouped[key][1] += float(row["admitted_count"])
        grouped[key][2] += float(row["target_fraction"]) * valid
    errors = [abs(admitted / valid - target / valid) for valid, admitted, target in grouped.values()]
    return {
        "groups": len(errors),
        "maximum_absolute_error": max(errors, default=float("inf")),
        "passed": bool(errors) and max(errors) <= 0.05 + 1.0e-12,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/analysis/v0_3")
    parser.add_argument(
        "--status-json", type=Path, default=PROJECT_ROOT / "reports/experiment_status/V0_3_FUNDUS_INTERNAL_GATE.json"
    )
    parser.add_argument(
        "--status-md", type=Path, default=PROJECT_ROOT / "reports/experiment_status/V0_3_FUNDUS_INTERNAL_GATE.md"
    )
    parser.add_argument(
        "--completion-json",
        type=Path,
        default=PROJECT_ROOT / "reports/experiment_status/V0_3_FUNDUS_MULTISEED_COMPLETION.json",
    )
    parser.add_argument(
        "--completion-md",
        type=Path,
        default=PROJECT_ROOT / "reports/experiment_status/V0_3_FUNDUS_MULTISEED_COMPLETION.md",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    outputs = [
        output_dir / "fundus_seedwise_metrics.csv",
        output_dir / "fundus_paired_differences.csv",
        output_dir / "fundus_classwise_metrics.csv",
        args.status_json,
        args.status_md,
        args.completion_json,
        args.completion_md,
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite an existing V0.3 internal-gate artifact")
    admission_csv = output_dir / "fundus_admission_analysis.csv"
    admission_json = output_dir / "fundus_admission_analysis.json"
    if not admission_csv.is_file() or not admission_json.is_file():
        raise FileNotFoundError("frozen hidden-GT admission analysis must complete before the internal gate")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {key: (args.run_root / name).resolve() for key, name in RUN_NAMES.items()}
    summaries = {key: _summary(run_dirs[key], *key) for key in RUN_NAMES}
    seedwise_rows = []
    aggregate_input = []
    for (seed, variant), payload in sorted(summaries.items()):
        metrics = {metric: float(payload["summary"][metric]) for metric in METRICS}
        row = {"seed": seed, "variant": variant, "run_dir": str(run_dirs[(seed, variant)]), **metrics}
        seedwise_rows.append(row)
        aggregate_input.append(row)
    paired = aggregate_paired_seed_metrics(aggregate_input)
    paired_rows = paired["paired"]

    matrices = {key: _csv(run_dirs[key] / "site_matrix_long.csv") for key in RUN_NAMES}
    class_rows = []
    site_class_deltas: dict[tuple[str, int], list[float]] = defaultdict(list)
    overall_class_deltas: dict[int, list[float]] = defaultdict(list)
    for seed in (0, 1, 2):
        final_by_variant = {}
        for variant in ("R0", "R1"):
            rows = matrices[(seed, variant)]
            final_index = max(int(row["trained_site_index"]) for row in rows)
            final_by_variant[variant] = {
                row["evaluation_site"]: row for row in rows if int(row["trained_site_index"]) == final_index
            }
        if set(final_by_variant["R0"]) != set(final_by_variant["R1"]):
            raise RuntimeError(f"site-matrix mismatch for seed {seed}")
        for site in sorted(final_by_variant["R0"]):
            for class_id in (1, 2):
                r0 = float(final_by_variant["R0"][site][f"dice_class_{class_id}"])
                r1 = float(final_by_variant["R1"][site][f"dice_class_{class_id}"])
                delta = r1 - r0
                class_rows.append(
                    {
                        "seed": seed,
                        "evaluation_site": site,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "R0_final_dice": r0,
                        "R1_final_dice": r1,
                        "delta_R1_minus_R0": delta,
                    }
                )
                site_class_deltas[(site, class_id)].append(delta)
        for class_id in (1, 2):
            overall_class_deltas[class_id].append(
                float(np.mean([row["delta_R1_minus_R0"] for row in class_rows if row["seed"] == seed and row["class_id"] == class_id]))
            )

    coverage = {str(seed): _coverage(run_dirs[(seed, "R1")]) for seed in (0, 1, 2)}
    admission_rows = [
        row
        for row in _csv(admission_csv)
        if row["gate_scope"] == "seed_foreground_class" and row["region"] == "all"
    ]
    admission_gate: dict[str, Any] = {}
    for class_id in (1, 2):
        selected = sorted((row for row in admission_rows if int(row["class_id"]) == class_id), key=lambda row: int(row["seed"]))
        if len(selected) != 3:
            raise RuntimeError(f"missing seed-level admission diagnostics for class {class_id}")
        gaps = [float(row["accuracy_gap_admitted_minus_candidate"]) for row in selected]
        admission_gate[str(class_id)] = {
            "class_name": CLASS_NAMES[class_id],
            "per_seed_accuracy_gap": gaps,
            "strictly_positive_seed_count": sum(value > 0 for value in gaps),
            "mean_accuracy_gap": float(np.mean(gaps)),
            "direction_passed": sum(value > 0 for value in gaps) >= 2,
            "magnitude_passed": float(np.mean(gaps)) >= 0.020,
        }

    metric_summary = paired["summary"]["metrics"]
    metric_thresholds = {
        "mean_delta_final": metric_summary["final_average_dice"]["mean"] >= 0.003,
        "mean_delta_bwt": metric_summary["bwt"]["mean"] >= 0.0,
        "mean_delta_incoming": metric_summary["incoming_dice"]["mean"] >= -0.005,
        "mean_delta_previous": metric_summary["previous_site_dice"]["mean"] >= 0.003,
        "final_positive_2_of_3": metric_summary["final_average_dice"]["positive_direction_count"] >= 2,
        "previous_positive_2_of_3": metric_summary["previous_site_dice"]["positive_direction_count"] >= 2,
        "bwt_nonnegative_2_of_3": metric_summary["bwt"]["nonnegative_direction_count"] >= 2,
    }
    class_means = {str(class_id): float(np.mean(values)) for class_id, values in overall_class_deltas.items()}
    site_class_means = {
        f"{site}:class_{class_id}": float(np.mean(values))
        for (site, class_id), values in sorted(site_class_deltas.items())
    }
    class_thresholds = {
        "optic_disc_rim_mean_delta_ge_minus_0_010": class_means["1"] >= -0.010,
        "optic_cup_mean_delta_ge_minus_0_010": class_means["2"] >= -0.010,
        "no_site_class_cross_seed_mean_drop_gt_0_015": min(site_class_means.values()) >= -0.015,
    }
    coverage_passed = all(result["passed"] for result in coverage.values())
    mechanism_passed = coverage_passed and all(
        result["direction_passed"] and result["magnitude_passed"] for result in admission_gate.values()
    )
    internal_passed = all(metric_thresholds.values()) and all(class_thresholds.values()) and mechanism_passed
    status = "FUNDUS_V0_3_INTERNAL_GATE_PASSED" if internal_passed else "FUNDUS_V0_3_INTERNAL_GATE_FAILED"

    write_csv(output_dir / "fundus_seedwise_metrics.csv", seedwise_rows)
    write_csv(output_dir / "fundus_paired_differences.csv", paired_rows)
    write_csv(output_dir / "fundus_classwise_metrics.csv", class_rows)
    gate = {
        "protocol_id": "lcrseg_v0_3",
        "status": status,
        "internal_gate_passed": internal_passed,
        "paired_summary": paired["summary"],
        "metric_thresholds": metric_thresholds,
        "class_mean_deltas": class_means,
        "site_class_mean_deltas": site_class_means,
        "class_thresholds": class_thresholds,
        "coverage": coverage,
        "coverage_passed": coverage_passed,
        "admission_gate": admission_gate,
        "mechanism_passed": mechanism_passed,
        "run_directories": {f"seed{seed}_{variant}": str(path) for (seed, variant), path in run_dirs.items()},
    }
    write_json(args.status_json, gate)
    write_text(
        args.status_md,
        "\n".join(
            [
                "# LCR-Seg V0.3 Fundus internal gate",
                "",
                f"**Status:** `{status}`",
                "",
                f"- Mean delta Final: `{metric_summary['final_average_dice']['mean']}`",
                f"- Mean delta BWT: `{metric_summary['bwt']['mean']}`",
                f"- Mean delta Incoming: `{metric_summary['incoming_dice']['mean']}`",
                f"- Mean delta Previous: `{metric_summary['previous_site_dice']['mean']}`",
                f"- Coverage gate: `{coverage_passed}`",
                f"- Admission mechanism gate: `{mechanism_passed}`",
                "",
            ]
        ),
    )
    completion = {
        **gate,
        "hard_stop": not internal_passed,
        "conditional_baselines_executed": False,
        "external_fundus_gate_evaluated": False,
        "p0_seeds_1_2_executed": False,
        "bootstrap_executed": False,
        "prostate_executed": False,
        "unexecuted_due_to_internal_gate": (
            [
                "Sequential-SSL seeds 1 and 2",
                "Uniform-KD seeds 1 and 2",
                "external Fundus gate",
                "P0 seeds 1 and 2",
                "patient-level bootstrap",
                "Prostate RUNMC to BMC pilot",
            ]
            if not internal_passed
            else []
        ),
        "p0_seed0_completion": str(
            PROJECT_ROOT / "reports/experiment_status/V0_3_P0_SEED0_COMPLETION.json"
        ),
        "admission_analysis": str(admission_csv),
    }
    write_json(args.completion_json, completion)
    write_text(
        args.completion_md,
        "\n".join(
            [
                "# LCR-Seg V0.3 Fundus multi-seed completion",
                "",
                f"**Status:** `{status}`",
                "",
                f"- Internal Fundus gate passed: `{internal_passed}`",
                f"- Hard stop: `{not internal_passed}`",
                f"- Conditional baselines executed: `False`",
                f"- P0 seeds 1 and 2 executed: `False`",
                f"- Prostate executed: `False`",
                "",
                "If the internal gate failed, the remaining conditional stages are prohibited by the preregistration.",
                "",
            ]
        ),
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
