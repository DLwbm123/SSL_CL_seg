#!/usr/bin/env python3
"""Compile frozen V0.4a Fundus post-hoc outputs and evaluate the internal gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.v0_3 import METRICS
from lcrseg.common import sha256_path, write_csv, write_json, write_text


R0_RUNS = {
    0: "fundus_seed0_lcrseg_uniform_relation_kd_full200e",
    1: "fundus_seed1_lcrseg_v0_3_r0_legacy_uniform_full200e",
    2: "fundus_seed2_lcrseg_v0_3_r0_legacy_uniform_full200e",
}
R1_RUNS = {
    0: "fundus_seed0_lcrseg_v0_2a_r1_progressive_uniform_full200e",
    1: "fundus_seed1_lcrseg_v0_3_r1_progressive_uniform_full200e",
    2: "fundus_seed2_lcrseg_v0_3_r1_progressive_uniform_full200e",
}
SRA_RUNS = {seed: f"fundus_seed{seed}_lcrseg_v0_4a_sra_uniform_full200e" for seed in (0, 1, 2)}
CLASS_NAMES = {1: "optic_disc_rim", 2: "optic_cup"}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _summary(run_dir: Path, seed: int, variant: str) -> dict[str, Any]:
    payload = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or int(payload.get("completed_global_steps", -1)) != 13400:
        raise RuntimeError(f"incomplete formal run: {run_dir}")
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"seed mismatch: {run_dir}")
    if variant == "SRA" and not (
        payload.get("method") == "lcrseg_v0_4a"
        and payload.get("protocol_id") == "lcrseg_v0_4a"
        and payload.get("variant_id") == "SRA"
    ):
        raise RuntimeError(f"V0.4a identity mismatch: {run_dir}")
    return payload


def _engineering(run_dir: Path, payload: dict[str, Any]) -> dict[str, bool]:
    rows = _csv(run_dir / "train_log.csv")
    finite = True
    for row in rows:
        for value in row.values():
            if value in ("", None):
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            finite = finite and math.isfinite(number)
    steps = [int(row["global_step"]) for row in rows]
    return {
        "rows_and_steps_exact": len(rows) == 13400 and steps == list(range(1, 13401)),
        "numeric_finite": finite,
        "hidden_gt_training_usage_zero": sum(float(row["hidden_gt_training_usage"]) for row in rows) == 0,
        "amp_skip_zero": sum(float(row["optimizer_step_skipped"]) for row in rows) == 0,
        "old_model_gradient_zero": sum(float(row["old_model_gradient_detected"]) for row in rows) == 0,
        "historical_anchor_mutation_zero": sum(float(row["historical_anchor_changed"]) for row in rows) == 0,
        "three_site_final_checkpoints": len(list(run_dir.glob("checkpoint_final_site*.pt"))) == 3,
        "final_checkpoint_present": (run_dir / "checkpoint_last.pt").is_file(),
        "steps_summary_exact": int(payload.get("new_optimizer_steps", -1)) == 13400,
    }


def _paired(rows: list[dict[str, Any]], comparison: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][str(row["variant"])] = row
    target = "SRA" if comparison == "SRA_minus_R0" else "R1"
    paired: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        record: dict[str, Any] = {"seed": seed, "comparison": comparison}
        for metric in METRICS:
            record[f"delta_{metric}"] = float(by_seed[seed][target][metric]) - float(by_seed[seed]["R0"][metric])
        paired.append(record)
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        values = np.asarray([row[f"delta_{metric}"] for row in paired], dtype=np.float64)
        metrics[metric] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "std_sample": float(values.std(ddof=1)),
            "positive_direction_count": int((values > 0).sum()),
            "nonnegative_direction_count": int((values >= 0).sum()),
        }
    return paired, {"comparison": comparison, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--analysis-dir", type=Path, default=PROJECT_ROOT / "reports/analysis/v0_4")
    parser.add_argument("--status-json", type=Path, default=PROJECT_ROOT / "reports/experiment_status/V0_4A_FUNDUS_COMPLETION.json")
    parser.add_argument("--status-md", type=Path, default=PROJECT_ROOT / "reports/experiment_status/V0_4A_FUNDUS_COMPLETION.md")
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    outputs = [
        analysis_dir / "v04a_seedwise_metrics.csv",
        analysis_dir / "v04a_paired_differences.csv",
        analysis_dir / "v04a_classwise_metrics.csv",
        analysis_dir / "v04a_mechanism_analysis.csv",
        args.status_json,
        args.status_md,
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite V0.4a internal-gate artifacts")
    detail_dir = analysis_dir / "v04a_detail"
    required = [
        detail_dir / "precision_coverage.csv",
        detail_dir / "boundary_coverage.csv",
        detail_dir / "precision_coverage_summary.json",
        detail_dir / "mode_coverage_k2.csv",
        detail_dir / "mode_coverage_k4.csv",
        detail_dir / "mode_coverage_summary.json",
        detail_dir / "v04a_mechanism_summary.json",
        detail_dir / "v04a_mechanism_raw.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V0.4a frozen post-hoc is incomplete: {missing}")

    run_maps = {"R0": R0_RUNS, "R1": R1_RUNS, "SRA": SRA_RUNS}
    run_dirs = {
        (seed, variant): (args.run_root / names[seed]).resolve()
        for variant, names in run_maps.items() for seed in (0, 1, 2)
    }
    summaries = {
        key: _summary(path, key[0], key[1]) for key, path in run_dirs.items()
    }
    engineering = {
        str(seed): _engineering(run_dirs[(seed, "SRA")], summaries[(seed, "SRA")])
        for seed in (0, 1, 2)
    }
    engineering_passed = all(all(checks.values()) for checks in engineering.values())

    seedwise_rows: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        for variant in ("R0", "R1", "SRA"):
            payload = summaries[(seed, variant)]
            seedwise_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "run_dir": str(run_dirs[(seed, variant)]),
                    "run_summary_sha256": sha256_path(run_dirs[(seed, variant)] / "run_summary.json"),
                    **{metric: float(payload["summary"][metric]) for metric in METRICS},
                }
            )
    sra_paired, sra_pair_summary = _paired(seedwise_rows, "SRA_minus_R0")
    r1_paired, r1_pair_summary = _paired(seedwise_rows, "R1_minus_R0")
    paired_rows = sra_paired + r1_paired
    metric = sra_pair_summary["metrics"]

    class_rows: list[dict[str, Any]] = []
    class_deltas: dict[int, list[float]] = defaultdict(list)
    site_class_deltas: dict[tuple[str, int], list[float]] = defaultdict(list)
    for seed in (0, 1, 2):
        matrices: dict[str, dict[str, dict[str, str]]] = {}
        for variant in ("R0", "R1", "SRA"):
            rows = _csv(run_dirs[(seed, variant)] / "site_matrix_long.csv")
            final_index = max(int(row["trained_site_index"]) for row in rows)
            matrices[variant] = {
                row["evaluation_site"]: row for row in rows if int(row["trained_site_index"]) == final_index
            }
        if not (set(matrices["R0"]) == set(matrices["R1"]) == set(matrices["SRA"])):
            raise RuntimeError(f"site matrix mismatch for seed {seed}")
        for site in sorted(matrices["R0"]):
            for class_id in (1, 2):
                values = {
                    variant: float(matrices[variant][site][f"dice_class_{class_id}"])
                    for variant in ("R0", "R1", "SRA")
                }
                delta_sra = values["SRA"] - values["R0"]
                class_deltas[class_id].append(delta_sra)
                site_class_deltas[(site, class_id)].append(delta_sra)
                class_rows.append(
                    {
                        "seed": seed,
                        "evaluation_site": site,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "R0_final_dice": values["R0"],
                        "R1_final_dice": values["R1"],
                        "SRA_final_dice": values["SRA"],
                        "delta_SRA_minus_R0": delta_sra,
                        "delta_R1_minus_R0": values["R1"] - values["R0"],
                    }
                )
    class_means = {str(class_id): float(np.mean(values)) for class_id, values in class_deltas.items()}
    site_class_means = {
        f"{site}:class_{class_id}": float(np.mean(values))
        for (site, class_id), values in sorted(site_class_deltas.items())
    }
    metric_checks = {
        "mean_delta_final_ge_0_003": metric["final_average_dice"]["mean"] >= 0.003,
        "mean_delta_bwt_ge_0": metric["bwt"]["mean"] >= 0.0,
        "mean_delta_incoming_ge_minus_0_005": metric["incoming_dice"]["mean"] >= -0.005,
        "mean_delta_previous_ge_0_003": metric["previous_site_dice"]["mean"] >= 0.003,
        "final_positive_at_least_2_of_3": metric["final_average_dice"]["positive_direction_count"] >= 2,
        "previous_positive_at_least_2_of_3": metric["previous_site_dice"]["positive_direction_count"] >= 2,
        "bwt_nonnegative_at_least_2_of_3": metric["bwt"]["nonnegative_direction_count"] >= 2,
    }
    class_checks = {
        "optic_disc_rim_mean_delta_ge_minus_0_010": class_means["1"] >= -0.010,
        "optic_cup_mean_delta_ge_minus_0_010": class_means["2"] >= -0.010,
        "no_site_class_cross_seed_mean_drop_gt_0_015": min(site_class_means.values()) >= -0.015,
    }
    stability_checks = {
        "std_delta_final_le_0_010": metric["final_average_dice"]["std_sample"] <= 0.010,
        "std_delta_previous_le_0_010": metric["previous_site_dice"]["std_sample"] <= 0.010,
    }
    mechanism = json.loads((detail_dir / "v04a_mechanism_summary.json").read_text(encoding="utf-8"))
    mechanism_checks = {key: bool(value) for key, value in mechanism["checks"].items()}
    mechanism_rows = [
        {"category": "mechanism", "check": key, "passed": value, "value": mechanism["raw"].get(key, "")}
        for key, value in mechanism_checks.items()
    ]
    mechanism_rows.extend(
        {"category": "mechanism_raw", "check": key, "passed": "", "value": value}
        for key, value in mechanism["raw"].items()
    )
    write_csv(analysis_dir / "v04a_seedwise_metrics.csv", seedwise_rows)
    write_csv(analysis_dir / "v04a_paired_differences.csv", paired_rows)
    write_csv(analysis_dir / "v04a_classwise_metrics.csv", class_rows)
    write_csv(analysis_dir / "v04a_mechanism_analysis.csv", mechanism_rows)

    research_passed = bool(
        all(metric_checks.values())
        and all(class_checks.values())
        and all(stability_checks.values())
        and all(mechanism_checks.values())
    )
    if not engineering_passed:
        status = "HARD_STOP_V0_4A_ENGINEERING_FAILURE"
    else:
        status = "FUNDUS_V0_4A_INTERNAL_GATE_PASSED" if research_passed else "FUNDUS_V0_4A_INTERNAL_GATE_FAILED"
    report = {
        "protocol_id": "lcrseg_v0_4a",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "engineering_gate_passed": engineering_passed,
        "internal_gate_passed": status == "FUNDUS_V0_4A_INTERNAL_GATE_PASSED",
        "hard_stop": status != "FUNDUS_V0_4A_INTERNAL_GATE_PASSED",
        "engineering": engineering,
        "paired_summary_sra_minus_r0": sra_pair_summary,
        "paired_summary_r1_minus_r0": r1_pair_summary,
        "metric_checks": metric_checks,
        "class_mean_deltas": class_means,
        "site_class_mean_deltas": site_class_means,
        "class_checks": class_checks,
        "stability_checks": stability_checks,
        "mechanism": mechanism,
        "posthoc_artifacts": {str(path.relative_to(PROJECT_ROOT)): sha256_path(path) for path in required},
        "conditional_external_baselines_executed": False,
        "external_gate_evaluated": False,
        "prostate_executed": False,
        "unexecuted_due_to_internal_gate": (
            ["Sequential-SSL seeds 1 and 2", "Uniform-KD seeds 1 and 2", "external Fundus gate", "Prostate RUNMC to BMC pilot"]
            if status != "FUNDUS_V0_4A_INTERNAL_GATE_PASSED" else []
        ),
    }
    write_json(args.status_json, report)
    write_text(
        args.status_md,
        "\n".join(
            [
                "# LCR-Seg V0.4a Fundus completion",
                "",
                f"**Status:** `{status}`",
                "",
                f"- Engineering gate: `{engineering_passed}`",
                f"- Mean delta Final: `{metric['final_average_dice']['mean']}`",
                f"- Mean delta BWT: `{metric['bwt']['mean']}`",
                f"- Mean delta Incoming: `{metric['incoming_dice']['mean']}`",
                f"- Mean delta Previous: `{metric['previous_site_dice']['mean']}`",
                f"- Mechanism gate: `{all(mechanism_checks.values())}`",
                f"- External baselines executed: `False`",
                f"- Prostate executed: `False`",
                "",
            ]
        ),
    )
    print(json.dumps({"status": status, "engineering": engineering_passed, "research": research_passed}, sort_keys=True))


if __name__ == "__main__":
    main()
