#!/usr/bin/env python3
"""Compile the registered six-run SR-GAS engineering and safety pilot gates."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import sha256_path, write_json, write_text  # noqa: E402


RUN_NAMES = {
    "A1": "fundus_seed0_srgas_a1_pilot1000",
    "A2": "fundus_seed0_srgas_a2_pilot1000",
    "A3": "fundus_seed0_srgas_a3_pilot1000",
    "A4": "fundus_seed0_srgas_a4_pilot1000",
    "A5": "fundus_seed0_srgas_a5_pilot1000",
    "A6": "fundus_seed0_srgas_a6_freeze_pilot1000",
}


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value in ("", None) else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=PROJECT_ROOT / "reports/experiment_status")
    args = parser.parse_args()
    parent_sha = sha256_path(args.parent_checkpoint)
    runs: dict[str, dict] = {}
    checks: dict[str, bool] = {}
    trajectory: dict[str, dict[int, dict[str, str]]] = {}
    for variant, name in RUN_NAMES.items():
        run = args.run_root / name
        required = ("run_summary.json", "train_log.csv", "pilot_trajectory.csv", "checkpoint_final.pt", "parent_artifact.json")
        missing = [filename for filename in required if not (run / filename).is_file()]
        if missing:
            raise FileNotFoundError(f"{variant} pilot incomplete: {missing}")
        rows = list(csv.DictReader((run / "train_log.csv").open()))
        if len(rows) != 1000 or int(rows[-1]["site_step"]) != 1000:
            raise RuntimeError(f"{variant} pilot did not complete exactly 1000 steps")
        parent = json.loads((run / "parent_artifact.json").read_text())
        parent_match = parent["initial_previous_checkpoint_sha256"] == parent_sha
        finite_losses = all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in row
            if key.startswith("loss_") and row[key] not in ("", None)
        )
        diagnostic_keys = (
            "classifier_weight_norm",
            "sampled_weight_norm",
            "perturbation_l2_ratio",
            "sensitivity_p10",
            "sensitivity_p50",
            "sensitivity_p90",
            "noise_scale_p10",
            "noise_scale_p50",
            "noise_scale_p90",
            "relation_head_drift",
            "training_step_seconds",
            "peak_memory_bytes",
        )
        finite_diagnostics = all(
            math.isfinite(_number(row, key)) for row in rows for key in diagnostic_keys
        )
        checks.update(
            {
                f"{variant}_parent_sha_exact": parent_match,
                f"{variant}_losses_finite": finite_losses,
                f"{variant}_diagnostics_finite": finite_diagnostics,
                f"{variant}_amp_skip_zero": sum(_number(row, "optimizer_step_skipped") for row in rows) == 0,
                f"{variant}_hidden_gt_zero": sum(_number(row, "hidden_gt_training_usage") for row in rows) == 0,
                f"{variant}_old_model_gradient_zero": sum(_number(row, "old_model_gradient_detected") for row in rows) == 0,
                f"{variant}_historical_anchor_unchanged": sum(_number(row, "historical_anchor_changed") for row in rows) == 0,
                f"{variant}_stochastic_eval_disabled": sum(_number(row, "stochastic_eval_enabled") for row in rows) == 0,
            }
        )
        if variant in {"A3", "A4", "A5"}:
            top = statistics.median(_number(row, "top_sensitivity_quartile_noise_median") for row in rows)
            bottom = statistics.median(_number(row, "bottom_sensitivity_quartile_noise_median") for row in rows)
            checks[f"{variant}_top_sensitive_noise_lower"] = top < bottom
        if variant == "A5":
            active = [row for row in rows if _number(row, "r2c_valid_count") > 0]
            checks.update(
                {
                    "A5_r2c_valid_positive": bool(active),
                    "A5_r2c_sensitivity_finite_nonzero": bool(active) and all(math.isfinite(_number(row, "s_r2c_mean")) for row in active) and any(_number(row, "s_r2c_mean") > 0 for row in active),
                    "A5_differs_from_A4_noise_scale": bool(active) and any(_number(row, "a5_a4_noise_scale_l1") > 0 for row in active),
                    "A5_projection_proxy_grad_zero": all(_number(row, "projection_head_proxy_grad_norm") == 0 for row in active),
                    "A5_r2c_objective_coefficient_zero": all(_number(row, "r2c_total_objective_coefficient") == 0 for row in active),
                }
            )
        trajectory_rows = list(csv.DictReader((run / "pilot_trajectory.csv").open()))
        if len(trajectory_rows) != 20:
            raise RuntimeError(f"{variant} pilot trajectory does not have 20 registered 50-step evaluations")
        trajectory[variant] = {int(row["site_step"]): row for row in trajectory_rows}
        runs[variant] = {
            "run": str(run.resolve()),
            "checkpoint_sha256": sha256_path(run / "checkpoint_final.pt"),
            "completed_steps": len(rows),
            "final_loss": _number(rows[-1], "loss_total"),
            "peak_memory_bytes": max(_number(row, "peak_memory_bytes") for row in rows),
            "median_step_seconds": statistics.median(_number(row, "training_step_seconds") for row in rows),
        }
    common_steps = sorted(set(trajectory["A1"]).intersection(trajectory["A5"]))
    refuge_drops = [
        float(trajectory["A1"][step]["refuge_mean_foreground_dice"])
        - float(trajectory["A5"][step]["refuge_mean_foreground_dice"])
        for step in common_steps
    ]
    rim_drops = [
        float(trajectory["A1"][step]["rim_one_mean_foreground_dice"])
        - float(trajectory["A5"][step]["rim_one_mean_foreground_dice"])
        for step in common_steps
    ]
    safety = {
        "maximum_refuge_trajectory_drop": max(refuge_drops),
        "maximum_rim_one_trajectory_drop": max(rim_drops),
        "maximum_allowed_drop": 0.015,
    }
    checks["A5_refuge_trajectory_drop_within_gate"] = safety["maximum_refuge_trajectory_drop"] <= 0.015
    checks["A5_rim_one_trajectory_drop_within_gate"] = safety["maximum_rim_one_trajectory_drop"] <= 0.015
    status = "SRGAS_PILOT_GATE_PASSED" if all(checks.values()) else "SRGAS_PILOT_GATE_FAILED"
    report = {
        "status": status,
        "parent_checkpoint": str(args.parent_checkpoint.resolve()),
        "parent_checkpoint_sha256": parent_sha,
        "runs": runs,
        "safety": safety,
        "checks": checks,
        "a6_contextual_only": True,
        "hyperparameters_changed_after_pilot": False,
    }
    json_path = args.report_root / "SRGAS_PILOT_REPORT.json"
    md_path = args.report_root / "SRGAS_PILOT_REPORT.md"
    if json_path.exists() or md_path.exists():
        raise FileExistsError("refusing to overwrite existing pilot report")
    write_json(json_path, report)
    lines = [
        "# SR-GAS Pilot Report",
        "",
        f"Status: `{status}`",
        "",
        f"Common parent SHA256: `{parent_sha}`",
        f"Maximum A5-vs-A1 REFUGE trajectory drop: `{safety['maximum_refuge_trajectory_drop']:.6f}`.",
        f"Maximum A5-vs-A1 RIM-ONE trajectory drop: `{safety['maximum_rim_one_trajectory_drop']:.6f}`.",
        "",
        "| Variant | Steps | Final loss | Median step seconds | Peak memory bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, values in runs.items():
        lines.append(f"| {variant} | {values['completed_steps']} | {values['final_loss']:.6f} | {values['median_step_seconds']:.6f} | {values['peak_memory_bytes']:.0f} |")
    write_text(md_path, "\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if status != "SRGAS_PILOT_GATE_PASSED":
        raise SystemExit(status)


if __name__ == "__main__":
    main()
