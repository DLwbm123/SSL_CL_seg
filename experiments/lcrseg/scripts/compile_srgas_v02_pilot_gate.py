#!/usr/bin/env python3
"""Compile the preregistered SR-GAS V0.2 seed-0 pilot gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable


RUN_NAMES = {variant: f"fundus_seed0_srgas_v02_{variant.lower()}_pilot1000" for variant in ("L0", "L1", "L2", "L3", "L4", "D1", "D2")}
SITE_COLUMNS = {"REFUGE": "refuge_mean_foreground_dice", "RIM_ONE_r3": "rim_one_mean_foreground_dice"}
PARENT_SHA256 = "8f188ba27074ecb09a689377982774e6cf59e8c1c652d3927be54fd7c377bf55"


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def _auc(rows: list[dict[str, str]], column: str) -> float:
    points = [(int(row["site_step"]), float(row[column])) for row in rows]
    area = sum((b[0] - a[0]) * (a[1] + b[1]) * 0.5 for a, b in zip(points, points[1:]))
    return area / float(points[-1][0] - points[0][0])


def _metric(summary: dict[str, Any], name: str) -> float:
    keys = {
        "Final": "final_average_dice",
        "BWT": "bwt",
        "Previous": "previous_site_dice",
        "Incoming": "incoming_dice",
    }
    value = summary["summary"][keys[name]]
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"missing/non-finite endpoint metric: {name}")
    return float(value)


def _delta(summaries: dict[str, dict[str, Any]], left: str, right: str, metric: str) -> float:
    return _metric(summaries[left], metric) - _metric(summaries[right], metric)


def compile_gate(run_root: Path, output_md: Path, output_json: Path, test_report: Path) -> dict[str, Any]:
    runs = {variant: run_root / name for variant, name in RUN_NAMES.items()}
    train: dict[str, list[dict[str, str]]] = {}
    trajectory: dict[str, list[dict[str, str]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    engineering_checks: dict[str, bool] = {}

    tests = _json(test_report)
    engineering_checks["v02_tests_passed"] = tests.get("status") == "SRGAS_V0_2_TESTS_PASSED"
    for variant, run in runs.items():
        required = [run / "train_log.csv", run / "pilot_trajectory.csv", run / "run_summary.json", run / "checkpoint_final.pt", run / "parent_artifact.json"]
        engineering_checks[f"{variant}_required_artifacts"] = all(path.is_file() for path in required)
        if not engineering_checks[f"{variant}_required_artifacts"]:
            continue
        train[variant] = _csv(run / "train_log.csv")
        trajectory[variant] = _csv(run / "pilot_trajectory.csv")
        summaries[variant] = _json(run / "run_summary.json")
        parent = _json(run / "parent_artifact.json")
        engineering_checks[f"{variant}_complete_1000"] = (
            summaries[variant].get("status") == "complete"
            and int(summaries[variant].get("completed_global_steps", -1)) == 9000
            and len(train[variant]) == 1000
            and [int(row["site_step"]) for row in trajectory[variant]] == list(range(50, 1001, 50))
        )
        engineering_checks[f"{variant}_parent_exact"] = parent.get("initial_previous_checkpoint_sha256") == PARENT_SHA256
        engineering_checks[f"{variant}_finite"] = all(_finite(value) for row in train[variant] for value in row.values())
        engineering_checks[f"{variant}_no_skip_hidden_oldgrad_anchor_change"] = all(
            float(row.get(key, 0) or 0) == 0.0
            for row in train[variant]
            for key in ("optimizer_step_skipped", "hidden_gt_training_usage", "old_model_gradient_detected", "historical_anchor_changed")
        )
        engineering_checks[f"{variant}_deterministic_eval"] = all(float(row.get("stochastic_eval_enabled", 0) or 0) == 0.0 for row in train[variant])

    if set(train) != set(RUN_NAMES):
        status = "HARD_STOP_SRGAS_V0_2_ENGINEERING_FAILURE"
        payload = {"status": status, "engineering_checks": engineering_checks, "missing_variants": sorted(set(RUN_NAMES).difference(train))}
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output_md.write_text(f"# SR-GAS V0.2 Seed-0 Pilot\n\nStatus: `{status}`\n")
        return payload

    warm_variants = ("L1", "L2", "L3", "L4", "D1")
    engineering_checks["warm_variants_first_step_zero"] = all(
        float(train[v][0]["noise_warmup_multiplier"]) == 0.0 and float(train[v][0]["perturbation_l2_ratio"]) == 0.0
        for v in warm_variants
    )
    engineering_checks["d2_is_registered_no_warm_ablation"] = float(train["D2"][0]["noise_warmup_multiplier"]) == 1.0
    engineering_checks["lag_first_step_ones_then_previous"] = all(
        float(train[v][0]["lagged_buffer_valid_before_step"]) == 0.0
        and float(train[v][1]["lagged_buffer_valid_before_step"]) == 1.0
        for v in ("L2", "L3", "L4", "D2")
    )
    engineering_checks["lag_buffer_not_parameter"] = all(float(row["lagged_buffer_is_parameter"]) == 0.0 for v in train for row in train[v])
    stochastic_variants = ("L1", "L2", "L3", "L4", "D1", "D2")
    engineering_checks["shared_raw_noise_stream_exact"] = all(
        len({train[v][index]["raw_noise_checksum"] for v in stochastic_variants}) == 1
        for index in range(1000)
    )
    for variant in ("L2", "L3", "L4"):
        active = train[variant][1:]
        top = median(float(row["top_sensitivity_quartile_noise_median"]) for row in active)
        bottom = median(float(row["bottom_sensitivity_quartile_noise_median"]) for row in active)
        engineering_checks[f"{variant}_adaptive_inverse_geometry"] = top < bottom
    engineering_checks["L4_r2c_valid_finite_nonzero"] = all(
        int(float(row["r2c_valid_count"])) > 0
        and float(row["relation_sensitivity_mass"]) > 0.0
        and float(row["r2c_total_objective_coefficient"]) == 0.0
        for row in train["L4"]
    )
    engineering_checks["L4_differs_from_L3_after_relation_active"] = any(
        abs(float(a["noise_scale_p50"]) - float(b["noise_scale_p50"])) > 0.0
        for a, b in zip(train["L4"][1:], train["L3"][1:])
    )

    engineering_passed = all(engineering_checks.values())
    trajectory_metrics: dict[str, Any] = {}
    safety_passed = True
    auc_passed = True
    for site, column in SITE_COLUMNS.items():
        deltas = [float(l4[column]) - float(l0[column]) for l4, l0 in zip(trajectory["L4"], trajectory["L0"])]
        worst_index = min(range(len(deltas)), key=deltas.__getitem__)
        worst_drop = max(0.0, -deltas[worst_index])
        auc_l4 = _auc(trajectory["L4"], column)
        auc_l0 = _auc(trajectory["L0"], column)
        trajectory_metrics[site] = {
            "worst_step": int(trajectory["L4"][worst_index]["site_step"]),
            "worst_drop": worst_drop,
            "normalized_auc_l4": auc_l4,
            "normalized_auc_l0": auc_l0,
            "normalized_auc_delta": auc_l4 - auc_l0,
        }
        safety_passed &= worst_drop <= 0.015
        auc_passed &= auc_l4 - auc_l0 >= -0.005

    endpoint_deltas = {metric: _delta(summaries, "L4", "L0", metric) for metric in ("Final", "BWT", "Previous", "Incoming")}
    endpoint_passed = endpoint_deltas["Final"] >= 0.005 and endpoint_deltas["BWT"] >= 0.020 and endpoint_deltas["Previous"] >= 0.020 and endpoint_deltas["Incoming"] >= -0.010
    relation_deltas = {metric: _delta(summaries, "L4", "L3", metric) for metric in ("Final", "BWT", "Previous", "Incoming")}
    relation_passed = relation_deltas["BWT"] >= 0.003 and relation_deltas["Previous"] >= 0.003 and relation_deltas["Final"] >= -0.002 and relation_deltas["Incoming"] >= -0.005
    adaptive_deltas = {metric: _delta(summaries, "L3", "L1", metric) for metric in ("Final", "BWT", "Previous", "Incoming")}
    adaptive_passed = (adaptive_deltas["BWT"] >= 0.003 or adaptive_deltas["Previous"] >= 0.003) and adaptive_deltas["Final"] >= -0.005 and adaptive_deltas["Incoming"] >= -0.010
    total_deltas = {metric: _delta(summaries, "L4", "L2", metric) for metric in ("Final", "BWT", "Previous", "Incoming")}
    total_passed = total_deltas["BWT"] >= 0.003 and total_deltas["Previous"] >= 0.003 and total_deltas["Final"] >= -0.002 and total_deltas["Incoming"] >= -0.005

    gates = {
        "engineering": engineering_passed,
        "original_worst_point_safety": safety_passed,
        "auc_safety": auc_passed,
        "endpoint": endpoint_passed,
        "l4_vs_l3_relation": relation_passed,
        "l3_vs_l1_adaptive": adaptive_passed,
        "l4_vs_l2_total_gas": total_passed,
    }
    status = (
        "HARD_STOP_SRGAS_V0_2_ENGINEERING_FAILURE"
        if not engineering_passed
        else "SRGAS_V0_2_SEED0_PILOT_PASSED"
        if all(gates.values())
        else "SRGAS_V0_2_SEED0_PILOT_FAILED"
    )
    endpoints = {
        variant: {metric: _metric(summaries[variant], metric) for metric in ("Final", "BWT", "Previous", "Incoming")}
        for variant in RUN_NAMES
    }
    payload = {
        "status": status,
        "gates": gates,
        "engineering_checks": engineering_checks,
        "trajectory": trajectory_metrics,
        "endpoints": endpoints,
        "l4_vs_l0": endpoint_deltas,
        "l4_vs_l3": relation_deltas,
        "l3_vs_l1": adaptive_deltas,
        "l4_vs_l2": total_deltas,
        "run_paths": {variant: str(path) for variant, path in runs.items()},
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# SR-GAS V0.2 Seed-0 Pilot Gate",
        "",
        f"Status: `{status}`",
        "",
        "| Gate | Passed |",
        "|---|---|",
        *[f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in gates.items()],
        "",
        "| Site | Worst L4-vs-L0 drop | Worst step | AUC delta |",
        "|---|---:|---:|---:|",
        *[
            f"| {site} | {values['worst_drop']:.6f} | {values['worst_step']} | {values['normalized_auc_delta']:+.6f} |"
            for site, values in trajectory_metrics.items()
        ],
        "",
        "| Variant | Final | BWT | Previous | Incoming |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {variant} | {values['Final']:.6f} | {values['BWT']:.6f} | {values['Previous']:.6f} | {values['Incoming']:.6f} |"
            for variant, values in endpoints.items()
        ],
    ]
    output_md.write_text("\n".join(lines) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--output-md", type=Path, default=Path(__file__).resolve().parents[1] / "reports/experiment_status/SRGAS_V0_2_SEED0_PILOT.md")
    parser.add_argument("--output-json", type=Path, default=Path(__file__).resolve().parents[1] / "reports/experiment_status/SRGAS_V0_2_SEED0_PILOT.json")
    parser.add_argument("--test-report", type=Path, default=Path(__file__).resolve().parents[1] / "reports/experiment_status/SRGAS_V0_2_TEST_REPORT.json")
    args = parser.parse_args()
    print(json.dumps(compile_gate(args.run_root, args.output_md, args.output_json, args.test_report), sort_keys=True))


if __name__ == "__main__":
    main()
