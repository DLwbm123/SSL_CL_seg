#!/usr/bin/env python3
"""Evaluate the immutable V0.2a R1-R3 1,000-step engineering pilot gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import sha256_path, write_json, write_text
from lcrseg.engine.checkpoint import load_checkpoint


def _nested_error(first: Any, second: Any) -> float:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        if first.shape != second.shape:
            return float("inf")
        if not first.is_floating_point() and not second.is_floating_point():
            return 0.0 if torch.equal(first, second) else float("inf")
        return float((first.detach().float() - second.detach().float()).abs().max().cpu()) if first.numel() else 0.0
    if isinstance(first, dict) and isinstance(second, dict):
        if set(first) != set(second):
            return float("inf")
        return max((_nested_error(first[key], second[key]) for key in first), default=0.0)
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            return float("inf")
        return max((_nested_error(left, right) for left, right in zip(first, second, strict=True)), default=0.0)
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return abs(float(first) - float(second))
    return 0.0 if first == second else float("inf")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return list(csv.DictReader(path.open()))


def _monotonic(values: list[float]) -> bool:
    return all(right + 1.0e-12 >= left for left, right in zip(values, values[1:]))


def _calibrator_check(run_dir: Path) -> dict[str, Any]:
    directory = run_dir / "site_RIM_ONE_r3"
    json_path = directory / "teacher_validity_calibrator.json"
    csv_path = directory / "teacher_validity_calibrator.csv"
    if not json_path.is_file() or not csv_path.is_file():
        return {"passed": False, "reason": "missing calibrator JSON or CSV"}
    state = json.loads(json_path.read_text())
    mappings = [state.get("global_mapping") or {}, *(state.get("class_mappings") or {}).values()]
    monotonic = bool(mappings) and all(
        mapping.get("probabilities") and _monotonic([float(value) for value in mapping["probabilities"]])
        for mapping in mappings
    )
    provenance = state.get("provenance") or {}
    csv_rows = _read_csv(csv_path)
    passed = (
        state.get("status") == "available_frozen_site_start"
        and int(state.get("fit_count", 0)) == 1
        and bool(csv_rows)
        and monotonic
        and provenance.get("fit_scope") == "current_site_train_labeled_only"
        and provenance.get("old_model_only") is True
        and int(provenance.get("hidden_gt_usage", -1)) == 0
        and provenance.get("frozen_during_site_training") is True
    )
    return {
        "passed": passed,
        "status": state.get("status"),
        "fit_count": state.get("fit_count"),
        "csv_rows": len(csv_rows),
        "mappings_monotonic": monotonic,
        "sample_counts_by_class": state.get("sample_counts_by_class"),
        "provenance": provenance,
        "json_sha256": sha256_path(json_path),
        "csv_sha256": sha256_path(csv_path),
    }


def _evaluate_variant(variant: str, run_dir: Path, parent: Path) -> dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    final_path = run_dir / "checkpoint_final.pt"
    if not summary_path.is_file() or not final_path.is_file():
        return {"variant": variant, "run_dir": str(run_dir), "passed": False, "reason": "incomplete run artifacts"}
    summary = json.loads(summary_path.read_text())
    protocol = json.loads((run_dir / "protocol.json").read_text())
    parent_artifact = json.loads((run_dir / "parent_artifact.json").read_text())
    rows = _read_csv(run_dir / "train_log.csv")
    branch_rows = _read_csv(run_dir / "branch_coverage.csv")
    final = load_checkpoint(final_path, map_location="cpu")
    parent_payload = load_checkpoint(parent, map_location="cpu")
    required_numeric = (
        "loss_total",
        "loss_sup",
        "loss_assim",
        "loss_relation",
        "relation_effective_sample_size",
        "relation_valid_count",
        "hidden_gt_training_usage",
        "current_old_js_gate_usage",
        "old_model_gradient_detected",
        "historical_anchor_changed",
    )
    finite = all(
        all(key in row and math.isfinite(float(row[key])) for key in required_numeric)
        for row in rows
    )
    expected_steps = list(range(8001, 9001))
    actual_steps = [int(row["global_step"]) for row in rows]
    exact_step_sequence = actual_steps == expected_steps
    runtime_guards = all(
        int(float(row[key])) == 0
        for row in rows
        for key in (
            "hidden_gt_training_usage",
            "current_old_js_gate_usage",
            "old_model_gradient_detected",
            "historical_anchor_changed",
        )
    )
    historical_anchor_error = _nested_error(
        parent_payload["current_anchor_state"], final["historical_anchor_state"]
    )
    resume_commands = sorted(run_dir.glob("resume_command_*.txt"))
    resume_configs = sorted(run_dir.glob("resume_config_*.yaml"))
    resume_exact = (
        len(resume_commands) == 1
        and len(resume_configs) == 1
        and exact_step_sequence
        and int(final["site_step"]) == 1000
        and int(final["global_step"]) == 9000
    )
    admission_groups: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for row in branch_rows:
        key = (int(row["epoch"]), int(row["predicted_class"]))
        valid_count = int(row["valid_count"])
        admission_groups[key][0] += valid_count
        admission_groups[key][1] += int(row["admitted_count"])
        admission_groups[key][2] += float(row["target_fraction"]) * valid_count
    admission_errors = [
        abs(selected / valid - target_weighted / valid)
        for valid, selected, target_weighted in admission_groups.values()
        if valid > 0
    ]
    max_admission_error = max(admission_errors, default=0.0)
    admission_passed = variant == "R2" or (bool(admission_errors) and max_admission_error <= 0.05 + 1.0e-12)
    rejection_fractions = [
        float(row["rejected_fraction"])
        for row in branch_rows
        if int(row["relation_valid_count"]) > 0
    ]
    max_rejection_fraction = max(rejection_fractions, default=0.0)
    rejection_passed = variant == "R1" or (bool(rejection_fractions) and max_rejection_fraction <= 0.20 + 1.0e-12)
    ess_ratios = [
        float(row["relation_effective_sample_size"]) / float(row["relation_valid_count"])
        for row in rows
        if float(row["relation_valid_count"]) > 0
    ]
    minimum_ess_ratio = min(ess_ratios, default=1.0)
    ess_passed = variant == "R1" or (bool(ess_ratios) and minimum_ess_ratio >= 0.80 - 1.0e-12)
    calibrator = (
        {"passed": True, "status": "not_applicable_uniform_relation"}
        if variant == "R1"
        else _calibrator_check(run_dir)
    )
    expected_modes = {
        "R1": ("progressive_admission", "uniform_relation"),
        "R2": ("legacy_continuous_v01", "calibrated_teacher_rejection"),
        "R3": ("progressive_admission", "calibrated_teacher_rejection"),
    }
    modes_exact = (
        protocol.get("variant_id") == variant
        and (protocol.get("assimilation_mode"), protocol.get("consolidation_mode")) == expected_modes[variant]
    )
    parent_hash_exact = parent_artifact.get("initial_previous_checkpoint_sha256") == sha256_path(parent)
    passed = all(
        (
            summary.get("status") == "complete",
            len(rows) == 1000,
            len(branch_rows) == 1000 * 3,
            finite,
            runtime_guards,
            historical_anchor_error == 0.0,
            resume_exact,
            admission_passed,
            rejection_passed,
            ess_passed,
            calibrator["passed"],
            modes_exact,
            parent_hash_exact,
        )
    )
    return {
        "variant": variant,
        "run_dir": str(run_dir),
        "status": summary.get("status"),
        "train_rows": len(rows),
        "branch_rows": len(branch_rows),
        "finite_losses_and_scalars": finite,
        "runtime_guards_zero": runtime_guards,
        "historical_anchor_max_abs": historical_anchor_error,
        "checkpoint_resume_exact": resume_exact,
        "global_step_sequence_exact": exact_step_sequence,
        "maximum_admission_fraction_error": max_admission_error,
        "admission_epoch_class_groups": len(admission_errors),
        "admission_passed": admission_passed,
        "maximum_rejection_fraction_per_class_step": max_rejection_fraction,
        "rejection_passed": rejection_passed,
        "minimum_relation_ess_ratio": minimum_ess_ratio,
        "relation_ess_passed": ess_passed,
        "calibrator": calibrator,
        "modes_exact": modes_exact,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": sha256_path(parent),
        "parent_hash_exact": parent_hash_exact,
        "final_checkpoint_sha256": sha256_path(final_path),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-parent", type=Path, required=True)
    parser.add_argument("--progressive-parent", type=Path, required=True)
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--r2-run", type=Path, required=True)
    parser.add_argument("--r3-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "reports/experiment_status/V0_2A_PILOT_REPORT.json")
    parser.add_argument("--output-md", type=Path, default=PROJECT_ROOT / "reports/experiment_status/V0_2A_PILOT_REPORT.md")
    args = parser.parse_args()
    records = [
        _evaluate_variant("R1", args.r1_run.resolve(), args.progressive_parent.resolve()),
        _evaluate_variant("R2", args.r2_run.resolve(), args.r0_parent.resolve()),
        _evaluate_variant("R3", args.r3_run.resolve(), args.progressive_parent.resolve()),
    ]
    passed = all(record["passed"] for record in records)
    report = {
        "protocol_id": "lcrseg_v0_2a",
        "status": "PASSED" if passed else "HARD_STOP_PILOT_GATE_FAILED",
        "thresholds": {
            "admission_fraction_absolute_error": 0.05,
            "rejection_fraction_per_class": 0.20,
            "relation_ess_fraction_of_valid_uniform_count": 0.80,
            "hidden_gt_training_usage": 0,
            "current_old_js_gate_usage": 0,
        },
        "records": records,
        "passed": passed,
    }
    write_json(args.output_json, report)
    lines = ["# LCR-Seg V0.2a 1,000-step pilot gate", "", f"**Status:** `{report['status']}`", ""]
    for record in records:
        lines.extend(
            [
                f"## {record['variant']}",
                "",
                f"- Passed: `{record['passed']}`",
                f"- Maximum admission error: `{record['maximum_admission_fraction_error']}`",
                f"- Maximum rejection fraction: `{record['maximum_rejection_fraction_per_class_step']}`",
                f"- Minimum relation ESS ratio: `{record['minimum_relation_ess_ratio']}`",
                f"- Checkpoint resume exact: `{record['checkpoint_resume_exact']}`",
                f"- Historical-anchor maximum error: `{record['historical_anchor_max_abs']}`",
                "",
            ]
        )
    lines.extend(["Formal R1-R3 remain blocked unless this status is `PASSED`.", ""])
    write_text(args.output_md, "\n".join(lines))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
