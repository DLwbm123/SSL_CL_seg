#!/usr/bin/env python3
"""Evaluate the preregistered 500-step V0.1/V0.2a paired bridge."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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


def _state_error(first: dict[str, Any], second: dict[str, Any]) -> float:
    if set(first) != set(second):
        return float("inf")
    return max((_nested_error(first[key], second[key]) for key in first), default=0.0)


def _rows(path: Path) -> dict[int, dict[str, str]]:
    return {int(row["global_step"]): row for row in csv.DictReader(path.open())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-run", type=Path, required=True)
    parser.add_argument("--amended-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "reports/experiment_status/V0_2A_R0_BRIDGE_REPORT.json")
    parser.add_argument("--output-md", type=Path, default=PROJECT_ROOT / "reports/experiment_status/V0_2A_R0_BRIDGE_REPORT.md")
    args = parser.parse_args()
    comparison_steps = list(range(8050, 8501, 50))
    records: list[dict[str, Any]] = []
    legacy_rows = _rows(args.legacy_run / "train_log.csv")
    amended_rows = _rows(args.amended_run / "train_log.csv")
    parameter_max = 0.0
    anchor_max = 0.0
    optimizer_max = 0.0
    scheduler_exact = True
    loss_max = 0.0
    denominator_max = 0.0
    counts_exact = True
    for step in comparison_steps:
        legacy_checkpoint = args.legacy_run / f"checkpoint_step_{step:06d}.pt"
        amended_checkpoint = args.amended_run / f"checkpoint_step_{step:06d}.pt"
        first = load_checkpoint(legacy_checkpoint, map_location="cpu")
        second = load_checkpoint(amended_checkpoint, map_location="cpu")
        parameter_error = _state_error(first["current_model_state"], second["current_model_state"])
        anchor_error = _state_error(first["current_anchor_state"], second["current_anchor_state"])
        optimizer_error = _nested_error(first["optimizer_state"], second["optimizer_state"])
        scheduler_error = _nested_error(first["scheduler_state"], second["scheduler_state"])
        parameter_max = max(parameter_max, parameter_error)
        anchor_max = max(anchor_max, anchor_error)
        optimizer_max = max(optimizer_max, optimizer_error)
        scheduler_exact = scheduler_exact and scheduler_error == 0.0
        left_row, right_row = legacy_rows[step], amended_rows[step]
        losses = {
            key: abs(float(left_row[key]) - float(right_row[key]))
            for key in ("loss_total", "loss_sup", "loss_assim", "loss_relation")
        }
        denominators = {
            key: abs(float(left_row[key]) - float(right_row[key]))
            for key in ("assimilation_denominator", "relation_denominator")
        }
        step_counts_exact = int(float(left_row["pseudo_valid_count"])) == int(float(right_row["pseudo_valid_count"]))
        loss_max = max(loss_max, *losses.values())
        denominator_max = max(denominator_max, *denominators.values())
        counts_exact = counts_exact and step_counts_exact
        records.append(
            {
                "global_step": step,
                "parameter_max_abs": parameter_error,
                "anchor_max_abs": anchor_error,
                "optimizer_state_max_abs": optimizer_error,
                "scheduler_state_exact": scheduler_error == 0.0,
                "loss_abs_error": losses,
                "denominator_abs_error": denominators,
                "integer_counts_exact": step_counts_exact,
                "legacy_checkpoint_sha256": sha256_path(legacy_checkpoint),
                "amended_checkpoint_sha256": sha256_path(amended_checkpoint),
            }
        )
    passed = (
        parameter_max <= 1.0e-6
        and anchor_max <= 1.0e-6
        and optimizer_max <= 1.0e-6
        and scheduler_exact
        and loss_max <= 1.0e-6
        and denominator_max <= 1.0e-6
        and counts_exact
    )
    report = {
        "protocol_id": "lcrseg_v0_2a",
        "status": "PASSED" if passed else "HARD_STOP_R0_BRIDGE_MISMATCH",
        "legacy_run": str(args.legacy_run),
        "amended_run": str(args.amended_run),
        "comparison_steps": comparison_steps,
        "max_parameter_abs": parameter_max,
        "max_anchor_abs": anchor_max,
        "max_optimizer_state_abs": optimizer_max,
        "scheduler_state_exact": scheduler_exact,
        "max_loss_abs": loss_max,
        "max_denominator_abs": denominator_max,
        "integer_counts_exact": counts_exact,
        "thresholds": {"parameter": 1.0e-6, "anchor": 1.0e-6, "loss": 1.0e-6, "counts": "exact", "optimizer_scheduler_steps": "exact"},
        "records": records,
        "passed": passed,
    }
    write_json(args.output_json, report)
    write_text(
        args.output_md,
        "\n".join(
            (
                "# LCR-Seg V0.2a R0 500-step bridge",
                "",
                f"**Status:** `{report['status']}`",
                "",
                f"- Maximum parameter absolute error: `{parameter_max}`",
                f"- Maximum anchor absolute error: `{anchor_max}`",
                f"- Maximum loss absolute error: `{loss_max}`",
                f"- Maximum denominator absolute error: `{denominator_max}`",
                f"- Optimizer maximum state error: `{optimizer_max}`",
                f"- Scheduler state exact: `{scheduler_exact}`",
                f"- Integer counts exact: `{counts_exact}`",
                "",
                "Formal R1-R3 remain blocked unless this status is `PASSED`.",
                "",
            )
        ),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
