#!/usr/bin/env python3
"""Validate the three registered V0.4a engineering pilots and freeze their gate."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import read_csv, sha256_path, utc_now, write_json, write_text
from lcrseg.engine.checkpoint import load_checkpoint
from scripts.run_v0_4a_experiment import PILOT_RUN_NAMES, R1_PARENT_SHA256, parent_checkpoint


def _state_max_error(first: Any, second: Any) -> float:
    if isinstance(first, dict) and isinstance(second, dict):
        if set(first) != set(second):
            return float("inf")
        return max((_state_max_error(first[key], second[key]) for key in first), default=0.0)
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        if first.shape != second.shape:
            return float("inf")
        return float((first.detach().cpu().float() - second.detach().cpu().float()).abs().max()) if first.numel() else 0.0
    return 0.0 if first == second else float("inf")


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite {key}")
    return value


def _validate_run(root: Path, seed: int) -> dict[str, Any]:
    run_dir = root / "runs" / PILOT_RUN_NAMES[seed]
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    rows = read_csv(run_dir / "train_log.csv")
    if summary.get("status") != "complete" or int(summary.get("completed_global_steps", -1)) != 9000:
        raise RuntimeError(f"seed {seed} pilot is not complete at global step 9000")
    if len(rows) != 1000 or [int(row["global_step"]) for row in rows] != list(range(8001, 9001)):
        raise RuntimeError(f"seed {seed} pilot does not contain exactly 1,000 contiguous training rows")
    if (run_dir / "failure_bundle").exists():
        raise RuntimeError(f"seed {seed} completed pilot retained a failure bundle")
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    expected_protocol = {
        "protocol_id": "lcrseg_v0_4a",
        "variant_id": "SRA",
        "assimilation_mode": "soft_reliability_allocation",
        "consolidation_mode": "uniform_relation",
        "single_anchor": True,
        "multi_agent": False,
        "ric": False,
        "teacher_rejection": False,
    }
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise RuntimeError(f"seed {seed} protocol fields differ from V0.4a")
    checkpoint_steps = [8250, 8500, 8750, 9000]
    interval = [run_dir / f"checkpoint_step_{step:06d}.pt" for step in checkpoint_steps]
    if not all(path.is_file() for path in interval):
        raise FileNotFoundError(f"seed {seed} lacks one or more interval checkpoints")
    resume_files = sorted(run_dir.glob("resume_config_*.yaml"))
    if len(resume_files) != 1:
        raise RuntimeError(f"seed {seed} must contain exactly one actual resume record")
    boundary = load_checkpoint(run_dir / "checkpoint_step_008500.pt", map_location="cpu")
    if (int(boundary["site_step"]), int(boundary["global_step"])) != (500, 8500):
        raise RuntimeError(f"seed {seed} resume boundary checkpoint is invalid")
    final_path = run_dir / "checkpoint_final_site1_RIM_ONE_r3.pt"
    final = load_checkpoint(final_path, map_location="cpu")
    parent_path = parent_checkpoint(root, seed)
    parent = load_checkpoint(parent_path, map_location="cpu")
    if sha256_path(parent_path) != R1_PARENT_SHA256[seed]:
        raise RuntimeError(f"seed {seed} parent changed after preregistration")
    historical_anchor_error = _state_max_error(parent["current_anchor_state"], final["historical_anchor_state"])

    numeric_keys = (
        "loss_total",
        "loss_sup",
        "loss_assim",
        "loss_relation",
        "sra_alpha_mean",
        "sra_hard_loss_mean",
        "sra_soft_loss_mean",
        "sra_weighted_hard_loss",
        "sra_weighted_soft_loss",
        "sra_anchor_update_mass",
    )
    for row in rows:
        for key in numeric_keys:
            _float(row, key)
    deciles = [json.loads(row["sra_alpha_deciles"]) for row in rows]
    deciles_valid = all(
        len(values) == 11
        and all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in values)
        and all(float(values[index]) <= float(values[index + 1]) for index in range(10))
        for values in deciles
    )
    hard_soft_ratios = [_float(row, "sra_hard_soft_loss_ratio") for row in rows]
    anchor_mass = [_float(row, "sra_anchor_update_mass") for row in rows]
    engineering = {
        "rows_exact_1000": len(rows) == 1000,
        "checkpoint_resume_actual": len(resume_files) == 1,
        "checkpoint_intervals_complete": all(path.is_file() for path in interval),
        "nan_inf_zero": True,
        "amp_skip_zero": max(_float(row, "optimizer_step_skipped") for row in rows) == 0.0,
        "hidden_gt_training_usage_zero": max(_float(row, "hidden_gt_training_usage") for row in rows) == 0.0,
        "old_model_gradient_zero": max(_float(row, "old_model_gradient_detected") for row in rows) == 0.0,
        "historical_anchor_mutation_zero": historical_anchor_error == 0.0,
        "gradient_finite": min(_float(row, "sra_gradient_finite") for row in rows) == 1.0,
        "historical_relation_exact_path": all(row["sra_historical_relation_exact_path"] == "frozen_uniform_relation_v0_2a" for row in rows),
        "current_relation_target_stopgrad": min(_float(row, "sra_current_relation_target_stopgrad") for row in rows) == 1.0,
        "alpha_deciles_valid": deciles_valid,
        "hard_soft_ratio_finite": all(math.isfinite(value) for value in hard_soft_ratios),
        "anchor_update_mass_positive": sum(anchor_mass) > 0.0,
    }
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "run_summary_sha256": sha256_path(run_dir / "run_summary.json"),
        "train_log_sha256": sha256_path(run_dir / "train_log.csv"),
        "final_checkpoint_sha256": sha256_path(final_path),
        "parent_checkpoint_sha256": sha256_path(parent_path),
        "manifest_sha256": (run_dir / "manifest_sha256.txt").read_text().strip(),
        "split_sha256": (run_dir / "split_sha256.txt").read_text().strip(),
        "rows": len(rows),
        "global_step": int(summary["completed_global_steps"]),
        "alpha_decile_mean": [sum(float(values[index]) for values in deciles) / len(deciles) for index in range(11)],
        "hard_soft_loss_ratio_mean": sum(hard_soft_ratios) / len(hard_soft_ratios),
        "anchor_update_mass_total": sum(anchor_mass),
        "historical_anchor_max_abs": historical_anchor_error,
        "resume_record": str(resume_files[0]),
        "engineering": engineering,
        "passed": all(engineering.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    project = args.project_root.resolve()
    status_dir = project / "reports/experiment_status"
    json_path = status_dir / "V0_4A_PILOT_REPORT.json"
    md_path = status_dir / "V0_4A_PILOT_REPORT.md"
    if json_path.exists() or md_path.exists():
        raise FileExistsError("refusing to overwrite completed V0.4a pilot report")
    runs = [_validate_run(args.root.resolve(), seed) for seed in (0, 1, 2)]
    passed = all(run["passed"] for run in runs)
    result = {
        "protocol_id": "lcrseg_v0_4a",
        "generated_at": utc_now(),
        "status": "V0_4A_PILOT_GATE_PASSED" if passed else "HARD_STOP_V0_4A_ENGINEERING_FAILURE",
        "engineering_gate_passed": passed,
        "tau": 0.10,
        "rank_schedule": "40_to_80_linear_per_site",
        "pilot_used_for_tuning": False,
        "runs": runs,
    }
    write_json(json_path, result)
    lines = [
        "# LCR-Seg V0.4a engineering pilot report",
        "",
        f"- Status: `{result['status']}`",
        f"- Engineering gate: `{passed}`",
        "- Pilot-based tuning: `False`",
        "",
    ]
    for run in runs:
        lines.extend(
            [
                f"## Seed {run['seed']}",
                "",
                f"- Run: `{run['run_dir']}`",
                f"- Rows/global step: `{run['rows']}` / `{run['global_step']}`",
                f"- Actual checkpoint resume: `{run['engineering']['checkpoint_resume_actual']}`",
                f"- Historical anchor max abs: `{run['historical_anchor_max_abs']}`",
                f"- Anchor update mass: `{run['anchor_update_mass_total']}`",
                f"- Passed: `{run['passed']}`",
                "",
            ]
        )
    write_text(md_path, "\n".join(lines))
    print(json.dumps({"status": result["status"], "passed": passed}, sort_keys=True))


if __name__ == "__main__":
    main()
