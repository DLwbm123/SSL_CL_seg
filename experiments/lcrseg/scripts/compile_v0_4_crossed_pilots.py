#!/usr/bin/env python3
"""Validate and compile all 12 preregistered V0.4 diagnostic-only crossed pilots."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.v0_4 import SITE_ORDER, load_frozen_method
from lcrseg.common import read_csv, sha256_path, write_csv, write_json
from lcrseg.engine.evaluator import evaluate_sites
from scripts.run_v0_4_crossed_pilot_queue import registered_jobs


def _float_rows(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value not in {"", "None", "nan"}:
            values.append(float(value))
    return values


def _run_row(root: Path, job: dict[str, object]) -> dict[str, Any]:
    run_dir = root / "runs" / str(job["run_name"])
    summary_path = run_dir / "run_summary.json"
    final_checkpoint = run_dir / "checkpoint_final_site1_RIM_ONE_r3.pt"
    if not summary_path.is_file() or not final_checkpoint.is_file():
        raise RuntimeError(f"crossed pilot is incomplete: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if not bool(config["experiment"].get("diagnostic_only")) or bool(config["experiment"].get("formal_method_result")):
        raise RuntimeError(f"pilot lacks diagnostic-only provenance: {run_dir}")
    if summary.get("status") != "complete" or int(summary.get("new_optimizer_steps", -1)) != 1000:
        raise RuntimeError(f"pilot completion contract failed: {run_dir}")
    train = read_csv(run_dir / "train_log.csv")
    if len(train) != 1000 or int(float(train[-1]["global_step"])) != 9000:
        raise RuntimeError(f"pilot train log is not exactly 1,000 continuation steps: {run_dir}")
    matrix = read_csv(run_dir / "site_matrix_long.csv")
    metrics = {row["evaluation_site"]: float(row["mean_foreground_dice"]) for row in matrix}
    if set(metrics) != set(SITE_ORDER):
        raise RuntimeError(f"pilot final validation sites are incomplete: {run_dir}")
    site_summary = json.loads((run_dir / "site_summary_1_RIM_ONE_r3.json").read_text(encoding="utf-8"))
    anchor_drift = np.asarray(site_summary["anchor_diagnostics"]["drift"], dtype=np.float64).reshape(-1)
    admission = _float_rows(train, "assim_selected_fraction")
    gradient = _float_rows(train, "gradient_cosine_assim_relation")
    hidden = _float_rows(train, "hidden_gt_training_usage")
    old_gradient = _float_rows(train, "old_model_gradient_detected")
    amp_skips = _float_rows(train, "amp_step_skipped")
    loss_total = np.asarray(_float_rows(train, "loss_total"), dtype=np.float64)
    if not np.isfinite(loss_total).all():
        raise RuntimeError(f"pilot contains NaN/Inf loss: {run_dir}")
    return {
        "record_type": "run_summary",
        **job,
        "parent": str(job["parent"]),
        "run_dir": str(run_dir),
        "status": "complete",
        "optimizer_steps": 1000,
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256_path(final_checkpoint),
        "final_average_dice": float(summary["summary"]["final_average_dice"]),
        "refuge_dice": metrics["REFUGE"],
        "rim_one_dice": metrics["RIM_ONE_r3"],
        "drishti_dice": metrics["Drishti_GS"],
        "admission_fraction_mean": float(np.mean(admission)) if admission else float("nan"),
        "admission_fraction_variance": float(np.var(admission)) if admission else float("nan"),
        "anchor_drift_mean": float(anchor_drift.mean()),
        "anchor_drift_std_by_class": float(anchor_drift.std()),
        "gradient_cosine_mean": float(np.mean(gradient)) if gradient else float("nan"),
        "gradient_cosine_variance": float(np.var(gradient)) if gradient else float("nan"),
        "hidden_gt_training_usage_max": max(hidden, default=0.0),
        "old_model_gradient_detected_max": max(old_gradient, default=0.0),
        "amp_skip_count": int(sum(value > 0 for value in amp_skips)),
        "nan_inf_count": 0,
        "split_hash": summary["split_hash"],
        "manifest_hash": summary["manifest_hash"],
    }


@torch.no_grad()
def _trajectory_rows(root: Path, job: dict[str, object], device: torch.device) -> list[dict[str, Any]]:
    run_dir = root / "runs" / str(job["run_name"])
    rows: list[dict[str, Any]] = []
    for global_step in (8250, 8500, 8750, 9000):
        checkpoint = run_dir / f"checkpoint_step_{global_step:06d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        method, _ = load_frozen_method(checkpoint, device)
        evaluation = evaluate_sites(
            method.model,
            data_root=root,
            seed=int(job["split_seed"]),
            dataset="fundus",
            sites=SITE_ORDER,
            num_classes=3,
            role="val",
            device=device,
            batch_size=4,
        )
        for metric in evaluation.per_site:
            rows.append(
                {
                    "record_type": "validation_trajectory",
                    **job,
                    "parent": str(job["parent"]),
                    "run_dir": str(run_dir),
                    "status": "complete",
                    "global_step": global_step,
                    "continuation_step": global_step - 8000,
                    "evaluation_site": metric["site"],
                    "trajectory_mean_foreground_dice": metric["mean_foreground_dice"],
                    "trajectory_dice_class_1": metric["dice_class_1"],
                    "trajectory_dice_class_2": metric["dice_class_2"],
                    "trajectory_checkpoint": str(checkpoint),
                    "trajectory_checkpoint_sha256": sha256_path(checkpoint),
                }
            )
    return rows


def _sensitivity(run_rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    selected = [row for row in run_rows if row["family"] == family]
    result: dict[str, Any] = {}
    for variant in ("R0", "R1"):
        rows = [row for row in selected if row["variant"] == variant]
        result[variant] = {
            key: float(np.std([float(row[key]) for row in rows], ddof=0))
            for key in (
                "final_average_dice",
                "refuge_dice",
                "rim_one_dice",
                "admission_fraction_mean",
                "anchor_drift_mean",
                "gradient_cosine_mean",
            )
        }
    result["r1_more_sensitive_final_dice"] = result["R1"]["final_average_dice"] > result["R0"]["final_average_dice"]
    result["r1_more_sensitive_any_mechanism"] = any(
        result["R1"][key] > result["R0"][key]
        for key in ("admission_fraction_mean", "anchor_drift_mean", "gradient_cosine_mean")
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "crossed_pilot_results.csv"
    json_path = output_dir / "crossed_pilot_summary.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("refusing to overwrite completed V0.4 crossed-pilot outputs")
    jobs = registered_jobs(root)
    run_rows = [_run_row(root, job) for job in jobs]
    trajectory: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        print(f"EVAL {index}/{len(jobs)} {job['run_name']}", flush=True)
        trajectory.extend(_trajectory_rows(root, job, torch.device(args.device)))
    all_rows = run_rows + trajectory
    fieldnames = list(dict.fromkeys(key for row in all_rows for key in row))
    write_csv(csv_path, all_rows, fieldnames=fieldnames)
    optimization = _sensitivity(run_rows, "O")
    split = _sensitivity(run_rows, "S")
    summary = {
        "protocol_id": "lcrseg_v0_4_failure_audit",
        "status": "complete",
        "diagnostic_only": True,
        "formal_method_result": False,
        "runs": run_rows,
        "trajectory_rows": len(trajectory),
        "optimization_sensitivity": optimization,
        "split_sensitivity": split,
        "r1_more_optimization_sensitive": bool(
            optimization["r1_more_sensitive_final_dice"] and optimization["r1_more_sensitive_any_mechanism"]
        ),
        "engineering": {
            "all_12_complete": len(run_rows) == 12,
            "hidden_gt_training_usage_zero": max(row["hidden_gt_training_usage_max"] for row in run_rows) == 0,
            "old_model_gradient_zero": max(row["old_model_gradient_detected_max"] for row in run_rows) == 0,
            "nan_inf_zero": max(row["nan_inf_count"] for row in run_rows) == 0,
            "amp_skip_zero": max(row["amp_skip_count"] for row in run_rows) == 0,
        },
    }
    write_json(json_path, summary)
    print(json.dumps({"status": "complete", "runs": len(run_rows), "trajectory_rows": len(trajectory)}, sort_keys=True))


if __name__ == "__main__":
    main()
