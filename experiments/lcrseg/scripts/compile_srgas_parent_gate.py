#!/usr/bin/env python3
"""Compile the immutable A1 REFUGE common-parent acceptance gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import sha256_path, write_json, write_text  # noqa: E402
from lcrseg.data import H5LabeledDataset, collate_labeled  # noqa: E402
from lcrseg.engine.checkpoint import load_checkpoint  # noqa: E402
from lcrseg.models import CosineSegmentationHead, UNet2D  # noqa: E402


def _row(path: Path) -> dict[str, str]:
    rows = [row for row in csv.DictReader(path.open()) if row["trained_site"] == "REFUGE" and row["evaluation_site"] == "REFUGE"]
    if len(rows) != 1:
        raise RuntimeError(f"expected one REFUGE-to-REFUGE row in {path}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs/fundus_seed0_srgas_a1_cosine_site1"))
    parser.add_argument("--a0-run", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs/fundus_seed0_lcrseg_uniform_relation_kd_full200e"))
    parser.add_argument("--overfit-run", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs/srgas_a1_two_case_overfit"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--report-root", type=Path, default=PROJECT_ROOT / "reports/experiment_status")
    args = parser.parse_args()
    required = ("run_summary.json", "site_matrix_long.csv", "train_log.csv", "checkpoint_final_site0_REFUGE.pt", "checkpoint_final.pt")
    missing = [name for name in required if not (args.parent_run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"parent is incomplete: {missing}")
    parent = _row(args.parent_run / "site_matrix_long.csv")
    a0 = _row(args.a0_run / "site_matrix_long.csv")
    train_rows = list(csv.DictReader((args.parent_run / "train_log.csv").open()))
    if not train_rows:
        raise RuntimeError("parent train log is empty")
    finite = all(
        math.isfinite(float(row[key]))
        for row in train_rows
        for key in row
        if key.startswith("loss_") and row[key] not in ("", None)
    )
    amp_skips = sum(float(row.get("optimizer_step_skipped") or 0.0) for row in train_rows)
    hidden = sum(float(row.get("hidden_gt_training_usage") or 0.0) for row in train_rows)
    overfit = json.loads((args.overfit_run / "two_case_overfit.json").read_text())
    final_checkpoint = args.parent_run / "checkpoint_final_site0_REFUGE.pt"
    payload = load_checkpoint(final_checkpoint, map_location="cpu")
    model = UNet2D(3, 3)
    model.segmentation_head = CosineSegmentationHead.from_conv2d(model.segmentation_head)
    model.load_state_dict(payload["current_model_state"], strict=True)
    model.eval()
    source = H5LabeledDataset(args.data_root, seed=0, dataset="fundus", sites=("REFUGE",), roles=("val",), transform=None)
    batch = collate_labeled([source[0], source[1]])
    with torch.no_grad():
        first = model(batch.image).logits
        second = model(batch.image).logits
    deterministic_error = float((first - second).abs().max())
    metrics = {}
    checks = {
        "parent_run_summary_present": True,
        "completed_exact_8000_steps": len(train_rows) == 8000 and int(train_rows[-1]["site_step"]) == 8000,
        "all_losses_finite": finite,
        "amp_skip_zero": amp_skips == 0,
        "hidden_gt_training_usage_zero": hidden == 0,
        "checkpoint_resume_deterministic": deterministic_error == 0.0,
        "overfit_foreground_dice_ge_0_95": float(overfit["final_mean_foreground_dice"]) >= 0.95,
    }
    thresholds = {"mean_foreground_dice": 0.015, "dice_class_1": 0.020, "dice_class_2": 0.020}
    for name, threshold in thresholds.items():
        baseline = float(a0[name])
        candidate = float(parent[name])
        drop = baseline - candidate
        metrics[name] = {"a0": baseline, "a1_parent": candidate, "drop": drop, "maximum_drop": threshold}
        checks[f"{name}_drop_within_gate"] = drop <= threshold
    status = "SRGAS_A1_PARENT_GATE_PASSED" if all(checks.values()) else "COSINE_PARENT_GATE_FAILED"
    report = {
        "status": status,
        "parent_run": str(args.parent_run.resolve()),
        "a0_run": str(args.a0_run.resolve()),
        "parent_checkpoint": str(final_checkpoint.resolve()),
        "parent_checkpoint_sha256": sha256_path(final_checkpoint),
        "manifest_sha256": (args.parent_run / "manifest_sha256.txt").read_text().strip(),
        "split_sha256": (args.parent_run / "split_sha256.txt").read_text().strip(),
        "completed_steps": len(train_rows),
        "amp_skip_count": amp_skips,
        "hidden_gt_training_usage": hidden,
        "checkpoint_deterministic_max_abs_error": deterministic_error,
        "overfit_foreground_dice": float(overfit["final_mean_foreground_dice"]),
        "metrics": metrics,
        "checks": checks,
    }
    json_path = args.report_root / "SRGAS_SITE1_PARENT_REPORT.json"
    md_path = args.report_root / "SRGAS_SITE1_PARENT_REPORT.md"
    if json_path.exists() or md_path.exists():
        raise FileExistsError("refusing to overwrite an existing parent gate report")
    write_json(json_path, report)
    lines = [
        "# SR-GAS Site-1 Parent Report",
        "",
        f"Status: `{status}`",
        "",
        f"Parent checkpoint: `{final_checkpoint}`",
        f"Checkpoint SHA256: `{report['parent_checkpoint_sha256']}`",
        f"Completed steps: `{len(train_rows)}`; AMP skips: `{amp_skips}`; hidden-GT use: `{hidden}`.",
        f"Two-case foreground Dice: `{report['overfit_foreground_dice']:.6f}`.",
        "",
        "| Metric | A0 | A1 parent | Drop | Maximum drop |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(f"| {name} | {values['a0']:.6f} | {values['a1_parent']:.6f} | {values['drop']:.6f} | {values['maximum_drop']:.6f} |")
    write_text(md_path, "\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if status != "SRGAS_A1_PARENT_GATE_PASSED":
        raise SystemExit(status)


if __name__ == "__main__":
    main()
