#!/usr/bin/env python3
"""Registered A1 cosine-classifier two-case overfit gate."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import sha256_path, write_json  # noqa: E402
from lcrseg.data import H5LabeledDataset, collate_labeled  # noqa: E402
from lcrseg.engine.checkpoint import checkpoint_payload, load_checkpoint, save_checkpoint  # noqa: E402
from lcrseg.engine.metrics import masked_cross_entropy, multiclass_dice, multiclass_dice_loss  # noqa: E402
from lcrseg.models import CosineSegmentationHead, UNet2D  # noqa: E402


def _two_indices(dataset: H5LabeledDataset) -> list[int]:
    by_case: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        by_case.setdefault(sample.row["case_id"], []).append(index)
    if len(by_case) < 2:
        raise RuntimeError("A1 overfit requires two labeled cases")
    return [indices[len(indices) // 2] for indices in list(by_case.values())[:2]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs/srgas_a1_two_case_overfit"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    source = H5LabeledDataset(args.root, seed=0, dataset="fundus", sites=("REFUGE",))
    batch = collate_labeled([source[index] for index in _two_indices(source)]).to(device)
    model = UNet2D(3, 3).to(device)
    model.segmentation_head = CosineSegmentationHead.from_conv2d(model.segmentation_head, temperature=10.0, eps=1.0e-8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    rows: list[dict[str, float]] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        context = torch.cuda.amp.autocast(enabled=True) if device.type == "cuda" else nullcontext()
        with context:
            output = model(batch.image)
            ce = masked_cross_entropy(output.logits, batch.label, batch.valid_mask)
            dice_loss = multiclass_dice_loss(output.logits, batch.label, batch.valid_mask)
            loss = ce + dice_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite A1 overfit loss at step {step}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            dice = multiclass_dice(output.logits, batch.label, batch.valid_mask)
        rows.append({"step": step + 1, "loss_total": float(loss), "mean_foreground_dice": float(dice.mean()), "dice_class_1": float(dice[0]), "dice_class_2": float(dice[1])})
    model.eval()
    with torch.no_grad():
        final_logits = model(batch.image).logits
        final_dice = multiclass_dice(final_logits, batch.label, batch.valid_mask).cpu()
    manifest = args.root / "manifests/training/lcrseg_v1_seed0.csv"
    split = args.root / "splits/fundus_seed0.json"
    payload = checkpoint_payload(
        method_name="srgas_a1_cosine_overfit",
        method_version="0.1a",
        git_commit="NO_GIT_WORKTREE",
        config_resolved={"protocol_id": "srgas_v0_1", "variant": "A1", "steps": args.steps},
        site_id="REFUGE", site_index=0, epoch=0, site_step=args.steps, global_step=args.steps,
        current_model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), scheduler_state={}, scaler_state=scaler.state_dict(),
        current_anchor_state={}, historical_anchor_state={}, bootstrap_state={"complete": False}, method_statistics={"hidden_gt_training_usage": 0},
        data_split_hash=sha256_path(split), manifest_hash=sha256_path(manifest),
    )
    checkpoint = args.output_dir / "checkpoint_final.pt"
    save_checkpoint(checkpoint, payload)
    restored = UNet2D(3, 3).to(device)
    restored.segmentation_head = CosineSegmentationHead.from_conv2d(restored.segmentation_head).to(device)
    restored.load_state_dict(load_checkpoint(checkpoint, map_location=device)["current_model_state"])
    restored.eval()
    with torch.no_grad():
        restore_error = float((restored(batch.image).logits - final_logits).abs().max())
    with (args.output_dir / "overfit_curve.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "SRGAS_A1_TWO_CASE_OVERFIT_PASSED" if float(final_dice.mean()) >= 0.95 and restore_error <= 1.0e-6 else "SRGAS_A1_TWO_CASE_OVERFIT_FAILED",
        "case_ids": batch.case_id,
        "steps": args.steps,
        "initial_loss": rows[0]["loss_total"],
        "final_loss": rows[-1]["loss_total"],
        "final_mean_foreground_dice": float(final_dice.mean()),
        "final_dice_class_1": float(final_dice[0]),
        "final_dice_class_2": float(final_dice[1]),
        "checkpoint_restore_max_abs_error": restore_error,
        "checkpoint_sha256": sha256_path(checkpoint),
        "hidden_gt_training_usage": 0,
    }
    write_json(args.output_dir / "two_case_overfit.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if report["status"].endswith("FAILED"):
        raise SystemExit("A1 cosine two-case overfit gate failed")


if __name__ == "__main__":
    main()
