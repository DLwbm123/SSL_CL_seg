#!/usr/bin/env python3
"""Supervised two-case relation-field overfit gate using the shared LCR method."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.contracts import UnlabeledBatch
from lcrseg.data import H5LabeledDataset, collate_labeled
from lcrseg.engine.checkpoint import checkpoint_payload, save_checkpoint
from lcrseg.engine.trainer import Trainer, TrainerState, build_optimizer, build_scheduler
from lcrseg.methods.components.relation_field import relation_field
from lcrseg.methods.lcrseg_v0_1 import LCRSegV01Method
from lcrseg.models import UNet2D


SPECS = {
    "fundus": (3, 3, "REFUGE"),
    "prostate": (1, 2, "RUNMC"),
    "mnms": (1, 4, "Siemens"),
}


def _choose(source: H5LabeledDataset) -> list[int]:
    selected: list[int] = []
    seen: set[str] = set()
    for index, sample in enumerate(source.samples):
        case_id = sample.row["case_id"]
        if case_id not in seen:
            selected.append(index)
            seen.add(case_id)
        if len(selected) == 2:
            return selected
    raise RuntimeError("relation overfit needs two distinct labeled cases")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("LCRSEG_DATA_ROOT", "/home/jiangsuiyang/SSL_CL")))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("LCRSEG_RUN_ROOT", "/home/jiangsuiyang/SSL_CL/runs")))
    parser.add_argument("--dataset", choices=tuple(SPECS), default="fundus")
    parser.add_argument("--site", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    channels, classes, default_site = SPECS[args.dataset]
    site = args.site or default_site
    root, run_root = args.root.resolve(), args.run_root.resolve()
    output = run_root / "m2" / f"two_case_relation_overfit_{args.dataset}_{site}_seed{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    source = H5LabeledDataset(root, seed=args.seed, dataset=args.dataset, sites=(site,))
    batch = collate_labeled([source[index] for index in _choose(source)])
    device = torch.device(args.device)
    batch = batch.to(device)
    unlabeled = UnlabeledBatch(
        weak_image=batch.image.detach().clone(), strong_image=batch.image.detach().clone(),
        strong_valid_mask=torch.ones_like(batch.valid_mask), case_id=[f"relation_gate_{index}" for index in range(len(batch.case_id))],
        patient_id=list(batch.patient_id), site=list(batch.site), slice_index=list(batch.slice_index), geometry_record=[{} for _ in batch.case_id],
    )
    method = LCRSegV01Method(
        UNet2D(channels, classes).to(device),
        config={
            "anchor_bootstrap_steps": 1,
            "anchor_min_support_pixels": 1,
            "anchor_max_pixels_per_class": 2048,
            "background_boundary_exclusion": 0,
            "lambda_assim": 0.0,
            "lambda_relation": 0.0,
            "assim_ramp_steps": 1,
            "relation_ramp_steps": 1,
            "tau_cls": 1.0,
            "tau_anchor": 1.0,
        },
    ).to(device)
    method.begin_site(site, None, args.steps)
    optimizer = build_optimizer(method, lr=args.learning_rate, weight_decay=1.0e-5)
    trainer = Trainer(method, optimizer=optimizer, scheduler=build_scheduler(optimizer, total_steps=args.steps), device=device, amp=device.type == "cuda")
    rows = []
    for step in range(args.steps):
        result = trainer.train_step(batch, unlabeled, state=TrainerState(global_step=step, site_step=step, epoch=0))
        rows.append({"step": step + 1, **{name: float(value.detach()) for name, value in result.losses.items()}})
    method.model.eval()
    with torch.no_grad():
        output_model = method.model(batch.image)
        relation = relation_field(output_model.relation_features, method.current_anchor_bank, temperature=float(method.config["relation_temperature"]))
        grid_label = F.interpolate(batch.label.unsqueeze(1).float(), size=relation.probabilities.shape[-2:], mode="nearest")[:, 0].long()
        relation_accuracy = float(relation.predicted_class.eq(grid_label).float().mean())
        segmentation_grid = F.interpolate(output_model.logits, size=grid_label.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
        agreement = float(segmentation_grid.eq(relation.predicted_class).float().mean())
    method.end_site(site)
    state = method.method_state_dict()
    manifest = root / "manifests" / "training" / f"lcrseg_v1_seed{args.seed}.csv"
    split = root / "splits" / f"{args.dataset}_seed{args.seed}.json"
    checkpoint = checkpoint_payload(
        method_name=method.method_name, method_version=method.method_version, git_commit="NO_GIT_WORKTREE",
        config_resolved={"dataset": args.dataset, "site": site, "steps": args.steps, "learning_rate": args.learning_rate},
        site_id=site, site_index=0, epoch=0, site_step=args.steps, global_step=args.steps,
        current_model_state=method.model.state_dict(), optimizer_state=optimizer.state_dict(), scheduler_state=trainer.scheduler.state_dict(), scaler_state=trainer.scaler.state_dict(),
        current_anchor_state=state["current_anchor_state"], historical_anchor_state=state["historical_anchor_state"], bootstrap_state=state["bootstrap_state"], method_statistics=state["method_statistics"],
        data_split_hash=__import__("hashlib").sha256(split.read_bytes()).hexdigest(), manifest_hash=__import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
    )
    save_checkpoint(output / "checkpoint_final.pt", checkpoint)
    with (output / "loss_curve.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "dataset": args.dataset, "site": site, "seed": args.seed, "steps": args.steps,
        "initial_loss": rows[0]["loss_sup"], "final_loss": rows[-1]["loss_sup"],
        "relation_accuracy": relation_accuracy, "segmentation_relation_agreement": agreement,
        "anchor_valid": method.current_anchor_bank.valid.detach().cpu().tolist(),
        "acceptance": {
            "loss_decreased": rows[-1]["loss_sup"] < rows[0]["loss_sup"],
            "relation_accuracy_ge_0_90": relation_accuracy >= 0.90,
            "segmentation_relation_agreement_ge_0_90": agreement >= 0.90,
            "all_anchors_valid": method.current_anchor_bank.all_classes_valid,
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device)},
        "output_dir": str(output),
    }
    (output / "relation_two_case_overfit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not all(report["acceptance"].values()):
        raise SystemExit("relation two-case overfit acceptance failed")


if __name__ == "__main__":
    main()
