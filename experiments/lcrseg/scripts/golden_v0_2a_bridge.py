#!/usr/bin/env python3
"""Real-checkpoint, real-batch golden bridge from V0.1 to V0.2a R0."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import sha256_path, write_json
from lcrseg.data import DeterministicBatcher, H5LabeledDataset, H5UnlabeledDataset, LabeledTransform, WeakStrongTransform, collate_labeled, collate_unlabeled
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from lcrseg.engine.trainer import seed_everything
from lcrseg.methods.lcrseg_v0_1 import LCRSegV01Method
from lcrseg.methods.lcrseg_v0_2a import LCRSegV02AMethod
from lcrseg.models import UNet2D


def _max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape:
        return float("inf")
    if first.dtype == torch.bool or not first.is_floating_point():
        return 0.0 if torch.equal(first, second) else float("inf")
    return float((first.detach().float() - second.detach().float()).abs().max().cpu())


def _environment() -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpus": query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/experiment_status/V0_2A_R0_GOLDEN_BRIDGE.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("deterministic golden bridge requires CUDA_VISIBLE_DEVICES=4")
    device = torch.device(args.device)
    seed_everything(0, deterministic=True)
    labeled_dataset = H5LabeledDataset(args.root, seed=0, dataset="fundus", sites=("RIM_ONE_r3",), transform=LabeledTransform(flip_probability=0.5))
    unlabeled_dataset = H5UnlabeledDataset(args.root, seed=0, dataset="fundus", sites=("RIM_ONE_r3",), transform=WeakStrongTransform())
    labeled_batcher = DeterministicBatcher(labeled_dataset, batch_size=2, seed=0, namespace="fundus:RIM_ONE_r3:labeled:RIM_ONE_r3", collate=collate_labeled)
    unlabeled_batcher = DeterministicBatcher(unlabeled_dataset, batch_size=4, seed=0, namespace="fundus:RIM_ONE_r3:unlabeled:RIM_ONE_r3", collate=collate_unlabeled)
    labeled = labeled_batcher.batch_at(0).to(device)
    unlabeled = unlabeled_batcher.batch_at(0).to(device)
    legacy = LCRSegV01Method(UNet2D(3, 3).to(device), config={"use_learnability": True, "use_compatibility": False}).to(device)
    amended = LCRSegV02AMethod(UNet2D(3, 3).to(device), config={"variant_id": "R0", "assimilation_mode": "legacy_continuous_v01", "consolidation_mode": "uniform_relation"}).to(device)
    legacy.set_site_index(1)
    amended.set_site_index(1)
    legacy.begin_site("RIM_ONE_r3", args.parent, 3200)
    amended.begin_site("RIM_ONE_r3", args.parent, 3200)
    state = capture_rng_state()
    first = legacy.training_step(labeled, unlabeled, global_step=8000, site_step=0)
    restore_rng_state(state)
    second = amended.training_step(labeled, unlabeled, global_step=8000, site_step=0)
    tensor_errors: dict[str, float] = {}
    assert first.maps is not None and second.maps is not None
    for key in ("pseudo_labels", "pseudo_valid", "learnability", "current_relation_probability", "old_relation_probability"):
        tensor_errors[key] = _max_abs(first.maps[key], second.maps[key])
    for update_index, (left, right) in enumerate(zip(legacy._pending_anchor_updates, amended._pending_anchor_updates, strict=True)):
        for tensor_index, (left_value, right_value) in enumerate(zip(left[:3], right[:3], strict=True)):
            tensor_errors[f"anchor_proposal_{update_index}_{tensor_index}"] = _max_abs(left_value, right_value)
    loss_errors = {
        key: abs(float(first.losses[key].detach()) - float(second.losses[key].detach()))
        for key in ("loss_sup", "loss_assim", "loss_relation")
    }
    loss_errors["total_loss"] = abs(float(first.total_loss.detach()) - float(second.total_loss.detach()))
    scalar_errors = {
        key: abs(float(first.scalars[key]) - float(second.scalars[key]))
        for key in ("assimilation_denominator", "relation_denominator", "pseudo_valid_count")
    }
    counts_exact = scalar_errors["pseudo_valid_count"] == 0.0
    passed = (
        max(tensor_errors.values(), default=0.0) <= 1.0e-6
        and max(loss_errors.values(), default=0.0) <= 1.0e-7
        and scalar_errors["assimilation_denominator"] <= 1.0e-6
        and scalar_errors["relation_denominator"] <= 1.0e-6
        and counts_exact
    )
    report = {
        "protocol_id": "lcrseg_v0_2a",
        "status": "PASSED" if passed else "HARD_STOP_R0_GOLDEN_BRIDGE_MISMATCH",
        "parent_checkpoint": str(args.parent),
        "parent_checkpoint_sha256": sha256_path(args.parent),
        "batch_case_ids": {"labeled": labeled.case_id, "unlabeled": unlabeled.case_id},
        "tensor_max_abs_error": tensor_errors,
        "loss_abs_error": loss_errors,
        "scalar_abs_error": scalar_errors,
        "integer_counts_exact": counts_exact,
        "thresholds": {"tensor_max_abs": 1.0e-6, "loss_abs": 1.0e-7, "integer_counts": "exact"},
        "environment": _environment(),
        "passed": passed,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
