#!/usr/bin/env python3
"""Exact V0.2a-R1 to V0.3-P0 REFUGE bridge using the frozen checkpoint."""
from __future__ import annotations

import argparse
import copy
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
from lcrseg.data import DeterministicBatcher, collate_labeled, collate_unlabeled
from lcrseg.engine.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state
from lcrseg.engine.continual_runner import ContinualRunner
from lcrseg.engine.trainer import seed_everything
from lcrseg.methods.lcrseg_v0_2a import LCRSegV02AMethod
from lcrseg.methods.lcrseg_v0_3 import FROZEN_V02A_R1_SITE0_SHA256, LCRSegV03Method
from lcrseg.models import UNet2D


def _max_abs(first: Any, second: Any) -> float:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        if first.shape != second.shape:
            return float("inf")
        if first.dtype == torch.bool or not first.is_floating_point():
            return 0.0 if torch.equal(first, second) else float("inf")
        return float((first.detach().float() - second.detach().float()).abs().max().cpu()) if first.numel() else 0.0
    if isinstance(first, dict) and isinstance(second, dict):
        if set(first) != set(second):
            return float("inf")
        return max((_max_abs(first[key], second[key]) for key in first), default=0.0)
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            return float("inf")
        return max((_max_abs(left, right) for left, right in zip(first, second, strict=True)), default=0.0)
    return 0.0 if first == second else float("inf")


def _environment() -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpus": query,
    }


def _load_frozen_site_state(method, payload: dict[str, Any], *, protocol_semantics: dict[str, Any] | None = None) -> None:
    method.set_site_index(0)
    method.begin_site("REFUGE", None, 8000)
    method.model.load_state_dict(payload["current_model_state"], strict=True)
    state = copy.deepcopy(payload)
    if protocol_semantics is not None:
        statistics = dict(state["method_statistics"])
        statistics["protocol_semantics"] = protocol_semantics
        state["method_statistics"] = statistics
    method.load_method_state_dict(state)
    method.set_training_context(epoch=199, steps_per_epoch=40)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/experiment_status/V0_3_P0_SITE1_BRIDGE.json",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite bridge report: {args.output}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("deterministic V0.3 P0 site-1 bridge requires CUDA_VISIBLE_DEVICES=4")
    parent = args.parent.resolve()
    if sha256_path(parent) != FROZEN_V02A_R1_SITE0_SHA256:
        raise ValueError("R1 REFUGE parent hash differs from the frozen preregistration")
    payload = load_checkpoint(parent, map_location="cpu")
    if (payload["site_id"], int(payload["site_index"]), int(payload["global_step"])) != ("REFUGE", 0, 8000):
        raise ValueError("bridge parent is not the frozen REFUGE site-end checkpoint")

    device = torch.device(args.device)
    seed_everything(0, deterministic=True)
    config_r1 = json.loads((PROJECT_ROOT / "configs/experiments/lcrseg_v0_2a_r1.yaml").read_text())
    runner = ContinualRunner(config_r1)
    labeled_dataset, unlabeled_dataset = runner._datasets(("REFUGE",))
    labeled_batcher = DeterministicBatcher(
        labeled_dataset,
        batch_size=2,
        seed=0,
        namespace="fundus:REFUGE:labeled:REFUGE",
        collate=collate_labeled,
    )
    unlabeled_batcher = DeterministicBatcher(
        unlabeled_dataset,
        batch_size=4,
        seed=0,
        namespace="fundus:REFUGE:unlabeled:REFUGE",
        collate=collate_unlabeled,
    )
    labeled = labeled_batcher.batch_at(0).to(device)
    unlabeled = unlabeled_batcher.batch_at(0).to(device)

    old_r1 = LCRSegV02AMethod(UNet2D(3, 3).to(device), config=config_r1["method"]).to(device)
    config_p0 = json.loads((PROJECT_ROOT / "configs/experiments/lcrseg_v0_3_p0.yaml").read_text())
    new_p0 = LCRSegV03Method(UNet2D(3, 3).to(device), config=config_p0["method"]).to(device)
    _load_frozen_site_state(old_r1, payload)
    _load_frozen_site_state(new_p0, payload, protocol_semantics=new_p0.protocol_semantics())

    with torch.no_grad():
        first_output = old_r1.model(labeled.image)
        second_output = new_p0.model(labeled.image)
        first_relation = old_r1._relation(first_output.relation_features, old_r1.current_anchor_bank)
        second_relation = new_p0._relation(second_output.relation_features, new_p0.current_anchor_bank)
    tensor_errors = {
        "logits": _max_abs(first_output.logits, second_output.logits),
        "relation_distribution": _max_abs(first_relation.probabilities, second_relation.probabilities),
    }
    rng = capture_rng_state()
    first = old_r1.training_step(labeled, unlabeled, global_step=7999, site_step=7999)
    restore_rng_state(rng)
    second = new_p0.training_step(labeled, unlabeled, global_step=7999, site_step=7999)
    assert first.maps is not None and second.maps is not None
    for key in ("learnability", "admission_mask", "pseudo_labels", "pseudo_valid", "current_relation_probability"):
        tensor_errors[key] = _max_abs(first.maps[key], second.maps[key])
    for index, (left, right) in enumerate(zip(old_r1._pending_anchor_updates, new_p0._pending_anchor_updates, strict=True)):
        for tensor_index, (left_value, right_value) in enumerate(zip(left[:3], right[:3], strict=True)):
            tensor_errors[f"anchor_proposal_{index}_{tensor_index}"] = _max_abs(left_value, right_value)
    loss_errors = {
        key: abs(float(first.losses[key].detach()) - float(second.losses[key].detach()))
        for key in ("loss_sup", "loss_assim", "loss_relation")
    }
    loss_errors["total_loss"] = abs(float(first.total_loss.detach()) - float(second.total_loss.detach()))
    count_errors = {
        key: abs(float(first.scalars[key]) - float(second.scalars[key]))
        for key in ("pseudo_valid_count", "assim_candidate_count", "assim_selected_count")
    }
    optimizer_error = _max_abs(payload["optimizer_state"], copy.deepcopy(payload["optimizer_state"]))
    scheduler_exact = _max_abs(payload["scheduler_state"], copy.deepcopy(payload["scheduler_state"])) == 0.0
    passed = (
        max(tensor_errors.values(), default=0.0) <= 1.0e-6
        and max(loss_errors.values(), default=0.0) <= 1.0e-7
        and max(count_errors.values(), default=0.0) == 0.0
        and optimizer_error == 0.0
        and scheduler_exact
    )
    report = {
        "protocol_id": "lcrseg_v0_3",
        "status": "PASSED" if passed else "HARD_STOP_P0_SITE1_MISMATCH",
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": sha256_path(parent),
        "batch_case_ids": {"labeled": labeled.case_id, "unlabeled": unlabeled.case_id},
        "tensor_max_abs_error": tensor_errors,
        "loss_abs_error": loss_errors,
        "count_abs_error": count_errors,
        "optimizer_max_state_error": optimizer_error,
        "scheduler_state_exact": scheduler_exact,
        "thresholds": {"tensor_max_abs": 1.0e-6, "loss_abs": 1.0e-7, "counts": "exact"},
        "environment": _environment(),
        "passed": passed,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

