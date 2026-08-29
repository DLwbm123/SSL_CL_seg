#!/usr/bin/env python3
"""Stateless functional virtual-step audit for frozen TARC R0 checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.func import functional_call
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.analysis.tarc_v0_1 import TRANSITIONS, labeled_loader, relation_probabilities  # noqa: E402
from lcrseg.analysis.v0_4 import load_frozen_method  # noqa: E402
from lcrseg.common import write_csv, write_json  # noqa: E402
from lcrseg.engine.metrics import masked_cross_entropy, multiclass_dice_loss  # noqa: E402
from lcrseg.methods.base import relation_supervision_loss  # noqa: E402


VIRTUAL_STEP_NORM = 1.0e-3
UPDATE_BATCHES = 32
VAL_BATCHES = 16


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _take_cycling(loader: Any, count: int, device: torch.device) -> list[Any]:
    result: list[Any] = []
    iterator = iter(loader)
    while len(result) < count:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        result.append(batch.to(device, non_blocking=True))
    return result


def _relation_logits(features: torch.Tensor, anchors: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    view = anchors[:, 0] if anchors.ndim == 3 else anchors
    return torch.einsum(
        "bdhw,cd->bchw",
        F.normalize(features.float(), dim=1),
        F.normalize(view.float(), dim=1),
    ) / float(temperature)


def _supervised_r0_loss(output: Any, batch: Any, native_anchors: torch.Tensor) -> torch.Tensor:
    ce = masked_cross_entropy(output.logits, batch.label, batch.valid_mask)
    dice = multiclass_dice_loss(output.logits, batch.label, batch.valid_mask)
    anchor = relation_supervision_loss(
        _relation_logits(output.relation_features, native_anchors), batch.label, batch.valid_mask
    )
    return ce + dice + 0.1 * anchor


@torch.no_grad()
def _baseline_loss(model: torch.nn.Module, batches: list[Any], native_anchors: torch.Tensor) -> float:
    return sum(float(_supervised_r0_loss(model(batch.image), batch, native_anchors)) for batch in batches) / len(batches)


@torch.no_grad()
def _functional_val_loss(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    batches: list[Any],
    native_anchors: torch.Tensor,
) -> float:
    total = 0.0
    for batch in batches:
        output = functional_call(model, state, (batch.image,))
        total += float(_supervised_r0_loss(output, batch, native_anchors))
    return total / len(batches)


def _variant_gradient(
    *,
    current_model: torch.nn.Module,
    old_model: torch.nn.Module,
    batch: Any,
    old_anchors: torch.Tensor,
    student_anchors: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], float, float]:
    with torch.no_grad():
        old_features = old_model(batch.image).relation_features
        target = relation_probabilities(old_features, old_anchors).detach()
    current_features = current_model(batch.image).relation_features
    student = relation_probabilities(current_features, student_anchors)
    loss = (target * (target.clamp_min(1.0e-8).log() - student.clamp_min(1.0e-8).log())).sum(dim=1).mean()
    parameters = tuple(current_model.parameters())
    raw_gradients = torch.autograd.grad(
        loss, parameters, retain_graph=False, create_graph=False, allow_unused=True
    )
    gradients = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, raw_gradients, strict=True)
    )
    norm = torch.sqrt(sum(gradient.detach().float().square().sum() for gradient in gradients))
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise FloatingPointError("virtual relation gradient is invalid")
    return tuple(gradient.detach() for gradient in gradients), float(norm), float(loss.detach())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_dir = args.output_dir.resolve() / f"seed{args.seed}"
    csv_path = seed_dir / "virtual_step_audit.csv"
    summary_path = seed_dir / "virtual_step_summary.json"
    for path in (csv_path, summary_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite TARC virtual-step audit: {path}")
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    transition_summaries: list[dict[str, Any]] = []
    for old_index, current_index in TRANSITIONS:
        bundle = torch.load(seed_dir / f"transport_{old_index}_{current_index}.pt", map_location="cpu")
        old_method, _ = load_frozen_method(Path(bundle["old_checkpoint"]), device)
        current_method, _ = load_frozen_method(Path(bundle["current_checkpoint"]), device)
        old_model = old_method.model.eval()
        current_model = current_method.model.eval()
        old_model.requires_grad_(False)
        before_hash = _model_hash(current_model)
        anchors = {
            "T0": bundle["native_current_anchors"].to(device),
            "T1": bundle["old_anchors"].to(device),
            "T2": bundle["global_anchors"].to(device),
            "T3": bundle["class_anchors"].to(device),
        }
        native_anchors = anchors["T0"]
        update_batches = _take_cycling(
            labeled_loader(args.data_root.resolve(), seed=args.seed, site_id=bundle["current_site_id"],
                           roles=("train_labeled",), batch_size=1, workers=args.workers),
            UPDATE_BATCHES, device,
        )
        previous_batches = _take_cycling(
            labeled_loader(args.data_root.resolve(), seed=args.seed, site_id=bundle["old_site_id"],
                           roles=("val",), batch_size=4, workers=args.workers),
            VAL_BATCHES, device,
        )
        current_batches = _take_cycling(
            labeled_loader(args.data_root.resolve(), seed=args.seed, site_id=bundle["current_site_id"],
                           roles=("val",), batch_size=4, workers=args.workers),
            VAL_BATCHES, device,
        )
        previous_before = _baseline_loss(current_model, previous_batches, native_anchors)
        current_before = _baseline_loss(current_model, current_batches, native_anchors)
        named_parameters = dict(current_model.named_parameters())
        named_buffers = dict(current_model.named_buffers())
        parameter_names = tuple(named_parameters)
        transition = f"{bundle['old_site_id']}->{bundle['current_site_id']}"
        for update_index, batch in enumerate(update_batches):
            for variant, student_anchors in anchors.items():
                gradients, raw_norm, update_loss = _variant_gradient(
                    current_model=current_model,
                    old_model=old_model,
                    batch=batch,
                    old_anchors=anchors["T1"],
                    student_anchors=student_anchors,
                )
                scale = VIRTUAL_STEP_NORM / raw_norm
                updated_parameters = {
                    name: named_parameters[name] - scale * gradient
                    for name, gradient in zip(parameter_names, gradients, strict=True)
                }
                functional_state = {**named_buffers, **updated_parameters}
                previous_after = _functional_val_loss(current_model, functional_state, previous_batches, native_anchors)
                current_after = _functional_val_loss(current_model, functional_state, current_batches, native_anchors)
                rows.append(
                    {
                        "seed": args.seed,
                        "transition": transition,
                        "old_site_id": bundle["old_site_id"],
                        "current_site_id": bundle["current_site_id"],
                        "update_batch": update_index,
                        "variant": variant,
                        "virtual_step_norm": VIRTUAL_STEP_NORM,
                        "raw_gradient_norm": raw_norm,
                        "relation_update_loss": update_loss,
                        "previous_val_loss_before": previous_before,
                        "previous_val_loss_after": previous_after,
                        "previous_val_loss_delta": previous_after - previous_before,
                        "current_val_loss_before": current_before,
                        "current_val_loss_after": current_after,
                        "current_val_loss_delta": current_after - current_before,
                        "old_model_gradient_nonnull": sum(parameter.grad is not None for parameter in old_model.parameters()),
                        "checkpoint_or_model_mutated": False,
                        "hidden_gt_usage": "none_visible_labeled_and_val_only",
                    }
                )
        after_hash = _model_hash(current_model)
        if before_hash != after_hash:
            raise AssertionError("functional virtual step mutated the frozen current model")
        transition_summaries.append(
            {
                "transition": transition,
                "model_sha256_before": before_hash,
                "model_sha256_after": after_hash,
                "model_unchanged": True,
                "update_batches": UPDATE_BATCHES,
                "previous_val_batches": VAL_BATCHES,
                "current_val_batches": VAL_BATCHES,
            }
        )
        del old_method, current_method, update_batches, previous_batches, current_batches
        torch.cuda.empty_cache()
    write_csv(csv_path, rows)
    write_json(
        summary_path,
        {
            "protocol_id": "tarcseg_v0_1",
            "seed": args.seed,
            "status": "TARC_VIRTUAL_STEP_SEED_AUDIT_COMPLETE",
            "rows": len(rows),
            "optimizer_steps": 0,
            "virtual_step_norm": VIRTUAL_STEP_NORM,
            "transitions": transition_summaries,
        },
    )
    print(json.dumps({"status": "complete", "seed": args.seed, "rows": len(rows), "csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
