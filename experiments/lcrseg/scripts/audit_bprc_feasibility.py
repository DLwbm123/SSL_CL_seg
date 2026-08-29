#!/usr/bin/env python3
"""Run BPRC Part-A gradient and functional-step feasibility for one seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from lcrseg.analysis.v0_4 import load_frozen_method, stable_seed  # noqa: E402
from lcrseg.common import canonical_json, sha256_bytes, sha256_path, write_csv, write_json  # noqa: E402
from lcrseg.data import (  # noqa: E402
    DeterministicBatcher,
    H5LabeledDataset,
    H5UnlabeledDataset,
    LabeledTransform,
    WeakStrongTransform,
    collate_labeled,
    collate_unlabeled,
)
from lcrseg.data.transforms import downsample_valid_mask  # noqa: E402
from lcrseg.engine.metrics import multiclass_dice_loss  # noqa: E402
from lcrseg.losses import pairwise_relation_consolidation  # noqa: E402
from lcrseg.methods.base import relation_supervision_loss  # noqa: E402
from lcrseg.methods.components.compatibility import zero_compatibility  # noqa: E402
from lcrseg.methods.components.learnability import compute_learnability  # noqa: E402
from lcrseg.methods.components.pseudo_label import build_pseudo_labels  # noqa: E402
from lcrseg.methods.components.routing import assimilation_loss, relation_consolidation_loss  # noqa: E402
from lcrseg.methods.lcrseg_v0_1 import _uniform_compatibility  # noqa: E402
from scripts.analyze_tarc_pairwise_geometry import _boundary_mask  # noqa: E402
from scripts.analyze_v0_4_gradient_utility import _diagnostic_labels, _region_masks  # noqa: E402
from scripts.audit_tarc_relation_fidelity import _margin  # noqa: E402
from scripts.audit_tarc_virtual_step import (  # noqa: E402
    _baseline_loss,
    _functional_val_loss,
    _model_hash,
    _supervised_r0_loss,
    _take_cycling,
)


UPDATE_BATCHES = 32
VAL_BATCHES = 16
VIRTUAL_STEP_NORM = 1.0e-3
VARIANT_MODES = {
    "B0": "categorical_pixel_mean",
    "B1": "categorical_class_balanced",
    "B2": "top2_pairwise_class_balanced",
    "B3": "all_pairwise_class_balanced",
}


def _batchers(root: Path, seed: int, site: str) -> tuple[DeterministicBatcher[Any], DeterministicBatcher[Any]]:
    labeled = H5LabeledDataset(
        root, seed=seed, dataset="fundus", sites=(site,), roles=("train_labeled",),
        transform=LabeledTransform(flip_probability=0.5),
    )
    unlabeled = H5UnlabeledDataset(
        root, seed=seed, dataset="fundus", sites=(site,),
        transform=WeakStrongTransform(
            flip_probability=0.5, strong_noise_std=0.03, brightness_delta=0.10,
            contrast_delta=0.10, cutout_probability=0.5, cutout_fraction=0.20,
        ),
    )
    order_seed = stable_seed("bprc-v0.1-fixed-order", seed, site)
    return (
        DeterministicBatcher(
            labeled, batch_size=2, seed=order_seed,
            namespace=f"bprc-v0.1:{seed}:{site}:labeled", collate=collate_labeled,
        ),
        DeterministicBatcher(
            unlabeled, batch_size=4, seed=order_seed,
            namespace=f"bprc-v0.1:{seed}:{site}:unlabeled", collate=collate_unlabeled,
        ),
    )


def _fixed_updates(root: Path, seed: int, site: str) -> tuple[list[tuple[Any, Any]], list[dict[str, Any]]]:
    labeled_batcher, unlabeled_batcher = _batchers(root, seed, site)
    batches: list[tuple[Any, Any]] = []
    manifest: list[dict[str, Any]] = []
    for index in range(UPDATE_BATCHES):
        transform_seed = stable_seed("bprc-v0.1-fixed-transform", seed, site, index)
        torch.manual_seed(transform_seed)
        labeled = labeled_batcher.batch_at(index)
        unlabeled = unlabeled_batcher.batch_at(index)
        entry = {
            "role": "current_update",
            "batch_index": index,
            "transform_seed": transform_seed,
            "labeled_case_ids": labeled.case_id,
            "unlabeled_case_ids": unlabeled.case_id,
        }
        entry["case_id_sha256"] = sha256_bytes(canonical_json(entry).encode("utf-8"))
        batches.append((labeled, unlabeled))
        manifest.append(entry)
    return batches, manifest


def _validation_manifest(role: str, batches: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, batch in enumerate(batches):
        entry = {"role": role, "batch_index": index, "case_ids": batch.case_id}
        entry["case_id_sha256"] = sha256_bytes(canonical_json(entry).encode("utf-8"))
        result.append(entry)
    return result


def _gradients(loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...]) -> tuple[torch.Tensor, ...]:
    raw = torch.autograd.grad(loss, parameters, retain_graph=True, create_graph=False, allow_unused=True)
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, raw, strict=True)
    )


def _norm(gradients: tuple[torch.Tensor, ...]) -> float:
    return float(torch.sqrt(sum(value.float().square().sum() for value in gradients)))


def _cosine(first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...]) -> float:
    first_norm = _norm(first)
    second_norm = _norm(second)
    if first_norm <= 0 or second_norm <= 0:
        return 0.0
    dot = sum((left.float() * right.float()).sum() for left, right in zip(first, second, strict=True))
    return float(dot / (first_norm * second_norm))


def _prepare_objectives(method: Any, payload: dict[str, Any], labeled: Any, unlabeled: Any) -> dict[str, Any]:
    method.model.train()
    if method.old_model is None or method.old_anchor_bank is None:
        raise RuntimeError("BPRC feasibility requires the frozen historical teacher")
    method.old_model.eval()
    labeled_output = method.model(labeled.image)
    labeled_relation = method._relation(labeled_output.relation_features, method.current_anchor_bank)
    sup = method._supervised_losses(
        labeled_output,
        labeled,
        relation_anchor_loss=relation_supervision_loss(
            labeled_relation.logits, labeled.label, labeled.valid_mask
        ),
    )["loss_sup"]
    with torch.no_grad():
        weak_output = method.model(unlabeled.weak_image)
        weak_relation = method._relation(weak_output.relation_features, method.current_anchor_bank)
        pseudo = build_pseudo_labels(
            weak_output.logits.softmax(dim=1), weak_relation,
            tau_cls=float(method.config["tau_cls"]), tau_anchor=float(method.config["tau_anchor"]),
            delta_anchor=float(method.config["delta_anchor"]), tau_spatial=float(method.config["tau_spatial"]),
            temperature_cls=float(method.config["temperature_cls"]),
            temperature_anchor=float(method.config["temperature_anchor"]),
            spatial_floor=float(method.config["spatial_floor"]),
        )
        site_step = max(0, int(payload["site_step"]) - 1)
        learnability = compute_learnability(
            weak_output.logits, weak_relation, pseudo,
            site_step=site_step, total_steps=max(1, int(method.total_steps)),
            rank_start=float(method.config["rank_start"]), rank_end=float(method.config["rank_end"]),
            rank_temperature=float(method.config["rank_temperature"]),
            relation_margin_center=float(method.config["relation_margin_center"]),
            relation_margin_temperature=float(method.config["relation_margin_temperature"]),
            min_rank_pixels=int(method.config["min_rank_pixels"]),
        )
        old_output = method.old_model(unlabeled.weak_image)
        old_relation = method._relation(old_output.relation_features, method.old_anchor_bank)
        compatibility = _uniform_compatibility(zero_compatibility(old_relation.probabilities))
    strong_output = method.model(unlabeled.strong_image)
    strong_relation = method._relation(strong_output.relation_features, method.current_anchor_bank)
    assim = assimilation_loss(
        strong_output.logits, pseudo, learnability, unlabeled.strong_valid_mask
    )
    b0_r0 = relation_consolidation_loss(
        strong_relation, old_relation, compatibility, unlabeled.strong_valid_mask,
        distill_temperature=float(method.config["distill_temperature"]),
    )
    relation_valid = downsample_valid_mask(
        unlabeled.strong_valid_mask, strong_relation.logits.shape[-2:]
    )
    primitive = {
        variant: pairwise_relation_consolidation(
            old_relation_scores=old_relation.logits,
            current_relation_scores=strong_relation.logits,
            valid_mask=relation_valid,
            mode=mode,
        )
        for variant, mode in VARIANT_MODES.items()
    }
    distill_scale = float(method.config["distill_temperature"]) ** 2
    relations = {"B0": b0_r0}
    relations.update({variant: primitive[variant].loss * distill_scale for variant in ("B1", "B2", "B3")})
    b0_primitive_scaled = primitive["B0"].loss * distill_scale
    b0_abs_error = abs(float(b0_r0.detach()) - float(b0_primitive_scaled.detach()))
    bootstrap_at = int(method.bootstrap_state.get("completed_at_site_step", -1))
    lambda_assim = float(method.config["lambda_assim"]) * method._assimilation_ramp(
        site_step, bootstrap_complete_at=bootstrap_at
    )
    lambda_relation = float(method.config["lambda_relation"]) * method._relation_ramp(site_step)
    return {
        "sup": sup,
        "assim": assim,
        "relations": relations,
        "primitive": primitive,
        "old_scores": old_relation.logits,
        "current_scores": strong_relation.logits,
        "relation_valid": relation_valid,
        "lambda_assim": lambda_assim,
        "lambda_relation": lambda_relation,
        "b0_abs_error": b0_abs_error,
    }


@torch.no_grad()
def _old_probabilities(method: Any, batches: list[Any]) -> list[torch.Tensor]:
    if method.old_model is None or method.old_anchor_bank is None:
        raise RuntimeError("missing historical teacher")
    result: list[torch.Tensor] = []
    method.old_model.eval()
    for batch in batches:
        output = method.old_model(batch.image)
        result.append(method._relation(output.relation_features, method.old_anchor_bank).probabilities.detach())
    return result


@torch.no_grad()
def _evaluate(
    *,
    model: torch.nn.Module,
    state: dict[str, torch.Tensor] | None,
    batches: list[Any],
    native_anchors: torch.Tensor,
    old_probabilities: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    loss_total = 0.0
    dice_total = 0.0
    margin_sum = {(class_id, region): 0.0 for class_id in range(3) for region in ("all", "boundary", "interior")}
    margin_count = {(class_id, region): 0 for class_id in range(3) for region in ("all", "boundary", "interior")}
    for index, batch in enumerate(batches):
        output = model(batch.image) if state is None else functional_call(model, state, (batch.image,))
        loss_total += float(_supervised_r0_loss(output, batch, native_anchors))
        dice_total += float(1.0 - multiclass_dice_loss(output.logits, batch.label, batch.valid_mask))
        if old_probabilities is None:
            continue
        current_probability = relation_probabilities(output.relation_features, native_anchors)
        old_probability = old_probabilities[index]
        target = F.interpolate(
            batch.label[:, None].float(), size=current_probability.shape[-2:], mode="nearest"
        )[:, 0].long()
        boundary = _boundary_mask(target)
        agreement = 1.0 - (_margin(current_probability) - _margin(old_probability)).abs()
        for class_id in range(3):
            class_mask = target.eq(class_id)
            for region, region_mask in (
                ("all", torch.ones_like(boundary)),
                ("boundary", boundary),
                ("interior", ~boundary),
            ):
                selected = class_mask & region_mask
                count = int(selected.sum())
                if count:
                    margin_sum[(class_id, region)] += float(agreement[selected].sum())
                    margin_count[(class_id, region)] += count
    margins = {
        f"class{class_id}_{region}": margin_sum[(class_id, region)] / margin_count[(class_id, region)]
        for class_id in range(3)
        for region in ("all", "boundary", "interior")
        if margin_count[(class_id, region)]
    }
    return {
        "loss": loss_total / len(batches),
        "dice": dice_total / len(batches),
        "margins": margins,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tarc-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_dir = args.output_dir.resolve() / f"seed{args.seed}"
    if seed_dir.exists():
        raise FileExistsError(f"refusing to overwrite BPRC feasibility seed: {seed_dir}")
    seed_dir.mkdir(parents=True)
    device = torch.device(args.device)
    gradient_rows: list[dict[str, Any]] = []
    virtual_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    batch_manifest: list[dict[str, Any]] = []
    transition_summaries: list[dict[str, Any]] = []
    for old_index, current_index in TRANSITIONS:
        tarc_seed_dir = args.tarc_analysis_dir.resolve() / f"seed{args.seed}"
        bundle = torch.load(tarc_seed_dir / f"transport_{old_index}_{current_index}.pt", map_location="cpu")
        method, payload = load_frozen_method(Path(bundle["current_checkpoint"]), device)
        method.train()
        if method.old_model is None or method.old_anchor_bank is None:
            raise RuntimeError("incremental R0 checkpoint lacks its frozen historical teacher")
        method.old_model.eval()
        transition = f"{bundle['old_site_id']}->{bundle['current_site_id']}"
        before_model_hash = _model_hash(method.model)
        before_old_hash = _model_hash(method.old_model)
        update_batches, update_manifest = _fixed_updates(
            args.data_root.resolve(), args.seed, bundle["current_site_id"]
        )
        previous_batches = _take_cycling(
            labeled_loader(
                args.data_root.resolve(), seed=args.seed, site_id=bundle["old_site_id"],
                roles=("val",), batch_size=4, workers=args.workers,
            ),
            VAL_BATCHES, device,
        )
        current_batches = _take_cycling(
            labeled_loader(
                args.data_root.resolve(), seed=args.seed, site_id=bundle["current_site_id"],
                roles=("val",), batch_size=4, workers=args.workers,
            ),
            VAL_BATCHES, device,
        )
        for row in update_manifest:
            batch_manifest.append({"seed": args.seed, "transition": transition, **row})
        batch_manifest.extend(
            {"seed": args.seed, "transition": transition, **row}
            for row in _validation_manifest("previous_val", previous_batches)
        )
        batch_manifest.extend(
            {"seed": args.seed, "transition": transition, **row}
            for row in _validation_manifest("current_val", current_batches)
        )
        native_anchors = method.current_anchor_bank.anchors.detach()
        old_probabilities = _old_probabilities(method, previous_batches)
        method.model.eval()
        previous_before = _evaluate(
            model=method.model, state=None, batches=previous_batches,
            native_anchors=native_anchors, old_probabilities=old_probabilities,
        )
        current_before = _evaluate(
            model=method.model, state=None, batches=current_batches,
            native_anchors=native_anchors,
        )
        exact_previous_before = _baseline_loss(method.model, previous_batches, native_anchors)
        exact_current_before = _baseline_loss(method.model, current_batches, native_anchors)
        if abs(previous_before["loss"] - exact_previous_before) > 1.0e-12 or abs(current_before["loss"] - exact_current_before) > 1.0e-12:
            raise AssertionError("combined evaluator diverges from exact TARC baseline loss")
        diagnostic_labels = _diagnostic_labels(
            args.data_root.resolve(), args.seed, bundle["current_site_id"]
        )
        named_parameters = dict(method.model.named_parameters())
        named_buffers = {name: value.detach() for name, value in method.model.named_buffers()}
        parameter_names = tuple(named_parameters)
        parameters = tuple(named_parameters.values())
        for update_index, (labeled_cpu, unlabeled_cpu) in enumerate(update_batches):
            labeled = labeled_cpu.to(device, non_blocking=True)
            unlabeled = unlabeled_cpu.to(device, non_blocking=True)
            objectives = _prepare_objectives(method, payload, labeled, unlabeled)
            if objectives["b0_abs_error"] > 1.0e-7:
                raise AssertionError(f"B0 primitive/R0 loss mismatch: {objectives['b0_abs_error']}")
            sup_gradient = _gradients(objectives["sup"], parameters)
            assim_gradient = _gradients(objectives["assim"], parameters)
            relation_gradients = {
                variant: _gradients(loss, parameters)
                for variant, loss in objectives["relations"].items()
            }
            b0_norm = _norm(relation_gradients["B0"])
            boundary, interior = _region_masks(
                diagnostic_labels, unlabeled_cpu,
                tuple(objectives["relation_valid"].shape[-2:]), device,
            )
            region_outputs: dict[str, dict[str, Any]] = {}
            for variant, mode in VARIANT_MODES.items():
                region_outputs[variant] = {}
                for region, mask in (("boundary", boundary), ("interior", interior)):
                    region_outputs[variant][region] = pairwise_relation_consolidation(
                        old_relation_scores=objectives["old_scores"],
                        current_relation_scores=objectives["current_scores"],
                        valid_mask=objectives["relation_valid"] & mask,
                        mode=mode,
                    )
            for variant, relation_loss in objectives["relations"].items():
                relation_gradient = relation_gradients[variant]
                relation_norm = _norm(relation_gradient)
                total_loss = (
                    objectives["sup"]
                    + objectives["lambda_assim"] * objectives["assim"]
                    + objectives["lambda_relation"] * relation_loss
                )
                total_gradient = _gradients(total_loss, parameters)
                total_norm = _norm(total_gradient)
                if not math.isfinite(total_norm) or total_norm <= 0:
                    raise FloatingPointError("invalid BPRC total virtual gradient")
                scale = VIRTUAL_STEP_NORM / total_norm
                updated_parameters = {
                    name: named_parameters[name].detach() - scale * gradient
                    for name, gradient in zip(parameter_names, total_gradient, strict=True)
                }
                functional_state = {**named_buffers, **updated_parameters}
                method.model.eval()
                previous_after = _evaluate(
                    model=method.model, state=functional_state, batches=previous_batches,
                    native_anchors=native_anchors, old_probabilities=old_probabilities,
                )
                current_after = _evaluate(
                    model=method.model, state=functional_state, batches=current_batches,
                    native_anchors=native_anchors,
                )
                if update_index == 0 and variant == "B0":
                    exact_previous = _functional_val_loss(
                        method.model, functional_state, previous_batches, native_anchors
                    )
                    exact_current = _functional_val_loss(
                        method.model, functional_state, current_batches, native_anchors
                    )
                    if abs(previous_after["loss"] - exact_previous) > 1.0e-12 or abs(current_after["loss"] - exact_current) > 1.0e-12:
                        raise AssertionError("combined evaluator diverges from exact TARC functional loss")
                primitive = objectives["primitive"][variant]
                distill_scale = float(method.config["distill_temperature"]) ** 2
                gradient_rows.append(
                    {
                        "seed": args.seed,
                        "transition": transition,
                        "update_batch": update_index,
                        "variant": variant,
                        "relation_mode": VARIANT_MODES[variant],
                        "relation_loss": float(relation_loss.detach()),
                        "relation_gradient_norm": relation_norm,
                        "relation_gradient_norm_ratio_to_b0": relation_norm / b0_norm,
                        "cos_relation_with_sup": _cosine(relation_gradient, sup_gradient),
                        "cos_relation_with_assim": _cosine(relation_gradient, assim_gradient),
                        "sup_gradient_norm": _norm(sup_gradient),
                        "assim_gradient_norm": _norm(assim_gradient),
                        "total_gradient_norm": total_norm,
                        "old_winner_count_class0": int(primitive.old_winner_counts[0]),
                        "old_winner_count_class1": int(primitive.old_winner_counts[1]),
                        "old_winner_count_class2": int(primitive.old_winner_counts[2]),
                        "per_class_loss0": float(primitive.per_class_loss[0]) * distill_scale,
                        "per_class_loss1": float(primitive.per_class_loss[1]) * distill_scale,
                        "per_class_loss2": float(primitive.per_class_loss[2]) * distill_scale,
                        "boundary_relation_loss": float(region_outputs[variant]["boundary"].loss) * distill_scale,
                        "interior_relation_loss": float(region_outputs[variant]["interior"].loss) * distill_scale,
                        "valid_count": int(primitive.valid_count),
                        "pair_count": int(primitive.pair_count),
                        "present_class_count": int(primitive.present_class_count),
                        "probability_sum_error": float(primitive.probability_sum_error),
                        "b0_r0_loss_abs_error": objectives["b0_abs_error"],
                        "all_finite": all(
                            math.isfinite(value)
                            for value in (relation_norm, total_norm, float(relation_loss.detach()))
                        ),
                        "hidden_gt_usage": "post_hoc_boundary_grouping_only",
                        "optimizer_step_called": False,
                    }
                )
                virtual_rows.append(
                    {
                        "seed": args.seed,
                        "transition": transition,
                        "update_batch": update_index,
                        "variant": variant,
                        "virtual_step_norm": VIRTUAL_STEP_NORM,
                        "previous_val_loss_before": previous_before["loss"],
                        "previous_val_loss_after": previous_after["loss"],
                        "previous_val_loss_delta": previous_after["loss"] - previous_before["loss"],
                        "previous_val_dice_before": previous_before["dice"],
                        "previous_val_dice_after": previous_after["dice"],
                        "previous_val_dice_delta": previous_after["dice"] - previous_before["dice"],
                        "current_val_loss_before": current_before["loss"],
                        "current_val_loss_after": current_after["loss"],
                        "current_val_loss_delta": current_after["loss"] - current_before["loss"],
                        "current_val_dice_before": current_before["dice"],
                        "current_val_dice_after": current_after["dice"],
                        "current_val_dice_delta": current_after["dice"] - current_before["dice"],
                        "old_model_gradient_nonnull": sum(parameter.grad is not None for parameter in method.old_model.parameters()),
                        "checkpoint_or_model_mutated": False,
                        "hidden_gt_usage": "none_visible_update_and_val_only",
                    }
                )
                for class_id in range(3):
                    for region in ("all", "boundary", "interior"):
                        key = f"class{class_id}_{region}"
                        if key not in previous_after["margins"]:
                            continue
                        margin_rows.append(
                            {
                                "seed": args.seed,
                                "transition": transition,
                                "update_batch": update_index,
                                "variant": variant,
                                "class_id": class_id,
                                "region": region,
                                "margin_agreement_before": previous_before["margins"][key],
                                "margin_agreement_after": previous_after["margins"][key],
                                "margin_agreement_delta": previous_after["margins"][key] - previous_before["margins"][key],
                                "margin_abs_error_after": 1.0 - previous_after["margins"][key],
                                "metric_function": "scripts.audit_tarc_relation_fidelity._margin",
                                "hidden_gt_usage": "post_hoc_visible_previous_val_only",
                            }
                        )
                method.model.train()
            del objectives, sup_gradient, assim_gradient, relation_gradients
        after_model_hash = _model_hash(method.model)
        after_old_hash = _model_hash(method.old_model)
        if before_model_hash != after_model_hash or before_old_hash != after_old_hash:
            raise AssertionError("BPRC feasibility mutated a frozen model")
        transition_summaries.append(
            {
                "transition": transition,
                "current_model_sha256_before": before_model_hash,
                "current_model_sha256_after": after_model_hash,
                "old_model_sha256_before": before_old_hash,
                "old_model_sha256_after": after_old_hash,
                "models_unchanged": True,
                "update_batches": UPDATE_BATCHES,
                "previous_val_batches": VAL_BATCHES,
                "current_val_batches": VAL_BATCHES,
                "checkpoint_sha256": sha256_path(Path(bundle["current_checkpoint"])),
            }
        )
        del method, update_batches, previous_batches, current_batches, old_probabilities
        torch.cuda.empty_cache()
    manifest_payload = {
        "protocol_id": "bprcseg_v0_1",
        "seed": args.seed,
        "entries": batch_manifest,
    }
    manifest_payload["combined_sha256"] = sha256_bytes(canonical_json(batch_manifest).encode("utf-8"))
    write_csv(seed_dir / "feasibility_gradient_scale.csv", gradient_rows)
    write_csv(seed_dir / "feasibility_virtual_steps.csv", virtual_rows)
    write_csv(seed_dir / "feasibility_margin_analysis.csv", margin_rows)
    write_json(seed_dir / "fixed_batch_lists.json", manifest_payload)
    write_json(
        seed_dir / "feasibility_summary.json",
        {
            "protocol_id": "bprcseg_v0_1",
            "status": "BPRC_FEASIBILITY_SEED_AUDIT_COMPLETE",
            "seed": args.seed,
            "optimizer_steps": 0,
            "gradient_rows": len(gradient_rows),
            "virtual_rows": len(virtual_rows),
            "margin_rows": len(margin_rows),
            "batch_list_sha256": manifest_payload["combined_sha256"],
            "transitions": transition_summaries,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "seed": args.seed,
                "gradient_rows": len(gradient_rows),
                "virtual_rows": len(virtual_rows),
                "margin_rows": len(margin_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
