#!/usr/bin/env python3
"""Run one seed of the preregistered SPARC-Seg V0.1 feasibility audit.

The ``visible`` phase uses training-visible data only and performs no optimizer
step.  The ``posthoc`` phase is a separate process mode; only that mode imports
the diagnostic-label resolver and it never contributes to an objective.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.func import functional_call
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.analysis.tarc_v0_1 import SITES, TRANSITIONS, checkpoint_path, labeled_loader  # noqa: E402
from lcrseg.analysis.v0_4 import load_frozen_method, stable_seed  # noqa: E402
from lcrseg.common import canonical_json, sha256_bytes, sha256_path, write_csv, write_json  # noqa: E402
from lcrseg.data import (  # noqa: E402
    DeterministicBatcher,
    H5LabeledDataset,
    H5UnlabeledDataset,
    collate_labeled,
    collate_unlabeled,
)
from lcrseg.engine.metrics import masked_cross_entropy, multiclass_dice_loss  # noqa: E402
from lcrseg.losses.stable_feature_maintaining import stable_feature_maintaining  # noqa: E402
from lcrseg.methods.base import relation_supervision_loss  # noqa: E402
from lcrseg.methods.components.compatibility import zero_compatibility  # noqa: E402
from lcrseg.methods.components.learnability import compute_learnability  # noqa: E402
from lcrseg.methods.components.progressive_admission import strict_relation_valid_mask  # noqa: E402
from lcrseg.methods.components.pseudo_label import build_pseudo_labels  # noqa: E402
from lcrseg.methods.components.routing import relation_consolidation_loss, weighted_mean  # noqa: E402
from lcrseg.methods.lcrseg_v0_1 import _uniform_compatibility  # noqa: E402
from lcrseg.semantics.anchored_validation import anchored_validation, partition_stable_plastic  # noqa: E402
from lcrseg.semantics.session_prototypes import SessionPrototypeSet, build_session_prototypes  # noqa: E402
from scripts.audit_aspr_relation_space import _workspace_hash  # noqa: E402


UPDATE_BATCHES = 32
VAL_BATCHES = 16
VIRTUAL_STEP_NORM = 1.0e-3
FOREGROUND_IDS = (1, 2)
VARIANTS = ("S0", "S1", "S2", "S3", "S4", "S5")


def _environment(physical_gpu: int, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "physical_gpu": int(physical_gpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    uuid, name, driver = (part.strip() for part in query.stdout.strip().split(",", maxsplit=2))
    result.update(uuid=uuid, name=name, driver=driver)
    return result


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if isinstance(value, torch.Tensor):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch_save(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite SPARC artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(value, temporary)
    os.replace(temporary, path)


def _take_cycling(loader: Iterable[Any], count: int, device: torch.device) -> list[Any]:
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


def _fixed_batchers(data_root: Path, seed: int, site_id: str) -> tuple[DeterministicBatcher[Any], DeterministicBatcher[Any]]:
    labeled = H5LabeledDataset(
        data_root, seed=seed, dataset="fundus", sites=(site_id,), roles=("train_labeled",), transform=None
    )
    unlabeled = H5UnlabeledDataset(data_root, seed=seed, dataset="fundus", sites=(site_id,), transform=None)
    order_seed = stable_seed("sparc-v0.1-feasibility-clean", seed, site_id)
    return (
        DeterministicBatcher(
            labeled,
            batch_size=2,
            seed=order_seed,
            namespace=f"sparc-v0.1:{seed}:{site_id}:labeled",
            collate=collate_labeled,
            shuffle=False,
        ),
        DeterministicBatcher(
            unlabeled,
            batch_size=4,
            seed=order_seed,
            namespace=f"sparc-v0.1:{seed}:{site_id}:unlabeled",
            collate=collate_unlabeled,
            shuffle=False,
        ),
    )


def _fixed_updates(
    data_root: Path, seed: int, site_id: str, device: torch.device
) -> tuple[list[tuple[Any, Any]], list[dict[str, Any]], str]:
    labeled, unlabeled = _fixed_batchers(data_root, seed, site_id)
    batches: list[tuple[Any, Any]] = []
    manifest: list[dict[str, Any]] = []
    for index in range(UPDATE_BATCHES):
        labeled_batch = labeled.batch_at(index).to(device, non_blocking=True)
        unlabeled_batch = unlabeled.batch_at(index).to(device, non_blocking=True)
        row = {
            "batch_index": index,
            "labeled_case_ids": labeled_batch.case_id,
            "unlabeled_case_ids": unlabeled_batch.case_id,
            "weak_equals_strong_clean_diagnostic": bool(torch.equal(unlabeled_batch.weak_image, unlabeled_batch.strong_image)),
        }
        row["row_sha256"] = sha256_bytes(canonical_json(row).encode("utf-8"))
        batches.append((labeled_batch, unlabeled_batch))
        manifest.append(row)
    return batches, manifest, sha256_bytes(canonical_json(manifest).encode("utf-8"))


def _class_semantics_sha() -> str:
    return sha256_path(ROOT / "reports" / "experiment_status" / "class_semantics.json")


def _prototype_loader(data_root: Path, seed: int, site_id: str, workers: int) -> Any:
    return labeled_loader(
        data_root, seed=seed, site_id=site_id, roles=("train_labeled",), batch_size=4, workers=workers
    )


def _build_prototypes(
    *,
    method: Any,
    payload: dict[str, Any],
    old_checkpoint: Path,
    current_checkpoint: Path,
    data_root: Path,
    seed: int,
    site_id: str,
    workers: int,
    device: torch.device,
) -> tuple[SessionPrototypeSet, SessionPrototypeSet]:
    if method.old_model is None:
        raise RuntimeError("SPARC transition lacks a frozen previous model")
    previous = build_session_prototypes(
        method.old_model,
        _prototype_loader(data_root, seed, site_id, workers),
        model_role="previous",
        site_id=site_id,
        epoch_id=-1,
        num_classes=3,
        class_semantics_sha256=_class_semantics_sha(),
        source_checkpoint_sha256=sha256_path(old_checkpoint),
        device=device,
    )
    current = build_session_prototypes(
        method.model,
        _prototype_loader(data_root, seed, site_id, workers),
        model_role="current",
        site_id=site_id,
        epoch_id=int(payload["epoch"]),
        num_classes=3,
        class_semantics_sha256=_class_semantics_sha(),
        source_checkpoint_sha256=sha256_path(current_checkpoint),
        device=device,
    )
    return current, previous


def _gated_assimilation(
    strong_logits: torch.Tensor,
    pseudo: Any,
    learnability: Any,
    strong_valid_mask: torch.Tensor,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    target = F.interpolate(pseudo.labels[:, None].float(), size=strong_logits.shape[-2:], mode="nearest")[:, 0].long()
    pseudo_valid = F.interpolate(pseudo.valid.float(), size=strong_logits.shape[-2:], mode="nearest").bool()
    weights = F.interpolate(
        learnability.score.detach(), size=strong_logits.shape[-2:], mode="bilinear", align_corners=False
    )
    valid = pseudo_valid & strong_valid_mask.detach().bool()
    if gate is not None:
        gate_full = F.interpolate(gate[:, None].detach().float(), size=strong_logits.shape[-2:], mode="nearest").bool()
        valid &= gate_full
    weights = weights * valid.float()
    pixel = F.cross_entropy(strong_logits, target, ignore_index=-100, reduction="none")[:, None]
    return weighted_mean(pixel, weights, reference=strong_logits)


def _gradients(loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...], *, retain_graph: bool) -> tuple[torch.Tensor, ...]:
    raw = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, create_graph=False, allow_unused=True)
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, raw, strict=True)
    )


def _grad_norm(gradients: tuple[torch.Tensor, ...]) -> float:
    return float(torch.sqrt(sum(value.float().square().sum() for value in gradients)))


def _grad_cosine(first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...]) -> float:
    first_norm, second_norm = _grad_norm(first), _grad_norm(second)
    if first_norm <= 0 or second_norm <= 0:
        return 0.0
    dot = sum((left.float() * right.float()).sum() for left, right in zip(first, second, strict=True))
    return float(dot / (first_norm * second_norm))


def _foreground_mask(classes: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(classes, dtype=torch.bool)
    for class_id in FOREGROUND_IDS:
        result |= classes.eq(class_id)
    return result


def _localization_rows(
    feature_output: Any,
    partition: Any,
    current_features: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_name in ("dec3", "dec1"):
        layer_loss = feature_output.layer_losses[layer_name]
        gradient = torch.autograd.grad(layer_loss, current_features[layer_name], retain_graph=True)[0]
        magnitude = gradient.detach().abs().sum(dim=1)
        size = tuple(magnitude.shape[-2:])
        stable = F.interpolate(partition.stable[:, None].float(), size=size, mode="nearest")[:, 0].bool()
        plastic = F.interpolate(partition.plastic[:, None].float(), size=size, mode="nearest")[:, 0].bool()
        rejected = F.interpolate(partition.rejected[:, None].float(), size=size, mode="nearest")[:, 0].bool()
        classes = F.interpolate(partition.current_class[:, None].float(), size=size, mode="nearest")[:, 0].long()
        foreground = _foreground_mask(classes)
        zones = {
            "stable_foreground": stable & foreground,
            "plastic": plastic,
            "rejected": rejected,
            "background": classes.eq(0),
        }
        for zone, mask in zones.items():
            rows.append(
                {
                    "layer": layer_name,
                    "zone": zone,
                    "cell_count": int(mask.sum()),
                    "gradient_abs_sum": float(magnitude[mask].sum()) if bool(mask.any()) else 0.0,
                    "gradient_abs_max": float(magnitude[mask].max()) if bool(mask.any()) else 0.0,
                }
            )
    return rows


def _prepare_objectives(
    method: Any,
    payload: dict[str, Any],
    labeled: Any,
    unlabeled: Any,
    current_prototypes: SessionPrototypeSet,
    previous_prototypes: SessionPrototypeSet,
) -> dict[str, Any]:
    if method.old_model is None or method.old_anchor_bank is None:
        raise RuntimeError("SPARC feasibility requires frozen previous model/anchors")
    method.model.train()
    method.old_model.eval()
    labeled_output = method.model(labeled.image)
    labeled_relation = method._relation(labeled_output.relation_features, method.current_anchor_bank)
    supervised = method._supervised_losses(
        labeled_output,
        labeled,
        relation_anchor_loss=relation_supervision_loss(labeled_relation.logits, labeled.label, labeled.valid_mask),
    )["loss_sup"]
    current_weak = method.model(unlabeled.weak_image)
    if current_weak.decoder_features is None:
        raise RuntimeError("SPARC decoder outputs are unavailable")
    current_relation = method._relation(current_weak.relation_features, method.current_anchor_bank)
    with torch.no_grad():
        old_weak = method.old_model(unlabeled.weak_image)
        if old_weak.decoder_features is None:
            raise RuntimeError("SPARC old decoder outputs are unavailable")
        old_relation = method._relation(old_weak.relation_features, method.old_anchor_bank)
    relation_valid = strict_relation_valid_mask(unlabeled.strong_valid_mask, tuple(current_weak.relation_features.shape[-2:]))
    current_validation = anchored_validation(
        current_weak.logits.detach(), current_weak.relation_features.detach(), current_prototypes, relation_valid
    )
    previous_validation = anchored_validation(
        old_weak.logits, old_weak.relation_features, previous_prototypes, relation_valid
    )
    partition = partition_stable_plastic(current_validation, previous_validation, relation_valid)
    pseudo = build_pseudo_labels(
        current_weak.logits.detach().softmax(dim=1),
        current_relation,
        tau_cls=float(method.config["tau_cls"]),
        tau_anchor=float(method.config["tau_anchor"]),
        delta_anchor=float(method.config["delta_anchor"]),
        tau_spatial=float(method.config["tau_spatial"]),
        temperature_cls=float(method.config["temperature_cls"]),
        temperature_anchor=float(method.config["temperature_anchor"]),
        spatial_floor=float(method.config["spatial_floor"]),
    )
    site_step = max(0, int(payload["site_step"]) - 1)
    learnability = compute_learnability(
        current_weak.logits.detach(),
        current_relation,
        pseudo,
        site_step=site_step,
        total_steps=max(1, int(method.total_steps)),
        rank_start=float(method.config["rank_start"]),
        rank_end=float(method.config["rank_end"]),
        rank_temperature=float(method.config["rank_temperature"]),
        relation_margin_center=float(method.config["relation_margin_center"]),
        relation_margin_temperature=float(method.config["relation_margin_temperature"]),
        min_rank_pixels=int(method.config["min_rank_pixels"]),
    )
    strong_output = method.model(unlabeled.strong_image)
    strong_relation = method._relation(strong_output.relation_features, method.current_anchor_bank)
    assimilation_r0 = _gated_assimilation(strong_output.logits, pseudo, learnability, unlabeled.strong_valid_mask, None)
    assimilation_pas = _gated_assimilation(
        strong_output.logits, pseudo, learnability, unlabeled.strong_valid_mask, current_validation.valid
    )
    compatibility = _uniform_compatibility(zero_compatibility(old_relation.probabilities))
    relation_loss = relation_consolidation_loss(
        strong_relation,
        old_relation,
        compatibility,
        unlabeled.strong_valid_mask,
        distill_temperature=float(method.config["distill_temperature"]),
    )
    stable_feature = stable_feature_maintaining(
        current_weak.decoder_features,
        old_weak.decoder_features,
        partition.stable,
        partition.current_class,
        partition.current_valid,
    )
    old_foreground = previous_validation.valid & _foreground_mask(previous_validation.predicted_class)
    old_only_feature = stable_feature_maintaining(
        current_weak.decoder_features,
        old_weak.decoder_features,
        old_foreground,
        previous_validation.predicted_class,
        partition.current_valid,
    )
    global_feature = stable_feature_maintaining(
        current_weak.decoder_features,
        old_weak.decoder_features,
        relation_valid[:, 0],
        partition.current_class,
        partition.current_valid,
        class_balanced=False,
        foreground_only=False,
    )
    bootstrap_at = int(method.bootstrap_state.get("completed_at_site_step", -1))
    lambda_assim = float(method.config["lambda_assim"]) * method._assimilation_ramp(
        site_step, bootstrap_complete_at=bootstrap_at
    )
    lambda_relation = float(method.config["lambda_relation"]) * method._relation_ramp(site_step)
    kappa = stable_feature.kappa
    totals = {
        "S0": supervised + lambda_assim * assimilation_r0 + lambda_relation * relation_loss,
        "S1": supervised + lambda_assim * assimilation_pas + lambda_relation * relation_loss,
        "S2": supervised + lambda_assim * assimilation_r0 + lambda_relation * (relation_loss + kappa * stable_feature.loss),
        "S3": supervised + lambda_assim * assimilation_pas + lambda_relation * (relation_loss + kappa * stable_feature.loss),
        "S4": supervised + lambda_assim * assimilation_pas + lambda_relation * (relation_loss + kappa * old_only_feature.loss),
        "S5": supervised + lambda_assim * assimilation_pas + lambda_relation * (relation_loss + kappa * global_feature.loss),
    }
    return {
        "totals": totals,
        "supervised": supervised,
        "assimilation_r0": assimilation_r0,
        "assimilation_pas": assimilation_pas,
        "relation_loss": relation_loss,
        "stable_feature": stable_feature,
        "old_only_feature": old_only_feature,
        "global_feature": global_feature,
        "partition": partition,
        "current_validation": current_validation,
        "previous_validation": previous_validation,
        "pseudo": pseudo,
        "current_features": current_weak.decoder_features,
        "lambda_assim": lambda_assim,
        "lambda_relation": lambda_relation,
    }


def _supervised_validation_loss(output: Any, batch: Any, anchors: torch.Tensor, method: Any) -> torch.Tensor:
    relation = method._relation(output.relation_features, method.current_anchor_bank)
    return (
        masked_cross_entropy(output.logits, batch.label, batch.valid_mask)
        + float(method.config["lambda_dice"]) * multiclass_dice_loss(output.logits, batch.label, batch.valid_mask)
        + float(method.config["lambda_anchor_sup"])
        * relation_supervision_loss(relation.logits, batch.label, batch.valid_mask)
    )


@torch.no_grad()
def _evaluate_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor] | None,
    batches: list[Any],
    method: Any,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    loss_total = 0.0
    dice_total = 0.0
    for batch in batches:
        output = model(batch.image) if state is None else functional_call(model, state, (batch.image,))
        loss_total += float(_supervised_validation_loss(output, batch, method.current_anchor_bank.anchors, method))
        dice_total += float(1.0 - multiclass_dice_loss(output.logits, batch.label, batch.valid_mask))
    model.train(was_training)
    return loss_total / len(batches), dice_total / len(batches)


def visible_phase(args: argparse.Namespace) -> int:
    seed_dir = args.output_dir.resolve() / f"seed{args.seed}"
    if seed_dir.exists():
        raise FileExistsError(f"refusing to overwrite SPARC feasibility seed: {seed_dir}")
    seed_dir.mkdir(parents=True)
    device = torch.device(args.device)
    gradient_rows: list[dict[str, Any]] = []
    virtual_rows: list[dict[str, Any]] = []
    batch_manifest: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for old_index, current_index in TRANSITIONS:
        old_checkpoint = checkpoint_path(args.run_root.resolve(), args.seed, old_index)
        current_checkpoint = checkpoint_path(args.run_root.resolve(), args.seed, current_index)
        method, payload = load_frozen_method(current_checkpoint, device)
        if method.old_model is None or method.old_anchor_bank is None:
            raise RuntimeError("current R0 checkpoint has no frozen previous model")
        old_payload = torch.load(old_checkpoint, map_location="cpu", weights_only=False)
        if _model_hash(method.old_model) != _state_hash(old_payload["current_model_state"]):
            raise RuntimeError("frozen previous model does not match the registered old checkpoint")
        current_prototypes, previous_prototypes = _build_prototypes(
            method=method,
            payload=payload,
            old_checkpoint=old_checkpoint,
            current_checkpoint=current_checkpoint,
            data_root=args.data_root.resolve(),
            seed=args.seed,
            site_id=SITES[current_index],
            workers=args.workers,
            device=device,
        )
        transition = f"{SITES[old_index]}->{SITES[current_index]}"
        prototype_path = seed_dir / f"prototype_bundle_{old_index}_{current_index}.pt"
        _atomic_torch_save(
            prototype_path,
            {
                "protocol_id": "sparcseg_v0_1",
                "seed": args.seed,
                "transition": transition,
                "current": current_prototypes.state_dict(),
                "previous": previous_prototypes.state_dict(),
            },
        )
        update_batches, update_manifest, update_sha = _fixed_updates(
            args.data_root.resolve(), args.seed, SITES[current_index], device
        )
        for row in update_manifest:
            batch_manifest.append({"seed": args.seed, "transition": transition, "role": "current_update", **row})
        previous_batches = _take_cycling(
            labeled_loader(
                args.data_root.resolve(), seed=args.seed, site_id=SITES[old_index], roles=("val",), batch_size=4, workers=args.workers
            ),
            VAL_BATCHES,
            device,
        )
        current_batches = _take_cycling(
            labeled_loader(
                args.data_root.resolve(), seed=args.seed, site_id=SITES[current_index], roles=("val",), batch_size=4, workers=args.workers
            ),
            VAL_BATCHES,
            device,
        )
        for role, batches in (("previous_val", previous_batches), ("current_val", current_batches)):
            for index, batch in enumerate(batches):
                row = {"seed": args.seed, "transition": transition, "role": role, "batch_index": index, "case_ids": batch.case_id}
                row["row_sha256"] = sha256_bytes(canonical_json(row).encode("utf-8"))
                batch_manifest.append(row)
        before_hash = _model_hash(method.model)
        old_before_hash = _model_hash(method.old_model)
        previous_before_loss, previous_before_dice = _evaluate_state(method.model, None, previous_batches, method)
        current_before_loss, current_before_dice = _evaluate_state(method.model, None, current_batches, method)
        named_parameters = dict(method.model.named_parameters())
        named_buffers = dict(method.model.named_buffers())
        parameter_names = tuple(named_parameters)
        parameters = tuple(named_parameters.values())
        for update_index, (labeled, unlabeled) in enumerate(update_batches):
            objectives = _prepare_objectives(
                method, payload, labeled, unlabeled, current_prototypes, previous_prototypes
            )
            lambda_relation = float(objectives["lambda_relation"])
            relation_gradients = _gradients(lambda_relation * objectives["relation_loss"], parameters, retain_graph=True)
            sfm_gradients = _gradients(
                lambda_relation * objectives["stable_feature"].kappa * objectives["stable_feature"].loss,
                parameters,
                retain_graph=True,
            )
            combined_gradients = _gradients(
                lambda_relation
                * (objectives["relation_loss"] + objectives["stable_feature"].kappa * objectives["stable_feature"].loss),
                parameters,
                retain_graph=True,
            )
            relation_norm = _grad_norm(relation_gradients)
            sfm_norm = _grad_norm(sfm_gradients)
            combined_norm = _grad_norm(combined_gradients)
            ratio = sfm_norm / relation_norm if relation_norm > 0 else float("inf")
            localization = _localization_rows(
                objectives["stable_feature"], objectives["partition"], objectives["current_features"]
            )
            loc = {(row["layer"], row["zone"]): row for row in localization}
            localization_pass = all(
                loc[(layer, "stable_foreground")]["gradient_abs_sum"] > 0
                and loc[(layer, "plastic")]["gradient_abs_sum"] == 0
                and loc[(layer, "rejected")]["gradient_abs_sum"] == 0
                and loc[(layer, "background")]["gradient_abs_sum"] == 0
                for layer in ("dec3", "dec1")
            )
            gradient_rows.append(
                {
                    "seed": args.seed,
                    "transition": transition,
                    "update_batch": update_index,
                    "lambda_relation": lambda_relation,
                    "kappa": float(objectives["stable_feature"].kappa),
                    "relation_gradient_norm": relation_norm,
                    "kappa_sfm_gradient_norm": sfm_norm,
                    "combined_historical_gradient_norm": combined_norm,
                    "sfm_to_relation_ratio": ratio,
                    "relation_sfm_gradient_cosine": _grad_cosine(relation_gradients, sfm_gradients),
                    "finite": bool(all(math.isfinite(value) for value in (relation_norm, sfm_norm, combined_norm, ratio))),
                    "localization_pass": localization_pass,
                    "dec3_stable_gradient": loc[("dec3", "stable_foreground")]["gradient_abs_sum"],
                    "dec3_plastic_gradient": loc[("dec3", "plastic")]["gradient_abs_sum"],
                    "dec3_rejected_gradient": loc[("dec3", "rejected")]["gradient_abs_sum"],
                    "dec3_background_gradient": loc[("dec3", "background")]["gradient_abs_sum"],
                    "dec1_stable_gradient": loc[("dec1", "stable_foreground")]["gradient_abs_sum"],
                    "dec1_plastic_gradient": loc[("dec1", "plastic")]["gradient_abs_sum"],
                    "dec1_rejected_gradient": loc[("dec1", "rejected")]["gradient_abs_sum"],
                    "dec1_background_gradient": loc[("dec1", "background")]["gradient_abs_sum"],
                    "old_model_gradient_nonnull": sum(parameter.grad is not None for parameter in method.old_model.parameters()),
                }
            )
            variant_gradients: dict[str, tuple[torch.Tensor, ...]] = {}
            for variant in VARIANTS:
                variant_gradients[variant] = _gradients(
                    objectives["totals"][variant], parameters, retain_graph=variant != VARIANTS[-1]
                )
            for variant in VARIANTS:
                gradients = variant_gradients[variant]
                raw_norm = _grad_norm(gradients)
                if not math.isfinite(raw_norm) or raw_norm <= 0:
                    raise FloatingPointError(f"invalid {variant} virtual gradient norm: {raw_norm}")
                scale = VIRTUAL_STEP_NORM / raw_norm
                updated = {
                    name: named_parameters[name] - scale * gradient
                    for name, gradient in zip(parameter_names, gradients, strict=True)
                }
                state = {**named_buffers, **updated}
                previous_after_loss, previous_after_dice = _evaluate_state(method.model, state, previous_batches, method)
                current_after_loss, current_after_dice = _evaluate_state(method.model, state, current_batches, method)
                virtual_rows.append(
                    {
                        "seed": args.seed,
                        "transition": transition,
                        "update_batch": update_index,
                        "variant": variant,
                        "virtual_step_norm": VIRTUAL_STEP_NORM,
                        "raw_gradient_norm": raw_norm,
                        "objective": float(objectives["totals"][variant].detach()),
                        "loss_sup": float(objectives["supervised"].detach()),
                        "loss_assim_r0": float(objectives["assimilation_r0"].detach()),
                        "loss_assim_pas": float(objectives["assimilation_pas"].detach()),
                        "loss_relation_r0": float(objectives["relation_loss"].detach()),
                        "loss_sfm_stable": float(objectives["stable_feature"].loss.detach()),
                        "kappa": float(objectives["stable_feature"].kappa),
                        "previous_val_loss_before": previous_before_loss,
                        "previous_val_loss_after": previous_after_loss,
                        "previous_val_loss_delta": previous_after_loss - previous_before_loss,
                        "previous_val_dice_before": previous_before_dice,
                        "previous_val_dice_after": previous_after_dice,
                        "previous_val_dice_delta": previous_after_dice - previous_before_dice,
                        "current_val_loss_before": current_before_loss,
                        "current_val_loss_after": current_after_loss,
                        "current_val_loss_delta": current_after_loss - current_before_loss,
                        "current_val_dice_before": current_before_dice,
                        "current_val_dice_after": current_after_dice,
                        "current_val_dice_delta": current_after_dice - current_before_dice,
                        "checkpoint_or_model_mutated": False,
                    }
                )
            del objectives, relation_gradients, sfm_gradients, combined_gradients, variant_gradients
        after_hash = _model_hash(method.model)
        old_after_hash = _model_hash(method.old_model)
        if before_hash != after_hash or old_before_hash != old_after_hash:
            raise AssertionError("SPARC feasibility mutated a frozen model")
        transitions.append(
            {
                "transition": transition,
                "old_checkpoint": str(old_checkpoint),
                "old_checkpoint_sha256": sha256_path(old_checkpoint),
                "current_checkpoint": str(current_checkpoint),
                "current_checkpoint_sha256": sha256_path(current_checkpoint),
                "prototype_bundle": str(prototype_path),
                "prototype_bundle_sha256": sha256_path(prototype_path),
                "current_prototype_case_counts": current_prototypes.case_counts.cpu().tolist(),
                "previous_prototype_case_counts": previous_prototypes.case_counts.cpu().tolist(),
                "current_prototype_pixel_counts": current_prototypes.pixel_counts.cpu().tolist(),
                "previous_prototype_pixel_counts": previous_prototypes.pixel_counts.cpu().tolist(),
                "current_prototype_source_case_ids_sha256": current_prototypes.source_case_ids_sha256,
                "previous_prototype_source_case_ids_sha256": previous_prototypes.source_case_ids_sha256,
                "update_batch_manifest_sha256": update_sha,
                "model_sha256_before": before_hash,
                "model_sha256_after": after_hash,
                "old_model_sha256_before": old_before_hash,
                "old_model_sha256_after": old_after_hash,
                "previous_val_batches": VAL_BATCHES,
                "current_val_batches": VAL_BATCHES,
            }
        )
        del method, update_batches, previous_batches, current_batches
        torch.cuda.empty_cache()
    write_csv(seed_dir / "gradient_scale.csv", gradient_rows)
    write_csv(seed_dir / "virtual_steps.csv", virtual_rows)
    write_json(seed_dir / "batch_manifest.json", batch_manifest)
    summary = {
        "protocol_id": "sparcseg_v0_1",
        "seed": args.seed,
        "status": "SPARC_VISIBLE_FEASIBILITY_COMPLETE",
        "optimizer_steps": 0,
        "hidden_gt_usage": "none",
        "transitions": transitions,
        "gradient_rows": len(gradient_rows),
        "virtual_rows": len(virtual_rows),
        "batch_manifest_sha256": sha256_bytes(canonical_json(batch_manifest).encode("utf-8")),
        "environment": _environment(args.physical_gpu, device),
        "training_manifest_sha256": sha256_path(
            args.data_root.resolve() / "manifests" / "training" / f"lcrseg_v1_seed{args.seed}.csv"
        ),
        "split_sha256": sha256_path(args.data_root.resolve() / "splits" / f"fundus_seed{args.seed}.json"),
        "workspace_hash": _workspace_hash(),
    }
    write_json(seed_dir / "visible_summary.json", summary)
    print(json.dumps({"status": summary["status"], "seed": args.seed, "output": str(seed_dir)}, indent=2))
    return 0


def _load_hidden_labels(data_root: Path, seed: int, site_id: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # Deliberately imported only in the independent post-hoc process mode.
    from lcrseg.analysis.diagnostics import _images_and_labels, diagnostic_records

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for record in diagnostic_records(data_root, seed=seed, dataset="fundus", site=site_id):
        pairs = list(_images_and_labels(record, "fundus"))
        if len(pairs) != 1:
            raise RuntimeError("fundus diagnostic case must yield exactly one image/label pair")
        result[record.case_id] = pairs[0]
    return result


def _boundary_maps(labels: torch.Tensor, sizes: Iterable[tuple[int, int]]) -> dict[tuple[int, int, int], torch.Tensor]:
    from lcrseg.analysis.v0_4 import signed_distance_and_component_size

    result: dict[tuple[int, int, int], torch.Tensor] = {}
    for class_id in FOREGROUND_IDS:
        full: list[torch.Tensor] = []
        for item in labels.cpu().numpy():
            signed, _ = signed_distance_and_component_size(item, class_id)
            full.append(torch.from_numpy(np.abs(signed) <= 3.0))
        full_tensor = torch.stack(full, dim=0)[:, None].float()
        for size in sizes:
            result[(class_id, size[0], size[1])] = F.interpolate(full_tensor, size=size, mode="nearest")[:, 0].bool()
    return result


def _metric_update(store: dict[Any, dict[str, float]], key: Any, selected: torch.Tensor, truth: torch.Tensor) -> None:
    values = store.setdefault(key, {"selected": 0.0, "correct": 0.0, "true": 0.0})
    values["selected"] += int(selected.sum())
    values["correct"] += int((selected & truth).sum())
    values["true"] += int(truth.sum())


def posthoc_phase(args: argparse.Namespace) -> int:
    seed_dir = args.output_dir.resolve() / f"seed{args.seed}"
    visible_summary_path = seed_dir / "visible_summary.json"
    if not visible_summary_path.is_file():
        raise FileNotFoundError(visible_summary_path)
    for name in ("prototype_validation_quality.csv", "partition_quality.csv", "feature_separation.csv", "posthoc_summary.json"):
        if (seed_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite SPARC post-hoc artifact: {seed_dir / name}")
    visible_summary = json.loads(visible_summary_path.read_text())
    device = torch.device(args.device)
    prototype_stats: dict[Any, dict[str, float]] = {}
    partition_stats: dict[Any, dict[str, float]] = {}
    feature_stats: dict[Any, dict[str, float]] = {}
    transition_records: list[dict[str, Any]] = []
    for old_index, current_index in TRANSITIONS:
        transition = f"{SITES[old_index]}->{SITES[current_index]}"
        current_checkpoint = checkpoint_path(args.run_root.resolve(), args.seed, current_index)
        method, payload = load_frozen_method(current_checkpoint, device)
        if method.old_model is None or method.old_anchor_bank is None:
            raise RuntimeError("post-hoc transition lacks frozen previous state")
        bundle_path = seed_dir / f"prototype_bundle_{old_index}_{current_index}.pt"
        bundle = torch.load(bundle_path, map_location=device, weights_only=False)
        current_prototypes = SessionPrototypeSet.from_state_dict(bundle["current"])
        previous_prototypes = SessionPrototypeSet.from_state_dict(bundle["previous"])
        hidden = _load_hidden_labels(args.data_root.resolve(), args.seed, SITES[current_index])
        _, unlabeled_batcher = _fixed_batchers(args.data_root.resolve(), args.seed, SITES[current_index])
        posthoc_manifest: list[dict[str, Any]] = []
        image_max_abs = 0.0
        method.model.eval()
        method.old_model.eval()
        with torch.inference_mode():
            for batch_index in range(UPDATE_BATCHES):
                batch = unlabeled_batcher.batch_at(batch_index).to(device, non_blocking=True)
                labels: list[torch.Tensor] = []
                for sample_index, case_id in enumerate(batch.case_id):
                    if case_id not in hidden:
                        raise KeyError(f"diagnostic label is unavailable for fixed case {case_id}")
                    image_np, label_np = hidden[case_id]
                    image_tensor = torch.from_numpy(image_np).to(batch.weak_image.device)
                    image_max_abs = max(image_max_abs, float((image_tensor - batch.weak_image[sample_index]).abs().max()))
                    labels.append(torch.from_numpy(label_np).long())
                full_labels = torch.stack(labels, dim=0).to(device)
                current_output = method.model(batch.weak_image)
                old_output = method.old_model(batch.weak_image)
                if current_output.decoder_features is None or old_output.decoder_features is None:
                    raise RuntimeError("decoder feature path missing in post-hoc")
                current_relation = method._relation(current_output.relation_features, method.current_anchor_bank)
                old_relation = method._relation(old_output.relation_features, method.old_anchor_bank)
                relation_valid = strict_relation_valid_mask(
                    batch.strong_valid_mask, tuple(current_output.relation_features.shape[-2:])
                )
                current_validation = anchored_validation(
                    current_output.logits, current_output.relation_features, current_prototypes, relation_valid
                )
                previous_validation = anchored_validation(
                    old_output.logits, old_output.relation_features, previous_prototypes, relation_valid
                )
                partition = partition_stable_plastic(current_validation, previous_validation, relation_valid)
                pseudo = build_pseudo_labels(
                    current_output.logits.softmax(dim=1),
                    current_relation,
                    tau_cls=float(method.config["tau_cls"]),
                    tau_anchor=float(method.config["tau_anchor"]),
                    delta_anchor=float(method.config["delta_anchor"]),
                    tau_spatial=float(method.config["tau_spatial"]),
                    temperature_cls=float(method.config["temperature_cls"]),
                    temperature_anchor=float(method.config["temperature_anchor"]),
                    spatial_floor=float(method.config["spatial_floor"]),
                )
                relation_size = tuple(current_output.relation_features.shape[-2:])
                truth_r = F.interpolate(full_labels[:, None].float(), size=relation_size, mode="nearest")[:, 0].long()
                boundary = _boundary_maps(full_labels, (relation_size, (256, 256)))
                selectors = {
                    "r0_candidate": (pseudo.valid[:, 0] & relation_valid[:, 0], pseudo.labels),
                    "confidence_only": (
                        current_validation.confidence.gt(0.70) & relation_valid[:, 0],
                        current_validation.predicted_class,
                    ),
                    "current_pas": (current_validation.valid, current_validation.predicted_class),
                    "old_pas": (previous_validation.valid, previous_validation.predicted_class),
                }
                for selector_name, (selector_mask, selector_class) in selectors.items():
                    for class_id in range(3):
                        _metric_update(
                            prototype_stats,
                            (transition, selector_name, class_id),
                            selector_mask & selector_class.eq(class_id),
                            truth_r.eq(class_id),
                        )
                partitions = {
                    "stable": partition.stable,
                    "plastic": partition.plastic,
                    "rejected": partition.rejected,
                }
                for partition_name, partition_mask in partitions.items():
                    for class_id in range(3):
                        selected = partition_mask & partition.current_class.eq(class_id)
                        truth = truth_r.eq(class_id)
                        key = (transition, partition_name, class_id)
                        values = partition_stats.setdefault(
                            key,
                            {
                                "selected": 0.0,
                                "correct": 0.0,
                                "true": 0.0,
                                "boundary_correct": 0.0,
                                "boundary_true": 0.0,
                                "interior_correct": 0.0,
                                "interior_true": 0.0,
                            },
                        )
                        values["selected"] += int(selected.sum())
                        values["correct"] += int((selected & truth).sum())
                        values["true"] += int(truth.sum())
                        if class_id in FOREGROUND_IDS:
                            band = boundary[(class_id, relation_size[0], relation_size[1])].to(device)
                            values["boundary_correct"] += int((selected & truth & band).sum())
                            values["boundary_true"] += int((truth & band).sum())
                            values["interior_correct"] += int((selected & truth & ~band).sum())
                            values["interior_true"] += int((truth & ~band).sum())
                for layer_name in ("dec3", "dec1"):
                    current_feature = current_output.decoder_features[layer_name].float()
                    old_feature = old_output.decoder_features[layer_name].float()
                    size = tuple(current_feature.shape[-2:])
                    current_unit = F.normalize(current_feature, p=2, dim=1, eps=1.0e-8)
                    old_unit = F.normalize(old_feature, p=2, dim=1, eps=1.0e-8)
                    cosine = (current_unit * old_unit).sum(dim=1)
                    l2 = (current_feature - old_feature).square().sum(dim=1).sqrt()
                    normalized_l2 = (current_unit - old_unit).square().sum(dim=1).sqrt()
                    class_map = F.interpolate(partition.current_class[:, None].float(), size=size, mode="nearest")[:, 0].long()
                    layer_partitions = {
                        name: F.interpolate(mask[:, None].float(), size=size, mode="nearest")[:, 0].bool()
                        for name, mask in partitions.items()
                    }
                    for partition_name, layer_mask in layer_partitions.items():
                        for class_id in range(3):
                            selected_class = layer_mask & class_map.eq(class_id)
                            regions = {"all": torch.ones_like(selected_class)}
                            if class_id in FOREGROUND_IDS:
                                band = boundary[(class_id, size[0], size[1])].to(device)
                                regions.update(boundary=band, interior=~band)
                            for region, region_mask in regions.items():
                                selected = selected_class & region_mask
                                key = (transition, layer_name, partition_name, class_id, region)
                                values = feature_stats.setdefault(
                                    key, {"count": 0.0, "cosine_sum": 0.0, "l2_sum": 0.0, "normalized_l2_sum": 0.0}
                                )
                                count = int(selected.sum())
                                values["count"] += count
                                if count:
                                    values["cosine_sum"] += float(cosine[selected].sum())
                                    values["l2_sum"] += float(l2[selected].sum())
                                    values["normalized_l2_sum"] += float(normalized_l2[selected].sum())
                row = {"batch_index": batch_index, "unlabeled_case_ids": batch.case_id}
                row["row_sha256"] = sha256_bytes(canonical_json(row).encode("utf-8"))
                posthoc_manifest.append(row)
        expected_transition = next(item for item in visible_summary["transitions"] if item["transition"] == transition)
        transition_records.append(
            {
                "transition": transition,
                "prototype_bundle_sha256": sha256_path(bundle_path),
                "prototype_bundle_matches_visible": sha256_path(bundle_path) == expected_transition["prototype_bundle_sha256"],
                "posthoc_unlabeled_manifest_sha256": sha256_bytes(canonical_json(posthoc_manifest).encode("utf-8")),
                "visible_update_manifest_sha256": expected_transition["update_batch_manifest_sha256"],
                "image_max_abs_vs_training_view": image_max_abs,
                "hidden_gt_usage": "post_hoc_metrics_only",
            }
        )
        del method
        torch.cuda.empty_cache()
    prototype_rows: list[dict[str, Any]] = []
    for (transition, selector, class_id), values in sorted(prototype_stats.items()):
        prototype_rows.append(
            {
                "seed": args.seed,
                "transition": transition,
                "selector": selector,
                "class_id": class_id,
                "selected_pixels": int(values["selected"]),
                "correct_selected_pixels": int(values["correct"]),
                "true_pixels": int(values["true"]),
                "precision": values["correct"] / values["selected"] if values["selected"] else float("nan"),
                "coverage": values["correct"] / values["true"] if values["true"] else float("nan"),
                "hidden_gt_usage": "post_hoc_only",
            }
        )
    partition_rows: list[dict[str, Any]] = []
    for (transition, partition_name, class_id), values in sorted(partition_stats.items()):
        partition_rows.append(
            {
                "seed": args.seed,
                "transition": transition,
                "partition": partition_name,
                "class_id": class_id,
                "selected_pixels": int(values["selected"]),
                "correct_selected_pixels": int(values["correct"]),
                "true_pixels": int(values["true"]),
                "precision": values["correct"] / values["selected"] if values["selected"] else float("nan"),
                "coverage": values["correct"] / values["true"] if values["true"] else float("nan"),
                "boundary_correct": int(values["boundary_correct"]),
                "boundary_true": int(values["boundary_true"]),
                "boundary_coverage": values["boundary_correct"] / values["boundary_true"] if values["boundary_true"] else float("nan"),
                "interior_correct": int(values["interior_correct"]),
                "interior_true": int(values["interior_true"]),
                "interior_coverage": values["interior_correct"] / values["interior_true"] if values["interior_true"] else float("nan"),
                "hidden_gt_usage": "post_hoc_only",
            }
        )
    feature_rows: list[dict[str, Any]] = []
    for (transition, layer, partition_name, class_id, region), values in sorted(feature_stats.items()):
        count = int(values["count"])
        feature_rows.append(
            {
                "seed": args.seed,
                "transition": transition,
                "layer": layer,
                "partition": partition_name,
                "class_id": class_id,
                "region": region,
                "cell_count": count,
                "old_current_cosine": values["cosine_sum"] / count if count else float("nan"),
                "old_current_l2": values["l2_sum"] / count if count else float("nan"),
                "old_current_normalized_l2": values["normalized_l2_sum"] / count if count else float("nan"),
                "hidden_gt_usage": "post_hoc_boundary_only" if region != "all" else "partition_only",
            }
        )
    write_csv(seed_dir / "prototype_validation_quality.csv", prototype_rows)
    write_csv(seed_dir / "partition_quality.csv", partition_rows)
    write_csv(seed_dir / "feature_separation.csv", feature_rows)
    summary = {
        "protocol_id": "sparcseg_v0_1",
        "seed": args.seed,
        "status": "SPARC_POSTHOC_FEASIBILITY_COMPLETE",
        "optimizer_steps": 0,
        "hidden_gt_usage": "independent_post_hoc_metrics_only",
        "training_objective_hidden_gt_usage": "none",
        "transitions": transition_records,
        "prototype_rows": len(prototype_rows),
        "partition_rows": len(partition_rows),
        "feature_rows": len(feature_rows),
        "environment": _environment(args.physical_gpu, device),
        "diagnostics_manifest_sha256": sha256_path(
            args.data_root.resolve() / "manifests" / "diagnostics" / f"lcrseg_v1_seed{args.seed}.csv"
        ),
        "workspace_hash": _workspace_hash(),
    }
    write_json(seed_dir / "posthoc_summary.json", summary)
    print(json.dumps({"status": summary["status"], "seed": args.seed, "output": str(seed_dir)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("visible", "posthoc"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--physical-gpu", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(stable_seed("sparc-v0.1-process", args.phase, args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(stable_seed("sparc-v0.1-process", args.phase, args.seed))
    return visible_phase(args) if args.phase == "visible" else posthoc_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
