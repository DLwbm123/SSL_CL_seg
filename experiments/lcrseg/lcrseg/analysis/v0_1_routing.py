"""Phase-A, post-hoc V0.1 routing diagnostics.

This module is deliberately analysis-only.  It reads frozen checkpoints and
the separate diagnostics manifest after training has completed; it is never
imported by the training data package, methods, trainer, or continual runner.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from ..common import write_csv, write_json
from ..data import H5LabeledDataset, H5UnlabeledDataset, LabeledTransform, WeakStrongTransform, collate_labeled, collate_unlabeled
from ..engine.checkpoint import load_checkpoint
from ..methods.components.compatibility import CompatibilityOutput, compute_compatibility, zero_compatibility
from ..methods.components.learnability import LearnabilityOutput, compute_learnability
from ..methods.components.pseudo_label import PseudoLabelOutput, build_pseudo_labels
from ..methods.components.relation_field import RelationOutput, relation_field
from ..methods.components.routing import assimilation_loss, relation_consolidation_loss
from ..methods.lcrseg_v0_1 import LCRSegV01Method
from ..models import UNet2D
from .diagnostics import _images_and_labels, diagnostic_records


DEFAULT_RUNS = (
    "fundus_seed0_lcrseg_v0_1_full200e",
    "fundus_seed0_lcrseg_uniform_relation_kd_full200e",
    "fundus_seed0_lcrseg_no_learnability_full200e",
    "fundus_seed0_sequential_ssl_full200e",
)


@dataclass(frozen=True)
class RoutingMaps:
    """Detached raw/applied routing state with the differentiable strong path."""

    strong_logits: torch.Tensor
    current_relation_strong: RelationOutput
    current_relation_weak: RelationOutput
    pseudo: PseudoLabelOutput
    raw_learnability: LearnabilityOutput
    applied_learnability: LearnabilityOutput
    raw_compatibility: CompatibilityOutput
    applied_compatibility: CompatibilityOutput
    old_relation_weak: RelationOutput | None
    current_full_prediction: torch.Tensor


def effective_sample_size(weights: np.ndarray, *, eps: float = 1.0e-12) -> float:
    """Return the standard importance-weight effective sample size."""

    flat = np.asarray(weights, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0
    numerator = float(flat.sum()) ** 2
    denominator = float(np.square(flat).sum()) + float(eps)
    return numerator / denominator if denominator > 0 else 0.0


def _quantiles(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(flat.mean()),
        "p10": float(np.quantile(flat, 0.10)),
        "p50": float(np.quantile(flat, 0.50)),
        "p90": float(np.quantile(flat, 0.90)),
    }


def _binary_boundary(label: np.ndarray) -> np.ndarray:
    """Return a 4-connected, two-sided class-boundary mask."""

    if label.ndim != 2:
        raise ValueError(f"expected a 2D label, got {label.shape}")
    boundary = np.zeros(label.shape, dtype=bool)
    vertical = label[1:, :] != label[:-1, :]
    horizontal = label[:, 1:] != label[:, :-1]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return boundary


def _interior_grid(label: np.ndarray, grid_shape: tuple[int, int], *, boundary_width: float = 3.0) -> np.ndarray:
    boundary = _binary_boundary(label)
    if bool(boundary.any()):
        interior = ndimage.distance_transform_edt(~boundary) > float(boundary_width)
    else:
        interior = np.ones_like(boundary, dtype=bool)
    tensor = torch.from_numpy(interior.astype(np.float32))[None, None]
    return F.interpolate(tensor, size=grid_shape, mode="nearest")[0, 0].bool().numpy()


def _component_areas(prediction: np.ndarray, num_classes: int) -> dict[int, np.ndarray]:
    structure = np.asarray(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8)
    result: dict[int, np.ndarray] = {}
    for class_id in range(num_classes):
        components, count = ndimage.label(prediction == class_id, structure=structure)
        if count == 0:
            result[class_id] = np.empty(0, dtype=np.int64)
            continue
        result[class_id] = np.bincount(components.reshape(-1))[1:].astype(np.int64, copy=False)
    return result


def _small_component_grid(
    prediction: np.ndarray,
    thresholds: dict[int, float],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    structure = np.asarray(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8)
    small = np.zeros(prediction.shape, dtype=bool)
    for class_id, threshold in thresholds.items():
        components, count = ndimage.label(prediction == class_id, structure=structure)
        if count == 0 or not np.isfinite(threshold):
            continue
        areas = np.bincount(components.reshape(-1))
        selected = np.flatnonzero(areas < float(threshold))
        selected = selected[selected != 0]
        if selected.size:
            small |= np.isin(components, selected)
    tensor = torch.from_numpy(small.astype(np.float32))[None, None]
    return F.interpolate(tensor, size=grid_shape, mode="nearest")[0, 0].bool().numpy()


def _unit_learnability(pseudo: PseudoLabelOutput, raw: LearnabilityOutput) -> LearnabilityOutput:
    return replace(raw, score=pseudo.valid.float().detach())


def _unit_compatibility(raw: CompatibilityOutput) -> CompatibilityOutput:
    return replace(raw, score=torch.ones_like(raw.score).detach())


def _load_lcr_method(checkpoint: Path, device: torch.device) -> tuple[LCRSegV01Method, dict[str, Any]]:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    if payload["method_name"] != "lcrseg_v0_1":
        raise ValueError(f"{checkpoint.name} is not an LCR-Seg V0.1 checkpoint")
    config = dict(payload["config_resolved"])
    model_config = dict(config["model"])
    method = LCRSegV01Method(
        UNet2D(
            int(model_config["in_channels"]),
            int(model_config["num_classes"]),
            base_channels=int(model_config.get("base_channels", 16)),
            relation_dim=int(model_config.get("relation_dim", 128)),
        ).to(device),
        config=dict(config.get("method", {})),
    ).to(device)
    method.model.load_state_dict(payload["current_model_state"], strict=True)
    method.load_method_state_dict(payload)
    method.site_id = str(payload["site_id"])
    method.site_index = int(payload["site_index"])
    method.total_steps = int((payload.get("method_statistics") or {}).get("active_site_total_steps") or max(1, int(payload["site_step"])))
    method.model.eval()
    if method.old_model is not None:
        method.old_model.eval()
    if not method.current_anchor_bank.all_classes_valid:
        raise RuntimeError(f"checkpoint has incomplete current anchors: {checkpoint}")
    return method, payload


def _routing_maps(
    method: LCRSegV01Method,
    *,
    weak_image: torch.Tensor,
    strong_image: torch.Tensor,
    site_step: int,
) -> RoutingMaps:
    """Recreate the frozen V0.1 routing tensors without an optimizer update."""

    with torch.no_grad():
        weak_output = method.model(weak_image)
        current_relation_weak = relation_field(
            weak_output.relation_features,
            method.current_anchor_bank,
            temperature=float(method.config["relation_temperature"]),
        )
        pseudo = build_pseudo_labels(
            weak_output.logits.detach().softmax(dim=1),
            current_relation_weak,
            tau_cls=float(method.config["tau_cls"]),
            tau_anchor=float(method.config["tau_anchor"]),
            delta_anchor=float(method.config["delta_anchor"]),
            tau_spatial=float(method.config["tau_spatial"]),
            temperature_cls=float(method.config["temperature_cls"]),
            temperature_anchor=float(method.config["temperature_anchor"]),
            spatial_floor=float(method.config["spatial_floor"]),
        )
        raw_learnability = compute_learnability(
            weak_output.logits,
            current_relation_weak,
            pseudo,
            site_step=int(site_step),
            total_steps=max(1, int(method.total_steps)),
            rank_start=float(method.config["rank_start"]),
            rank_end=float(method.config["rank_end"]),
            rank_temperature=float(method.config["rank_temperature"]),
            relation_margin_center=float(method.config["relation_margin_center"]),
            relation_margin_temperature=float(method.config["relation_margin_temperature"]),
            min_rank_pixels=int(method.config["min_rank_pixels"]),
        )
        current_full_prediction = weak_output.logits.argmax(dim=1).detach()
        old_relation: RelationOutput | None = None
        raw_compatibility = zero_compatibility(current_relation_weak.probabilities)
        if method.old_model is not None and method.old_anchor_bank is not None:
            old_output = method.old_model(weak_image)
            old_relation = relation_field(
                old_output.relation_features,
                method.old_anchor_bank,
                temperature=float(method.config["relation_temperature"]),
            )
            raw_compatibility = compute_compatibility(
                current_relation_weak,
                old_relation,
                old_margin_center=float(method.config["old_margin_center"]),
                old_margin_temperature=float(method.config["old_margin_temperature"]),
                js_temperature=float(method.config["js_temperature"]),
                spatial_floor=float(method.config["spatial_floor"]),
            )
    strong_output = method.model(strong_image)
    current_relation_strong = relation_field(
        strong_output.relation_features,
        method.current_anchor_bank,
        temperature=float(method.config["relation_temperature"]),
    )
    applied_learnability = raw_learnability if bool(method.config["use_learnability"]) else _unit_learnability(pseudo, raw_learnability)
    applied_compatibility = raw_compatibility
    if old_relation is not None and not bool(method.config["use_compatibility"]):
        applied_compatibility = _unit_compatibility(raw_compatibility)
    return RoutingMaps(
        strong_logits=strong_output.logits,
        current_relation_strong=current_relation_strong,
        current_relation_weak=current_relation_weak,
        pseudo=pseudo,
        raw_learnability=raw_learnability,
        applied_learnability=applied_learnability,
        raw_compatibility=raw_compatibility,
        applied_compatibility=applied_compatibility,
        old_relation_weak=old_relation,
        current_full_prediction=current_full_prediction,
    )


def _equal_frequency_rows(
    *,
    metadata: dict[str, Any],
    route: str,
    class_id: int,
    score: np.ndarray,
    weights: np.ndarray,
    valid: np.ndarray,
    correctness: np.ndarray,
    bins: int,
) -> list[dict[str, Any]]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    correctness = np.asarray(correctness, dtype=bool).reshape(-1)
    if not (score.shape == weights.shape == valid.shape == correctness.shape):
        raise ValueError("calibration inputs must have the same shape")
    order = np.argsort(score, kind="stable")
    rows: list[dict[str, Any]] = []
    for bin_index, selected in enumerate(np.array_split(order, int(bins))):
        chosen_score = score[selected]
        chosen_weight = weights[selected]
        chosen_valid = valid[selected]
        chosen_correct = correctness[selected][chosen_valid]
        row = dict(metadata)
        row.update(
            {
                "route": route,
                "class_id": int(class_id),
                "bin": int(bin_index),
                "pixel_count": int(selected.size),
                "valid_pixel_count": int(chosen_valid.sum()),
                "coverage": float(chosen_valid.mean()) if chosen_valid.size else float("nan"),
                "accuracy": float(chosen_correct.mean()) if chosen_correct.size else float("nan"),
                "weight_sum": float(chosen_weight.sum()),
                "effective_pixel_count": effective_sample_size(chosen_weight),
                **{f"score_{key}": value for key, value in _quantiles(chosen_score).items()},
                **{f"weight_{key}": value for key, value in _quantiles(chosen_weight).items()},
            }
        )
        rows.append(row)
    return rows


def _ess_row(
    *,
    metadata: dict[str, Any],
    route: str,
    scope: str,
    class_id: int | None,
    weights: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid_mask = np.asarray(valid, dtype=bool).reshape(-1)
    if weight.shape != valid_mask.shape:
        raise ValueError("ESS weight/valid shape mismatch")
    effective_weights = weight * valid_mask.astype(np.float64)
    valid_count = int(valid_mask.sum())
    n_eff = effective_sample_size(effective_weights)
    row = dict(metadata)
    row.update(
        {
            "route": route,
            "scope": scope,
            "class_id": "" if class_id is None else int(class_id),
            "valid_count": valid_count,
            "weighted_pixel_count": float(effective_weights.sum()),
            "effective_pixel_count": n_eff,
            "effective_fraction_of_valid": n_eff / valid_count if valid_count else float("nan"),
            "weight_mean_over_valid": float(effective_weights[valid_mask].mean()) if valid_count else float("nan"),
        }
    )
    return row


def _norm_and_vector(loss: torch.Tensor, parameters: list[torch.nn.Parameter], *, retain_graph: bool) -> tuple[float, torch.Tensor]:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    # Keep a fixed parameter-space layout for all loss branches.  A
    # segmentation-only loss and a relation-only loss naturally leave
    # different heads unused; omitting those entries would make their vectors
    # incomparable and would corrupt the cosine.
    pieces = [
        (gradient.detach().float() if gradient is not None else torch.zeros_like(parameter, dtype=torch.float32)).reshape(-1)
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]
    if not pieces:
        return 0.0, torch.empty(0, device=loss.device)
    vector = torch.cat(pieces)
    return float(torch.linalg.vector_norm(vector).cpu()), vector


def _gradient_diagnostic(
    *,
    root: Path,
    checkpoint: Path,
    dataset: str,
    site: str,
    seed: int,
    device: torch.device,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Measure unmodified V0.1 loss gradients on one deterministic golden batch."""

    method, payload = _load_lcr_method(checkpoint, device)
    labeled = H5LabeledDataset(root, seed=seed, dataset=dataset, sites=(site,), transform=LabeledTransform(flip_probability=0.0))
    unlabeled = H5UnlabeledDataset(
        root,
        seed=seed,
        dataset=dataset,
        sites=(site,),
        transform=WeakStrongTransform(
            flip_probability=0.0,
            strong_noise_std=0.0,
            brightness_delta=0.0,
            contrast_delta=0.0,
            cutout_probability=0.0,
        ),
    )
    labeled_batch = collate_labeled([labeled[index] for index in range(min(2, len(labeled)))]).to(device)
    unlabeled_batch = collate_unlabeled([unlabeled[index] for index in range(min(2, len(unlabeled)))]).to(device)
    method.model.eval()
    maps = _routing_maps(
        method,
        weak_image=unlabeled_batch.weak_image,
        strong_image=unlabeled_batch.strong_image,
        site_step=max(0, int(payload["site_step"]) - 1),
    )
    assimilation = assimilation_loss(maps.strong_logits, maps.pseudo, maps.applied_learnability, unlabeled_batch.strong_valid_mask)
    relation = (
        relation_consolidation_loss(
            maps.current_relation_strong,
            maps.old_relation_weak,
            maps.applied_compatibility,
            unlabeled_batch.strong_valid_mask,
            distill_temperature=float(method.config["distill_temperature"]),
        )
        if maps.old_relation_weak is not None
        else maps.strong_logits.sum() * 0.0
    )
    raw_l = maps.raw_learnability.score.detach()
    pseudo_valid = maps.pseudo.valid.detach()
    if bool(pseudo_valid.any()):
        l_threshold = torch.median(raw_l[pseudo_valid])
        high_l_mask = pseudo_valid & raw_l.ge(l_threshold)
        low_l_mask = pseudo_valid & raw_l.lt(l_threshold)
    else:
        high_l_mask = torch.zeros_like(pseudo_valid)
        low_l_mask = torch.zeros_like(pseudo_valid)
    high_l = assimilation_loss(
        maps.strong_logits,
        maps.pseudo,
        replace(maps.applied_learnability, score=(maps.applied_learnability.score * high_l_mask.float()).detach()),
        unlabeled_batch.strong_valid_mask,
    )
    low_l = assimilation_loss(
        maps.strong_logits,
        maps.pseudo,
        replace(maps.applied_learnability, score=(maps.applied_learnability.score * low_l_mask.float()).detach()),
        unlabeled_batch.strong_valid_mask,
    )
    raw_c = maps.raw_compatibility.score.detach()
    if maps.old_relation_weak is not None:
        c_threshold = torch.median(raw_c)
        high_c_mask = raw_c.ge(c_threshold)
        low_c_mask = raw_c.lt(c_threshold)
        high_c = relation_consolidation_loss(
            maps.current_relation_strong,
            maps.old_relation_weak,
            replace(maps.applied_compatibility, score=(maps.applied_compatibility.score * high_c_mask.float()).detach()),
            unlabeled_batch.strong_valid_mask,
            distill_temperature=float(method.config["distill_temperature"]),
        )
        low_c = relation_consolidation_loss(
            maps.current_relation_strong,
            maps.old_relation_weak,
            replace(maps.applied_compatibility, score=(maps.applied_compatibility.score * low_c_mask.float()).detach()),
            unlabeled_batch.strong_valid_mask,
            distill_temperature=float(method.config["distill_temperature"]),
        )
    else:
        high_c = maps.strong_logits.sum() * 0.0
        low_c = maps.strong_logits.sum() * 0.0
    parameters = [parameter for parameter in method.model.parameters() if parameter.requires_grad]
    norm_assim, grad_assim = _norm_and_vector(assimilation, parameters, retain_graph=True)
    norm_relation, grad_relation = _norm_and_vector(relation, parameters, retain_graph=True)
    norm_assim_high_l, _ = _norm_and_vector(high_l, parameters, retain_graph=True)
    norm_assim_low_l, _ = _norm_and_vector(low_l, parameters, retain_graph=True)
    norm_relation_high_c, _ = _norm_and_vector(high_c, parameters, retain_graph=True)
    norm_relation_low_c, _ = _norm_and_vector(low_c, parameters, retain_graph=False)
    if grad_assim.numel() and grad_relation.numel() and norm_assim > 1.0e-12 and norm_relation > 1.0e-12:
        cosine = float(torch.dot(grad_assim, grad_relation).div(torch.linalg.vector_norm(grad_assim) * torch.linalg.vector_norm(grad_relation)).clamp(-1.0, 1.0).cpu())
    else:
        cosine = float("nan")
    if method.old_model is not None and any(parameter.grad is not None for parameter in method.old_model.parameters()):
        raise AssertionError("post-hoc gradient diagnostic gave the old model a gradient")
    row = dict(metadata)
    row.update(
        {
            "golden_labeled_case_ids": list(labeled_batch.case_id),
            "golden_unlabeled_case_ids": list(unlabeled_batch.case_id),
            "loss_assim": float(assimilation.detach()),
            "loss_relation": float(relation.detach()),
            "gradient_norm_assim": norm_assim,
            "gradient_norm_relation": norm_relation,
            "gradient_cosine_assim_relation": cosine,
            "gradient_norm_assim_high_l": norm_assim_high_l,
            "gradient_norm_assim_low_l": norm_assim_low_l,
            "gradient_norm_relation_high_c": norm_relation_high_c,
            "gradient_norm_relation_low_c": norm_relation_low_c,
            "old_model_grad_free": True,
        }
    )
    return row


def _component_thresholds(
    *,
    method: LCRSegV01Method,
    root: Path,
    dataset: str,
    site: str,
    seed: int,
    device: torch.device,
) -> dict[int, float]:
    """Compute per-class predicted-component p10 over all current-site unlabeled data."""

    records = diagnostic_records(root, seed=seed, dataset=dataset, site=site)
    areas: dict[int, list[np.ndarray]] = {class_id: [] for class_id in range(method.num_classes)}
    with torch.no_grad():
        for record in records:
            for image, _ in _images_and_labels(record, dataset):
                tensor = torch.from_numpy(image).unsqueeze(0).to(device)
                prediction = method.model(tensor).logits.argmax(dim=1)[0].detach().cpu().numpy()
                for class_id, values in _component_areas(prediction, method.num_classes).items():
                    if values.size:
                        areas[class_id].append(values)
    return {
        class_id: float(np.quantile(np.concatenate(values), 0.10)) if values else float("nan")
        for class_id, values in areas.items()
    }


def _metadata(run_name: str, checkpoint: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "method_name": str(payload["method_name"]),
        "site_id": str(payload["site_id"]),
        "site_index": int(payload["site_index"]),
    }


def _region_row(
    *,
    metadata: dict[str, Any],
    region: str,
    class_id: int,
    mask: np.ndarray,
    total_pixels: int,
    raw_l: np.ndarray,
    applied_l: np.ndarray,
    pseudo_valid: np.ndarray,
    pseudo_correct: np.ndarray,
    raw_c: np.ndarray | None,
    applied_c: np.ndarray | None,
    old_correct: np.ndarray | None,
    relation_js: np.ndarray | None,
    agreement: np.ndarray | None,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    valid = pseudo_valid[selected]
    pseudo_accuracy = pseudo_correct[selected][valid]
    row = dict(metadata)
    row.update(
        {
            "region": region,
            "class_id": int(class_id),
            "pixel_count": int(selected.sum()),
            "pixel_fraction": float(selected.sum() / total_pixels) if total_pixels else float("nan"),
            "pseudo_valid_coverage": float(valid.mean()) if valid.size else float("nan"),
            "pseudo_label_accuracy": float(pseudo_accuracy.mean()) if pseudo_accuracy.size else float("nan"),
            **{f"raw_l_{key}": value for key, value in _quantiles(raw_l[selected]).items()},
            **{f"assim_weight_{key}": value for key, value in _quantiles(applied_l[selected]).items()},
            "assim_effective_sample_size": effective_sample_size(applied_l[selected] * valid.astype(np.float64)),
        }
    )
    if raw_c is None or applied_c is None or old_correct is None or relation_js is None or agreement is None:
        row.update(
            {
                "compatibility_available": False,
                "old_relation_accuracy": float("nan"),
                "current_old_agreement": float("nan"),
                "cons_effective_sample_size": float("nan"),
                **{f"raw_c_{key}": float("nan") for key in ("mean", "p10", "p50", "p90")},
                **{f"cons_weight_{key}": float("nan") for key in ("mean", "p10", "p50", "p90")},
                **{f"relation_js_{key}": float("nan") for key in ("mean", "p10", "p50", "p90")},
            }
        )
        return row
    row.update(
        {
            "compatibility_available": True,
            "old_relation_accuracy": float(old_correct[selected].mean()) if bool(selected.any()) else float("nan"),
            "current_old_agreement": float(agreement[selected].mean()) if bool(selected.any()) else float("nan"),
            "cons_effective_sample_size": effective_sample_size(applied_c[selected]),
            **{f"raw_c_{key}": value for key, value in _quantiles(raw_c[selected]).items()},
            **{f"cons_weight_{key}": value for key, value in _quantiles(applied_c[selected]).items()},
            **{f"relation_js_{key}": value for key, value in _quantiles(relation_js[selected]).items()},
        }
    )
    return row


def _analyze_lcr_checkpoint(
    *,
    root: Path,
    run_name: str,
    checkpoint: Path,
    dataset: str,
    seed: int,
    device: torch.device,
    bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    method, payload = _load_lcr_method(checkpoint, device)
    metadata = _metadata(run_name, checkpoint, payload)
    site = str(payload["site_id"])
    thresholds = _component_thresholds(method=method, root=root, dataset=dataset, site=site, seed=seed, device=device)
    records = diagnostic_records(root, seed=seed, dataset=dataset, site=site)
    class_l: dict[int, dict[str, list[np.ndarray]]] = {class_id: {"score": [], "weight": [], "valid": [], "correct": []} for class_id in range(method.num_classes)}
    class_c: dict[int, dict[str, list[np.ndarray]]] = {class_id: {"score": [], "weight": [], "valid": [], "correct": []} for class_id in range(method.num_classes)}
    region_data: dict[tuple[str, int], dict[str, list[np.ndarray]]] = {}
    ess_data: dict[str, dict[str, list[np.ndarray]]] = {"assimilation": {"weight": [], "valid": [], "class": []}, "consolidation": {"weight": [], "valid": [], "class": []}}
    total_pixels = 0
    has_old = method.old_model is not None and method.old_anchor_bank is not None
    with torch.no_grad():
        for record in records:
            for image, label in _images_and_labels(record, dataset):
                image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
                maps = _routing_maps(
                    method,
                    weak_image=image_tensor,
                    strong_image=image_tensor,
                    site_step=max(0, int(payload["site_step"]) - 1),
                )
                grid_shape = tuple(maps.current_relation_weak.probabilities.shape[-2:])
                grid_label = F.interpolate(torch.from_numpy(label).to(device).unsqueeze(0).unsqueeze(0).float(), size=grid_shape, mode="nearest")[0, 0].long()
                current_class = maps.current_relation_weak.predicted_class[0].detach().cpu().numpy().reshape(-1)
                pseudo_valid = maps.pseudo.valid[0, 0].detach().cpu().numpy().astype(bool).reshape(-1)
                pseudo_correct = maps.pseudo.labels[0].eq(grid_label).detach().cpu().numpy().astype(bool).reshape(-1)
                raw_l = maps.raw_learnability.score[0, 0].detach().cpu().numpy().reshape(-1)
                applied_l = maps.applied_learnability.score[0, 0].detach().cpu().numpy().reshape(-1)
                full_prediction = maps.current_full_prediction[0].detach().cpu().numpy()
                interior = _interior_grid(label, grid_shape).reshape(-1)
                small = _small_component_grid(full_prediction, thresholds, grid_shape).reshape(-1)
                boundary = ~interior
                total_pixels += current_class.size
                raw_c: np.ndarray | None = None
                applied_c: np.ndarray | None = None
                old_correct: np.ndarray | None = None
                old_class: np.ndarray | None = None
                relation_js: np.ndarray | None = None
                agreement: np.ndarray | None = None
                if maps.old_relation_weak is not None:
                    old_class = maps.old_relation_weak.predicted_class[0].detach().cpu().numpy().reshape(-1)
                    old_correct = maps.old_relation_weak.predicted_class[0].eq(grid_label).detach().cpu().numpy().astype(bool).reshape(-1)
                    raw_c = maps.raw_compatibility.score[0, 0].detach().cpu().numpy().reshape(-1)
                    applied_c = maps.applied_compatibility.score[0, 0].detach().cpu().numpy().reshape(-1)
                    relation_js = maps.raw_compatibility.js_divergence[0, 0].detach().cpu().numpy().reshape(-1)
                    agreement = maps.raw_compatibility.agreement[0, 0].detach().cpu().numpy().astype(bool).reshape(-1)
                for class_id in range(method.num_classes):
                    class_mask = current_class == class_id
                    if bool(class_mask.any()):
                        class_l[class_id]["score"].append(raw_l[class_mask])
                        class_l[class_id]["weight"].append(applied_l[class_mask])
                        class_l[class_id]["valid"].append(pseudo_valid[class_mask])
                        class_l[class_id]["correct"].append(pseudo_correct[class_mask])
                    if old_class is not None and raw_c is not None and applied_c is not None and old_correct is not None:
                        old_mask = old_class == class_id
                        if bool(old_mask.any()):
                            class_c[class_id]["score"].append(raw_c[old_mask])
                            class_c[class_id]["weight"].append(applied_c[old_mask])
                            class_c[class_id]["valid"].append(np.ones(int(old_mask.sum()), dtype=bool))
                            class_c[class_id]["correct"].append(old_correct[old_mask])
                for region_name, region_mask in (("all", np.ones_like(interior, dtype=bool)), ("interior", interior), ("boundary", boundary), ("small_component", small)):
                    for class_id in range(method.num_classes):
                        selected = region_mask & (current_class == class_id)
                        if not bool(selected.any()):
                            continue
                        holder = region_data.setdefault((region_name, class_id), {"mask": [], "raw_l": [], "applied_l": [], "pseudo_valid": [], "pseudo_correct": [], "raw_c": [], "applied_c": [], "old_correct": [], "relation_js": [], "agreement": []})
                        holder["mask"].append(selected)
                        holder["raw_l"].append(raw_l)
                        holder["applied_l"].append(applied_l)
                        holder["pseudo_valid"].append(pseudo_valid)
                        holder["pseudo_correct"].append(pseudo_correct)
                        if raw_c is not None and applied_c is not None and old_correct is not None and relation_js is not None and agreement is not None:
                            holder["raw_c"].append(raw_c)
                            holder["applied_c"].append(applied_c)
                            holder["old_correct"].append(old_correct)
                            holder["relation_js"].append(relation_js)
                            holder["agreement"].append(agreement)
                ess_data["assimilation"]["weight"].append(applied_l)
                ess_data["assimilation"]["valid"].append(pseudo_valid)
                ess_data["assimilation"]["class"].append(current_class)
                if raw_c is not None and applied_c is not None and old_class is not None:
                    ess_data["consolidation"]["weight"].append(applied_c)
                    ess_data["consolidation"]["valid"].append(np.ones_like(applied_c, dtype=bool))
                    ess_data["consolidation"]["class"].append(old_class)
    class_rows: list[dict[str, Any]] = []
    for class_id, values in class_l.items():
        if values["score"]:
            class_rows.extend(_equal_frequency_rows(metadata=metadata, route="learnability", class_id=class_id, score=np.concatenate(values["score"]), weights=np.concatenate(values["weight"]), valid=np.concatenate(values["valid"]), correctness=np.concatenate(values["correct"]), bins=bins))
    for class_id, values in class_c.items():
        if values["score"]:
            class_rows.extend(_equal_frequency_rows(metadata=metadata, route="compatibility", class_id=class_id, score=np.concatenate(values["score"]), weights=np.concatenate(values["weight"]), valid=np.concatenate(values["valid"]), correctness=np.concatenate(values["correct"]), bins=bins))
    region_rows: list[dict[str, Any]] = []
    for (region_name, class_id), values in sorted(region_data.items()):
        mask = np.concatenate(values["mask"])
        raw_c_values = np.concatenate(values["raw_c"]) if values["raw_c"] else None
        applied_c_values = np.concatenate(values["applied_c"]) if values["applied_c"] else None
        old_correct_values = np.concatenate(values["old_correct"]) if values["old_correct"] else None
        relation_js_values = np.concatenate(values["relation_js"]) if values["relation_js"] else None
        agreement_values = np.concatenate(values["agreement"]) if values["agreement"] else None
        region_rows.append(_region_row(metadata=metadata, region=region_name, class_id=class_id, mask=mask, total_pixels=total_pixels, raw_l=np.concatenate(values["raw_l"]), applied_l=np.concatenate(values["applied_l"]), pseudo_valid=np.concatenate(values["pseudo_valid"]), pseudo_correct=np.concatenate(values["pseudo_correct"]), raw_c=raw_c_values, applied_c=applied_c_values, old_correct=old_correct_values, relation_js=relation_js_values, agreement=agreement_values))
    ess_rows: list[dict[str, Any]] = []
    for route, values in ess_data.items():
        if not values["weight"]:
            continue
        weight = np.concatenate(values["weight"])
        valid = np.concatenate(values["valid"])
        classes = np.concatenate(values["class"])
        ess_rows.append(_ess_row(metadata=metadata, route=route, scope="global", class_id=None, weights=weight, valid=valid))
        for class_id in range(method.num_classes):
            selected = classes == class_id
            if bool(selected.any()):
                ess_rows.append(_ess_row(metadata=metadata, route=route, scope="class", class_id=class_id, weights=weight[selected], valid=valid[selected]))
    gradient = _gradient_diagnostic(root=root, checkpoint=checkpoint, dataset=dataset, site=site, seed=seed, device=device, metadata=metadata)
    checkpoint_summary = {
        **metadata,
        "records": len(records),
        "pixels": total_pixels,
        "has_historical_model": has_old,
        "component_area_p10": thresholds,
    }
    return class_rows, region_rows, ess_rows, {"summary": checkpoint_summary, "gradient": gradient}


def _discover_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints = sorted(run_dir.glob("checkpoint_final_site*_*.pt"))
    if checkpoints:
        return checkpoints
    final = run_dir / "checkpoint_final.pt"
    return [final] if final.is_file() else []


def _png_bar(path: Path, rows: list[dict[str, Any]], *, title: str, value_key: str) -> None:
    """Write a compact dependency-light overview figure with no patient data."""

    from PIL import Image, ImageDraw

    width, height, margin = 960, 420, 60
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), title, fill="black")
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    finite = [float(row[value_key]) for row in rows if row.get(value_key) not in (None, "") and np.isfinite(float(row[value_key]))]
    maximum = max([1.0, *finite])
    count = max(1, len(rows))
    step = (width - 2 * margin) / count
    for index, row in enumerate(rows):
        value = float(row.get(value_key, 0.0) or 0.0)
        value = value if np.isfinite(value) else 0.0
        left = margin + index * step + 3
        top = height - margin - (value / maximum) * (height - 2 * margin)
        draw.rectangle((left, top, left + max(2, step - 6), height - margin), fill=(54, 118, 191))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def analyze_v0_1_routing(
    *,
    root: Path,
    run_root: Path,
    output_dir: Path,
    dataset: str = "fundus",
    seed: int = 0,
    run_names: Iterable[str] = DEFAULT_RUNS,
    device: str | torch.device = "cuda",
    bins: int = 10,
) -> dict[str, Any]:
    """Run the prerequisite V0.1 routing audit against frozen checkpoints."""

    root = Path(root).resolve()
    run_root = Path(run_root).resolve()
    output_dir = Path(output_dir).resolve()
    frozen_h5 = (root / "h5" / "v1").resolve()
    if output_dir == frozen_h5 or frozen_h5 in output_dir.parents:
        raise ValueError("routing analysis may not write beneath frozen HDF5")
    if int(bins) != 10:
        raise ValueError("the V0.2 preregistration fixes Phase-A calibration at 10 bins")
    device_obj = torch.device(device)
    class_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    ess_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for run_name in run_names:
        run_dir = run_root / str(run_name)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"required routing-diagnostic run is missing: {run_dir}")
        checkpoints = _discover_checkpoints(run_dir)
        if not checkpoints:
            raise FileNotFoundError(f"no final/site checkpoints found in {run_dir}")
        for checkpoint in checkpoints:
            payload = load_checkpoint(checkpoint, map_location="cpu")
            basic = _metadata(str(run_name), checkpoint, payload)
            if payload["method_name"] != "lcrseg_v0_1":
                checkpoint_rows.append({**basic, "routing_status": "not_applicable_no_lcr_relation_state", "reason": "baseline checkpoint has no LCR semantic-anchor or old-relation state"})
                continue
            try:
                current_class_rows, current_region_rows, current_ess_rows, detail = _analyze_lcr_checkpoint(root=root, run_name=str(run_name), checkpoint=checkpoint, dataset=dataset, seed=seed, device=device_obj, bins=int(bins))
            except BaseException as exc:
                errors.append({**basic, "error_type": type(exc).__name__, "error": str(exc)})
                raise
            class_rows.extend(current_class_rows)
            region_rows.extend(current_region_rows)
            ess_rows.extend(current_ess_rows)
            gradient_rows.append(detail["gradient"])
            checkpoint_rows.append({**detail["summary"], "routing_status": "analyzed", "reason": ""})
    output_dir.mkdir(parents=True, exist_ok=True)
    class_fields = ["run_name", "checkpoint", "checkpoint_name", "method_name", "site_id", "site_index", "route", "class_id", "bin", "pixel_count", "valid_pixel_count", "coverage", "accuracy", "weight_sum", "effective_pixel_count", "score_mean", "score_p10", "score_p50", "score_p90", "weight_mean", "weight_p10", "weight_p50", "weight_p90"]
    region_fields = ["run_name", "checkpoint", "checkpoint_name", "method_name", "site_id", "site_index", "region", "class_id", "pixel_count", "pixel_fraction", "pseudo_valid_coverage", "pseudo_label_accuracy", "raw_l_mean", "raw_l_p10", "raw_l_p50", "raw_l_p90", "assim_weight_mean", "assim_weight_p10", "assim_weight_p50", "assim_weight_p90", "assim_effective_sample_size", "compatibility_available", "raw_c_mean", "raw_c_p10", "raw_c_p50", "raw_c_p90", "old_relation_accuracy", "relation_js_mean", "relation_js_p10", "relation_js_p50", "relation_js_p90", "current_old_agreement", "cons_weight_mean", "cons_weight_p10", "cons_weight_p50", "cons_weight_p90", "cons_effective_sample_size"]
    ess_fields = ["run_name", "checkpoint", "checkpoint_name", "method_name", "site_id", "site_index", "route", "scope", "class_id", "valid_count", "weighted_pixel_count", "effective_pixel_count", "effective_fraction_of_valid", "weight_mean_over_valid"]
    gradient_fields = ["run_name", "checkpoint", "checkpoint_name", "method_name", "site_id", "site_index", "golden_labeled_case_ids", "golden_unlabeled_case_ids", "loss_assim", "loss_relation", "gradient_norm_assim", "gradient_norm_relation", "gradient_cosine_assim_relation", "gradient_norm_assim_high_l", "gradient_norm_assim_low_l", "gradient_norm_relation_high_c", "gradient_norm_relation_low_c", "old_model_grad_free"]
    checkpoint_fields = ["run_name", "checkpoint", "checkpoint_name", "method_name", "site_id", "site_index", "routing_status", "reason", "records", "pixels", "has_historical_model", "component_area_p10"]
    write_csv(output_dir / "classwise_calibration.csv", class_rows, fieldnames=class_fields)
    write_csv(output_dir / "regionwise_calibration.csv", region_rows, fieldnames=region_fields)
    write_csv(output_dir / "effective_sample_size.csv", ess_rows, fieldnames=ess_fields)
    write_csv(output_dir / "gradient_diagnostics.csv", gradient_rows, fieldnames=gradient_fields)
    write_csv(output_dir / "checkpoint_inventory.csv", checkpoint_rows, fieldnames=checkpoint_fields)
    _png_bar(output_dir / "classwise_calibration.png", [row for row in class_rows if row["route"] == "learnability" and row["class_id"] == 0], title="V0.1 learnability-bin pseudo accuracy (class 0 overview)", value_key="accuracy")
    _png_bar(output_dir / "regionwise_routing.png", region_rows, title="V0.1 routing region pseudo accuracy", value_key="pseudo_label_accuracy")
    _png_bar(output_dir / "gradient_diagnostics.png", gradient_rows, title="V0.1 gradient cosine diagnostic", value_key="gradient_cosine_assim_relation")
    summary = {
        "dataset": dataset,
        "seed": int(seed),
        "run_names": list(run_names),
        "frozen_input_root": str(root),
        "output_dir": str(output_dir),
        "classwise_rows": len(class_rows),
        "regionwise_rows": len(region_rows),
        "ess_rows": len(ess_rows),
        "gradient_rows": len(gradient_rows),
        "checkpoint_rows": len(checkpoint_rows),
        "errors": errors,
        "hidden_gt_usage": "post_hoc_diagnostics_process_only",
        "sequential_ssl_policy": "inventory_only_not_applicable_no_lcr_relation_state",
    }
    write_json(output_dir / "routing_diagnostic_summary.json", summary)
    return summary
