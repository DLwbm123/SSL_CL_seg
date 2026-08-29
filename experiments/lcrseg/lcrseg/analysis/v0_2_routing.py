"""Post-hoc, hidden-GT diagnostics for one frozen LCR-Seg V0.2 Fundus run.

The training runner never imports this module.  It is invoked only after a
checkpoint is complete, and it reads diagnostic labels only in this separate
analysis process.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ..common import sha256_path, write_csv, write_json
from ..data import H5LabeledDataset, H5UnlabeledDataset, LabeledTransform, WeakStrongTransform, collate_labeled, collate_unlabeled
from ..engine.checkpoint import load_checkpoint
from ..methods.components.compatibility import CompatibilityOutput, compute_compatibility, zero_compatibility
from ..methods.components.learnability import LearnabilityOutput, compute_learnability
from ..methods.components.pseudo_label import PseudoLabelOutput, build_pseudo_labels
from ..methods.components.rejection_only_routing import RejectionOnlyOutput
from ..methods.components.relation_field import RelationOutput
from ..methods.lcrseg_v0_2 import LCRSegV02Method
from ..models import UNet2D
from .diagnostics import _images_and_labels, diagnostic_records
from .v0_1_routing import _component_areas, _interior_grid, _norm_and_vector, _small_component_grid, effective_sample_size


@dataclass(frozen=True)
class V02RoutingMaps:
    weak_relation: RelationOutput
    pseudo: PseudoLabelOutput
    raw_learnability: LearnabilityOutput
    admission_mask: torch.Tensor
    raw_compatibility: CompatibilityOutput
    old_relation: RelationOutput | None
    consolidation: RejectionOnlyOutput
    full_prediction: torch.Tensor


def _quantiles(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if not flat.size:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(flat.mean()),
        "p10": float(np.quantile(flat, 0.10)),
        "p50": float(np.quantile(flat, 0.50)),
        "p90": float(np.quantile(flat, 0.90)),
    }


def _load_method(checkpoint: Path, device: torch.device) -> tuple[LCRSegV02Method, dict[str, Any]]:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    if payload["method_name"] != "lcrseg_v0_2" or str(payload["method_version"]) != "0.2":
        raise ValueError(f"{checkpoint.name} is not an LCR-Seg V0.2 checkpoint")
    config = dict(payload["config_resolved"])
    model_config = dict(config["model"])
    model = UNet2D(
        int(model_config["in_channels"]),
        int(model_config["num_classes"]),
        base_channels=int(model_config.get("base_channels", 16)),
        relation_dim=int(model_config.get("relation_dim", 128)),
    ).to(device)
    method = LCRSegV02Method(model, config=dict(config.get("method", {}))).to(device)
    method.model.load_state_dict(payload["current_model_state"], strict=True)
    method.load_method_state_dict(payload)
    method.site_id = str(payload["site_id"])
    method.site_index = int(payload["site_index"])
    method.total_steps = int((payload.get("method_statistics") or {}).get("active_site_total_steps") or max(1, int(payload["site_step"])))
    if not method.current_anchor_bank.all_classes_valid:
        raise RuntimeError(f"checkpoint has incomplete current anchors: {checkpoint}")
    method.model.eval()
    if method.old_model is not None:
        method.old_model.eval()
    return method, payload


@torch.no_grad()
def _routing_maps(method: LCRSegV02Method, *, image: torch.Tensor, site_step: int) -> V02RoutingMaps:
    """Recreate raw routing plus applied V0.2 masks without an optimizer update."""

    weak_output = method.model(image)
    weak_relation = method._relation(weak_output.relation_features, method.current_anchor_bank)
    pseudo = build_pseudo_labels(
        weak_output.logits.detach().softmax(dim=1),
        weak_relation,
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
        weak_relation,
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
    strong_valid = torch.ones((image.shape[0], 1, image.shape[-2], image.shape[-1]), device=image.device, dtype=torch.bool)
    admission = method._admission(pseudo, raw_learnability, strong_valid, site_step=int(site_step))
    old_relation: RelationOutput | None = None
    raw_compatibility: CompatibilityOutput = zero_compatibility(weak_relation.probabilities)
    if method.old_model is not None and method.old_anchor_bank is not None:
        old_output = method.old_model(image)
        old_relation = method._relation(old_output.relation_features, method.old_anchor_bank)
        raw_compatibility = compute_compatibility(
            weak_relation,
            old_relation,
            old_margin_center=float(method.config["old_margin_center"]),
            old_margin_temperature=float(method.config["old_margin_temperature"]),
            js_temperature=float(method.config["js_temperature"]),
            spatial_floor=float(method.config["spatial_floor"]),
        )
    consolidation = method._consolidation(raw_compatibility, old_relation, strong_valid)
    return V02RoutingMaps(
        weak_relation=weak_relation,
        pseudo=pseudo,
        raw_learnability=raw_learnability,
        admission_mask=admission.mask.detach(),
        raw_compatibility=raw_compatibility,
        old_relation=old_relation,
        consolidation=consolidation,
        full_prediction=weak_output.logits.argmax(dim=1).detach(),
    )


def _component_thresholds(
    *,
    method: LCRSegV02Method,
    root: Path,
    dataset: str,
    site: str,
    seed: int,
    device: torch.device,
) -> dict[int, float]:
    records = diagnostic_records(root, seed=seed, dataset=dataset, site=site)
    areas: dict[int, list[np.ndarray]] = {class_id: [] for class_id in range(method.num_classes)}
    with torch.no_grad():
        for record in records:
            for image, _ in _images_and_labels(record, dataset):
                prediction = method.model(torch.from_numpy(image).unsqueeze(0).to(device)).logits.argmax(dim=1)[0].cpu().numpy()
                for class_id, values in _component_areas(prediction, method.num_classes).items():
                    if values.size:
                        areas[class_id].append(values)
    return {
        class_id: float(np.quantile(np.concatenate(values), 0.10)) if values else float("nan")
        for class_id, values in areas.items()
    }


def _metadata(run_dir: Path, checkpoint: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": run_dir.name,
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": sha256_path(checkpoint),
        "method_name": str(payload["method_name"]),
        "site_id": str(payload["site_id"]),
        "site_index": int(payload["site_index"]),
    }


def _bin_rows(
    *,
    metadata: dict[str, Any],
    route: str,
    class_id: int,
    score: np.ndarray,
    valid: np.ndarray,
    correctness: np.ndarray,
    applied_weight: np.ndarray,
    bins: int,
) -> list[dict[str, Any]]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    correctness = np.asarray(correctness, dtype=bool).reshape(-1)
    applied_weight = np.asarray(applied_weight, dtype=np.float64).reshape(-1)
    if not (score.shape == valid.shape == correctness.shape == applied_weight.shape):
        raise ValueError("V0.2 calibration inputs must have equal shape")
    order = np.argsort(score, kind="stable")
    rows: list[dict[str, Any]] = []
    for bin_index, selected in enumerate(np.array_split(order, int(bins))):
        chosen_valid = valid[selected]
        chosen_correct = correctness[selected][chosen_valid]
        chosen_weight = applied_weight[selected]
        row = dict(metadata)
        row.update(
            {
                "route": route,
                "class_id": int(class_id),
                "bin": int(bin_index),
                "pixel_count": int(selected.size),
                "valid_pixel_count": int(chosen_valid.sum()),
                "valid_coverage": float(chosen_valid.mean()) if selected.size else float("nan"),
                "correctness": float(chosen_correct.mean()) if chosen_correct.size else float("nan"),
                "applied_count": int(np.count_nonzero(chosen_weight[chosen_valid] > 0.0)),
                "applied_fraction": float(np.mean(chosen_weight[chosen_valid] > 0.0)) if bool(chosen_valid.any()) else float("nan"),
                "weight_mean_over_valid": float(chosen_weight[chosen_valid].mean()) if bool(chosen_valid.any()) else float("nan"),
                "effective_pixel_count": effective_sample_size(chosen_weight * chosen_valid.astype(np.float64)),
                **{f"score_{key}": value for key, value in _quantiles(score[selected]).items()},
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
        raise ValueError("V0.2 ESS weight and valid inputs must match")
    effective = weight * valid_mask.astype(np.float64)
    valid_count = int(valid_mask.sum())
    n_eff = effective_sample_size(effective)
    row = dict(metadata)
    row.update(
        {
            "route": route,
            "scope": scope,
            "class_id": "" if class_id is None else int(class_id),
            "valid_count": valid_count,
            "weighted_pixel_count": float(effective.sum()),
            "effective_pixel_count": n_eff,
            "effective_fraction_of_valid": n_eff / valid_count if valid_count else float("nan"),
            "weight_mean_over_valid": float(effective[valid_mask].mean()) if valid_count else float("nan"),
        }
    )
    return row


def _region_row(
    *,
    metadata: dict[str, Any],
    region: str,
    class_id: int,
    mask: np.ndarray,
    total_pixels: int,
    raw_learnability: np.ndarray,
    pseudo_valid: np.ndarray,
    pseudo_correct: np.ndarray,
    admission: np.ndarray,
    raw_compatibility: np.ndarray | None,
    calibrated_compatibility: np.ndarray | None,
    old_correct: np.ndarray | None,
    rejection: np.ndarray | None,
    consolidation_weight: np.ndarray | None,
    relation_valid: np.ndarray | None,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    pseudo_is_valid = pseudo_valid[selected]
    row = dict(metadata)
    row.update(
        {
            "region": region,
            "class_id": int(class_id),
            "pixel_count": int(selected.sum()),
            "pixel_fraction": float(selected.sum() / total_pixels) if total_pixels else float("nan"),
            "pseudo_valid_coverage": float(pseudo_is_valid.mean()) if pseudo_is_valid.size else float("nan"),
            "pseudo_label_accuracy": float(pseudo_correct[selected][pseudo_is_valid].mean()) if bool(pseudo_is_valid.any()) else float("nan"),
            "admission_fraction": float(admission[selected][pseudo_is_valid].mean()) if bool(pseudo_is_valid.any()) else float("nan"),
            "assim_effective_sample_size": effective_sample_size(admission[selected] * pseudo_is_valid.astype(np.float64)),
            **{f"raw_l_{key}": value for key, value in _quantiles(raw_learnability[selected]).items()},
        }
    )
    if raw_compatibility is None or calibrated_compatibility is None or old_correct is None or rejection is None or consolidation_weight is None or relation_valid is None:
        row.update(
            {
                "compatibility_available": False,
                "old_relation_accuracy": float("nan"),
                "rejection_fraction": float("nan"),
                "consolidation_weight_mean": float("nan"),
                "cons_effective_sample_size": float("nan"),
                **{f"raw_c_{key}": float("nan") for key in ("mean", "p10", "p50", "p90")},
                **{f"calibrated_c_{key}": float("nan") for key in ("mean", "p10", "p50", "p90")},
            }
        )
        return row
    relation_is_valid = relation_valid[selected]
    row.update(
        {
            "compatibility_available": True,
            "old_relation_accuracy": float(old_correct[selected][relation_is_valid].mean()) if bool(relation_is_valid.any()) else float("nan"),
            "rejection_fraction": float(rejection[selected][relation_is_valid].mean()) if bool(relation_is_valid.any()) else float("nan"),
            "consolidation_weight_mean": float(consolidation_weight[selected][relation_is_valid].mean()) if bool(relation_is_valid.any()) else float("nan"),
            "cons_effective_sample_size": effective_sample_size(consolidation_weight[selected] * relation_is_valid.astype(np.float64)),
            **{f"raw_c_{key}": value for key, value in _quantiles(raw_compatibility[selected]).items()},
            **{f"calibrated_c_{key}": value for key, value in _quantiles(calibrated_compatibility[selected]).items()},
        }
    )
    return row


def _gradient_row(
    *,
    root: Path,
    checkpoint: Path,
    dataset: str,
    site: str,
    seed: int,
    device: torch.device,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Measure the exact V0.2 assimilation/relation gradients on a fixed train batch."""

    method, payload = _load_method(checkpoint, device)
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
    result = method.training_step(
        labeled_batch,
        unlabeled_batch,
        global_step=int(payload["global_step"]),
        site_step=max(0, int(payload["site_step"]) - 1),
    )
    parameters = [parameter for parameter in method.model.parameters() if parameter.requires_grad]
    norm_assim, grad_assim = _norm_and_vector(result.losses["loss_assim"], parameters, retain_graph=True)
    norm_relation, grad_relation = _norm_and_vector(result.losses["loss_relation"], parameters, retain_graph=False)
    cosine = float("nan")
    if grad_assim.numel() and grad_relation.numel() and norm_assim > 1.0e-12 and norm_relation > 1.0e-12:
        cosine = float(torch.dot(grad_assim, grad_relation).div(torch.linalg.vector_norm(grad_assim) * torch.linalg.vector_norm(grad_relation)).clamp(-1.0, 1.0).cpu())
    if method.old_model is not None and any(parameter.grad is not None for parameter in method.old_model.parameters()):
        raise AssertionError("V0.2 gradient diagnostic gave the old model a gradient")
    row = dict(metadata)
    row.update(
        {
            "golden_labeled_case_ids": list(labeled_batch.case_id),
            "golden_unlabeled_case_ids": list(unlabeled_batch.case_id),
            "loss_assim": float(result.losses["loss_assim"].detach()),
            "loss_relation": float(result.losses["loss_relation"].detach()),
            "gradient_norm_assim": norm_assim,
            "gradient_norm_relation": norm_relation,
            "gradient_cosine_assim_relation": cosine,
            "old_model_grad_free": True,
        }
    )
    return row


def _branch_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("branch_statistics_site*.json")):
        payload = json.loads(path.read_text())
        stats = dict(payload.get("statistics") or {})
        candidate = list(stats.get("candidate_counts_by_class") or [])
        selected = list(stats.get("selected_counts_by_class") or [])
        relation = list(stats.get("relation_counts_by_class") or [])
        rejected = list(stats.get("rejected_counts_by_class") or [])
        num_classes = max(len(candidate), len(selected), len(relation), len(rejected))
        for class_id in range(num_classes):
            candidate_count = int(candidate[class_id]) if class_id < len(candidate) else 0
            selected_count = int(selected[class_id]) if class_id < len(selected) else 0
            relation_count = int(relation[class_id]) if class_id < len(relation) else 0
            rejected_count = int(rejected[class_id]) if class_id < len(rejected) else 0
            rows.append(
                {
                    "site_id": payload.get("site_id", ""),
                    "site_index": payload.get("site_index", ""),
                    "class_id": class_id,
                    "calibrator_status": payload.get("calibrator_status", ""),
                    "calibrator_last_update_epoch": payload.get("calibrator_last_update_epoch", ""),
                    "pseudo_candidate_count": candidate_count,
                    "assim_selected_count": selected_count,
                    "assim_selected_fraction": selected_count / candidate_count if candidate_count else float("nan"),
                    "relation_valid_count": relation_count,
                    "compat_rejected_count": rejected_count,
                    "compat_rejected_fraction": rejected_count / relation_count if relation_count else float("nan"),
                }
            )
    for path in sorted(run_dir.glob("calibration_site*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                calibration_rows.append({"source_file": path.name, **row})
    return rows, calibration_rows


def _analyze_checkpoint(
    *,
    root: Path,
    run_dir: Path,
    checkpoint: Path,
    dataset: str,
    seed: int,
    device: torch.device,
    bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    method, payload = _load_method(checkpoint, device)
    metadata = _metadata(run_dir, checkpoint, payload)
    site = str(payload["site_id"])
    records = diagnostic_records(root, seed=seed, dataset=dataset, site=site)
    thresholds = _component_thresholds(method=method, root=root, dataset=dataset, site=site, seed=seed, device=device)
    l_data = {class_id: {"score": [], "valid": [], "correct": [], "weight": []} for class_id in range(method.num_classes)}
    c_data = {class_id: {"raw": [], "calibrated": [], "valid": [], "correct": [], "weight": []} for class_id in range(method.num_classes)}
    region_data: dict[tuple[str, int], dict[str, list[np.ndarray]]] = {}
    ess_data = {
        "assimilation": {"weight": [], "valid": [], "class": []},
        "consolidation": {"weight": [], "valid": [], "class": []},
    }
    total_pixels = 0
    with torch.no_grad():
        for record in records:
            for image, label in _images_and_labels(record, dataset):
                route = _routing_maps(method, image=torch.from_numpy(image).unsqueeze(0).to(device), site_step=max(0, int(payload["site_step"]) - 1))
                grid_shape = tuple(route.weak_relation.probabilities.shape[-2:])
                grid_label = F.interpolate(torch.from_numpy(label).to(device).unsqueeze(0).unsqueeze(0).float(), size=grid_shape, mode="nearest")[0, 0].long()
                current_class = route.weak_relation.predicted_class[0].cpu().numpy().reshape(-1)
                pseudo_class = route.pseudo.labels[0].cpu().numpy().reshape(-1)
                pseudo_valid = route.pseudo.valid[0, 0].cpu().numpy().astype(bool).reshape(-1)
                pseudo_correct = route.pseudo.labels[0].eq(grid_label).cpu().numpy().astype(bool).reshape(-1)
                raw_l = route.raw_learnability.score[0, 0].cpu().numpy().reshape(-1)
                admission = route.admission_mask[0, 0].cpu().numpy().astype(np.float64).reshape(-1)
                full_prediction = route.full_prediction[0].cpu().numpy()
                interior = _interior_grid(label, grid_shape).reshape(-1)
                small = _small_component_grid(full_prediction, thresholds, grid_shape).reshape(-1)
                total_pixels += current_class.size
                old_class: np.ndarray | None = None
                raw_c: np.ndarray | None = None
                calibrated_c: np.ndarray | None = None
                old_correct: np.ndarray | None = None
                rejection: np.ndarray | None = None
                consolidation_weight: np.ndarray | None = None
                relation_valid: np.ndarray | None = None
                if route.old_relation is not None:
                    old_class = route.old_relation.predicted_class[0].cpu().numpy().reshape(-1)
                    old_correct = route.old_relation.predicted_class[0].eq(grid_label).cpu().numpy().astype(bool).reshape(-1)
                    raw_c = route.raw_compatibility.score[0, 0].cpu().numpy().reshape(-1)
                    calibrated_c = route.consolidation.calibrated_compatibility[0, 0].cpu().numpy().reshape(-1)
                    rejection = route.consolidation.rejection_mask[0, 0].cpu().numpy().astype(bool).reshape(-1)
                    consolidation_weight = route.consolidation.weights[0, 0].cpu().numpy().reshape(-1)
                    relation_valid = route.consolidation.relation_valid_mask[0, 0].cpu().numpy().astype(bool).reshape(-1)
                for class_id in range(method.num_classes):
                    pseudo_mask = pseudo_class == class_id
                    if bool(pseudo_mask.any()):
                        l_data[class_id]["score"].append(raw_l[pseudo_mask])
                        l_data[class_id]["valid"].append(pseudo_valid[pseudo_mask])
                        l_data[class_id]["correct"].append(pseudo_correct[pseudo_mask])
                        l_data[class_id]["weight"].append(admission[pseudo_mask])
                    if old_class is not None and raw_c is not None and calibrated_c is not None and old_correct is not None and consolidation_weight is not None and relation_valid is not None:
                        old_mask = old_class == class_id
                        if bool(old_mask.any()):
                            c_data[class_id]["raw"].append(raw_c[old_mask])
                            c_data[class_id]["calibrated"].append(calibrated_c[old_mask])
                            c_data[class_id]["valid"].append(relation_valid[old_mask])
                            c_data[class_id]["correct"].append(old_correct[old_mask])
                            c_data[class_id]["weight"].append(consolidation_weight[old_mask])
                for region_name, region_mask in (("all", np.ones_like(current_class, dtype=bool)), ("interior", interior), ("boundary", ~interior), ("small_component", small)):
                    for class_id in range(method.num_classes):
                        selected = region_mask & (current_class == class_id)
                        if not bool(selected.any()):
                            continue
                        holder = region_data.setdefault(
                            (region_name, class_id),
                            {"mask": [], "raw_l": [], "pseudo_valid": [], "pseudo_correct": [], "admission": [], "raw_c": [], "calibrated_c": [], "old_correct": [], "rejection": [], "weight": [], "relation_valid": []},
                        )
                        holder["mask"].append(selected)
                        holder["raw_l"].append(raw_l)
                        holder["pseudo_valid"].append(pseudo_valid)
                        holder["pseudo_correct"].append(pseudo_correct)
                        holder["admission"].append(admission)
                        if raw_c is not None and calibrated_c is not None and old_correct is not None and rejection is not None and consolidation_weight is not None and relation_valid is not None:
                            holder["raw_c"].append(raw_c)
                            holder["calibrated_c"].append(calibrated_c)
                            holder["old_correct"].append(old_correct)
                            holder["rejection"].append(rejection)
                            holder["weight"].append(consolidation_weight)
                            holder["relation_valid"].append(relation_valid)
                ess_data["assimilation"]["weight"].append(admission)
                ess_data["assimilation"]["valid"].append(pseudo_valid)
                ess_data["assimilation"]["class"].append(np.where(pseudo_class >= 0, pseudo_class, current_class))
                if old_class is not None and consolidation_weight is not None and relation_valid is not None:
                    ess_data["consolidation"]["weight"].append(consolidation_weight)
                    ess_data["consolidation"]["valid"].append(relation_valid)
                    ess_data["consolidation"]["class"].append(old_class)
    class_rows: list[dict[str, Any]] = []
    for class_id, values in l_data.items():
        if values["score"]:
            class_rows.extend(_bin_rows(metadata=metadata, route="learnability", class_id=class_id, score=np.concatenate(values["score"]), valid=np.concatenate(values["valid"]), correctness=np.concatenate(values["correct"]), applied_weight=np.concatenate(values["weight"]), bins=bins))
    for class_id, values in c_data.items():
        if values["raw"]:
            shared = {"metadata": metadata, "class_id": class_id, "valid": np.concatenate(values["valid"]), "correctness": np.concatenate(values["correct"]), "applied_weight": np.concatenate(values["weight"]), "bins": bins}
            class_rows.extend(_bin_rows(route="raw_compatibility", score=np.concatenate(values["raw"]), **shared))
            class_rows.extend(_bin_rows(route="calibrated_compatibility", score=np.concatenate(values["calibrated"]), **shared))
    region_rows: list[dict[str, Any]] = []
    for (region, class_id), values in sorted(region_data.items()):
        raw_c_values = np.concatenate(values["raw_c"]) if values["raw_c"] else None
        calibrated_c_values = np.concatenate(values["calibrated_c"]) if values["calibrated_c"] else None
        old_correct_values = np.concatenate(values["old_correct"]) if values["old_correct"] else None
        rejection_values = np.concatenate(values["rejection"]) if values["rejection"] else None
        weight_values = np.concatenate(values["weight"]) if values["weight"] else None
        relation_valid_values = np.concatenate(values["relation_valid"]) if values["relation_valid"] else None
        region_rows.append(
            _region_row(
                metadata=metadata,
                region=region,
                class_id=class_id,
                mask=np.concatenate(values["mask"]),
                total_pixels=total_pixels,
                raw_learnability=np.concatenate(values["raw_l"]),
                pseudo_valid=np.concatenate(values["pseudo_valid"]),
                pseudo_correct=np.concatenate(values["pseudo_correct"]),
                admission=np.concatenate(values["admission"]),
                raw_compatibility=raw_c_values,
                calibrated_compatibility=calibrated_c_values,
                old_correct=old_correct_values,
                rejection=rejection_values,
                consolidation_weight=weight_values,
                relation_valid=relation_valid_values,
            )
        )
    ess_rows: list[dict[str, Any]] = []
    for route_name, values in ess_data.items():
        if not values["weight"]:
            continue
        weight = np.concatenate(values["weight"])
        valid = np.concatenate(values["valid"])
        classes = np.concatenate(values["class"])
        ess_rows.append(_ess_row(metadata=metadata, route=route_name, scope="global", class_id=None, weights=weight, valid=valid))
        for class_id in range(method.num_classes):
            selected = classes == class_id
            if bool(selected.any()):
                ess_rows.append(_ess_row(metadata=metadata, route=route_name, scope="class", class_id=class_id, weights=weight[selected], valid=valid[selected]))
    gradient = _gradient_row(root=root, checkpoint=checkpoint, dataset=dataset, site=site, seed=seed, device=device, metadata=metadata)
    summary = {**metadata, "records": len(records), "pixels": total_pixels, "has_historical_model": method.old_model is not None, "component_area_p10": thresholds}
    return class_rows, region_rows, ess_rows, {"summary": summary, "gradient": gradient}


def analyze_v0_2_run(
    *,
    root: Path,
    run_dir: Path,
    output_dir: Path,
    dataset: str = "fundus",
    seed: int = 0,
    device: str | torch.device = "cuda",
    bins: int = 10,
) -> dict[str, Any]:
    """Write all post-hoc routing diagnostics required for one completed V0.2 run."""

    root = Path(root).resolve()
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    frozen_h5 = (root / "h5" / "v1").resolve()
    if output_dir == frozen_h5 or frozen_h5 in output_dir.parents:
        raise ValueError("V0.2 analysis output may not be written into frozen HDF5")
    run_summary = json.loads((run_dir / "run_summary.json").read_text())
    if run_summary.get("status") != "complete" or run_summary.get("method") != "lcrseg_v0_2":
        raise RuntimeError("V0.2 post-hoc analysis requires a complete LCR-Seg V0.2 run")
    if dataset != "fundus" or run_summary.get("dataset") != dataset or int(run_summary.get("seed", -1)) != int(seed):
        raise ValueError("V0.2 analysis input does not match the preregistered Fundus seed-0 scope")
    checkpoints = sorted(run_dir.glob("checkpoint_final_site*_*.pt"))
    if not checkpoints:
        raise FileNotFoundError("no final site checkpoints found for V0.2 analysis")
    device_obj = torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    ess_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        class_part, region_part, ess_part, extra = _analyze_checkpoint(root=root, run_dir=run_dir, checkpoint=checkpoint, dataset=dataset, seed=seed, device=device_obj, bins=bins)
        class_rows.extend(class_part)
        region_rows.extend(region_part)
        ess_rows.extend(ess_part)
        checkpoint_rows.append(extra["summary"])
        gradient_rows.append(extra["gradient"])
    branch_rows, calibration_rows = _branch_rows(run_dir)
    class_fields = ["run_name", "checkpoint", "checkpoint_name", "checkpoint_sha256", "method_name", "site_id", "site_index", "route", "class_id", "bin", "pixel_count", "valid_pixel_count", "valid_coverage", "correctness", "applied_count", "applied_fraction", "weight_mean_over_valid", "effective_pixel_count", "score_mean", "score_p10", "score_p50", "score_p90"]
    region_fields = ["run_name", "checkpoint", "checkpoint_name", "checkpoint_sha256", "method_name", "site_id", "site_index", "region", "class_id", "pixel_count", "pixel_fraction", "pseudo_valid_coverage", "pseudo_label_accuracy", "admission_fraction", "assim_effective_sample_size", "compatibility_available", "old_relation_accuracy", "rejection_fraction", "consolidation_weight_mean", "cons_effective_sample_size", "raw_l_mean", "raw_l_p10", "raw_l_p50", "raw_l_p90", "raw_c_mean", "raw_c_p10", "raw_c_p50", "raw_c_p90", "calibrated_c_mean", "calibrated_c_p10", "calibrated_c_p50", "calibrated_c_p90"]
    ess_fields = ["run_name", "checkpoint", "checkpoint_name", "checkpoint_sha256", "method_name", "site_id", "site_index", "route", "scope", "class_id", "valid_count", "weighted_pixel_count", "effective_pixel_count", "effective_fraction_of_valid", "weight_mean_over_valid"]
    gradient_fields = ["run_name", "checkpoint", "checkpoint_name", "checkpoint_sha256", "method_name", "site_id", "site_index", "golden_labeled_case_ids", "golden_unlabeled_case_ids", "loss_assim", "loss_relation", "gradient_norm_assim", "gradient_norm_relation", "gradient_cosine_assim_relation", "old_model_grad_free"]
    checkpoint_fields = ["run_name", "checkpoint", "checkpoint_name", "checkpoint_sha256", "method_name", "site_id", "site_index", "records", "pixels", "has_historical_model", "component_area_p10"]
    branch_fields = ["site_id", "site_index", "class_id", "calibrator_status", "calibrator_last_update_epoch", "pseudo_candidate_count", "assim_selected_count", "assim_selected_fraction", "relation_valid_count", "compat_rejected_count", "compat_rejected_fraction"]
    calibration_fields = ["source_file", "site_id", "site_index", "epoch", "scope", "class_id", "bin", "upper_edge", "pixel_count", "correct_count", "laplace_accuracy", "pava_probability", "pava_weight"]
    write_csv(output_dir / "classwise_calibration.csv", class_rows, fieldnames=class_fields)
    write_csv(output_dir / "regionwise_calibration.csv", region_rows, fieldnames=region_fields)
    write_csv(output_dir / "effective_sample_size.csv", ess_rows, fieldnames=ess_fields)
    write_csv(output_dir / "gradient_diagnostics.csv", gradient_rows, fieldnames=gradient_fields)
    write_csv(output_dir / "checkpoint_inventory.csv", checkpoint_rows, fieldnames=checkpoint_fields)
    write_csv(output_dir / "branch_coverage.csv", branch_rows, fieldnames=branch_fields)
    write_csv(output_dir / "calibration_tables.csv", calibration_rows, fieldnames=calibration_fields)
    summary = {
        "run_dir": str(run_dir),
        "dataset": dataset,
        "seed": int(seed),
        "output_dir": str(output_dir),
        "checkpoint_count": len(checkpoints),
        "classwise_rows": len(class_rows),
        "regionwise_rows": len(region_rows),
        "ess_rows": len(ess_rows),
        "gradient_rows": len(gradient_rows),
        "branch_rows": len(branch_rows),
        "calibration_rows": len(calibration_rows),
        "hidden_gt_usage": "post_hoc_diagnostics_process_only",
        "frozen_input_root": str(root),
    }
    write_json(output_dir / "routing_analysis_summary.json", summary)
    return summary
