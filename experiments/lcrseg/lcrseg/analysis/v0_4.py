"""Shared, post-hoc-only primitives for the preregistered V0.4 failure audit.

This module may load diagnostics labels, but it is never imported by the
training data package, methods, trainer, or continual runner.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from ..engine.checkpoint import load_checkpoint
from ..methods.components.learnability import compute_learnability
from ..methods.components.progressive_admission import ProgressiveAdmissionOutput
from ..methods.components.pseudo_label import PseudoLabelOutput, build_pseudo_labels
from ..methods.lcrseg_v0_1 import LCRSegV01Method
from ..methods.lcrseg_v0_2a import LCRSegV02AMethod
from ..methods.lcrseg_v0_3 import LCRSegV03Method
from ..methods.lcrseg_v0_4a import LCRSegV04AMethod
from ..models import UNet2D


SITE_ORDER = ("REFUGE", "RIM_ONE_r3", "Drishti_GS")
RUN_NAMES = {
    (0, "R0"): "fundus_seed0_lcrseg_uniform_relation_kd_full200e",
    (0, "R1"): "fundus_seed0_lcrseg_v0_2a_r1_progressive_uniform_full200e",
    (1, "R0"): "fundus_seed1_lcrseg_v0_3_r0_legacy_uniform_full200e",
    (1, "R1"): "fundus_seed1_lcrseg_v0_3_r1_progressive_uniform_full200e",
    (2, "R0"): "fundus_seed2_lcrseg_v0_3_r0_legacy_uniform_full200e",
    (2, "R1"): "fundus_seed2_lcrseg_v0_3_r1_progressive_uniform_full200e",
}


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def checkpoint_variant(payload: Mapping[str, Any]) -> str:
    method = dict(dict(payload.get("config_resolved") or {}).get("method") or {})
    variant = str(method.get("variant_id") or "").upper()
    if variant in {"R0", "R1", "SRA"}:
        return variant
    if payload.get("method_name") == "lcrseg_v0_1":
        return "R0"
    raise ValueError("checkpoint does not identify a frozen R0/R1 variant")


def load_frozen_method(
    checkpoint: Path, device: torch.device | str
) -> tuple[LCRSegV01Method, dict[str, Any]]:
    """Load a frozen V0.3-input checkpoint without modifying it."""

    checkpoint = Path(checkpoint).resolve()
    payload = load_checkpoint(checkpoint, map_location="cpu")
    config = dict(payload["config_resolved"])
    model_config = dict(config["model"])
    model = UNet2D(
        int(model_config["in_channels"]),
        int(model_config["num_classes"]),
        base_channels=int(model_config.get("base_channels", 16)),
        relation_dim=int(model_config.get("relation_dim", 128)),
    ).to(device)
    method_config = dict(config.get("method") or {})
    name = str(payload["method_name"])
    if name == "lcrseg_v0_1":
        method: LCRSegV01Method = LCRSegV01Method(model, config=method_config).to(device)
    elif name == "lcrseg_v0_2a":
        method = LCRSegV02AMethod(model, config=method_config).to(device)
    elif name == "lcrseg_v0_3":
        method = LCRSegV03Method(model, config=method_config).to(device)
    elif name == "lcrseg_v0_4a":
        method = LCRSegV04AMethod(model, config=method_config).to(device)
    else:
        raise ValueError(f"unsupported V0.4 audit checkpoint method: {name}")
    method.model.load_state_dict(payload["current_model_state"], strict=True)
    method.load_method_state_dict(payload)
    method.site_id = str(payload["site_id"])
    method.site_index = int(payload["site_index"])
    statistics = dict(payload.get("method_statistics") or {})
    method.total_steps = int(statistics.get("active_site_total_steps") or payload["site_step"])
    method.model.eval()
    if method.old_model is not None:
        method.old_model.eval()
    if not method.current_anchor_bank.all_classes_valid:
        raise RuntimeError("frozen checkpoint has incomplete current anchors")
    return method, payload


def _uniform_admission(pseudo: PseudoLabelOutput, *, num_classes: int) -> ProgressiveAdmissionOutput:
    mask = pseudo.valid.detach().bool()
    counts = tuple(int((mask[:, 0] & pseudo.labels.eq(class_id)).sum()) for class_id in range(num_classes))
    return ProgressiveAdmissionOutput(
        mask=mask,
        candidate_mask=mask,
        site_progress=1.0,
        target_fraction=1.0,
        candidate_counts=counts,
        selected_counts=counts,
        learnability_thresholds=tuple(float("nan") for _ in counts),
    )


@dataclass(frozen=True)
class DiagnosticSnapshot:
    features: torch.Tensor
    logits: torch.Tensor
    pseudo: PseudoLabelOutput
    learnability: torch.Tensor
    admission: ProgressiveAdmissionOutput
    assimilation_weight: torch.Tensor
    relation_probabilities: torch.Tensor
    relation_margin: torch.Tensor
    relation_entropy: torch.Tensor
    logit_margin: torch.Tensor
    spatial_weight: torch.Tensor


@torch.no_grad()
def diagnostic_snapshot(
    method: LCRSegV01Method,
    payload: Mapping[str, Any],
    image: torch.Tensor,
) -> DiagnosticSnapshot:
    """Evaluate one already-processed image at the checkpoint's final step."""

    output = method.model(image)
    relation = method._relation(output.relation_features, method.current_anchor_bank)
    pseudo = build_pseudo_labels(
        output.logits.detach().softmax(dim=1),
        relation,
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
        output.logits.detach(),
        relation,
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
    variant = checkpoint_variant(payload)
    if variant in {"R1", "SRA"} and hasattr(method, "_compute_admission"):
        valid = torch.ones(
            (image.shape[0], 1, *image.shape[-2:]), dtype=torch.bool, device=image.device
        )
        admission = method._compute_admission(  # type: ignore[attr-defined]
            pseudo, learnability, valid, site_step=site_step
        )
    else:
        admission = _uniform_admission(pseudo, num_classes=method.model.num_classes)
    if variant == "SRA":
        allocation = getattr(method, "_sra_allocation", None)
        if allocation is None:
            raise AssertionError("SRA post-hoc allocation was not produced")
        assimilation_weight = allocation.alpha.detach()
    else:
        assimilation_weight = admission.mask.detach().float()
    top2 = output.logits.detach().topk(k=2, dim=1).values
    logit_margin = F.interpolate(
        (top2[:, :1] - top2[:, 1:2]),
        size=relation.probabilities.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    probability = relation.probabilities.detach().float().clamp_min(1.0e-8)
    entropy = -(probability * probability.log()).sum(dim=1, keepdim=True)
    features = F.normalize(output.relation_features.detach().float(), p=2, dim=1, eps=1.0e-8)
    return DiagnosticSnapshot(
        features=features,
        logits=output.logits.detach(),
        pseudo=pseudo,
        learnability=learnability.score.detach(),
        admission=admission,
        assimilation_weight=assimilation_weight,
        relation_probabilities=probability,
        relation_margin=relation.margin.detach(),
        relation_entropy=entropy.detach(),
        logit_margin=logit_margin.detach(),
        spatial_weight=pseudo.spatial_weight.detach(),
    )


def signed_distance_and_component_size(label: np.ndarray, class_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return processed-pixel signed distance and true-component size maps."""

    foreground = np.asarray(label == int(class_id), dtype=bool)
    if not foreground.any():
        return -ndimage.distance_transform_edt(~foreground).astype(np.float32), np.zeros_like(label, dtype=np.int64)
    inside = ndimage.distance_transform_edt(foreground)
    outside = ndimage.distance_transform_edt(~foreground)
    signed = np.where(foreground, inside, -outside).astype(np.float32)
    components, count = ndimage.label(foreground)
    sizes = np.bincount(components.reshape(-1), minlength=count + 1)
    component_size = sizes[components]
    component_size[~foreground] = 0
    return signed, component_size.astype(np.int64)


def resize_numpy(value: np.ndarray, size: tuple[int, int], *, mode: str) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(value))[None, None].float()
    kwargs = {"align_corners": False} if mode in {"bilinear", "bicubic"} else {}
    result = F.interpolate(tensor, size=size, mode=mode, **kwargs)[0, 0]
    return result.cpu().numpy()


def jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("JSD inputs must be same-shaped vectors")
    if first.sum() <= 0 or second.sum() <= 0:
        return float("nan")
    first = first / first.sum()
    second = second / second.sum()
    middle = 0.5 * (first + second)

    def kl(value: np.ndarray, reference: np.ndarray) -> float:
        mask = value > 0
        return float(np.sum(value[mask] * np.log(value[mask] / reference[mask])))

    return 0.5 * (kl(first, middle) + kl(second, middle))


def _kmeans_plus_plus(features: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = [features[int(rng.integers(0, len(features)))]]
    for _ in range(1, k):
        similarity = features @ np.stack(centers).T
        distance = np.maximum(0.0, 1.0 - similarity.max(axis=1))
        total = float(distance.sum())
        index = int(rng.integers(0, len(features))) if total <= 0 else int(rng.choice(len(features), p=distance / total))
        centers.append(features[index])
    return np.stack(centers)


def spherical_kmeans(
    features: np.ndarray,
    *,
    k: int,
    seed: int,
    restarts: int = 5,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministic CPU spherical k-means used only for V0.4 diagnostics."""

    # Float32 matches the frozen projection representation while keeping the
    # largest preregistered (3 sites x 200k pixels) audit groups tractable on
    # CPU.  Determinism is guaranteed by fixed ordering and RNG seeds.
    value = np.asarray(features, dtype=np.float32)
    if value.ndim != 2 or len(value) < k or k < 2:
        raise ValueError("spherical k-means requires [N,D] with N >= K >= 2")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12) or not np.isfinite(value).all():
        raise ValueError("spherical k-means received invalid features")
    value = value / norms
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for restart in range(int(restarts)):
        rng = np.random.default_rng(stable_seed(seed, k, restart))
        centers = _kmeans_plus_plus(value, k, rng)
        labels = np.zeros(len(value), dtype=np.int64)
        for _ in range(int(max_iterations)):
            new_labels = np.argmax(value @ centers.T, axis=1)
            new_centers = []
            similarity = value @ centers.T
            for class_index in range(k):
                selected = value[new_labels == class_index]
                if not len(selected):
                    replacement = value[int(np.argmin(similarity.max(axis=1)))]
                    center = replacement
                else:
                    center = selected.mean(axis=0)
                center = center / max(1.0e-12, float(np.linalg.norm(center)))
                new_centers.append(center)
            new_centers_array = np.stack(new_centers)
            converged = np.array_equal(labels, new_labels) and np.allclose(centers, new_centers_array, atol=1.0e-7)
            labels, centers = new_labels, new_centers_array
            if converged:
                break
        objective = float(np.sum(1.0 - np.sum(value * centers[labels], axis=1), dtype=np.float64))
        candidate = (labels.copy(), centers.copy(), objective)
        if best is None or objective < best[2] - 1.0e-12:
            best = candidate
    if best is None:
        raise AssertionError("spherical k-means produced no restart")
    return best
