"""Audit-only CRISP channel-role primitives.

No function in this module performs an optimizer step or reads hidden labels.
The tensors are computed from a frozen previous-site model and detached before
they are assembled into a site-static role state.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F


PROTOCOL_ID = "crispseg_v0_1"
FEATURE_LAYERS = ("dec3", "dec1")
ROLE_EPSILON = 1.0e-8
ZERO_EVIDENCE_THRESHOLD = 1.0e-12


def canonical_ids_sha256(values: Sequence[str]) -> str:
    canonical = json.dumps(sorted(map(str, values)), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def stable_half_assignment(patient_case_ids: Sequence[tuple[str, str]]) -> dict[str, str]:
    """Assign complete patient groups to deterministic hash-sorted halves."""

    if not patient_case_ids:
        raise ValueError("cannot split an empty role dataset")
    grouped: dict[str, list[str]] = {}
    for patient_id, case_id in patient_case_ids:
        patient_key = str(patient_id or case_id)
        grouped.setdefault(patient_key, []).append(str(case_id))
    if len(grouped) < 2:
        raise ValueError("split-half audit requires at least two patient/case groups")
    ordered = sorted(
        grouped,
        key=lambda patient: (
            hashlib.sha256(f"{PROTOCOL_ID}\0split-half\0{patient}".encode("utf-8")).hexdigest(),
            patient,
        ),
    )
    midpoint = int(math.ceil(len(ordered) / 2.0))
    first = set(ordered[:midpoint])
    return {
        case_id: ("A" if patient_id in first else "B")
        for patient_id, case_ids in grouped.items()
        for case_id in case_ids
    }


def _aligned_valid_mask(feature: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if feature.ndim != 4:
        raise ValueError(f"feature must be [B,D,H,W], got {tuple(feature.shape)}")
    if valid_mask is None:
        return torch.ones(
            (feature.shape[0], 1, *feature.shape[-2:]), dtype=torch.bool, device=feature.device
        )
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[:, None]
    if valid_mask.ndim != 4 or valid_mask.shape[0] != feature.shape[0] or valid_mask.shape[1] != 1:
        raise ValueError("valid mask must be [B,1,H,W] and batch-aligned")
    return F.interpolate(valid_mask.detach().float(), size=feature.shape[-2:], mode="nearest").bool()


def centered_channel_unit(
    feature: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    epsilon: float = ROLE_EPSILON,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center and spatially L2-normalize each batch/channel map."""

    mask = _aligned_valid_mask(feature, valid_mask)
    float_feature = feature.float()
    float_mask = mask.float()
    counts = float_mask.sum(dim=(-2, -1), keepdim=True)
    sample_valid = counts[:, 0, 0, 0].gt(0)
    means = (float_feature * float_mask).sum(dim=(-2, -1), keepdim=True) / counts.clamp_min(1.0)
    centered = (float_feature - means) * float_mask
    squared_norms = centered.square().sum(dim=(-2, -1), keepdim=True)
    # Keep the exact ||x||_2 + epsilon forward formula while defining the
    # zero-vector subgradient as zero.  A direct sqrt(0) backward produces
    # 0/0 in PyTorch even though the normalized forward value is finite.
    positive = squared_norms.gt(0)
    safe_sqrt = torch.where(positive, squared_norms, torch.ones_like(squared_norms)).sqrt()
    norms = torch.where(positive, safe_sqrt, torch.zeros_like(safe_sqrt))
    return centered / (norms + float(epsilon)), sample_valid


def content_relevance_case(feature: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """Return one case's squared activation-gradient score per channel."""

    if feature.shape != gradient.shape or feature.ndim != 4:
        raise ValueError("feature/gradient must be same-shaped [B,D,H,W] tensors")
    score = (feature.float() * gradient.float()).square().mean(dim=(0, 2, 3))
    if not bool(torch.isfinite(score).all()):
        raise FloatingPointError("non-finite content relevance")
    return score.detach()


def style_sensitivity_case(
    clean_feature: torch.Tensor,
    style_feature: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    epsilon: float = ROLE_EPSILON,
) -> torch.Tensor:
    """Return one case's centered-map squared style distance per channel."""

    if clean_feature.shape != style_feature.shape:
        raise ValueError("clean/style features must be shape-aligned")
    clean_unit, clean_valid = centered_channel_unit(clean_feature, valid_mask, epsilon=epsilon)
    style_unit, style_valid = centered_channel_unit(style_feature, valid_mask, epsilon=epsilon)
    usable = clean_valid & style_valid
    if not bool(usable.any()):
        raise ValueError("style case has no valid spatial support")
    distance = (clean_unit - style_unit).square().sum(dim=(-2, -1))
    score = distance[usable].mean(dim=0)
    if not bool(torch.isfinite(score).all()):
        raise FloatingPointError("non-finite style sensitivity")
    return score.detach()


def case_equal_mean(case_scores: Sequence[torch.Tensor]) -> torch.Tensor:
    if not case_scores:
        raise ValueError("case-equal aggregation requires at least one case")
    shape = case_scores[0].shape
    if any(value.shape != shape for value in case_scores):
        raise ValueError("case scores have inconsistent channel shapes")
    result = torch.stack([value.detach().float() for value in case_scores], dim=0).mean(dim=0)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("non-finite case-equal aggregate")
    return result


def continuous_channel_roles(
    content_scores: torch.Tensor,
    style_scores: torch.Tensor,
    *,
    epsilon: float = ROLE_EPSILON,
    zero_threshold: float = ZERO_EVIDENCE_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the preregistered continuous alpha/beta allocation."""

    if content_scores.shape != style_scores.shape or content_scores.ndim != 1:
        raise ValueError("content/style scores must be same-shaped channel vectors")
    content = content_scores.detach().float()
    style = style_scores.detach().float()
    if bool((content < 0).any()) or bool((style < 0).any()):
        raise ValueError("content/style scores must be non-negative")
    if not bool(torch.isfinite(content).all() and torch.isfinite(style).all()):
        raise FloatingPointError("non-finite channel evidence")
    content_normalized = content / (content.mean() + float(epsilon))
    style_normalized = style / (style.mean() + float(epsilon))
    zero = content.lt(float(zero_threshold)) & style.lt(float(zero_threshold))
    alpha = content_normalized / (content_normalized + style_normalized + float(epsilon))
    alpha = torch.where(zero, torch.full_like(alpha, 0.5), alpha).clamp(0.0, 1.0).detach()
    beta = (1.0 - alpha).detach()
    if float((alpha + beta - 1.0).abs().max()) > 1.0e-7:
        raise AssertionError("alpha/beta complement error exceeds protocol tolerance")
    return alpha, beta, zero.detach()


def hard_rank_roles(alpha: torch.Tensor, *, invariant_fraction: float = 0.60) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha.ndim != 1 or not 0.0 < invariant_fraction < 1.0:
        raise ValueError("invalid hard-rank role input")
    count = int(math.ceil(float(invariant_fraction) * alpha.numel()))
    order = sorted(range(alpha.numel()), key=lambda index: (-float(alpha[index]), index))
    invariant = torch.zeros_like(alpha, dtype=torch.float32)
    invariant[order[:count]] = 1.0
    return invariant.detach(), (1.0 - invariant).detach()


def uniform_half_roles(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha.ndim != 1:
        raise ValueError("alpha must be a channel vector")
    invariant = torch.full_like(alpha, 0.5, dtype=torch.float32)
    return invariant.detach(), invariant.clone().detach()


def effective_sample_size(weights: torch.Tensor, *, epsilon: float = ROLE_EPSILON) -> float:
    value = weights.detach().float()
    return float(value.sum().square() / (value.square().sum() + float(epsilon)))


def quartile_indices(values: torch.Tensor, *, largest: bool) -> tuple[int, ...]:
    if values.ndim != 1:
        raise ValueError("quartiles require a channel vector")
    count = max(1, int(math.ceil(values.numel() * 0.25)))
    order = sorted(
        range(values.numel()),
        key=(lambda index: (-float(values[index]), index)) if largest else (lambda index: (float(values[index]), index)),
    )
    return tuple(order[:count])


def spearman_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape or first.ndim != 1 or first.numel() < 2:
        raise ValueError("Spearman inputs must be same-shaped channel vectors")

    def ranks(value: torch.Tensor) -> torch.Tensor:
        entries = sorted((float(item), index) for index, item in enumerate(value.detach().cpu()))
        result = torch.empty(len(entries), dtype=torch.float64)
        start = 0
        while start < len(entries):
            end = start + 1
            while end < len(entries) and entries[end][0] == entries[start][0]:
                end += 1
            average_rank = 0.5 * (start + end - 1)
            for _, index in entries[start:end]:
                result[index] = average_rank
            start = end
        return result

    left, right = ranks(first), ranks(second)
    left, right = left - left.mean(), right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator) if float(denominator) > 0 else 0.0


def jaccard(first: Sequence[int], second: Sequence[int]) -> float:
    left, right = set(map(int, first)), set(map(int, second))
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@dataclass(frozen=True)
class ChannelRoleState:
    protocol_id: str
    site_id: str
    source_checkpoint_sha256: str
    labeled_case_ids_sha256: str
    unlabeled_case_ids_sha256: str
    style_probe_sha256: str
    content_scores: dict[str, torch.Tensor]
    style_scores: dict[str, torch.Tensor]
    invariant_weights: dict[str, torch.Tensor]
    plastic_weights: dict[str, torch.Tensor]
    zero_evidence_masks: dict[str, torch.Tensor]
    feature_shapes: dict[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if self.protocol_id != PROTOCOL_ID:
            raise ValueError(f"unexpected channel-role protocol: {self.protocol_id}")
        mappings: tuple[Mapping[str, Any], ...] = (
            self.content_scores,
            self.style_scores,
            self.invariant_weights,
            self.plastic_weights,
            self.zero_evidence_masks,
            self.feature_shapes,
        )
        if any(tuple(mapping) != FEATURE_LAYERS for mapping in mappings):
            raise ValueError(f"channel-role state must contain layers in order {FEATURE_LAYERS}")
        for layer in FEATURE_LAYERS:
            tensors = (
                self.content_scores[layer],
                self.style_scores[layer],
                self.invariant_weights[layer],
                self.plastic_weights[layer],
                self.zero_evidence_masks[layer],
            )
            if any(value.ndim != 1 for value in tensors) or len({value.numel() for value in tensors}) != 1:
                raise ValueError(f"invalid role tensor shapes for {layer}")
            if self.feature_shapes[layer][1] != tensors[0].numel():
                raise ValueError(f"feature/channel mismatch for {layer}")
            if not bool(torch.isfinite(torch.cat([value.detach().float() for value in tensors])).all()):
                raise FloatingPointError(f"non-finite role state for {layer}")
            if float((self.invariant_weights[layer] + self.plastic_weights[layer] - 1.0).abs().max()) > 1.0e-7:
                raise ValueError(f"non-complementary roles for {layer}")

    def state_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "site_id": self.site_id,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "labeled_case_ids_sha256": self.labeled_case_ids_sha256,
            "unlabeled_case_ids_sha256": self.unlabeled_case_ids_sha256,
            "style_probe_sha256": self.style_probe_sha256,
            "content_scores": {key: value.detach().cpu().clone() for key, value in self.content_scores.items()},
            "style_scores": {key: value.detach().cpu().clone() for key, value in self.style_scores.items()},
            "invariant_weights": {key: value.detach().cpu().clone() for key, value in self.invariant_weights.items()},
            "plastic_weights": {key: value.detach().cpu().clone() for key, value in self.plastic_weights.items()},
            "zero_evidence_masks": {key: value.detach().cpu().clone() for key, value in self.zero_evidence_masks.items()},
            "feature_shapes": {key: tuple(value) for key, value in self.feature_shapes.items()},
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any], *, device: torch.device | str = "cpu") -> "ChannelRoleState":
        tensor_fields = ("content_scores", "style_scores", "invariant_weights", "plastic_weights", "zero_evidence_masks")
        values = {
            field: {layer: tensor.detach().clone().to(device) for layer, tensor in state[field].items()}
            for field in tensor_fields
        }
        return cls(
            protocol_id=str(state["protocol_id"]),
            site_id=str(state["site_id"]),
            source_checkpoint_sha256=str(state["source_checkpoint_sha256"]),
            labeled_case_ids_sha256=str(state["labeled_case_ids_sha256"]),
            unlabeled_case_ids_sha256=str(state["unlabeled_case_ids_sha256"]),
            style_probe_sha256=str(state["style_probe_sha256"]),
            feature_shapes={layer: tuple(shape) for layer, shape in state["feature_shapes"].items()},
            **values,
        )


def build_channel_role_state(
    *,
    site_id: str,
    source_checkpoint_sha256: str,
    labeled_case_ids: Sequence[str],
    unlabeled_case_ids: Sequence[str],
    style_probe_sha256: str,
    content_scores: Mapping[str, torch.Tensor],
    style_scores: Mapping[str, torch.Tensor],
    feature_shapes: Mapping[str, tuple[int, ...]],
) -> ChannelRoleState:
    invariant: dict[str, torch.Tensor] = {}
    plastic: dict[str, torch.Tensor] = {}
    zero: dict[str, torch.Tensor] = {}
    for layer in FEATURE_LAYERS:
        invariant[layer], plastic[layer], zero[layer] = continuous_channel_roles(
            content_scores[layer], style_scores[layer]
        )
    return ChannelRoleState(
        protocol_id=PROTOCOL_ID,
        site_id=str(site_id),
        source_checkpoint_sha256=str(source_checkpoint_sha256),
        labeled_case_ids_sha256=canonical_ids_sha256(labeled_case_ids),
        unlabeled_case_ids_sha256=canonical_ids_sha256(unlabeled_case_ids),
        style_probe_sha256=str(style_probe_sha256),
        content_scores={layer: content_scores[layer].detach().float().clone() for layer in FEATURE_LAYERS},
        style_scores={layer: style_scores[layer].detach().float().clone() for layer in FEATURE_LAYERS},
        invariant_weights=invariant,
        plastic_weights=plastic,
        zero_evidence_masks=zero,
        feature_shapes={layer: tuple(feature_shapes[layer]) for layer in FEATURE_LAYERS},
    )


__all__ = [
    "ChannelRoleState",
    "FEATURE_LAYERS",
    "PROTOCOL_ID",
    "ROLE_EPSILON",
    "ZERO_EVIDENCE_THRESHOLD",
    "build_channel_role_state",
    "canonical_ids_sha256",
    "case_equal_mean",
    "centered_channel_unit",
    "content_relevance_case",
    "continuous_channel_roles",
    "effective_sample_size",
    "hard_rank_roles",
    "jaccard",
    "quartile_indices",
    "spearman_correlation",
    "stable_half_assignment",
    "style_sensitivity_case",
    "uniform_half_roles",
]
