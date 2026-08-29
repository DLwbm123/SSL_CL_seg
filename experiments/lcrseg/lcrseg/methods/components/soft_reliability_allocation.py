"""Detached Soft Reliability Allocation for the preregistered V0.4a method."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ...contracts import differentiable_zero
from .progressive_admission import site_progress
from .pseudo_label import IGNORE_INDEX, PseudoLabelOutput
from .relation_field import RelationOutput


@dataclass(frozen=True)
class SoftReliabilityAllocationOutput:
    """Continuous hard/soft allocation on the frozen relation grid."""

    percentile: torch.Tensor
    alpha: torch.Tensor
    candidate_mask: torch.Tensor
    site_progress: float
    quantile_boundary: float
    target_hard_fraction: float
    candidate_counts: tuple[int, ...]
    fallback_counts: tuple[int, ...]


@dataclass(frozen=True)
class SoftAssimilationLossOutput:
    """SRA objective plus detached diagnostic components."""

    loss: torch.Tensor
    hard_mean: torch.Tensor
    soft_mean: torch.Tensor
    weighted_hard_mean: torch.Tensor
    weighted_soft_mean: torch.Tensor
    valid_count: int


def _empirical_cdf(scores: torch.Tensor) -> torch.Tensor:
    """Return F(x)=P(X<=x) with equal values receiving equal percentiles."""

    if scores.ndim != 1:
        raise ValueError("empirical CDF expects a flat score tensor")
    if not scores.numel():
        return torch.empty_like(scores)
    sorted_scores = torch.sort(scores.detach().float(), stable=True).values
    ranks = torch.searchsorted(sorted_scores, scores.detach().float(), right=True)
    return ranks.to(dtype=scores.dtype) / float(scores.numel())


@torch.no_grad()
def classwise_empirical_cdf(
    raw_learnability: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    num_classes: int,
    minimum_pixels_for_class_cdf: int = 32,
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    """Compute detached class-wise percentiles with the registered global fallback."""

    if raw_learnability.shape != candidate_mask.shape:
        raise ValueError("learnability and candidate mask must share [B,1,H,W] geometry")
    if labels.shape != (raw_learnability.shape[0], *raw_learnability.shape[-2:]):
        raise ValueError("labels must share the learnability relation grid")
    if num_classes < 1 or minimum_pixels_for_class_cdf < 1:
        raise ValueError("class count and CDF minimum must be positive")
    flat_scores = raw_learnability.detach()[:, 0].reshape(-1)
    flat_labels = labels.detach().reshape(-1)
    flat_candidates = candidate_mask.detach().bool()[:, 0].reshape(-1)
    result = torch.zeros_like(flat_scores)
    global_indices = torch.nonzero(flat_candidates, as_tuple=False).flatten()
    global_cdf = _empirical_cdf(flat_scores.index_select(0, global_indices))
    global_lookup = torch.zeros_like(flat_scores)
    if global_indices.numel():
        global_lookup[global_indices] = global_cdf
    candidate_counts: list[int] = []
    fallback_counts: list[int] = []
    for class_id in range(int(num_classes)):
        indices = torch.nonzero(flat_candidates & flat_labels.eq(class_id), as_tuple=False).flatten()
        count = int(indices.numel())
        candidate_counts.append(count)
        fallback = count > 0 and count < int(minimum_pixels_for_class_cdf)
        fallback_counts.append(count if fallback else 0)
        if not count:
            continue
        values = global_lookup.index_select(0, indices) if fallback else _empirical_cdf(flat_scores.index_select(0, indices))
        result[indices] = values
    return result.reshape_as(raw_learnability).detach(), tuple(candidate_counts), tuple(fallback_counts)


@torch.no_grad()
def soft_reliability_allocation(
    pseudo: PseudoLabelOutput,
    raw_learnability: torch.Tensor,
    relation_valid_mask: torch.Tensor,
    *,
    num_classes: int,
    site_step: int,
    total_site_steps: int,
    start_hard_fraction: float = 0.40,
    end_hard_fraction: float = 0.80,
    tau: float = 0.10,
    minimum_pixels_for_class_cdf: int = 32,
) -> SoftReliabilityAllocationOutput:
    """Map class-wise learnability percentiles to a detached smooth hard coefficient."""

    if not 0.0 <= start_hard_fraction <= end_hard_fraction <= 1.0:
        raise ValueError("hard fractions must satisfy 0 <= start <= end <= 1")
    if tau <= 0:
        raise ValueError("SRA tau must be positive")
    if raw_learnability.shape != pseudo.valid.shape or relation_valid_mask.shape != pseudo.valid.shape:
        raise ValueError("SRA inputs must share [B,1,H,W] geometry")
    progress = site_progress(site_step=site_step, total_site_steps=total_site_steps)
    hard_fraction = float(start_hard_fraction) + (float(end_hard_fraction) - float(start_hard_fraction)) * progress
    quantile_boundary = 1.0 - hard_fraction
    candidates = (pseudo.valid.detach().bool() & relation_valid_mask.detach().bool()).detach()
    percentile, counts, fallback_counts = classwise_empirical_cdf(
        raw_learnability,
        pseudo.labels,
        candidates,
        num_classes=num_classes,
        minimum_pixels_for_class_cdf=minimum_pixels_for_class_cdf,
    )
    alpha = torch.sigmoid((percentile - quantile_boundary) / float(tau))
    alpha = torch.where(candidates, alpha, torch.zeros_like(alpha)).detach()
    return SoftReliabilityAllocationOutput(
        percentile=percentile.detach(),
        alpha=alpha,
        candidate_mask=candidates,
        site_progress=progress,
        quantile_boundary=float(quantile_boundary),
        target_hard_fraction=float(hard_fraction),
        candidate_counts=counts,
        fallback_counts=fallback_counts,
    )


def soft_reliability_assimilation_loss(
    strong_logits: torch.Tensor,
    pseudo: PseudoLabelOutput,
    current_relation_weak: RelationOutput,
    current_relation_strong: RelationOutput,
    allocation: SoftReliabilityAllocationOutput,
    strong_valid_mask: torch.Tensor,
    *,
    current_relation_temperature: float = 1.0,
) -> SoftAssimilationLossOutput:
    """Blend hard pseudo-label CE and weak-to-strong current-relation KL per candidate."""

    temperature = float(current_relation_temperature)
    if temperature <= 0:
        raise ValueError("current relation temperature must be positive")
    grid_shape = tuple(current_relation_strong.logits.shape[-2:])
    if current_relation_weak.logits.shape != current_relation_strong.logits.shape:
        raise ValueError("weak/strong current relation geometry mismatch")
    if allocation.alpha.shape != (strong_logits.shape[0], 1, *grid_shape):
        raise ValueError("SRA allocation and current relation geometry mismatch")
    target = pseudo.labels.detach()
    hard_logits = F.interpolate(strong_logits, size=grid_shape, mode="bilinear", align_corners=False)
    hard = F.cross_entropy(hard_logits, target, ignore_index=IGNORE_INDEX, reduction="none").unsqueeze(1)
    weak_probability = F.softmax(current_relation_weak.logits.detach().float() / temperature, dim=1).clamp_min(1.0e-8)
    strong_log_probability = F.log_softmax(current_relation_strong.logits.float() / temperature, dim=1)
    soft = (weak_probability * (weak_probability.log() - strong_log_probability)).sum(dim=1, keepdim=True)
    soft = temperature**2 * soft
    strict_valid = F.adaptive_avg_pool2d(strong_valid_mask.detach().float(), grid_shape).eq(1.0)
    valid = allocation.candidate_mask.detach().bool() & strict_valid
    count = int(valid.sum())
    if not count:
        zero = differentiable_zero(strong_logits) + differentiable_zero(current_relation_strong.logits)
        detached = zero.detach()
        return SoftAssimilationLossOutput(zero, detached, detached, detached, detached, 0)
    alpha = allocation.alpha.detach()
    valid_float = valid.float()
    denominator = valid_float.sum().clamp_min(1.0)
    hard_mean = (hard * valid_float).sum() / denominator
    soft_mean = (soft * valid_float).sum() / denominator
    weighted_hard = (alpha * hard * valid_float).sum() / denominator
    weighted_soft = ((1.0 - alpha) * soft * valid_float).sum() / denominator
    return SoftAssimilationLossOutput(
        loss=weighted_hard + weighted_soft,
        hard_mean=hard_mean,
        soft_mean=soft_mean,
        weighted_hard_mean=weighted_hard,
        weighted_soft_mean=weighted_soft,
        valid_count=count,
    )


def anchor_update_weights(
    pseudo: PseudoLabelOutput,
    current_relation_weak: RelationOutput,
    allocation: SoftReliabilityAllocationOutput,
) -> torch.Tensor:
    """Detached alpha times pseudo/current-relation agreement for anchor writes."""

    agreement = pseudo.labels.detach().eq(current_relation_weak.predicted_class.detach()).unsqueeze(1)
    return (
        allocation.alpha.detach()
        * agreement.float()
        * allocation.candidate_mask.detach().float()
    ).detach()
