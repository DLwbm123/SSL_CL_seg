"""Continuous assimilation and relation-consolidation losses."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ...contracts import differentiable_zero
from ...data.transforms import downsample_valid_mask
from .compatibility import CompatibilityOutput
from .learnability import LearnabilityOutput
from .pseudo_label import IGNORE_INDEX, PseudoLabelOutput
from .relation_field import RelationOutput


def weighted_mean(values: torch.Tensor, weights: torch.Tensor, *, reference: torch.Tensor | None = None) -> torch.Tensor:
    if values.shape != weights.shape:
        raise ValueError(f"weighted mean shape mismatch: {tuple(values.shape)} vs {tuple(weights.shape)}")
    denominator = weights.sum()
    if not bool(denominator.detach().gt(0)):
        return differentiable_zero(reference if reference is not None else values)
    return (values * weights).sum() / denominator.clamp_min(1e-8)


def assimilation_loss(
    strong_logits: torch.Tensor,
    pseudo: PseudoLabelOutput,
    learnability: LearnabilityOutput,
    strong_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """L_i-weighted strong-view CE; deferred/cutout locations never contribute."""

    target = F.interpolate(pseudo.labels.unsqueeze(1).float(), size=strong_logits.shape[-2:], mode="nearest")[:, 0].long().detach()
    pseudo_valid = F.interpolate(pseudo.valid.float(), size=strong_logits.shape[-2:], mode="nearest").bool()
    weight = F.interpolate(learnability.score.detach(), size=strong_logits.shape[-2:], mode="bilinear", align_corners=False)
    valid = pseudo_valid & strong_valid_mask.bool()
    weight = weight * valid.float()
    per_pixel = F.cross_entropy(strong_logits, target, ignore_index=IGNORE_INDEX, reduction="none").unsqueeze(1)
    return weighted_mean(per_pixel, weight, reference=strong_logits)


def relation_consolidation_loss(
    current_relation_strong: RelationOutput,
    old_relation_weak: RelationOutput,
    compatibility: CompatibilityOutput,
    strong_valid_mask: torch.Tensor,
    *,
    distill_temperature: float,
) -> torch.Tensor:
    """C_i-weighted KL(old weak teacher || current strong student)."""

    if distill_temperature <= 0:
        raise ValueError("distill temperature must be positive")
    old_probability = old_relation_weak.probabilities.detach().float().clamp_min(1e-8)
    current_probability = current_relation_strong.probabilities.float().clamp_min(1e-8)
    if old_probability.shape != current_probability.shape:
        raise ValueError("old/current relation geometry mismatch")
    kl = (old_probability * (old_probability.log() - current_probability.log())).sum(dim=1, keepdim=True)
    valid = downsample_valid_mask(strong_valid_mask, current_probability.shape[-2:])
    weights = compatibility.score.detach() * valid.float()
    return float(distill_temperature) ** 2 * weighted_mean(kl, weights, reference=current_relation_strong.logits)
