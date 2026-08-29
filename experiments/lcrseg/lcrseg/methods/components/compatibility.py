"""Historical compatibility maps for V2 of LCR-Seg V0.1."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .pseudo_label import spatial_weight
from .relation_field import RelationOutput


@dataclass(frozen=True)
class CompatibilityOutput:
    score: torch.Tensor
    js_divergence: torch.Tensor
    old_margin_weight: torch.Tensor
    agreement: torch.Tensor
    spatial_weight: torch.Tensor


@torch.no_grad()
def zero_compatibility(reference: torch.Tensor) -> CompatibilityOutput:
    zeros = torch.zeros((reference.shape[0], 1, *reference.shape[-2:]), device=reference.device, dtype=reference.dtype)
    return CompatibilityOutput(score=zeros, js_divergence=zeros, old_margin_weight=zeros, agreement=zeros, spatial_weight=zeros)


@torch.no_grad()
def compute_compatibility(
    current_relation: RelationOutput,
    old_relation: RelationOutput,
    *,
    old_margin_center: float,
    old_margin_temperature: float,
    js_temperature: float,
    spatial_floor: float,
    eps: float = 1e-8,
) -> CompatibilityOutput:
    """Score only historically clear and semantically compatible relations."""

    if old_margin_temperature <= 0 or js_temperature <= 0:
        raise ValueError("compatibility temperatures must be positive")
    current = current_relation.probabilities.detach().float()
    old = old_relation.probabilities.detach().float()
    if current.shape != old.shape:
        raise ValueError("current and old relation distributions must have matching shape")
    current = current.clamp_min(eps)
    old = old.clamp_min(eps)
    midpoint = 0.5 * (current + old)
    js = 0.5 * (current * (current.log() - midpoint.log())).sum(dim=1, keepdim=True)
    js += 0.5 * (old * (old.log() - midpoint.log())).sum(dim=1, keepdim=True)
    old_margin_weight = torch.sigmoid(
        (old_relation.margin.detach().float() - float(old_margin_center)) / float(old_margin_temperature)
    )
    js_weight = torch.exp(-js / float(js_temperature))
    agreement = current_relation.predicted_class.detach().eq(old_relation.predicted_class.detach()).unsqueeze(1).float()
    all_valid = torch.ones_like(agreement, dtype=torch.bool)
    old_spatial_weight, _ = spatial_weight(
        old_relation.predicted_class.detach(),
        all_valid,
        num_classes=current.shape[1],
        floor=spatial_floor,
    )
    score = (old_margin_weight * js_weight * agreement * old_spatial_weight).clamp(0.0, 1.0).detach()
    return CompatibilityOutput(
        score=score,
        js_divergence=js.detach(),
        old_margin_weight=old_margin_weight.detach(),
        agreement=agreement.detach(),
        spatial_weight=old_spatial_weight.detach(),
    )
