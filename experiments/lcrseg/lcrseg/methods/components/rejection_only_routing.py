"""Conservative compatibility rejection for LCR-Seg V0.2 consolidation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ...contracts import differentiable_zero
from ...data.transforms import downsample_valid_mask
from .relation_field import RelationOutput


@dataclass(frozen=True)
class RejectionOnlyOutput:
    calibrated_compatibility: torch.Tensor
    rejection_mask: torch.Tensor
    weights: torch.Tensor
    relation_valid_mask: torch.Tensor
    candidate_counts: tuple[int, ...]
    rejected_counts: tuple[int, ...]
    calibrator_available: bool

    @property
    def rejected_fraction_by_class(self) -> tuple[float, ...]:
        return tuple(
            (rejected / candidate) if candidate else 0.0
            for rejected, candidate in zip(self.rejected_counts, self.candidate_counts, strict=True)
        )


@torch.no_grad()
def rejection_only_weights(
    calibrated_compatibility: torch.Tensor,
    old_predicted_class: torch.Tensor,
    relation_valid_mask: torch.Tensor,
    *,
    num_classes: int,
    calibrator_available: bool,
    probability_threshold: float = 0.7,
    max_reject_fraction_per_class: float = 0.2,
    rejected_weight_floor: float = 0.5,
) -> RejectionOnlyOutput:
    """Return unit weights except for the capped lowest calibrated-C pixels."""

    if calibrated_compatibility.shape != relation_valid_mask.shape:
        raise ValueError("compatibility and relation-valid masks must share geometry")
    if old_predicted_class.shape != calibrated_compatibility[:, 0].shape:
        raise ValueError("old predicted class must be [B,H,W]")
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("compatibility probability threshold must be in [0, 1]")
    if not 0.0 <= max_reject_fraction_per_class <= 1.0:
        raise ValueError("rejection cap must be in [0, 1]")
    if not 0.0 < rejected_weight_floor <= 1.0:
        raise ValueError("rejected weight floor must be in (0, 1]")
    score = calibrated_compatibility.detach().float().clamp(0.0, 1.0)
    valid = relation_valid_mask.detach().bool()
    weights = torch.ones_like(score)
    rejected = torch.zeros_like(valid)
    candidate_counts: list[int] = []
    rejected_counts: list[int] = []
    if not calibrator_available:
        flat_valid = valid[:, 0].reshape(-1)
        flat_class = old_predicted_class.detach().reshape(-1)
        return RejectionOnlyOutput(
            calibrated_compatibility=score,
            rejection_mask=rejected.detach(),
            weights=weights.detach(),
            relation_valid_mask=valid.detach(),
            candidate_counts=tuple(
                int((flat_valid & flat_class.eq(class_id)).sum())
                for class_id in range(num_classes)
            ),
            rejected_counts=tuple(0 for _ in range(num_classes)),
            calibrator_available=False,
        )
    flat_score = score[:, 0].reshape(-1)
    flat_valid = valid[:, 0].reshape(-1)
    flat_class = old_predicted_class.detach().reshape(-1)
    flat_rejected = rejected[:, 0].reshape(-1)
    for class_id in range(int(num_classes)):
        class_indices = torch.nonzero(flat_valid & flat_class.eq(class_id), as_tuple=False).flatten()
        class_count = int(class_indices.numel())
        candidate_counts.append(class_count)
        cap = int(math.floor(float(max_reject_fraction_per_class) * class_count))
        if class_count == 0 or cap == 0:
            rejected_counts.append(0)
            continue
        candidate_indices = class_indices[flat_score.index_select(0, class_indices).lt(float(probability_threshold))]
        if candidate_indices.numel() == 0:
            rejected_counts.append(0)
            continue
        order = torch.argsort(flat_score.index_select(0, candidate_indices), descending=False, stable=True)
        chosen = candidate_indices.index_select(0, order[:cap])
        flat_rejected[chosen] = True
        rejected_counts.append(int(chosen.numel()))
    rejected = flat_rejected.reshape_as(old_predicted_class).unsqueeze(1)
    weights = torch.where(rejected, torch.full_like(weights, float(rejected_weight_floor)), weights)
    return RejectionOnlyOutput(
        calibrated_compatibility=score.detach(),
        rejection_mask=rejected.detach(),
        weights=weights.detach(),
        relation_valid_mask=valid.detach(),
        candidate_counts=tuple(candidate_counts),
        rejected_counts=tuple(rejected_counts),
        calibrator_available=True,
    )


def rejection_only_relation_loss(
    current_relation_strong: RelationOutput,
    old_relation_weak: RelationOutput,
    routing: RejectionOnlyOutput,
    strong_valid_mask: torch.Tensor,
    *,
    distill_temperature: float,
) -> torch.Tensor:
    """KL(old weak relation || current strong relation) with floor-preserving weights."""

    if distill_temperature <= 0:
        raise ValueError("distill temperature must be positive")
    old_probability = old_relation_weak.probabilities.detach().float().clamp_min(1.0e-8)
    current_probability = current_relation_strong.probabilities.float().clamp_min(1.0e-8)
    if old_probability.shape != current_probability.shape:
        raise ValueError("old/current relation geometry mismatch")
    valid = downsample_valid_mask(strong_valid_mask, current_probability.shape[-2:]).bool()
    weights = routing.weights.detach() * routing.relation_valid_mask.detach().float() * valid.float()
    if not bool(weights.sum().detach().gt(0)):
        return differentiable_zero(current_relation_strong.logits)
    kl = (old_probability * (old_probability.log() - current_probability.log())).sum(dim=1, keepdim=True)
    return float(distill_temperature) ** 2 * (kl * weights).sum() / weights.sum().clamp_min(1.0e-8)
