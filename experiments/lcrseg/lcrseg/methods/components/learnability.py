"""Current learnability maps for V1 of LCR-Seg V0.1."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .pseudo_label import PseudoLabelOutput
from .relation_field import RelationOutput


@dataclass(frozen=True)
class LearnabilityOutput:
    score: torch.Tensor
    robust_progress_index: torch.Tensor
    percentile_rank: torch.Tensor
    progress_weight: torch.Tensor
    relation_weight: torch.Tensor
    spatial_weight: torch.Tensor
    source_weight: torch.Tensor


def _percentile_rank(values: torch.Tensor, predicted_class: torch.Tensor, *, min_rank_pixels: int) -> torch.Tensor:
    """Deterministic class-conditional ranks with a global small-class fallback."""

    if values.ndim != 3 or predicted_class.shape != values.shape:
        raise ValueError("rank inputs must be [B,H,W]")
    if min_rank_pixels < 1:
        raise ValueError("min_rank_pixels must be positive")
    flat_values = values.detach().reshape(-1)
    flat_class = predicted_class.detach().reshape(-1)
    ranks = torch.zeros_like(flat_values)

    def assign(indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return torch.empty_like(indices, dtype=flat_values.dtype)
        order = torch.argsort(flat_values.index_select(0, indices), stable=True)
        result = torch.empty(indices.numel(), dtype=flat_values.dtype, device=flat_values.device)
        if indices.numel() == 1:
            result[order] = 1.0
        else:
            result[order] = torch.arange(indices.numel(), device=flat_values.device, dtype=flat_values.dtype) / float(indices.numel() - 1)
        return result

    global_indices = torch.arange(flat_values.numel(), device=flat_values.device)
    global_rank = assign(global_indices)
    unique_classes = torch.unique(flat_class).tolist()
    for class_index in unique_classes:
        indices = torch.nonzero(flat_class.eq(int(class_index)), as_tuple=False).flatten()
        if indices.numel() < min_rank_pixels:
            ranks.index_copy_(0, indices, global_rank.index_select(0, indices))
        else:
            ranks.index_copy_(0, indices, assign(indices))
    return ranks.reshape_as(values)


@torch.no_grad()
def compute_learnability(
    weak_segmentation_logits: torch.Tensor,
    current_relation: RelationOutput,
    pseudo: PseudoLabelOutput,
    *,
    site_step: int,
    total_steps: int,
    rank_start: float,
    rank_end: float,
    rank_temperature: float,
    relation_margin_center: float,
    relation_margin_temperature: float,
    min_rank_pixels: int,
    eps: float = 1e-8,
) -> LearnabilityOutput:
    """Compute the detached product defining the current learnability score."""

    if total_steps < 1 or site_step < 0:
        raise ValueError("site step and total steps are invalid")
    if rank_temperature <= 0 or relation_margin_temperature <= 0:
        raise ValueError("learnability temperatures must be positive")
    resized_logits = F.interpolate(
        weak_segmentation_logits.detach().float(),
        size=current_relation.probabilities.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    probabilities = resized_logits.softmax(dim=1)
    values, predicted = resized_logits.topk(k=min(2, resized_logits.shape[1]), dim=1)
    p_values, _ = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1)
    margin = values[:, 0] - (values[:, 1] if values.shape[1] > 1 else torch.zeros_like(values[:, 0]))
    top1 = p_values[:, 0]
    top2 = p_values[:, 1] if p_values.shape[1] > 1 else torch.zeros_like(top1)
    kappa = top1 + top2 - (top1 - top2).square()
    robust = margin.abs() / kappa.clamp_min(eps)
    rank = _percentile_rank(robust, predicted[:, 0], min_rank_pixels=min_rank_pixels)
    progress = min(1.0, float(site_step + 1) / float(total_steps))
    threshold = (1.0 - progress) * float(rank_start) + progress * float(rank_end)
    progress_weight = torch.sigmoid((rank - threshold) / float(rank_temperature)).unsqueeze(1)
    relation_weight = torch.sigmoid(
        (current_relation.margin.detach() - float(relation_margin_center)) / float(relation_margin_temperature)
    )
    score = pseudo.valid.float() * progress_weight * relation_weight * pseudo.spatial_weight.detach() * pseudo.source_weight.detach()
    score = score.clamp(0.0, 1.0).detach()
    return LearnabilityOutput(
        score=score,
        robust_progress_index=robust.unsqueeze(1).detach(),
        percentile_rank=rank.unsqueeze(1).detach(),
        progress_weight=progress_weight.detach(),
        relation_weight=relation_weight.detach(),
        spatial_weight=pseudo.spatial_weight.detach(),
        source_weight=pseudo.source_weight.detach(),
    )
