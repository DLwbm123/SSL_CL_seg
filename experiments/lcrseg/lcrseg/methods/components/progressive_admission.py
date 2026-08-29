"""Class-wise, detached learnability admission for LCR-Seg V0.2."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ...contracts import differentiable_zero
from .pseudo_label import IGNORE_INDEX, PseudoLabelOutput


@dataclass(frozen=True)
class ProgressiveAdmissionOutput:
    """Frozen class-wise top-k selection on the relation grid."""

    mask: torch.Tensor
    candidate_mask: torch.Tensor
    site_progress: float
    target_fraction: float
    candidate_counts: tuple[int, ...]
    selected_counts: tuple[int, ...]
    learnability_thresholds: tuple[float, ...]

    @property
    def selected_fraction_by_class(self) -> tuple[float, ...]:
        return tuple(
            (selected / candidate) if candidate else 0.0
            for selected, candidate in zip(self.selected_counts, self.candidate_counts, strict=True)
        )


def site_progress(*, site_step: int, total_site_steps: int) -> float:
    """Map the zero-based current-site step to the closed interval [0, 1]."""

    if site_step < 0 or total_site_steps < 1:
        raise ValueError("site_step must be nonnegative and total_site_steps positive")
    if total_site_steps == 1:
        return 0.0
    return min(1.0, float(site_step) / float(total_site_steps - 1))


def strict_relation_valid_mask(strong_valid_mask: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    """Keep a relation cell only when every underlying strong-view pixel is valid."""

    if strong_valid_mask.ndim != 4 or strong_valid_mask.shape[1] != 1:
        raise ValueError("strong_valid_mask must be [B,1,H,W]")
    pooled = F.adaptive_avg_pool2d(strong_valid_mask.detach().float(), grid_shape)
    return pooled.eq(1.0).detach()


@torch.no_grad()
def classwise_progressive_admission(
    pseudo: PseudoLabelOutput,
    raw_learnability: torch.Tensor,
    relation_valid_mask: torch.Tensor,
    *,
    num_classes: int,
    site_step: int,
    total_site_steps: int,
    pi_start: float = 0.4,
    pi_end: float = 0.8,
    minimum_pixels_for_class_quantile: int = 1,
    minimum_admitted_per_present_class: int = 1,
) -> ProgressiveAdmissionOutput:
    """Select the top detached L pixels independently within each pseudo class."""

    if not 0.0 <= pi_start <= pi_end <= 1.0:
        raise ValueError("admission fractions must satisfy 0 <= start <= end <= 1")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if minimum_pixels_for_class_quantile < 1 or minimum_admitted_per_present_class < 1:
        raise ValueError("class-quantile and admission minima must be positive")
    if raw_learnability.shape != pseudo.valid.shape or relation_valid_mask.shape != pseudo.valid.shape:
        raise ValueError("admission inputs must share [B,1,H,W] geometry")
    progress = site_progress(site_step=site_step, total_site_steps=total_site_steps)
    fraction = float(pi_start) + (float(pi_end) - float(pi_start)) * progress
    candidates = pseudo.valid.detach().bool() & relation_valid_mask.detach().bool()
    flat_candidates = candidates[:, 0].reshape(-1)
    flat_labels = pseudo.labels.detach().reshape(-1)
    flat_scores = raw_learnability.detach()[:, 0].reshape(-1)
    flat_selected = torch.zeros_like(flat_candidates)
    candidate_counts: list[int] = []
    selected_counts: list[int] = []
    thresholds: list[float] = []
    global_indices = torch.nonzero(flat_candidates, as_tuple=False).flatten()
    global_threshold: torch.Tensor | None = None
    if global_indices.numel():
        global_selected_count = max(1, int(math.ceil(fraction * int(global_indices.numel()))))
        global_order = torch.argsort(
            flat_scores.index_select(0, global_indices),
            descending=True,
            stable=True,
        )
        global_chosen = global_indices.index_select(0, global_order[:global_selected_count])
        global_threshold = flat_scores.index_select(0, global_chosen).min()
    for class_id in range(int(num_classes)):
        indices = torch.nonzero(flat_candidates & flat_labels.eq(class_id), as_tuple=False).flatten()
        candidate_count = int(indices.numel())
        candidate_counts.append(candidate_count)
        if candidate_count == 0:
            selected_counts.append(0)
            thresholds.append(float("nan"))
            continue
        class_scores = flat_scores.index_select(0, indices)
        order = torch.argsort(class_scores, descending=True, stable=True)
        if candidate_count < int(minimum_pixels_for_class_quantile):
            if global_threshold is None:
                raise AssertionError("present class cannot have an empty global candidate set")
            chosen = indices[class_scores.ge(global_threshold)]
            required = min(candidate_count, int(minimum_admitted_per_present_class))
            if int(chosen.numel()) < required:
                chosen = indices.index_select(0, order[:required])
        else:
            selected_count = max(
                int(minimum_admitted_per_present_class),
                int(math.ceil(fraction * candidate_count)),
            )
            selected_count = min(candidate_count, selected_count)
            # stable=True makes ties resolve by the original row-major pixel
            # index, which is deterministic and gradient-independent.
            chosen = indices.index_select(0, order[:selected_count])
        flat_selected[chosen] = True
        selected_counts.append(int(chosen.numel()))
        thresholds.append(float(flat_scores.index_select(0, chosen).min().cpu()))
    mask = flat_selected.reshape_as(pseudo.labels).unsqueeze(1).detach()
    return ProgressiveAdmissionOutput(
        mask=mask,
        candidate_mask=candidates.detach(),
        site_progress=progress,
        target_fraction=fraction,
        candidate_counts=tuple(candidate_counts),
        selected_counts=tuple(selected_counts),
        learnability_thresholds=tuple(thresholds),
    )


def admission_assimilation_loss(
    strong_logits: torch.Tensor,
    pseudo: PseudoLabelOutput,
    admission: ProgressiveAdmissionOutput,
    strong_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Unit-weight CE over exactly the admitted, non-cutout pixels."""

    target = F.interpolate(
        pseudo.labels.detach().unsqueeze(1).float(),
        size=strong_logits.shape[-2:],
        mode="nearest",
    )[:, 0].long()
    selected = F.interpolate(admission.mask.detach().float(), size=strong_logits.shape[-2:], mode="nearest").bool()
    valid = selected & strong_valid_mask.detach().bool()
    if not bool(valid.any()):
        return differentiable_zero(strong_logits)
    pixel_ce = F.cross_entropy(strong_logits, target, ignore_index=IGNORE_INDEX, reduction="none").unsqueeze(1)
    weights = valid.float()
    return (pixel_ce * weights).sum() / weights.sum().clamp_min(1.0)
