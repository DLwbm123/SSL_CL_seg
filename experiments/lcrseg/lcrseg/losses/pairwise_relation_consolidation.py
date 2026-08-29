"""Audit-only BPRC categorical and pairwise relation objectives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F


Mode = Literal[
    "categorical_pixel_mean",
    "categorical_class_balanced",
    "top2_pairwise_class_balanced",
    "all_pairwise_class_balanced",
]


@dataclass(frozen=True)
class PairwiseRelationOutput:
    loss: torch.Tensor
    valid_count: torch.Tensor
    present_class_count: torch.Tensor
    per_class_loss: torch.Tensor
    old_winner_counts: torch.Tensor
    probability_sum_error: torch.Tensor
    pair_count: torch.Tensor


def _pair_probability(winner_score: torch.Tensor, competitor_score: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(winner_score.float() - competitor_score.float())


def _bernoulli_kl(old_probability: torch.Tensor, current_probability: torch.Tensor, eps: float) -> torch.Tensor:
    old = old_probability.detach().float().clamp(eps, 1.0 - eps)
    current = current_probability.float().clamp(eps, 1.0 - eps)
    return old * (old.log() - current.log()) + (1.0 - old) * ((1.0 - old).log() - (1.0 - current).log())


def _class_balanced_reduce(
    per_pixel: torch.Tensor,
    old_winner: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_classes: int,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    losses: list[torch.Tensor] = []
    per_class: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    for class_id in range(num_classes):
        selected = valid & old_winner.eq(class_id)
        count = selected.sum()
        counts.append(count)
        if bool(count.detach().gt(0)):
            value = per_pixel[selected].mean()
            losses.append(value)
            per_class.append(value.detach())
        else:
            per_class.append(torch.full((), float("nan"), device=reference.device))
    loss = torch.stack(losses).mean() if losses else reference.sum() * 0.0
    return loss, torch.stack(per_class), torch.stack(counts), torch.tensor(len(losses), device=reference.device, dtype=torch.int64)


def pairwise_relation_consolidation(
    *,
    old_relation_scores: torch.Tensor,
    current_relation_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: Mode,
    probability_eps: float = 1.0e-6,
) -> PairwiseRelationOutput:
    """Compare frozen old relation competition with current native scores.

    Scores are the existing temperature-scaled cosine logits. This function
    never applies another temperature and never reads labels.
    """

    if old_relation_scores.shape != current_relation_scores.shape or old_relation_scores.ndim != 4:
        raise ValueError("old/current relation scores must share [B,C,H,W] shape")
    if valid_mask.shape != (old_relation_scores.shape[0], 1, *old_relation_scores.shape[-2:]):
        raise ValueError("valid mask must have [B,1,H,W] relation-grid shape")
    if old_relation_scores.shape[1] < 2:
        raise ValueError("pairwise relation consolidation requires at least two classes")
    if not 0.0 < probability_eps < 0.5:
        raise ValueError("probability_eps must lie in (0,0.5)")
    allowed = {
        "categorical_pixel_mean",
        "categorical_class_balanced",
        "top2_pairwise_class_balanced",
        "all_pairwise_class_balanced",
    }
    if mode not in allowed:
        raise ValueError(f"unknown BPRC relation mode: {mode}")
    if not torch.isfinite(old_relation_scores).all() or not torch.isfinite(current_relation_scores).all():
        raise ValueError("relation scores contain non-finite values")

    old_scores = old_relation_scores.detach().float()
    current_scores = current_relation_scores.float()
    valid = valid_mask[:, 0].detach().bool()
    num_classes = old_scores.shape[1]
    old_winner = old_scores.argmax(dim=1).detach()
    old_probability = old_scores.softmax(dim=1).detach()
    current_probability = current_scores.softmax(dim=1)
    probability_sum_error = torch.maximum(
        (old_probability.sum(dim=1) - 1.0).abs().max(),
        (current_probability.sum(dim=1) - 1.0).abs().max(),
    ).detach()

    if mode.startswith("categorical"):
        # B0/B1 retain the frozen R0 categorical KL numerical floor. The
        # preregistered probability_eps applies only to Bernoulli pair math.
        categorical_eps = 1.0e-8
        per_pixel = (
            old_probability.clamp_min(categorical_eps)
            * (
                old_probability.clamp_min(categorical_eps).log()
                - current_probability.clamp_min(categorical_eps).log()
            )
        ).sum(dim=1)
        pair_multiplier = 1
    else:
        winner_index = old_winner.unsqueeze(1)
        old_winner_score = old_scores.gather(1, winner_index)[:, 0]
        current_winner_score = current_scores.gather(1, winner_index)[:, 0]
        competitor_mask = F.one_hot(old_winner, num_classes=num_classes).permute(0, 3, 1, 2).bool()
        if mode == "top2_pairwise_class_balanced":
            old_competitor_scores = old_scores.masked_fill(competitor_mask, -torch.inf)
            competitor = old_competitor_scores.argmax(dim=1, keepdim=True).detach()
            old_competitor = old_scores.gather(1, competitor)[:, 0]
            current_competitor = current_scores.gather(1, competitor)[:, 0]
            per_pixel = _bernoulli_kl(
                _pair_probability(old_winner_score, old_competitor),
                _pair_probability(current_winner_score, current_competitor),
                probability_eps,
            )
            pair_multiplier = 1
        else:
            old_winner_expanded = old_winner_score.unsqueeze(1).expand_as(old_scores)
            current_winner_expanded = current_winner_score.unsqueeze(1).expand_as(current_scores)
            pair_kl = _bernoulli_kl(
                _pair_probability(old_winner_expanded, old_scores),
                _pair_probability(current_winner_expanded, current_scores),
                probability_eps,
            )
            pair_kl = pair_kl.masked_fill(competitor_mask, 0.0)
            per_pixel = pair_kl.sum(dim=1) / float(num_classes - 1)
            pair_multiplier = num_classes - 1

    class_balanced = mode != "categorical_pixel_mean"
    if class_balanced:
        loss, per_class, winner_counts, present_count = _class_balanced_reduce(
            per_pixel, old_winner, valid, num_classes=num_classes, reference=current_scores
        )
    else:
        winner_counts = torch.stack([(valid & old_winner.eq(class_id)).sum() for class_id in range(num_classes)])
        present_count = winner_counts.gt(0).sum()
        per_class_values: list[torch.Tensor] = []
        for class_id in range(num_classes):
            selected = valid & old_winner.eq(class_id)
            per_class_values.append(
                per_pixel[selected].mean().detach()
                if bool(selected.any())
                else torch.full((), float("nan"), device=current_scores.device)
            )
        per_class = torch.stack(per_class_values)
        loss = per_pixel[valid].mean() if bool(valid.any()) else current_scores.sum() * 0.0

    valid_count = valid.sum()
    pair_count = valid_count * int(pair_multiplier)
    if not torch.isfinite(loss):
        raise FloatingPointError("BPRC relation loss is non-finite")
    return PairwiseRelationOutput(
        loss=loss,
        valid_count=valid_count.detach(),
        present_class_count=present_count.detach(),
        per_class_loss=per_class.detach(),
        old_winner_counts=winner_counts.detach(),
        probability_sum_error=probability_sum_error,
        pair_count=pair_count.detach(),
    )
