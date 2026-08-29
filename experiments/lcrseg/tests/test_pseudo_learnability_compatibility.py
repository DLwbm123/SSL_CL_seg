from __future__ import annotations

import torch

from lcrseg.methods.components.compatibility import compute_compatibility, zero_compatibility
from lcrseg.methods.components.learnability import compute_learnability
from lcrseg.methods.components.pseudo_label import IGNORE_INDEX, PseudoLabelOutput, build_pseudo_labels
from lcrseg.methods.components.relation_field import RelationOutput


def _relation(probabilities: torch.Tensor) -> RelationOutput:
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    top, indexes = probabilities.topk(k=2, dim=1)
    return RelationOutput(
        logits=probabilities.log(),
        probabilities=probabilities,
        predicted_class=indexes[:, 0],
        top1=top[:, :1],
        top2=top[:, 1:2],
        margin=top[:, :1] - top[:, 1:2],
        valid_class_mask=torch.ones(probabilities.shape[1], dtype=torch.bool),
    )


def _pseudo(labels: torch.Tensor, *, valid: bool = True) -> PseudoLabelOutput:
    mask = torch.full((labels.shape[0], 1, *labels.shape[-2:]), valid, dtype=torch.bool)
    return PseudoLabelOutput(
        labels=labels,
        valid=mask,
        source=torch.ones_like(labels),
        source_weight=mask.float(),
        spatial_weight=mask.float(),
        spatial_agreement=mask.float(),
    )


def test_classifier_easy_anchor_recoverable_and_deferred_branches() -> None:
    seg_classifier = torch.zeros((1, 3, 3, 3))
    seg_classifier[:, 1] = 6.0
    relation_classifier = _relation(torch.tensor([[[[0.05] * 3] * 3, [[0.90] * 3] * 3, [[0.05] * 3] * 3]]))
    output = build_pseudo_labels(
        seg_classifier.softmax(1), relation_classifier,
        tau_cls=0.8, tau_anchor=0.8, delta_anchor=0.1, tau_spatial=0.5,
        temperature_cls=0.05, temperature_anchor=0.05, spatial_floor=0.25,
    )
    assert bool(output.valid.all())
    assert torch.all(output.source.eq(1))
    assert torch.all(output.labels.eq(1))

    seg_anchor = torch.zeros((1, 3, 3, 3))
    relation_anchor = _relation(torch.tensor([[[[0.05] * 3] * 3, [[0.90] * 3] * 3, [[0.05] * 3] * 3]]))
    output = build_pseudo_labels(
        seg_anchor.softmax(1), relation_anchor,
        tau_cls=0.95, tau_anchor=0.8, delta_anchor=0.1, tau_spatial=0.5,
        temperature_cls=0.05, temperature_anchor=0.05, spatial_floor=0.25,
    )
    assert bool(output.valid.all())
    assert torch.all(output.source.eq(2))
    assert torch.all(output.labels.eq(1))

    relation_deferred = _relation(torch.full((1, 3, 3, 3), 1.0 / 3.0))
    output = build_pseudo_labels(
        seg_anchor.softmax(1), relation_deferred,
        tau_cls=0.95, tau_anchor=0.8, delta_anchor=0.1, tau_spatial=0.5,
        temperature_cls=0.05, temperature_anchor=0.05, spatial_floor=0.25,
    )
    assert not bool(output.valid.any())
    assert torch.all(output.source.eq(0))
    assert torch.all(output.labels.eq(IGNORE_INDEX))


def test_progressive_learnability_rank_fallback_and_detach() -> None:
    logits = torch.tensor([[[[0.0, 0.1, 0.2, 0.3]], [[0.2, 0.4, 0.6, 2.0]], [[0.0, 0.0, 0.0, 0.0]]]], requires_grad=True)
    relation = _relation(torch.tensor([[[[0.05, 0.05, 0.05, 0.05]], [[0.90, 0.90, 0.90, 0.90]], [[0.05, 0.05, 0.05, 0.05]]]]))
    pseudo = _pseudo(torch.ones((1, 1, 4), dtype=torch.long))
    early = compute_learnability(
        logits, relation, pseudo, site_step=4, total_steps=100,
        rank_start=0.8, rank_end=0.2, rank_temperature=0.1,
        relation_margin_center=0.1, relation_margin_temperature=0.05, min_rank_pixels=128,
    )
    late = compute_learnability(
        logits, relation, pseudo, site_step=94, total_steps=100,
        rank_start=0.8, rank_end=0.2, rank_temperature=0.1,
        relation_margin_center=0.1, relation_margin_temperature=0.05, min_rank_pixels=128,
    )
    assert not early.score.requires_grad
    assert torch.all(early.score.ge(0) & early.score.le(1))
    low_rank = early.percentile_rank.argmin()
    assert late.score.reshape(-1)[low_rank] > early.score.reshape(-1)[low_rank]
    assert torch.equal(early.percentile_rank, compute_learnability(
        logits, relation, pseudo, site_step=4, total_steps=100,
        rank_start=0.8, rank_end=0.2, rank_temperature=0.1,
        relation_margin_center=0.1, relation_margin_temperature=0.05, min_rank_pixels=128,
    ).percentile_rank)


def test_compatibility_identical_conflict_margin_monotonic_and_detach() -> None:
    old = _relation(torch.tensor([[[[0.05]], [[0.90]], [[0.05]]]]))
    same = compute_compatibility(old, old, old_margin_center=0.1, old_margin_temperature=0.05, js_temperature=0.2, spatial_floor=0.25)
    assert not same.score.requires_grad
    assert same.js_divergence.abs().max() < 1e-6
    assert float(same.score) > 0.9
    conflict_relation = _relation(torch.tensor([[[[0.90]], [[0.05]], [[0.05]]]]))
    conflict = compute_compatibility(conflict_relation, old, old_margin_center=0.1, old_margin_temperature=0.05, js_temperature=0.2, spatial_floor=0.25)
    assert float(conflict.score) == 0.0
    low_margin_old = _relation(torch.tensor([[[[0.30]], [[0.40]], [[0.30]]]]))
    low_margin = compute_compatibility(low_margin_old, low_margin_old, old_margin_center=0.1, old_margin_temperature=0.05, js_temperature=0.2, spatial_floor=0.25)
    assert float(low_margin.score) < float(same.score)
    slightly_shifted = _relation(torch.tensor([[[[0.10]], [[0.85]], [[0.05]]]]))
    strongly_shifted = _relation(torch.tensor([[[[0.40]], [[0.55]], [[0.05]]]]))
    slight = compute_compatibility(slightly_shifted, old, old_margin_center=0.1, old_margin_temperature=0.05, js_temperature=0.2, spatial_floor=0.25)
    strong = compute_compatibility(strongly_shifted, old, old_margin_center=0.1, old_margin_temperature=0.05, js_temperature=0.2, spatial_floor=0.25)
    assert float(strong.js_divergence) > float(slight.js_divergence)
    assert float(strong.score) < float(slight.score)
    zeros = zero_compatibility(torch.zeros((2, 3, 4, 5)))
    assert zeros.score.shape == (2, 1, 4, 5)
    assert not bool(zeros.score.any())
