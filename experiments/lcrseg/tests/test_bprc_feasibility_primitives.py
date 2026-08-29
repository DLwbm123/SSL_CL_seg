from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from lcrseg.losses.pairwise_relation_consolidation import (
    _bernoulli_kl,
    _pair_probability,
    pairwise_relation_consolidation,
)


MODES = (
    "categorical_pixel_mean",
    "categorical_class_balanced",
    "top2_pairwise_class_balanced",
    "all_pairwise_class_balanced",
)


def _scores() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    old = torch.tensor(
        [[[[4.0, 4.0, 1.0, 1.0]], [[1.0, 1.0, 4.0, 1.0]], [[0.0, 0.0, 0.0, 4.0]]]],
        requires_grad=True,
    )
    current = (old.detach() + torch.tensor([[[[0.2]], [[-0.1]], [[0.1]]]])).requires_grad_()
    valid = torch.ones((1, 1, 1, 4), dtype=torch.bool)
    return old, current, valid


@pytest.mark.parametrize("mode", MODES)
def test_bprc_old_scores_stopgrad_current_receives_grad(mode: str) -> None:
    old, current, valid = _scores()
    output = pairwise_relation_consolidation(
        old_relation_scores=old, current_relation_scores=current, valid_mask=valid, mode=mode
    )
    output.loss.backward()
    assert old.grad is None
    assert current.grad is not None and torch.isfinite(current.grad).all()


def test_bprc_pair_probability_matches_two_class_softmax() -> None:
    winner = torch.tensor([2.0, -1.0])
    competitor = torch.tensor([-0.5, 0.5])
    expected = torch.stack((winner, competitor), dim=1).softmax(dim=1)[:, 0]
    assert torch.allclose(_pair_probability(winner, competitor), expected)


def test_bprc_bernoulli_kl_zero_when_equal_and_clamp_finite() -> None:
    probability = torch.tensor([0.0, 0.25, 1.0])
    value = _bernoulli_kl(probability, probability.clone().requires_grad_(), 1.0e-6)
    assert torch.isfinite(value).all()
    assert torch.allclose(value, torch.zeros_like(value), atol=1.0e-7)


def test_bprc_b0_matches_categorical_pixel_mean() -> None:
    old, current, valid = _scores()
    output = pairwise_relation_consolidation(
        old_relation_scores=old, current_relation_scores=current, valid_mask=valid,
        mode="categorical_pixel_mean",
    )
    old_probability = old.detach().softmax(dim=1)
    current_probability = current.softmax(dim=1)
    expected = (old_probability * (old_probability.log() - current_probability.log())).sum(dim=1).mean()
    assert torch.allclose(output.loss, expected, atol=1.0e-7)


def test_bprc_top2_and_all_pair_counts() -> None:
    old, current, valid = _scores()
    top2 = pairwise_relation_consolidation(
        old_relation_scores=old, current_relation_scores=current, valid_mask=valid,
        mode="top2_pairwise_class_balanced",
    )
    all_pair = pairwise_relation_consolidation(
        old_relation_scores=old, current_relation_scores=current, valid_mask=valid,
        mode="all_pairwise_class_balanced",
    )
    assert int(top2.pair_count) == 4
    assert int(all_pair.pair_count) == 8


def test_bprc_old_winner_counts_include_background_equal_class_mass() -> None:
    old, current, valid = _scores()
    output = pairwise_relation_consolidation(
        old_relation_scores=old, current_relation_scores=current, valid_mask=valid,
        mode="all_pairwise_class_balanced",
    )
    assert output.old_winner_counts.tolist() == [2, 1, 1]
    assert int(output.present_class_count) == 3
    assert torch.isfinite(output.per_class_loss).all()


def test_bprc_class_balanced_is_not_pixel_mean() -> None:
    old, current, valid = _scores()
    pixel = pairwise_relation_consolidation(
        old_relation_scores=old, current_relation_scores=current, valid_mask=valid,
        mode="categorical_pixel_mean",
    )
    balanced = pairwise_relation_consolidation(
        old_relation_scores=old, current_relation_scores=current, valid_mask=valid,
        mode="categorical_class_balanced",
    )
    expected = balanced.per_class_loss.mean()
    assert torch.allclose(balanced.loss, expected)
    assert not torch.allclose(pixel.loss, balanced.loss)


def test_bprc_absent_class_and_empty_mask_are_safe() -> None:
    old = torch.tensor([3.0, 1.0, 0.0]).reshape(1, 3, 1, 1)
    current = old.clone().requires_grad_()
    valid = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    output = pairwise_relation_consolidation(
        old_relation_scores=old, current_relation_scores=current, valid_mask=valid,
        mode="all_pairwise_class_balanced",
    )
    assert int(output.present_class_count) == 1
    assert torch.isnan(output.per_class_loss[1:]).all()
    empty = pairwise_relation_consolidation(
        old_relation_scores=old, current_relation_scores=current, valid_mask=torch.zeros_like(valid),
        mode="all_pairwise_class_balanced",
    )
    empty.loss.backward()
    assert float(empty.loss) == 0.0 and int(empty.valid_count) == 0
    assert current.grad is not None


def test_bprc_probability_sum_error_and_no_trainable_state() -> None:
    old, current, valid = _scores()
    output = pairwise_relation_consolidation(
        old_relation_scores=old, current_relation_scores=current, valid_mask=valid,
        mode="top2_pairwise_class_balanced",
    )
    assert float(output.probability_sum_error) <= 1.0e-6
    assert not any(isinstance(value, torch.nn.Parameter) for value in vars(output).values())


def test_bprc_rejects_nonfinite_and_does_not_accept_temperature_or_weights() -> None:
    old, current, valid = _scores()
    with pytest.raises(ValueError):
        pairwise_relation_consolidation(
            old_relation_scores=old * float("nan"), current_relation_scores=current,
            valid_mask=valid, mode="categorical_pixel_mean",
        )
    with pytest.raises(TypeError):
        pairwise_relation_consolidation(
            old_relation_scores=old, current_relation_scores=current,
            valid_mask=valid, mode="categorical_pixel_mean", temperature=0.5,  # type: ignore[call-arg]
        )
