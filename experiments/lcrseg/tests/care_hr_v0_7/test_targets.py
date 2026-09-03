import inspect

import numpy as np
import pytest

from care_hr_v0_7.contracts import REVIEW_STATUS, ReviewBlocked
from care_hr_v0_7.targets import c9_regional_oracle, proposal_targets


def one_hot(hard):
    return np.eye(3, dtype=np.float64)[hard].transpose(2, 0, 1)


def test_positive_revision_has_positive_gains_and_zero_harm():
    truth = np.ones((4, 4), dtype=int)
    result = proposal_targets(one_hot(np.zeros_like(truth)), one_hot(truth), truth)
    assert result["gain_fg"] > 0 and result["gain_rim"] > 0 and result["harm"] == 0


def test_harm_sign_is_maximum_negative_gain():
    truth = np.ones((4, 4), dtype=int)
    result = proposal_targets(one_hot(truth), one_hot(np.zeros_like(truth)), truth)
    assert result["gain_fg"] < 0 and result["gain_rim"] < 0
    assert result["harm"] == max(0, -result["gain_fg"], -result["gain_rim"], -result["gain_cup"])


def test_cup_gain_sign_is_measured_separately():
    truth = np.full((4, 4), 2, dtype=int)
    assert proposal_targets(one_hot(np.zeros_like(truth)), one_hot(truth), truth)["gain_cup"] > 0


def test_unchanged_revision_has_zero_targets():
    truth = np.zeros((4, 4), dtype=int); probability = one_hot(truth)
    assert set(proposal_targets(probability, probability, truth).values()) == {0.0}


def test_c9_is_always_review_locked_even_with_boolean_override():
    probability = one_hot(np.zeros((2, 2), dtype=int))
    with pytest.raises(ReviewBlocked, match=REVIEW_STATUS):
        c9_regional_oracle(probability, probability, [], np.zeros((2, 2)), evaluator_authorized=True)


def test_ground_truth_parameter_only_occurs_in_evaluator_helper():
    assert "ground_truth" in inspect.signature(proposal_targets).parameters
    assert "ground_truth" in inspect.signature(c9_regional_oracle).parameters
