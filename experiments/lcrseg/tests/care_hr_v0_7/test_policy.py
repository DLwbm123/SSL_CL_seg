from dataclasses import replace

import numpy as np
import pytest

from care_hr_v0_7.contracts import Proposal, candidate_grid
from care_hr_v0_7.policy import (FAILURE, accept_regions, apply_anchored_revision,
                                c0_current, c3_shor_whole_case, c4_ppc_whole_case,
                                c5_disagreement_veto, c6_classwise_anchored,
                                c7_confidence_regions, c8_care_hr,
                                candidate_sort_key, select_candidate)


def probabilities():
    current = np.zeros((3, 30, 30), dtype=np.float64); current[0] = 1
    historical = current.copy(); historical[0, :4, :4] = 0; historical[1, :4, :4] = 1
    mask = np.zeros((30, 30), dtype=bool); mask[:4, :4] = True
    return current, historical, Proposal("p", 1, "add", 16, 1.5, 1.5, mask)


def test_outside_region_is_bitwise_identical():
    current, historical, proposal = probabilities()
    revised = apply_anchored_revision(current, historical, [proposal], 0.5)
    assert np.array_equal(revised[:, ~proposal.mask], current[:, ~proposal.mask])


def test_no_accepted_proposal_is_bitwise_current():
    current, historical, _ = probabilities()
    assert np.array_equal(apply_anchored_revision(current, historical, [], 0.5), current)


@pytest.mark.parametrize("value", (0.0, 0.25, 1.0))
def test_primary_blend_lambda_is_only_half_or_three_quarters(value):
    current, historical, proposal = probabilities()
    with pytest.raises(ValueError, match="lambda"):
        apply_anchored_revision(current, historical, [proposal], value)


def test_revised_probability_remains_finite_nonnegative_normalized():
    current, historical, proposal = probabilities()
    revised = apply_anchored_revision(current, historical, [proposal], 0.75)
    assert np.all(np.isfinite(revised)) and np.all(revised >= 0)
    assert np.allclose(revised.sum(axis=0), 1)


def test_acceptance_requires_finite_not_ood_and_consensus():
    _, _, proposal = probabilities()
    accepted = accept_regions([proposal] * 3, [0.1, np.nan, 0.1], [0.0] * 3,
                              [8, 8, 8], [10, 10, 10], [False, False, True],
                              0.0, 0.005, 0.8, 1000, 10000)
    assert len(accepted) == 1


def test_case_and_class_budgets_are_enforced():
    _, _, proposal = probabilities()
    proposals = [replace(proposal, proposal_id=str(i), area=8) for i in range(6)]
    accepted = accept_regions(proposals, [1] * 6, [0] * 6, [200] * 6, [200] * 6,
                              [False] * 6, 0, 0.005, 0.8, 1000, 10000)
    assert len(accepted) == 3


def test_changed_pixel_budgets_are_enforced():
    _, _, proposal = probabilities()
    assert not accept_regions([proposal], [1], [0], [200], [200], [False],
                              0, 0.005, 0.8, 100, 900)


def test_candidate_sort_is_safety_first_then_registered_order():
    base = {"all_inner_safety_gates": True, "catastrophic_current_event": False,
            "shared_gain_p10": 0.2, "historical_gain_p10": 0.3, "current_drop_p90": 0.01,
            "maximum_seed_domain_drop_p90": 0.02, "blend_lambda": 0.5,
            "delta_harm": 0.005, "epsilon_gain": 0.0, "candidate_id": "a"}
    worse = {**base, "shared_gain_p10": 0.1, "candidate_id": "b"}
    assert candidate_sort_key(base) < candidate_sort_key(worse)
    assert select_candidate([worse, base]) == base


def test_no_candidate_does_not_fall_back():
    row = {"all_inner_safety_gates": False, "catastrophic_current_event": False}
    with pytest.raises(RuntimeError, match=FAILURE):
        select_candidate([row])


def test_c8_has_no_whole_case_path_and_returns_regions():
    current, historical, proposal = probabilities(); candidate = candidate_grid()[0]
    revised, accepted = c8_care_hr(current, historical, [proposal], [0.1], [0], [200], [200],
                                   [False], candidate, 1000, 10000)
    assert accepted == (proposal,) and not np.array_equal(revised, current)


def test_all_non_oracle_controls_are_pure_and_available():
    current, historical, proposal = probabilities()
    assert np.array_equal(c0_current(current), current)
    assert np.array_equal(c3_shor_whole_case(current, historical, True), historical)
    assert np.array_equal(c4_ppc_whole_case(current, historical, False), current)
    assert np.array_equal(c5_disagreement_veto(current, historical, 1.0, 0.5), current)
    assert c6_classwise_anchored(current, historical, [proposal]).shape == current.shape
    assert c7_confidence_regions(current, historical, [proposal], [0.9], 0.8).shape == current.shape
