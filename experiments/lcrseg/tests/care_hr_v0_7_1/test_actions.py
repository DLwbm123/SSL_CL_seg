from dataclasses import replace

import numpy as np
import pytest

from care_hr_v0_7.contracts import Proposal
from care_hr_v0_7_1.actions import (apply_action, enumerate_actions, probability_pair,
                                    proposals_for_route, validate_proposals, validate_vote_counts)


def regions(count, shape=(80, 80), area=8, classes=(1, 2)):
    result = []
    for i in range(count):
        mask = np.zeros(shape, dtype=bool)
        mask[i, :area] = True
        rr, cc = np.nonzero(mask)
        result.append(Proposal(f"p{i:02d}", classes[i % len(classes)], "add", area,
                               float(rr.mean()), float(cc.mean()), mask))
    return tuple(result)


def test_nested_sets_fixed_lambda_and_count_caps():
    proposals = regions(12)
    current = np.ones((80, 80), dtype=np.int64)
    cap = set(enumerate_actions(proposals, current, "O_CAP", 0.5))
    no_area = set(enumerate_actions(proposals, current, "O_NO_AREA", 0.5))
    free = set(enumerate_actions(proposals, current, "O_FREE_SUBSET", 0.5))
    assert cap <= no_area <= free and len(free) == 4096
    assert len(list(enumerate_actions(proposals, current, "O_FREE_SUBSET"))) == 8191
    assert len(list(enumerate_actions(proposals, current))) <= 1587
    assert max(len(indices) for indices, _ in no_area) == 4
    for indices, _ in no_area:
        assert max(sum(proposals[i].target_class == c for i in indices) for c in (1, 2)) <= 3


def test_zero_foreground_allows_only_noop():
    p = regions(2)
    assert list(enumerate_actions(p, np.zeros((80, 80), dtype=np.int64))) == [((), None)]


def test_exact_foreground_budget_boundary_and_image_budget_boundary():
    p = regions(1, shape=(40, 40), area=24)
    current = np.zeros((40, 40), dtype=np.int64)
    current.flat[:160] = 1
    assert len(list(enumerate_actions(p, current))) == 3
    current.flat[159] = 0
    assert len(list(enumerate_actions(p, current))) == 1
    p = regions(1, shape=(20, 20), area=8)
    assert len(list(enumerate_actions(p, np.ones((20, 20), dtype=np.int64)))) == 3
    p = regions(1, shape=(20, 20), area=9)
    assert len(list(enumerate_actions(p, np.ones((20, 20), dtype=np.int64)))) == 1


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_blend_dtype_argmax_tie_outside_bytes_and_input_immutability(dtype):
    current = np.zeros((3, 80, 80), dtype=dtype); current[1] = 1
    history = np.zeros_like(current); history[2] = 1
    p = regions(1)
    before = (current.tobytes(), history.tobytes(), p[0].mask.tobytes())
    noop = apply_action(current, history, p, ((), None))
    assert noop.tobytes() == current.tobytes() and not np.shares_memory(noop, current)
    for coefficient in (0.5, 0.75):
        out = apply_action(current, history, p, ((0,), coefficient))
        assert out.dtype == dtype and out[:, ~p[0].mask].tobytes() == current[:, ~p[0].mask].tobytes()
        assert np.all(out.argmax(axis=0)[p[0].mask] == (1 if coefficient == 0.5 else 2))
    assert before == (current.tobytes(), history.tobytes(), p[0].mask.tobytes())


def test_overlaps_duplicates_area_and_cap_are_rejected():
    p = regions(2)
    bad = ((p[0], p[0]), (p[0], replace(p[0], proposal_id="other")),
           (replace(p[0], area=9),), (replace(p[0], centroid_row=3),), regions(13))
    for values in bad:
        with pytest.raises(ValueError): validate_proposals(values, (80, 80))


@pytest.mark.parametrize("votes,finite", (([1], [0]), ([-1], [1]), ([1], [201]), ([1], []), ([1.0], [1])))
def test_invalid_vote_counts(votes, finite):
    with pytest.raises(ValueError): validate_vote_counts(votes, finite, 1)


def test_gt_sentinel_and_probability_errors_do_not_enter_blind_api():
    p = np.zeros((3, 10, 10), dtype=np.float32); p[0] = 1
    with pytest.raises(TypeError): proposals_for_route(p, p, 2, ground_truth=object())
    assert proposals_for_route(p, p, 2) == ()
    for x in (np.full_like(p, np.nan), -p, p * 2, p[:2]):
        with pytest.raises(ValueError): probability_pair(x, p)
