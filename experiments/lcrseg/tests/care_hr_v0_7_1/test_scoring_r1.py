import numpy as np
import pytest

from care_hr_v0_7_1.actions import apply_action, proposals_for_route
from care_hr_v0_7_1.blind_r1 import prepare_case
from care_hr_v0_7_1.oracle_r1 import score_actions, select_oracles, tie_key
from care_hr_v0_7_1.resolution_r1 import resolution_audit
from care_hr_v0_7_1.scoring_r1 import action_target, case_metrics, confusion, gains
from care_hr_v0_7_1.semantic_audit import frozen_functions, one_hot


def test_all_1296_two_pixel_combinations_exact_golden_parity():
    report = resolution_audit()
    assert report["two_pixel_combinations"] == 1296
    assert report["unresolved_semantics_count"] == 0 and report["historical_difference_detected"]


@pytest.mark.parametrize("current,revised,truth,expected_harm", (
    ([1, 0], [1, 1], [1, 255], 0), ([1, 1], [0, 0], [255, 255], 0),
    ([1, 1], [1, 2], [1, 1], 1)))
def test_required_ignore_and_empty_class_examples(current, revised, truth, expected_harm):
    c, r, t = [np.asarray([a]) for a in (current, revised, truth)]
    delta = gains(case_metrics(c, t), case_metrics(r, t))
    assert delta["harm"] == expected_harm
    if expected_harm == 0: assert [delta[k] for k in ("gain_macro_fg", "gain_rim", "gain_cup")] == [0, 0, 0]
    else:
        assert case_metrics(c, t)["cup_dice"] == 1 and case_metrics(r, t)["cup_dice"] == 0


def test_macro_target_is_action_conditioned_and_not_union():
    c, t = np.asarray([[1, 1, 2, 2]]), np.asarray([[1, 2, 2, 2]])
    for coefficient in (0.5, 0.75):
        target = action_target(one_hot(c), one_hot(t), t, "example", coefficient)
        assert target["gain_macro_fg"] == pytest.approx(4 / 15)
        assert target["gain_union_fg"] == 0 and target["blend_lambda"] == coefficient
        assert "gain_fg" not in target


@pytest.mark.parametrize("p,t", (([], []), ([[0]], []), ([[0]], [[0, 0]]),
    ([[255]], [[0]]), ([[0]], [[3]]), ([[0.0]], [[0]]), ([[0]], None), ([0], [0])))
def test_invalid_labels_never_become_all_ignore(p, t):
    with pytest.raises(ValueError): case_metrics(p, t)


def fixture():
    truth = np.ones((40, 40), dtype=np.int64); truth[20:] = 2
    history = truth.copy(); history.flat[0:8] = 0; history.flat[16:24] = 0
    c, h = one_hot(truth), one_hot(history)
    proposals = proposals_for_route(c, h, 0)
    assert len(proposals) == 2
    return c, h, truth, proposals


def test_nonadditive_final_combination_harm_and_safe_noop():
    c, h, truth, proposals = fixture()
    rows = score_actions(prepare_case(c, h, 0), truth)
    single = [r for r in rows if len(r["indices"]) == 1]
    combined = [r for r in rows if len(r["indices"]) == 2]
    assert all(r["harm"] == pytest.approx(0.00502512562814) and r["harm"] <= 0.01 for r in single)
    assert all(r["harm"] == pytest.approx(0.01010101010101) and r["harm"] > 0.01 for r in combined)
    assert combined[0]["harm"] != sum(r["harm"] for r in single[:2])
    assert not select_oracles(rows)["O_SAFE0_envelope"]["indices"]


def test_every_combination_count_score_equals_pixel_bruteforce():
    c, h, truth, proposals = fixture()
    # Include ignored pixels while preserving blind proposal masks/budgets.
    truth = truth.copy(); truth[0, 0:4] = 255; truth[25:] = 0
    prepared = prepare_case(c, h, 0)
    gold, _ = frozen_functions("shor_v0_4_test.py", ("case_metrics",))
    before = case_metrics(c.argmax(axis=0), truth)
    for row in score_actions(prepared, truth):
        revised = apply_action(c, h, proposals, (tuple(row["indices"]), row["blend_lambda"]))
        hard = revised.argmax(axis=0)
        pixel = gold["case_metrics"](hard, truth)
        assert all(row[k] == pixel[k] for k in pixel)
        delta = gains(before, case_metrics(hard, truth))
        assert all(row[k] == delta[k] for k in delta)
        assert confusion(hard, truth).dtype == np.int64
    assert prepared["proposals"] == prepare_case(c, h, 0)["proposals"]


def test_nesting_optima_all_ignore_noop_and_no_gt_budget_feedback():
    c, h, truth, _ = fixture()
    prepared = prepare_case(c, h, 0)
    action_before = prepared["actions"]
    for target in (truth, h.argmax(axis=0), np.full_like(truth, 255)):
        selected = select_oracles(score_actions(prepared, target))
        for name in ("lambda050", "lambda075", "envelope"):
            assert selected["O_CAP_" + name]["macro_fg_dice"] <= selected["O_NO_AREA_" + name]["macro_fg_dice"] <= selected["O_FREE_SUBSET_" + name]["macro_fg_dice"]
        if np.all(target == 255):
            assert all(not r["indices"] and not r["has_evaluable_gt"] and r["macro_fg_dice"] == 1 for r in selected.values())
        assert prepared["actions"] == action_before
    with pytest.raises(TypeError): prepare_case(c, h, 0, valid_mask=truth != 255)


def test_safe_constraint_has_no_epsilon_and_tie_order_is_exact():
    c, h, truth, _ = fixture()
    rows = score_actions(prepare_case(c, h, 0), truth)
    assert all(not r["O_SAFE0"] for r in rows if r["gain_rim"] < 0)
    a = dict(rows[0], macro_fg_dice=0.5)
    b = dict(a, macro_fg_dice=np.nextafter(0.5, 1.0), mask_union_pixels=100)
    assert tie_key(b) < tie_key(a)


@pytest.mark.parametrize("prediction,truth", (([[0, 0]], [[0, 0]]), ([[1, 0]], [[255, 0]]),
    ([[2, 2]], [[255, 2]]), ([[0, 0]], [[1, 2]])))
def test_noop_and_mixed_support(prediction, truth):
    p, t = np.asarray(prediction), np.asarray(truth)
    m = case_metrics(p, t)
    assert gains(m, m)["harm"] == 0
    assert m["has_evaluable_gt"] and m["n_valid_pixels"] == int(np.count_nonzero(t != 255))
