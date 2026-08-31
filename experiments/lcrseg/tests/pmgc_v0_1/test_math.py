import copy

import numpy as np
import pytest
import torch

from di_dmpa_gate1c_v2 import binding as b
from di_dmpa_gate1_v2.features import split_support
from di_dmpa_gate1_v2.geometry import fit
from di_dmpa_gate1.sampling import sample_layout
from di_dmpa_gate1.feature_extraction import state_hash
from pmgc_v0_1 import core as c, modes as m, evaluator as ev
from pmgc_v0_1.testing import synthetic_models, synthetic_geometry


def support(count=64):
    return [dict(class_id=i//2, mode=i%2, center_active=True, active_pixels=count, old_correct_pixels=count, KD_active=count >= 32) for i in range(6)]


def test_exact_class_mode_assignment_and_lowest_slot_tie():
    centers = np.stack([np.eye(16)[:2]]*3)
    features = np.zeros((1, 16, 1, 4)); features[0, 0, 0, :] = 1; features[0, 1, 0, 1] = 2
    labels = np.array([[[0, 1, 2, 2]]])
    modes, active = m.assignments(features, labels, centers, np.ones((3, 2), bool))
    assert modes.tolist() == [[[0, 3, 4, 4]]] and active.all()


def test_null_and_ignore_keep_separate_slots():
    centers = np.stack([np.eye(16)[:2]]*3)
    modes, active = m.assignments(np.zeros((1, 16, 1, 2)), np.array([[[1, 255]]]), centers, np.ones((3, 2), bool))
    assert modes.tolist() == [[[-1, -2]]] and not active.any()


def test_all_null_fit_never_reduces_K_or_fabricates_direction():
    found = fit(np.zeros((4, 16)), np.zeros(4, bool), np.ones(4)/4, 2, seed=0, stage=1, class_id=1)
    assert found["K"] == 2 and found["centers"].shape == (2, 16) and not found["active"].any()
    assert found["original_null_count"] == 4


def test_case_pixel_balancing():
    unit = dict(cases=[dict(case_id="a", classes=[dict(sampled_pixels=1, coordinates=[[0, 0]], boundary=[False])]),
                       dict(case_id="b", classes=[dict(sampled_pixels=2, coordinates=[[0, 0], [0, 1]], boundary=[False, False])])])
    layout = sample_layout(unit, 0)
    np.testing.assert_array_equal(layout["weights"], [.5, .25, .25])
    assert len(layout["uids"]) == 3


def test_old_correct_has_no_confidence_threshold():
    p = np.array([[[[.34, .32]], [[.33, .35]], [[.33, .33]]]], np.float32)
    assert m.old_correct(p, np.array([[[0, 1]]])).all()
    assert not m.old_correct(p, np.array([[[2, 0]]])).any()


def test_old_posterior_mean_determinism_and_no_grad():
    models, _ = synthetic_models(); old = models["old"]
    image = torch.randn(2, 3, 8, 8)
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        first = old(image, stochastic_classifier=False)[0]
        second = old(image, stochastic_classifier=False)[0]
    assert torch.equal(first, second) and torch.equal(torch.get_rng_state(), rng)
    assert not first.requires_grad and all(p.grad is None for p in old.parameters())


def test_supervised_CE_and_forward_KL_formula():
    logits = torch.tensor([[[[2., 0.]], [[0., 2.]], [[-1., -1.]]]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([[[[.6, .1]], [[.3, .8]], [[.1, .1]]]], dtype=torch.float64)
    y = torch.tensor([[[0, 1]]])
    ce, kl = m.loss_maps(logits, y, p)
    torch.testing.assert_close(ce, -logits.log_softmax(1).gather(1, y[:, None]).squeeze(1))
    torch.testing.assert_close(kl, (p*(p.log()-logits.log_softmax(1))).sum(1))


def test_teacher_target_rejects_gradients():
    with pytest.raises(c.Blocked):
        m.loss_maps(torch.randn(1, 3, 1, 1), torch.zeros(1, 1, 1, dtype=torch.long), torch.ones(1, 3, 1, 1, requires_grad=True)/3)


def test_mode_VJP_denominators_and_explicit_None_zeros():
    logits = torch.nn.Parameter(torch.randn(1, 3, 2, 3, dtype=torch.float64))
    unused = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    labels = torch.tensor([[[0, 0, 1], [1, 2, 2]]]); modes = torch.arange(6).reshape(1, 2, 3)
    p = torch.softmax(torch.ones_like(logits), 1).detach()
    sup, old, masks = m.guard_vjps(logits, labels, modes, torch.ones_like(modes, dtype=torch.bool), p,
                                   [("logits", logits), ("unused", unused)], support(1))
    assert sup.shape == old.shape == (6, 20) and all(mask == [False, True] for mask in masks)
    assert not sup[:, -2:].any() and not old[:, -2:].any() and logits.grad is unused.grad is None


def test_empty_mode_graph_connected_zero():
    logits = torch.nn.Parameter(torch.randn(1, 3, 1, 1, dtype=torch.float64))
    sup, old, _ = m.guard_vjps(logits, torch.zeros(1, 1, 1, dtype=torch.long), torch.full((1, 1, 1), -1),
                              torch.zeros(1, 1, 1, dtype=torch.bool), torch.ones(1, 3, 1, 1)/3, [("p", logits)], support(0))
    assert not sup.any() and not old.any() and logits.grad is None


@pytest.mark.parametrize("candidate,count", [("P0", 0), ("P1", 1), ("P2", 7), ("P3", 7), ("P4", 13)])
def test_five_fixed_comparator_constraint_sets(candidate, count):
    guards = c.constraint_sets(np.ones(3), np.arange(18).reshape(6, 3), np.ones((6, 3)), support())
    assert len(guards[candidate]) == count
    if candidate == "P1": assert guards[candidate][0][0] == "global_supervised"
    if candidate == "P2": assert all("class" in name for name, _ in guards[candidate][1:])
    if candidate == "P3": assert not any(name.startswith("old") for name, _ in guards[candidate])


def test_inactive_KD_is_not_merged_into_primary():
    rows = support(20)
    guards = c.constraint_sets(np.ones(3), np.ones((6, 3)), np.ones((6, 3)), rows)
    assert len(guards["P4"]) == 7 and len(guards["P2"]) == 7


@pytest.mark.parametrize("scale", [.001, .25, 1., 9., 1000.])
def test_constraint_positive_scaling_invariance(scale):
    v, cert = c.solve_cone(np.array([-1., 2.]), [("h", np.array([scale, 0.]))])
    np.testing.assert_allclose(v, [0, 2], atol=1e-12)
    assert cert["primal_feasible"] and cert["dual_feasible"]


def test_deterministic_duplicate_rank_deficient_KKT():
    guards = [("a", np.array([1., 0.])), ("b", np.array([2., 0.])), ("zero", np.zeros(2))]
    v, cert = c.solve_cone(np.array([-2., 3.]), guards)
    other, repeated = c.solve_cone(np.array([-2., 3.]), guards)
    assert np.array_equal(v, other) and cert == repeated and cert["rank_deficient"]
    assert cert["condition_number"] is None and cert["gram_rank"] == 1 and cert["zero_constraint_indices"] == [2]
    assert cert["complementarity_max"] <= 1e-12 and cert["duality_gap"] <= 1e-12


@pytest.mark.parametrize("guards", [[], [("z", np.zeros(2))]])
def test_zero_constraints_leave_raw_direction(guards):
    raw = np.array([-2., 3.]); v, cert = c.solve_cone(raw, guards)
    np.testing.assert_array_equal(raw, v)
    assert cert["objective"] == 0 and not cert["fallback_used"]


@pytest.mark.parametrize("seed", range(12))
def test_minimum_distance_projection_on_orthant(seed):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=6)
    guards = [(str(i), np.eye(6)[i]) for i in range(6)]
    v, cert = c.solve_cone(raw, guards)
    np.testing.assert_allclose(v, np.maximum(raw, 0), atol=1e-12)
    assert cert["stationarity_residual"] <= 1e-12
    assert cert["objective"] == pytest.approx(.5*np.minimum(raw, 0) @ np.minimum(raw, 0))


def test_cone_can_return_zero_but_retention_marks_failure():
    guards = [("plus", np.array([1.])), ("minus", np.array([-1.]))]
    projections, arrays = c.projections(np.array([-1.]), {key: guards for key in c.CANDIDATES})
    assert projections["P4"]["zero_direction"] and projections["P4"]["norm_ratio"] == 0
    np.testing.assert_array_equal(arrays["P0"], [-1.])


def test_reject_more_than_thirteen_constraints():
    with pytest.raises(c.Blocked): c.solve_cone(np.ones(2), [(str(i), np.ones(2)) for i in range(14)])


def test_nonfinite_constraint_is_blocked():
    with pytest.raises(b.NonfiniteEvidence): c.solve_cone(np.ones(2), [("bad", np.array([np.nan, 0]))])


def test_stateless_virtual_step_exact_norm_and_model_immutability():
    models, _ = synthetic_models(); model = models["gradient_student"]
    before = state_hash(model.state_dict()); n = sum(p.numel() for p in model.parameters())
    direction = np.linspace(-1, 1, n)
    mapping, receipt = c.displacement(model, direction, raw_norm=float(np.linalg.norm(direction)))
    assert receipt["step_valid"] and receipt["step_norm"] == pytest.approx(.001, abs=1e-15)
    logits, _ = c.functional_forward(model, mapping, torch.ones(2, 3, 8, 8))
    assert torch.isfinite(logits).all() and state_hash(model.state_dict()) == before
    assert all(p.grad is None for p in model.parameters()) and not logits.requires_grad


def test_zero_virtual_step_explicit_identity_without_epsilon():
    models, _ = synthetic_models(); model = models["gradient_student"]
    n = sum(p.numel() for p in model.parameters())
    _, receipt = c.displacement(model, np.zeros(n), raw_norm=1.)
    assert not receipt["step_valid"] and receipt["step_norm"] == 0


def test_full_real_model_parameter_inventory():
    from tests.gate0.test_classifier_stochasticity import _model
    inventory = c.full_inventory(c.parameters(_model()))
    assert len(inventory) == 51 and sum(r["elements"] for r in inventory) == 484016


def test_inventory_does_not_silently_drop_None_parameters():
    models, _ = synthetic_models()
    with pytest.raises(c.Blocked): c.full_inventory(c.parameters(models["student"]))


@pytest.mark.parametrize("role", ["train_unlabeled", "test", "unknown"])
def test_mode_construction_rejects_hidden_test_and_validation_roles(role):
    with pytest.raises(c.Blocked): m.fit_bank(dict(role=role), {})


def test_fit_rejects_validation_GT():
    with pytest.raises(c.Blocked): m.fit_bank(dict(role="val"), {})


@pytest.mark.parametrize("role", ["previous", "current"])
def test_validation_evaluator_role_isolation(role):
    with pytest.raises(c.Blocked): ev.load_panel(dict(role="train_unlabeled"), "/forbidden", "cpu", role="val")


def test_per_mode_foreground_Dice_aggregation():
    labels = torch.tensor([[[0, 1, 2], [0, 1, 2]]]); modes = torch.tensor([[[0, 2, 4], [1, 3, 5]]])
    logits = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).double()*5
    p = logits.softmax(1).detach()
    stats = ev.statistics(logits, labels, modes, torch.ones_like(labels, dtype=torch.bool), p)
    value = ev.aggregate([stats, stats])
    assert value["foreground_Dice"] == 1 and value["mode_Dice"] == [1]*6
    assert value["totals"]["pixels"] == 12 and value["totals"]["mode_pixels"] == [2]*6
    assert all(x is not None for x in value["mode_CE"])


def test_mode_UID_counts_and_null_support_are_deterministic():
    labels = np.array([[0, 1, 2], [0, 1, 2]], np.uint8)
    caches = {case: dict(labels=labels) for case in ("a", "b")}
    modes = np.array([[0, 2, 4], [1, -1, 5]], np.int8)
    maps = {case: dict(modes=modes, active=modes >= 0, old_correct=np.ones_like(labels, bool)) for case in caches}
    fits = [dict(UID_order_sha256="s", fits={"2":dict(centers=np.eye(16)[:2].tolist(), center_norms=[1., 1.], converged=True, iterations=2)}) for _ in range(3)]
    rows = m.support_rows(list(caches), caches, maps, np.ones((3, 2), bool), fits)
    assert rows == m.support_rows(list(caches), caches, maps, np.ones((3, 2), bool), fits)
    assert rows[2]["active_pixels"] == 2 and rows[2]["case_count"] == 2 and rows[2]["occupancy"] == 1
    assert rows[3]["active_pixels"] == 0 and rows[3]["null_count"] == 2
    assert rows[2]["UID_sha256"] != rows[3]["UID_sha256"]


def test_real_forward_counter_blocks_before_extra_dispatch():
    models, _ = synthetic_models()
    with pytest.raises(c.Blocked), c.measured(models, dict.fromkeys(c.COUNT_KEYS, 0)):
        models["student"](torch.ones(1, 3, 8, 8), stochastic_classifier=False)


def test_autograd_counter_blocks_unregistered_call():
    models, _ = synthetic_models(); parameter = next(models["student"].parameters())
    with pytest.raises(c.Blocked), c.measured(models, dict.fromkeys(c.COUNT_KEYS, 0)):
        torch.autograd.grad(parameter.sum(), [parameter])
