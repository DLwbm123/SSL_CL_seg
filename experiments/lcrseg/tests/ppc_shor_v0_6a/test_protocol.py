import inspect
import subprocess

import numpy as np
import pytest

import ppc_shor_v0_6a as ppc
from shor_jascl_v0_3.core import shor_routes


def synthetic_rows(count=90):
    rows = []
    for index in range(count):
        domain = index % 3
        alpha = np.asarray([0.8, 0.1, 0.1]) if domain == 0 else (
            np.asarray([0.1, 0.8, 0.1]) if domain == 1 else np.asarray([0.1, 0.1, 0.8]))
        rows.append({"seed": index % 3, "patient_id": "p%d" % (index // 2),
                     "domain_index": domain, "alpha": alpha, "row_index": index})
    return rows


def test_global_row_index_alignment_is_append_order_invariant():
    rows = [{"row_index": index, "value": index} for index in [3, 0, 2, 1]]
    assert [row["value"] for row in ppc.aligned_global(rows, 4)] == [0, 1, 2, 3]
    with pytest.raises(ppc.ProtocolViolation):
        ppc.aligned_global(rows[:-1], 4)


def test_bayesian_bootstrap_preserves_support_and_patient_weight_contract():
    rows = []
    for domain in range(3):
        for patient in range(8):
            for seed in (0, 1):
                rows.append({"patient_id": "d%d-p%d" % (domain, patient), "domain_index": domain,
                             "seed": seed})
    weights = ppc.bootstrap_weights(rows, 17, 0)
    assert np.all(weights > 0)
    for patient in {row["patient_id"] for row in rows}:
        selected = np.asarray([row["patient_id"] == patient for row in rows])
        assert len(set(weights[selected])) == 1
    base = ppc.patient_row_weights(rows)
    assert all(base[np.asarray([row["patient_id"] == patient for row in rows])].sum() == 1
               for patient in {row["patient_id"] for row in rows})


def test_consensus_denominator_is_finite_predictions_and_acceptance_meets_rho():
    rows = [{"alpha": np.asarray([0.8, 0.1, 0.1])}]
    prediction = np.full((200, 1, 2), np.nan); prediction[:160, 0, 0] = 0.99
    route, _, consensus, denominator = ppc.route_policy(rows, prediction, 0.95, minimum_predictions=150)
    assert denominator.tolist() == [160] and consensus.tolist() == [1.0] and route.tolist() == [0]
    prediction[:32, 0, 0] = 0.1
    route, _, consensus, _ = ppc.route_policy(rows, prediction, 0.95, minimum_predictions=150)
    assert consensus[0] == pytest.approx(0.8) and consensus[0] >= ppc.RHO and route.tolist() == [0]


def test_kappa_and_tau_change_routes_and_duplicates_are_explicit():
    rows = [{"seed": 0, "alpha": np.asarray([0.8, 0.1, 0.1])}]
    constant = lambda p: {"upper": [0.0], "probability": [p], "fit_weight": [1.0], "blocks": 1}
    state = {"feasible": {"0": True, "1": True},
             "pooled": {"0": constant(0.90), "1": constant(0.90)},
             "local": {"0:0": {**constant(1.0), "fallback": False, "n_eff": 100.0},
                       "0:1": {**constant(1.0), "fallback": False, "n_eff": 100.0}}}
    low = ppc.calibrated_probabilities(state, rows, 10)[None, ...]
    high = ppc.calibrated_probabilities(state, rows, 100)[None, ...]
    assert ppc.route_policy(rows, low, 0.98, minimum_predictions=1)[0].tolist() == [0]
    assert ppc.route_policy(rows, high, 0.98, minimum_predictions=1)[0].tolist() == [2]
    assert ppc.route_policy(rows, low, 0.995, minimum_predictions=1)[0].tolist() == [2]
    assert ppc.mark_duplicates({"a": "x", "b": "x", "c": "y"}) == {"a": "a", "b": "a", "c": None}


def test_final_and_realization_use_one_route_function_without_global_early_return():
    source = inspect.getsource(ppc.run_calibration_fold)
    assert source.count("route_policy(") >= 4
    assert "feasible.sum()" not in inspect.getsource(ppc.route_policy)


def test_feasibility_is_per_expert_and_zero_route_precision_is_null():
    rows = synthetic_rows(90); weights = ppc.patient_row_weights(rows)
    result = ppc.fit_calibrators(rows, weights)
    assert set(result["feasible"]) == {"0", "1"}
    metrics = ppc.calibration_metrics(np.full(90, 2), rows, weights)
    assert metrics["route_precision"] is None
    assert metrics["route_precision_numerator"] == metrics["route_precision_denominator"] == 0


def test_historical_recall_denominator_excludes_current_and_wrong_top1():
    rows = synthetic_rows(90); weights = ppc.patient_row_weights(rows)
    routes = np.full(90, 2); routes[0] = 0; routes[1] = 1
    metrics = ppc.calibration_metrics(routes, rows, weights)
    top = np.argmax(np.stack([row["alpha"] for row in rows]), axis=1)
    expected = sum(weight for row, weight, winner in zip(rows, weights, top)
                   if row["domain_index"] < 2 and winner == row["domain_index"])
    assert metrics["historical_recall_denominator"] == pytest.approx(expected)


def test_seal_tamper_and_preseal_GT_access_hard_fail(tmp_path):
    artifact = tmp_path / "route.npy"; np.save(artifact, np.arange(3))
    seal = {"status": "PASS_OUTER_CANDIDATES_SEALED_BEFORE_GT", "outer_GT_reads": 0,
            "outer_domain_reads": 0, "sealed_files": {artifact.name: ppc.sha256_file(artifact)}}
    path = ppc.write_json_new(tmp_path / "seal.json", seal)
    ppc.verify_candidate_seal(path, tmp_path)
    artifact.write_bytes(b"tampered")
    with pytest.raises(ppc.ProtocolViolation, match="post-seal"):
        ppc.verify_candidate_seal(path, tmp_path)
    with pytest.raises(ppc.ProtocolViolation, match="before verified seal"):
        ppc.EvaluatorAccess().require(0)


def test_exact_C3_and_no_training_update_path():
    alpha = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    thresholds = {0: 0.0, 1: 0.0}
    assert shor_routes(alpha, stage=2, thresholds=thresholds).tolist() == [0, 1, 2]
    source = inspect.getsource(ppc)
    assert ".backward(" not in source and ".step(" not in source and ".train(" not in source


def test_predecessor_files_are_unchanged():
    base = "71d7e164390e10cce1619df9fcd18c17dc677939"
    for path in ("experiments/lcrseg/docs/shor_jascl_v0_3",
                 "experiments/lcrseg/docs/shor_jascl_v0_3_1",
                 "experiments/lcrseg/docs/shor_v0_4_fixed_policy_test",
                 "experiments/lcrseg/docs/rc_shor_v0_5",
                 "experiments/lcrseg/rc_shor_v0_5.py",
                 "experiments/lcrseg/tests/rc_shor_v0_5"):
        assert subprocess.run(["git", "-C", str(ppc.REPO), "diff", "--quiet", base, "--", path]).returncode == 0
