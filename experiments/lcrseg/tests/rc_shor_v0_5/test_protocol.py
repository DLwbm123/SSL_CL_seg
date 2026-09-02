import inspect
import json
import subprocess

import numpy as np
import pytest

import rc_shor_v0_5 as rc


def synthetic(count=30):
    rng = np.random.Generator(np.random.PCG64(7))
    x = rng.normal(size=(count, 6))
    y = 0.2 * x[:, 0] - 0.1 * x[:, 1]
    groups = np.asarray(["p%02d" % index for index in range(count)])
    return x, y, groups


def test_rc_route_api_rejects_domain_fields():
    forbidden = {"domain", "domain_index", "site", "vendor"}
    for function in (rc.route_from_base, rc.route_from_ensemble):
        assert not forbidden & set(inspect.signature(function).parameters)


def test_outer_truth_refused_before_seal(tmp_path):
    access = rc.FoldAccess(0, ["train"], ["eval"])
    with pytest.raises(rc.ProtocolViolation):
        access.permit("eval", True)
    seal = tmp_path / "seal.json"
    rc.write_json_new(seal, {"status": "PASS_OUTER_CANDIDATES_SEALED_BEFORE_GT",
                             "outer_GT_reads": 0, "outer_domain_reads": 0})
    access.mark_sealed(seal)
    access.permit("eval", True)


def test_cross_seed_duplicate_patient_stays_in_one_fold():
    rows = [{"seed": seed, "case_id": "same", "patient_id": "patient",
             "image_h5_relpath": "x", "image_sha256": "0" * 64} for seed in range(3)]
    rows += [{"seed": 0, "case_id": "c%d" % i, "patient_id": "p%d" % i,
              "image_h5_relpath": "x", "image_sha256": "0" * 64} for i in range(40)]
    folds = rc.fold_assignments(rows)
    assert len(set(folds[:3])) == 1


def test_active_support_compression_preserves_inactive_nan():
    x, y, groups = synthetic(30)
    weights = np.ones(30); weights[[1, 7, 19]] = 0
    output = rc.oof_predict(x, y, weights, groups, 0.1, "active")
    assert np.isfinite(output[weights > 0]).all()
    assert np.isnan(output[weights == 0]).all()


def test_bootstrap_draw_is_exactly_deterministic():
    groups = np.asarray(["p%d" % i for i in range(30)])
    domains = np.repeat(np.arange(3), 10)
    first = rc.bootstrap_multiplicity(groups, domains, 123, 9)
    second = rc.bootstrap_multiplicity(groups, domains, 123, 9)
    assert np.array_equal(first[0], second[0]) and first[1] == second[1]


def test_tie_rule_is_lowest_historical_index():
    lcb = np.ones((100, 2, 2), dtype=np.float64)
    route, _, consensus = rc.route_from_ensemble(lcb, np.ones(100, dtype=bool),
                                                  np.full(100, 20), 0.0, 0.7)
    assert route.tolist() == [0, 0]
    assert np.all(consensus[:, 0] == 1.0)


def test_conformal_higher_quantile_is_exact():
    prediction = np.arange(18, dtype=np.float64)
    truth = np.zeros(18)
    seeds = np.repeat(np.arange(3), 6)
    domains = np.tile(np.repeat(np.arange(3), 2), 3)
    q, group = rc.conformal_q(prediction, truth, np.ones(18), seeds, domains)
    assert q == 17.0 and len(group) == 9


def test_low_support_forces_abstention():
    lcb = np.zeros((100, 3, 2)); lcb[:, :, 0] = 1.0
    route, _, _ = rc.route_from_ensemble(lcb, np.ones(100, dtype=bool),
                                         np.full(100, 14), 0.0, 0.7)
    assert route.tolist() == [2, 2, 2]


def test_nonfinite_active_model_input_hard_fails():
    x, y, groups = synthetic(30)
    features = np.stack([x, x], axis=1); features[0, 0, 0] = np.nan
    with pytest.raises(rc.ProtocolViolation, match="nonfinite active model input"):
        rc.fit_bootstrap_ensemble(features, np.stack([y, y], axis=1), groups,
                                  np.repeat(np.arange(3), 10), np.tile(np.arange(3), 10),
                                  features[:2], [0.1, 0.1], 1, "bad")


def test_oracles_are_only_built_after_seal():
    source = inspect.getsource(rc.execute_outer_folds)
    assert source.index("access.mark_sealed") < source.index("c7 = evaluated") < source.index("c8 =")


def test_frozen_forwards_have_no_training_or_optimizer_path():
    source = inspect.getsource(rc.materialize_blind_inputs)
    assert "with no_updates()" in source
    assert ".backward(" not in source and ".step(" not in source and ".train(" not in source


def test_c3_is_exact_frozen_shor_behavior():
    alpha = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]])
    thresholds = {0: {0: 0.0, 1: 0.0}}
    observed = rc.c3_routes(alpha, np.zeros(3, dtype=np.int64), thresholds)
    expected = rc.shor_routes(alpha, stage=2, thresholds=thresholds[0])
    assert observed.tobytes() == expected.tobytes()


def test_same_input_route_and_hash_are_identical():
    lcb = np.zeros((100, 4, 2)); lcb[:, :, 1] = 0.2
    args = (lcb, np.ones(100, dtype=bool), np.full(100, 20), 0.01, 0.8)
    first = rc.route_from_ensemble(*args)[0]
    second = rc.route_from_ensemble(*args)[0]
    assert first.tobytes() == second.tobytes() and rc.array_hash(first) == rc.array_hash(second)


def test_existing_status_refuses_second_evaluation(tmp_path):
    path = tmp_path / "public"; path.mkdir()
    (path / "RC_SHOR_V0_5_STATUS.json").write_text("{}")
    with pytest.raises(rc.ProtocolViolation, match="REFUSED_AFTER_STATUS_EXISTS"):
        rc.refuse_occupied_output(tmp_path)


def test_frozen_history_is_byte_unchanged():
    protocol, audit = rc.load_protocol()
    assert audit["history"]["v0_3_1"]["status_sha256"] == rc.sha256_file(
        rc.ROOT / "docs/shor_jascl_v0_3_1/SHOR_V0_3_1_STATUS.json")
    assert audit["history"]["v0_4"]["status_sha256"] == rc.sha256_file(
        rc.ROOT / "docs/shor_v0_4_fixed_policy_test/SHOR_V0_4_TEST_STATUS.json")
    for path in ("experiments/lcrseg/docs/shor_jascl_v0_3",
                 "experiments/lcrseg/docs/shor_jascl_v0_3_1",
                 "experiments/lcrseg/docs/shor_v0_4_fixed_policy_test"):
        assert subprocess.run(["git", "-C", str(rc.REPO), "diff", "--quiet",
                               protocol["base_commit"], "--", path]).returncode == 0


def test_preregistration_is_complete_and_v0_4_private_is_not_an_input():
    protocol, audit = rc.load_protocol()
    assert protocol["formal_attempts_authorized"] == 1
    assert protocol["router_bootstrap"]["replicates"] == 100
    assert protocol["model"]["lambda_grid"] == [0.0001,0.001,0.01,0.1,1,10,100]
    assert audit["history"]["v0_4"]["formal_03_content_reads"] == 0
    assert protocol["frozen_inputs"]["forbidden_v0_4_root"] not in json.dumps(
        {key: value for key, value in protocol["frozen_inputs"].items() if key != "forbidden_v0_4_root"})
