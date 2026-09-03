import inspect
import json
import subprocess

import numpy as np
import pytest

import ppc_shor_v0_6a as v6a
import ppc_shor_v0_6b as v6b


def synthetic_rows(count=90):
    rows = []
    for index in range(count):
        domain = index % 3
        alpha = np.full(3, 0.1)
        alpha[domain] = 0.8
        rows.append({"seed": (index // 3) % 3, "patient_id": "p%d" % index,
                     "domain_index": domain, "alpha": alpha, "row_index": index})
    return rows


def test_prediction_validity_is_separate_from_route_eligibility():
    rows = [{"alpha": np.asarray([0.1, 0.1, 0.8])}]
    predictions = np.ones((200, 1, 2), dtype=np.float64)
    validity, eligibility = v6b.prediction_accounting(rows, predictions)
    assert validity.tolist() == [[200, 200]]
    assert eligibility.tolist() == [0]
    assert validity.min() >= 190


def test_effective_PAV_levels_merge_only_adjacent_bitwise_equal_values():
    assert v6b.effective_probability_levels({"probability": [0.0, 0.0, 1.0, 1.0]}) == 2
    left = np.float64(0.5)
    right = np.nextafter(left, np.float64(1.0))
    assert v6b.effective_probability_levels({"probability": [left, right]}) == 2


def test_accounting_change_preserves_fitted_probabilities_routes_and_hashes():
    rows = synthetic_rows()
    weights = v6a.patient_row_weights(rows)
    original = v6b._fit_calibrators_v0_6a(rows, weights)
    corrected = v6b.fit_calibrators(rows, weights)
    old = v6b.probability_fields(original)
    new = v6b.probability_fields(corrected)
    assert old.keys() == new.keys()
    assert all(np.array_equal(old[key].view(np.uint64), new[key].view(np.uint64)) for key in old)
    old_probability = v6a.calibrated_probabilities(original, rows, 10)
    new_probability = v6a.calibrated_probabilities(corrected, rows, 10)
    old_route = v6a.route_policy(rows, old_probability[None], 0.9, minimum_predictions=1)[0]
    new_route = v6a.route_policy(rows, new_probability[None], 0.9, minimum_predictions=1)[0]
    assert np.array_equal(old_route, new_route)
    assert v6b.array_hash(old_route) == v6b.array_hash(new_route)


def test_registered_corrected_parameter_ratios_are_exact():
    protocol = v6b.load_protocol()
    assert protocol["recovery_qualification"]["expected_parameter_ratio_fold_maxima"] == list(v6b.EXPECTED_RATIOS)
    old = json.loads((v6a.DOCS / "PPC_SHOR_V0_6A_STATUS.json").read_text())
    assert old["postflight_interpretation"]["corrected_parameter_ratio"]["per_fold_max"] == list(v6b.EXPECTED_RATIOS)


def test_fixed_selected_candidates_are_exact():
    assert v6b.load_protocol()["v0_6a_recovery_binding"]["fixed_selected_candidates"] == list(v6b.EXPECTED_SELECTED)


def test_registered_stitched_route_is_155_of_198():
    stitched = v6b.load_protocol()["v0_6a_recovery_binding"]["stitched_C6"]
    assert stitched["route_count"] == 155
    assert stitched["population"] == 198
    assert stitched["route_frequency"] == 155 / 198
    assert len(stitched["route_sha256"]) == 64


def test_modal_disagreement_is_one_minus_route_agreement():
    routes = np.asarray([[0, 0, 2], [0, 1, 2], [0, 1, 1], [0, 1, 1]])
    result = v6b.stability_diagnostics(routes)
    assert result["modal_disagreement"] == pytest.approx(1 - result["route_agreement"])


def test_any_flip_is_diagnostic_and_distinct_from_modal_disagreement():
    routes = np.zeros((200, 2), dtype=np.int64)
    routes[0, 0] = 1
    result = v6b.stability_diagnostics(routes)
    assert result["any_flip_case_fraction"] == 0.5
    assert result["modal_disagreement"] == pytest.approx(0.0025)


def test_constituent_routes_are_not_named_full_policy_realizations():
    result = v6b.stability_diagnostics(np.zeros((200, 2), dtype=np.int64))
    assert result["constituent_route_semantics"] == "constituent_bayesian_routes"
    assert "full_policy" not in result["constituent_route_semantics"]


def test_final_ensemble_requires_200_predictions_tau_and_rho():
    rows = [{"alpha": np.asarray([0.8, 0.1, 0.1])}]
    prediction = np.full((200, 1, 2), np.nan)
    prediction[:160, 0, 0] = 0.99
    prediction[160:, 0, 0] = 0.1
    route, _, consensus, denominator = v6b.final_ensemble_route(rows, prediction, 0.95, 0.8)
    assert route.tolist() == [0] and consensus.tolist() == [0.8] and denominator.tolist() == [200]
    with pytest.raises(v6b.ProtocolViolation, match="200 predictions"):
        v6b.final_ensemble_route(rows, prediction[:-1], 0.95, 0.8)


def test_outer_blind_controller_has_no_domain_or_label_fields():
    row = {"case_id": "c", "patient_id": "p", "seed": 0, "alpha": [0.8, 0.1, 0.1],
           "image_h5_relpath": "x", "image_sha256": "0" * 64, "row_index": 0, "fold": 0}
    assert v6b.validate_outer_blind_rows([row])
    with pytest.raises(v6b.ProtocolViolation):
        v6b.validate_outer_blind_rows([{**row, "domain_index": 0}])


def test_outer_domain_is_not_used_for_fit_or_selection():
    fit_source = inspect.getsource(v6b.fit_calibrators)
    selection_source = inspect.getsource(v6a.candidate_sort_key)
    assert "outer" not in fit_source and "domain" not in selection_source
    protocol = v6b.load_protocol()
    assert protocol["role_isolation"]["outer_domain_used_for_fit"] == 0
    assert protocol["role_isolation"]["outer_domain_used_for_selection"] == 0


def test_asset_SHA_tamper_blocks_image_label_checkpoint_and_probability(tmp_path):
    for role in ("image", "label", "checkpoint", "probability"):
        path = tmp_path / (role + ".bin")
        path.write_bytes(b"frozen")
        digest = v6b.sha256_file(path)
        v6a.verify_file(path, digest)
        path.write_bytes(b"tampered")
        with pytest.raises(v6a.ProtocolViolation):
            v6a.verify_file(path, digest)


@pytest.mark.parametrize("mutation", ("nan", "negative", "sum"))
def test_invalid_expert_probability_cache_blocks(tmp_path, mutation):
    path = tmp_path / (mutation + ".npy")
    value = np.zeros((1, 3, 384, 384), dtype=np.float32)
    value[:, 0] = 1
    if mutation == "nan":
        value[0, 0, 0, 0] = np.nan
    elif mutation == "negative":
        value[0, 0, 0, 0] = -0.1
    else:
        value[0, 0, 0, 0] = 0.5
    np.save(path, value, allow_pickle=False)
    with pytest.raises(v6b.ProtocolViolation, match="invalid expert"):
        v6b.validate_probability_cache(path, 1)


@pytest.mark.parametrize("name", ("expert.npy", "prediction.npy"))
def test_candidate_seal_tamper_blocks_expert_or_prediction_change(tmp_path, name):
    artifact = tmp_path / name
    artifact.write_bytes(b"sealed")
    seal = {"status": "PASS_OUTER_CANDIDATES_SEALED_BEFORE_GT", "outer_GT_reads": 0,
            "outer_domain_reads": 0, "sealed_files": {name: v6b.sha256_file(artifact)}}
    path = v6b.write_json_new(tmp_path / (name + ".seal.json"), seal)
    v6a.verify_candidate_seal(path, tmp_path)
    artifact.write_bytes(b"changed")
    with pytest.raises(v6a.ProtocolViolation, match="post-seal"):
        v6a.verify_candidate_seal(path, tmp_path)


def test_fold_append_shuffle_does_not_change_global_metrics():
    rows = [{"row_index": index, "value": float(index)} for index in (3, 0, 2, 1)]
    ordered = v6a.aligned_global(rows, 4)
    shuffled = v6a.aligned_global(list(reversed(rows)), 4)
    assert [row["value"] for row in ordered] == [row["value"] for row in shuffled] == [0, 1, 2, 3]


def test_zero_route_precision_is_null_with_explicit_counts():
    rows = synthetic_rows()
    utility = np.zeros((len(rows), 2))
    classes = np.zeros((len(rows), 2, 2))
    result = v6a.route_metrics(np.full(len(rows), 2), utility, classes,
                               np.asarray([row["seed"] for row in rows]),
                               np.asarray([row["domain_index"] for row in rows]),
                               np.asarray([row["patient_id"] for row in rows]))
    assert result["route_precision"] is None
    assert result["route_precision_numerator"] == result["route_precision_denominator"] == 0


def test_runtime_guard_blocks_backward_and_records_no_unrequested_update():
    torch = v6a.v4.torch
    guard = v6b.NoUpdateGuard()
    with guard:
        with pytest.raises(v6b.ProtocolViolation, match="backward"):
            torch.tensor(1.0, requires_grad=True).backward()
    assert guard.calls == {"backward": 1, "optimizer_construction": 0, "optimizer_step": 0}


def test_existing_reservation_refuses_second_attempt(tmp_path):
    path = tmp_path / "FORMAL_GT_ACCESS_RESERVATION.json"
    v6b.create_reservation(path, {"status": "FORMAL_GT_ACCESS_RESERVED"})
    with pytest.raises(FileExistsError):
        v6b.create_reservation(path, {"status": "SECOND"})


def test_all_frozen_predecessor_files_are_byte_unchanged():
    base = "89e05bac567614137fade175f1edf5c434dd9461"
    paths = (
        "experiments/lcrseg/docs/shor_jascl_v0_3",
        "experiments/lcrseg/docs/shor_jascl_v0_3_1",
        "experiments/lcrseg/docs/shor_v0_4_fixed_policy_test",
        "experiments/lcrseg/docs/rc_shor_v0_5",
        "experiments/lcrseg/docs/rc_shor_v0_5_erratum",
        "experiments/lcrseg/docs/ppc_shor_v0_6a",
        "experiments/lcrseg/rc_shor_v0_5.py",
        "experiments/lcrseg/ppc_shor_v0_6a.py",
        "experiments/lcrseg/tests/rc_shor_v0_5",
        "experiments/lcrseg/tests/ppc_shor_v0_6a",
    )
    for path in paths:
        assert subprocess.run(["git", "-C", str(v6b.REPO), "diff", "--quiet", base, "--", path]).returncode == 0
