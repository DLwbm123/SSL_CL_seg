from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pytest

from pres_dsr_sf_v0_2.core import fit_router
from shor_jascl_v0_3 import REGISTRATION
from shor_jascl_v0_3.core import (adjudicate, bootstrap_weights, calibration, historical_score, one_hot,
                                  reconstruct_oof, select_threshold, shor_routes, top1_lowest)
from shor_jascl_v0_3.protocol import (AUTHORITY, PRIVATE_BYTES, PRIVATE_CONTENT_SHA, PRIVATE_FILES,
                                      PHASES, compile_call_graph)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/shor_jascl_v0_3"
PACKAGE = ROOT / "shor_jascl_v0_3"


def source(name):
    return (PACKAGE / name).read_text()


def binary_unit():
    labels = np.repeat(np.arange(2), 50)
    alpha = np.empty((100, 2), dtype=np.float64)
    alpha[labels == 0] = (.99, .01)
    alpha[labels == 1] = (.01, .99)
    return alpha, labels


@lru_cache(maxsize=1)
def oof_unit():
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(2), 20)
    value = rng.normal(0, .05, (40, 102))
    value[labels == 0, 0] -= 1
    value[labels == 1, 0] += 1
    ids = tuple(f"domain{label}_case{i:03d}" for i, label in enumerate(labels))
    return value, labels, ids, reconstruct_oof(value, labels, ids)


def passing_evidence():
    return dict(calibration=dict(all_units_feasible=True, all_finite=True),
                current_safety=dict(current_domain_drop=.010, maximum_current_class_drop=.015,
                                    maximum_seed_domain_drop=.020),
                value=dict(three_domain_gain=.100, historical_gain=.150, oracle_gap=.060,
                           positive_seed_count=3, REFUGE_mean_gain=1e-9, RIM_ONE_r3_mean_gain=1e-9),
                repair=dict(current_domain_drop_reduction=.020, maximum_seed_domain_drop_reduction=.020,
                            shared_gain_loss=.060, historical_gain_loss=.080),
                stability=dict(shared_gain_p10=.080, historical_gain_p10=.120, current_domain_drop_p90=.015,
                               maximum_seed_domain_drop_p90=.025,
                               every_unit_feasible_in_at_least_4_of_5=True, all_finite=True),
                isolation=True)


def test_01_v021_closure_binding():
    closure = json.loads((DOCS / "PRES_DSR_SF_V0_2_1_FINAL_CLOSURE.json").read_text())
    assert AUTHORITY[0][0] == "9feee43c5e34c427356ceaaafa6f691dd14186a3"
    assert closure["status"] == "CLOSED" and closure["next_protocol"] == "SHOR_JASCL_V0_3"
    assert closure["soft_expert_fusion_status"] == "FAIL_SOFT_EXPERT_FUSION_VALUE"


def test_02_private_bundle_hash_binding():
    assert (PRIVATE_FILES, PRIVATE_BYTES, PRIVATE_CONTENT_SHA) == (
        183, 4386018614, "05c9008ad4496ccbdc51df6103638024d49fae4b3b4cdc2a9f829c5f3ab165bb")


def test_03_no_model_import_or_forward():
    text = source("run.py")
    assert "load_models" not in text and "model(" not in text and ".forward(" not in text
    assert "new_model_forwards=0" in text


def test_04_exact_train_oof_reconstruction():
    _, labels, _, (_, oof) = oof_unit()
    assert oof.shape == (40, 2) and np.isfinite(oof).all() and np.allclose(oof.sum(1), 1)
    assert np.mean(np.argmax(oof, 1) == labels) > .70


def test_05_ridge_formula_parity():
    value, labels, ids, (model, _) = oof_unit()
    reference = fit_router(value, labels, ids)
    assert model["selected_lambda"] == reference["selected_lambda"]
    assert model["selected_temperature"] == reference["selected_temperature"]
    assert np.array_equal(model["weights"], reference["weights"])


def test_06_score_log_odds_formula():
    alpha = np.array([[.8, .2], [.25, .75]])
    assert np.allclose(historical_score(alpha, 1, 0), np.log(alpha[:, 0] + 1e-12) - np.log(alpha[:, 1] + 1e-12))


def test_07_historical_top1_requirement():
    alpha, labels = binary_unit()
    alpha[0] = (.49, .51)
    row = calibration(alpha, labels, stage=1, domain=0, threshold=-100)
    assert row["accepted_count"] == 49


def test_08_current_fallback():
    alpha = np.array([[.4, .6]])
    assert shor_routes(alpha, stage=1, thresholds={0: dict(threshold=-100.)}).tolist() == [1]


def test_09_stage1_fallback_expert1():
    alpha = np.array([[.9, .1]])
    assert shor_routes(alpha, stage=1, thresholds={0: None}).tolist() == [1]


def test_10_stage2_fallback_expert2():
    alpha = np.array([[.8, .1, .1]])
    assert shor_routes(alpha, stage=2, thresholds={0: None, 1: None}).tolist() == [2]


def test_11_future_expert_exclusion():
    alpha = np.array([[.8, .1, .1], [.1, .8, .1]])
    routed = shor_routes(alpha, stage=2, thresholds={0: dict(threshold=-100.), 1: dict(threshold=-100.)})
    assert routed.tolist() == [0, 1] and routed.max() <= 2


def test_12_unique_threshold_generation():
    alpha, labels = binary_unit()
    _, rows = select_threshold(alpha, labels, stage=1, domain=0)
    finite = [row["threshold"] for row in rows if np.isfinite(row["threshold"])]
    assert len(finite) == len(set(finite)) == 1


def test_13_infinity_threshold_handling():
    alpha, labels = binary_unit()
    _, rows = select_threshold(alpha, labels, stage=1, domain=0)
    assert rows[-1]["threshold"] == np.inf and rows[-1]["accepted_count"] == 0 and not rows[-1]["feasible"]


def test_14_precision_calculation():
    alpha, labels = binary_unit()
    assert calibration(alpha, labels, stage=1, domain=0, threshold=0)["precision"] == 1


def test_15_recall_calculation():
    alpha, labels = binary_unit()
    assert calibration(alpha, labels, stage=1, domain=0, threshold=0)["historical_recall"] == 1


def test_16_current_false_override():
    alpha, labels = binary_unit()
    alpha[50] = (.9, .1)
    assert calibration(alpha, labels, stage=1, domain=0, threshold=0)["current_false_override"] == .02


def test_17_accepted_count_rule():
    alpha, labels = binary_unit()
    mult = np.zeros(100); mult[:14] = 1; mult[50:] = 1
    assert calibration(alpha, labels, stage=1, domain=0, threshold=0, multiplicity=mult)["accepted_count"] == 14
    assert not calibration(alpha, labels, stage=1, domain=0, threshold=0, multiplicity=mult)["feasible"]


def test_18_deterministic_threshold_selection():
    alpha, labels = binary_unit()
    first, _ = select_threshold(alpha, labels, stage=1, domain=0)
    second, _ = select_threshold(alpha, labels, stage=1, domain=0)
    assert first == second and first["feasible"]


def test_19_threshold_tie_rule():
    alpha = np.array([[.5, .5], [.5, .5]])
    assert top1_lowest(alpha).tolist() == [0, 0]
    assert shor_routes(alpha, stage=1, thresholds={0: dict(threshold=0.)}).tolist() == [0, 0]


def test_20_no_validation_in_threshold_fitting():
    text = source("core.py")
    section = text[text.index("def reconstruct_oof"):text.index("def bootstrap_weights")]
    assert "validation" not in section


def test_21_no_segmentation_gt_in_routing():
    text = source("core.py")
    assert "segmentation" not in text and "label_h5" not in text


def test_22_shor_hard_one_hot_route():
    encoded = one_hot(np.array([0, 2, 1]))
    assert np.array_equal(encoded, np.eye(3)[[0, 2, 1]]) and np.all(encoded.sum(1) == 1)


def test_23_no_soft_mixing_in_s3():
    text = source("run.py")
    s3 = text[text.index('"S3_SHOR": routes'):text.index("formal_predictions[seed][stage] = {}")]
    assert "probability_fusion" not in s3


def test_24_s0_definition():
    assert '"S0_SHARED": np.full(len(ids), stage' in source("run.py")


def test_25_s1_definition():
    assert '"S1_RIDGE_HARD": top' in source("run.py")


def test_26_s2_frozen_definition():
    text = source("run.py")
    assert "S2_RIDGE_SOFT" in text and "probability_fusion(padded(alpha" in text


def test_27_s4_fixed_oracle_mapping():
    text = source("run.py")
    assert 'predictions["S4_ORACLE"] = hard_predictions(experts[seed], positions, truth)' in text


def test_28_candidate_seal_before_gt():
    text = source("run.py")
    assert text.index('phase_barrier(args.output, "candidate_prediction_seal"') < text.index("contract = gate1c_contract()")
    assert "PHASE_candidate_prediction_seal_MANIFEST.json" in text


def test_29_per_case_failure_attribution():
    text = source("run.py")
    for field in ("top1_correct", "soft_vs_current_regret", "soft_vs_hard_regret", "current_domain_regret",
                  "correct_route_current_case", "misrouted_current_case"):
        assert field in text


def test_30_exact_seed_domain_aggregation():
    text = source("run.py")
    assert "len(segmentation_rows) == 75" in text and "len(utility_rows) == 15" in text


def test_31_bootstrap_determinism():
    ids = {0: [f"a{i}" for i in range(20)], 1: [f"b{i}" for i in range(20)]}
    first = bootstrap_weights(ids, seed=1, stage=1, replicate=3)
    second = bootstrap_weights(ids, seed=1, stage=1, replicate=3)
    assert np.array_equal(first[0], second[0]) and first[1] == second[1]


def test_32_formal_threshold_not_reused_in_bootstrap():
    text = source("run.py")
    assert "select_threshold(boot_oof" in text and 'boot_thresholds[domain] = selected' in text


@pytest.mark.parametrize("mutate,expected", [
    (lambda x: x["calibration"].update(all_units_feasible=False), "FAIL_SELECTIVE_OVERRIDE_CALIBRATION"),
    (lambda x: x["current_safety"].update(current_domain_drop=.0100001), "FAIL_SELECTIVE_OVERRIDE_CURRENT_SAFETY"),
    (lambda x: x["value"].update(three_domain_gain=.099999), "FAIL_SELECTIVE_OVERRIDE_VALUE"),
    (lambda x: x["repair"].update(current_domain_drop_reduction=.019999), "FAIL_SELECTIVE_OVERRIDE_VALUE"),
    (lambda x: x["stability"].update(shared_gain_p10=.079999), "FAIL_SELECTIVE_OVERRIDE_STABILITY"),
    (lambda x: x.update(isolation=False), "BLOCKED_PROTOCOL_OR_LEAKAGE"),
])
def test_33_h1_h6_exact_boundaries(mutate, expected):
    evidence = passing_evidence(); mutate(evidence)
    assert adjudicate(evidence)["scientific_status"] == expected
    assert adjudicate(passing_evidence())["scientific_status"] == "PASS_SHOR_JASCL_VALIDATION_FEASIBILITY"


def test_34_controls_cannot_rescue():
    evidence = passing_evidence(); evidence["calibration"]["all_units_feasible"] = False
    evidence["controls"] = {"S0": "perfect", "S1": "perfect", "S2": "perfect", "S4": "perfect"}
    assert adjudicate(evidence)["scientific_status"] == "FAIL_SELECTIVE_OVERRIDE_CALIBRATION"


def test_35_no_test_construction():
    text = source("run.py")
    assert 'b.records(data_root, contract, seed, domain, "val")' in text and '"test"' not in text


def test_36_no_optimizer():
    text = source("run.py") + source("core.py")
    assert ".step(" not in text and "torch.optim" not in text


def test_37_no_autograd():
    assert "torch.autograd" not in source("run.py") and "autograd." not in source("run.py")


def test_38_no_backward():
    assert ".backward(" not in source("run.py")


def test_39_no_parameter_grad():
    assert ".grad" not in source("run.py")


def test_40_input_immutability():
    text = source("protocol.py")
    assert "verify_private_bundle" in text and "private artifact changed" in text and "d.write_new(root" not in text


def test_41_output_key_set():
    text = source("run.py")
    assert "required.issubset(names)" in text and "BLOCKED_OUTPUT_KEYSET_MISMATCH" in text


def test_42_create_only_state():
    text = source("run.py")
    assert '.open("x"' in text and 'mode="w+"' in text and "not path.exists()" in text


def test_43_durable_exit_receipt():
    text = source("postflight.py")
    assert "PROCESS_EXIT.json" in text and 'actual_child_exit_code"] == 0' in text


def test_44_artifact_manifest():
    text = source("run.py")
    assert "SHOR_ARTIFACT_MANIFEST.json" in text and "required_outputs_complete=True" in text


def test_45_private_archive_audit():
    text = source("postflight.py")
    assert "PASS_SHOR_PRIVATE_ARCHIVE_AUDIT" in text and "verify_private_bundle(args.private_root)" in text


def test_46_report_compiler_fail_closed():
    text = source("postflight.py")
    assert 'status["scientific_status"].startswith(("PASS_", "FAIL_"))' in text
    assert REGISTRATION == "SHOR_JASCL_V0_3_SELECTIVE_HISTORICAL_OVERRIDE" and len(PHASES) == 9


def test_47_frozen_validation_cardinality(tmp_path):
    graph = compile_call_graph(tmp_path, {1: 140, 2: 165}, "code")
    assert (graph["formal_route_rows"], graph["formal_candidate_case_predictions"],
            graph["bootstrap_candidate_case_predictions"], graph["failure_attribution_rows"]) == (
                915, 3660, 4575, 915)
