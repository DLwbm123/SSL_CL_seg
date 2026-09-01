from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from di_dmpa_gate1c_v3 import durable as d
from pres_dsr_sf_v0_2 import REGISTRATION
from pres_dsr_sf_v0_2 import postflight, protocol, run, testing
from pres_dsr_sf_v0_2.core import (Blocked, DOMAINS, adjudicate, balanced_weights,
                                   bootstrap_multiplicity, case_folds, fit_router, fit_standardizer,
                                   hard_routes, probability_fusion, raw_style_block, raw_style_descriptors,
                                   ridge_fit, select_memory, softmax)
from pres_jascl_v0_1.run import deterministic_backend_state


def _restore_backend(state):
    torch.use_deterministic_algorithms(state[0])
    torch.backends.cudnn.deterministic = state[1]
    torch.backends.cudnn.benchmark = state[2]
    torch.backends.cuda.matmul.allow_tf32 = state[3]
    torch.backends.cudnn.allow_tf32 = state[4]


def _passing_evidence():
    return dict(E1=True, E6=True,
                oracle=dict(three_domain_gain=.02, historical_gain=.03, positive_seed_count=2,
                            maximum_domain_drop=.005),
                ridge_hard=dict(stage1_macro=.95, stage1_per_domain=[.9, .9], stage2_macro=.9,
                                stage2_per_domain=[.85, .85, .85]),
                ridge_soft=dict(oracle_gap=.02, shared_gain=.13, historical_gain=.2, gain_over_m1_hard=.01,
                                positive_seed_count=3, maximum_seed_domain_drop=.02, current_domain_drop=.01),
                stability=dict(hard_macro_p10=.85, soft_gain_p10=.1, soft_oracle_gap_p90=.03,
                               all_domains_nonempty=True, all_finite=True))


def _router_data():
    labels = np.repeat(np.arange(2), 10)
    case_ids = [f"d{label}c{i}" for label in range(2) for i in range(10)]
    return np.zeros((20, 102), dtype=np.float64), labels, case_ids


def test_01_v01_closure_binding():
    prereg = protocol.authority()
    assert prereg["registration_id"] == REGISTRATION


def test_02_formal_report_guard_commit_binding():
    assert protocol.BASE_HEAD == "ab71694ad6b3134fe1b45bd479658349e619fdc5"
    assert [row[0] for row in protocol.AUTHORITY] == [
        "607a067319a6e8f0bfc1b8d6a305f014cd6ab676",
        "c4767688e01ee9106d172a88a95f7e6c8a5de0eb",
        "78427b35ae5101c0576863386df0c434f77d2734",
    ]


def test_03_backend_import_order_isolation():
    original = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        audit = protocol.freeze_backend(lambda: setattr(torch.backends.cudnn, "benchmark", True))
        assert audit["after_pinned_import"]["cudnn_benchmark_disabled"] is False
        assert all(audit["registered"].values())
    finally:
        _restore_backend(original)


def test_04_no_forward_before_backend_freeze():
    assert "with forbid_forwards():" in inspect.getsource(protocol.freeze_backend)
    assert "model_forwards_before_freeze=0" in inspect.getsource(protocol.freeze_backend)


def test_05_backend_state_phase_invariance():
    original = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        protocol.freeze_backend(lambda: None)
        assert all(protocol.require_backend("test")["state"].values())
    finally:
        _restore_backend(original)


def test_06_exact_nine_checkpoint_identities():
    assert len(protocol.gate1c_contract()["immutable_baseline"]["checkpoint_inputs"]) == 9


def test_07_fixed_expert_domain_mapping():
    assert DOMAINS == ("REFUGE", "RIM_ONE_r3", "Drishti_GS")


def test_08_oracle_never_selects_best_expert():
    source = inspect.getsource(run.materialize_candidates)
    assert '"C1_ORACLE": one_hot_routes(truth)' in source
    assert "argmax" not in source.split('"C1_ORACLE"')[1].split('"C2_M1_HARD"')[0]


def test_09_raw_rgb_descriptor():
    value = torch.tensor([[[[1.0, 3.0], [1.0, 3.0]]]])
    block = raw_style_block(value)
    assert torch.allclose(block[:, :1], torch.tensor([[2.0]], dtype=torch.float64))


def test_10_raw_enc1_descriptor():
    rgb, enc1, enc2 = torch.zeros(1, 3, 2, 2), torch.ones(1, 16, 2, 2), torch.zeros(1, 32, 1, 1)
    descriptor = raw_style_descriptors(rgb, enc1, enc2)
    assert np.all(descriptor[0, 6:22] == 1.0)


def test_11_raw_enc2_descriptor():
    descriptor = raw_style_descriptors(torch.zeros(1, 3, 2, 2), torch.zeros(1, 16, 2, 2),
                                       torch.full((1, 32, 1, 1), 2.0))
    assert np.all(descriptor[0, 38:70] == 2.0)


def test_12_log_std_formula():
    value = torch.tensor([[[[0.0, 2.0]]]])
    assert np.isclose(raw_style_block(value)[0, 1].item(), np.log(1.0 + 1e-6))


def test_13_expected_dimension_102():
    value = raw_style_descriptors(torch.zeros(2, 3, 2, 2), torch.zeros(2, 16, 1, 1),
                                  torch.zeros(2, 32, 1, 1))
    assert value.shape == (2, 102)


def test_14_train_only_standardization():
    state = fit_standardizer(np.vstack((np.zeros((1, 102)), np.full((1, 102), 2.0))))
    assert np.allclose(state["mean"], 1.0) and np.allclose(state["std"], 1.0)


def test_15_constant_dimension_handling():
    state = fit_standardizer(np.ones((3, 102)))
    assert state["constant"].all() and np.all(state["scale"] == 1.0)


def test_16_descriptor_determinism():
    tensors = (torch.rand(2, 3, 4, 4), torch.rand(2, 16, 2, 2), torch.rand(2, 32, 1, 1))
    assert np.array_equal(raw_style_descriptors(*tensors), raw_style_descriptors(*tensors))


def test_17_no_path_or_domain_feature():
    assert tuple(inspect.signature(raw_style_descriptors).parameters) == ("rgb", "enc1", "enc2")


def test_18_no_segmentation_gt_in_router():
    source = inspect.getsource(run.fit_ridge_routers)
    assert "label_h5" not in source and "read_labels" not in source


def test_19_memory_cap():
    selected, hashes = select_memory([{"case_id": f"c{i}"} for i in range(600)])
    assert len(selected) == len(hashes) == 512


def test_20_historical_memory_immutability():
    assert "historical memory bytes changed" in inspect.getsource(run.build_memory)


def test_21_domain_balanced_weights():
    weights = balanced_weights([0, 0, 0, 1])
    assert np.isclose(weights[:3].sum(), .5) and np.isclose(weights[3:].sum(), .5)


def test_22_deterministic_five_folds():
    ids = [f"d{d}c{i}" for d in range(2) for i in range(10)]
    labels = np.repeat(np.arange(2), 10)
    first = case_folds(ids, labels)
    assert np.array_equal(first, case_folds(ids, labels))
    assert all(np.bincount(first[labels == d], minlength=5).tolist() == [2] * 5 for d in range(2))


def test_23_ridge_closed_form_solution():
    value = np.vstack((np.full((5, 2), -1.0), np.full((5, 2), 1.0)))
    labels = np.repeat(np.arange(2), 5)
    weights = ridge_fit(value, labels, .01)
    assert weights.shape == (3, 2) and np.isfinite(weights).all()


def test_24_bias_unregularized():
    weights = ridge_fit(np.zeros((10, 2)), np.repeat(np.arange(2), 5), 1.0)
    assert np.allclose(weights[-1], [.5, .5])


def test_25_lambda_selection_ties_choose_largest():
    value, labels, ids = _router_data()
    assert fit_router(value, labels, ids)["selected_lambda"] == 1.0


def test_26_temperature_selection_ties_choose_one():
    value, labels, ids = _router_data()
    assert fit_router(value, labels, ids)["selected_temperature"] == 1.0


def test_27_validation_excluded_from_selection():
    source = inspect.getsource(run.fit_ridge_routers)
    assert source.index("model = fit_router(train, labels, ids)") < source.index('row["role"] == "val"')


def test_28_stage1_future_domain_exclusion():
    assert set(hard_routes(np.array([[.2, .8]]), (0, 1))) <= {0, 1}


def test_29_stage2_three_domain_inclusion():
    assert hard_routes(np.array([[.1, .2, .7]]), (0, 1, 2)).tolist() == [2]


def test_30_hard_routing_tie_lowest():
    assert hard_routes(np.array([[.5, .5]]), (0, 1)).tolist() == [0]


def test_31_soft_weights_sum_one():
    probability = softmax(np.array([[1.0, 2.0]]), 2.0)
    assert np.allclose(probability.sum(1), 1.0)


def test_32_probability_fusion():
    experts = np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    assert np.allclose(probability_fusion(np.array([[.25, .75]]), experts), [[[.25, .75]]])


def test_33_no_logit_fusion():
    source = inspect.getsource(run.materialize_candidates)
    assert "probability_fusion" in source and "logit" not in source


def test_34_m1_clean_reproduction():
    assert "clean_states[seed][1]" in inspect.getsource(run.fit_ridge_routers)


def test_35_m2_clean_reproduction():
    assert "clean_states[seed][2]" in inspect.getsource(run.fit_ridge_routers)


def test_36_shared_final_definition():
    assert '"C0_SHARED": one_hot_routes(np.full(len(ids), 2))' in inspect.getsource(run.materialize_candidates)


def test_37_oracle_definition():
    assert '"C1_ORACLE": one_hot_routes(truth)' in inspect.getsource(run.materialize_candidates)


def test_38_uniform_control():
    assert '1.0 / (stage + 1)' in inspect.getsource(run.materialize_candidates)


def test_39_routing_confusion():
    assert "len(confusion_rows) == 117" in inspect.getsource(run.fit_ridge_routers)


def test_40_segmentation_aggregation():
    prediction = np.array([[0, 1], [2, 2]])
    target = np.array([[0, 1], [1, 2]])
    from pres_jascl_v0_1.core import pixel_confusion, segmentation_metrics
    metrics = segmentation_metrics(pixel_confusion(prediction, target))
    assert metrics["mean_foreground_dice"] > 0


def test_41_bootstrap_determinism():
    ids = [f"c{i}" for i in range(20)]
    first = bootstrap_multiplicity(ids, seed=0, stage=2, domain=1, replicate=3)
    second = bootstrap_multiplicity(ids, seed=0, stage=2, domain=1, replicate=3)
    assert np.array_equal(first[0], second[0]) and first[1] == second[1]


def test_42_E1_E6_exact_boundaries():
    result = adjudicate(_passing_evidence())
    assert result["scientific_status"] == "PASS_PRES_DSR_SF_FEASIBILITY" and all(result[f"E{i}"] for i in range(1, 7))


def test_43_controls_cannot_rescue_primary():
    evidence = _passing_evidence()
    evidence["ridge_soft"]["shared_gain"] = .129999
    assert adjudicate(evidence)["scientific_status"] == "FAIL_SOFT_EXPERT_FUSION_VALUE"


def test_44_no_test_construction():
    assert 'test_GT_reads=0' in inspect.getsource(run.report)


def test_45_validation_gt_isolation():
    source = inspect.getsource(run.main)
    assert source.index("materialize_candidates") < source.index("evaluate_segmentation")


def test_46_no_model_optimizer():
    assert "torch.optim" not in inspect.getsource(run)


def test_47_no_model_autograd():
    assert "model_autograd_calls=0" in inspect.getsource(run.main)


def test_48_no_backward():
    assert ".backward(" not in inspect.getsource(run)


def test_49_model_checkpoint_immutability():
    assert "ImmutableModels" in inspect.getsource(run.extract_descriptors)
    assert "ImmutableModels" in inspect.getsource(run.predict_expert_probabilities)


def test_50_call_graph_compiler(tmp_path: Path):
    counts = {"train_labeled": (40, 16, 10), "train_unlabeled": (160, 63, 41), "val": (100, 40, 25)}
    records = {seed: {stage: {role: [{}] * counts[role][stage] for role in counts} for stage in range(3)}
               for seed in range(3)}
    graph = protocol.compile_call_graph(tmp_path, records, "a" * 40)
    assert graph["expert_probability_forwards"] == 189 and graph["total_output_rows"] == 1356


def test_51_create_only_state(tmp_path: Path):
    path = tmp_path / "cache.npz"
    run.npz_new(path, x=np.zeros(1))
    with pytest.raises(FileExistsError):
        run.npz_new(path, x=np.ones(1))


def test_52_durable_child_exit(tmp_path: Path):
    d.write_new(tmp_path / "EXECUTION_COMPLETION.json", {"status": "COMMAND_COMPLETED", "actual_child_exit_code": 0})
    d.write_new(tmp_path / "PROCESS_EXIT.json", {"actual_child_exit_code": 0})
    assert postflight.validate_durable_completion(tmp_path) is None


def test_53_artifact_manifest_contract():
    source = inspect.getsource(run.artifact_manifest)
    assert "PRES_DSR_SF_ARTIFACT_MANIFEST.json" in source and "required_outputs_complete=True" in source


def test_54_private_archive_audit():
    source = inspect.getsource(postflight.main)
    assert "PASS_PRIVATE_ARCHIVE_AUDIT" in source and "PRES_DSR_SF_PRIVATE_BUNDLE_MANIFEST.json" in source


def test_55_report_compiler_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    junit = tmp_path / "pytest.xml"
    junit.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase classname="x" name="x"/></testsuite>')
    output = tmp_path / "output.txt"
    output.write_text("1 passed")
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(testing, "source_gate", lambda commit: {"code_commit": commit})
    monkeypatch.setattr(sys, "argv", ["testing", "--junit", str(junit), "--pytest-output", str(output),
                                      "--output", str(receipt), "--code-commit", "a" * 40,
                                      "--exact-command", "python -m pytest"])
    with pytest.raises(Blocked):
        testing.main()
    assert not receipt.exists()
