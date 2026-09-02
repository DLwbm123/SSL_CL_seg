import ast
import copy
import inspect
import json

import numpy as np
import pytest

import shor_v0_4_test as v4


def calls(function):
    tree = ast.parse(inspect.getsource(function))
    return {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Attribute, ast.Name))}


def indexed_strings(function):
    tree = ast.parse(inspect.getsource(function))
    return {node.slice.value for node in ast.walk(tree) if isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)}


def test_frozen_threshold_hashes_are_required(tmp_path):
    protocol = copy.deepcopy(v4.load_protocol())
    protocol["frozen_inputs"]["v0_3_1_threshold_manifest"]["path"] = str(tmp_path / "missing.json")
    with pytest.raises(v4.RequiredInputMissing):
        v4.load_frozen_policy(protocol)
    assert len(protocol["frozen_inputs"]["v0_3_1_threshold_manifest"]["sha256"]) == 64


def test_no_ridge_temperature_or_threshold_fitting():
    tree = ast.parse(v4.Path(v4.__file__).read_text())
    called = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
              for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, (ast.Attribute, ast.Name))}
    assert not {"fit_router", "ridge_fit", "fit_standardizer", "select_threshold"} & called


def test_s3_has_no_test_domain_input():
    assert tuple(inspect.signature(v4.route_s3).parameters) == ("alpha", "thresholds")
    alpha = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    assert v4.route_s3(alpha, {0: 0.0, 1: 0.0}).tolist() == [0, 2]


def test_test_truth_is_refused_before_candidate_seal(tmp_path):
    access = v4.TestAccess()
    with pytest.raises(v4.ProtocolViolation):
        access.require_evaluator()
    seal = tmp_path / "TEST_CANDIDATE_SEAL.json"
    seal.write_text(json.dumps({"status": "PASS_TEST_CANDIDATES_SEALED_BEFORE_GT",
                                "test_GT_reads": 0, "test_domain_reads": 0}))
    access.mark_sealed(seal)
    access.require_evaluator()


def test_s0_s3_use_identical_cases():
    v4.validate_eval_alignment(["a", "b"], ["a", "b"])
    with pytest.raises(v4.ProtocolViolation):
        v4.validate_eval_alignment(["a", "b"], ["b", "a"])


def test_expert_inference_is_deterministic():
    state = v4.configure_determinism()
    assert state == {"deterministic_algorithms": True, "cudnn_deterministic": True,
                     "cudnn_benchmark": False, "matmul_tf32": False, "cudnn_tf32": False}
    assert "stochastic_classifier=False" in inspect.getsource(v4.predict_expert)


def test_paired_hierarchical_bootstrap_is_reproducible():
    groups = {(seed, domain): np.asarray([0.01, 0.02, -0.01, 0.03])
              for seed in range(3) for domain in range(3)}
    first = v4.paired_bootstrap(groups, replicates=40, seed=17)
    second = v4.paired_bootstrap(groups, replicates=40, seed=17)
    assert first == second


def test_test_domain_is_only_materialized_by_evaluator():
    blind = indexed_strings(v4.blind_test_rows)
    evaluator = indexed_strings(v4.full_test_rows)
    assert not {"site_or_vendor", "label_h5_relpath", "label_sha256"} & blind
    assert {"site_or_vendor", "label_h5_relpath", "label_sha256"} <= evaluator


def test_no_model_update_optimizer_backward_or_training_path():
    assert "no_updates" in calls(v4.phase_a)
    all_calls = calls(v4.phase_a) | calls(v4.extract_test_descriptors) | calls(v4.predict_expert)
    assert not {"step", "backward", "grad", "train"} & all_calls


def test_repeated_final_evaluation_is_refused(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    (public / "SHOR_V0_4_TEST_STATUS.json").write_text("{}")
    with pytest.raises(v4.ProtocolViolation, match="REFUSED_AFTER_STATUS_EXISTS"):
        v4.refuse_occupied_output(tmp_path)
