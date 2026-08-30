import copy
import json

import pytest

from scripts.compile_gate0_reports import audit_log, validate_report, OBJECTIVE


def _rows():
    common = {"loss_total": 0.2, "loss_supervised": 0.1, "hidden_gt_training_usage": "none",
              "objective_name": OBJECTIVE, "config_hash": "hash", "git_commit": "sha", "domain": "REFUGE"}
    return [
        {**common, "phase": "supervised", "lr": 0.001, "global_step": 1},
        {**common, "phase": "unlabeled", "global_step": 2, "loss_consistency": 0.2,
         "pas_joint_valid_pixels": 4, "pas_joint_coverage": 0.5, "consistency_requires_grad": True,
         "student_unsupervised_gradient_norm": 0.1, "student_total_gradient_norm": 1.0,
         "teacher_nonnull_gradient_count": 0, "prototype_requires_grad": False,
         "stochastic_classifier_train_mode": True, "teacher_forward_no_grad": True, "optimizer_step_executed": True},
    ]


def _audit(tmp_path, rows):
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return audit_log(path, config_hash="hash", git_commit="sha", domains=["REFUGE"])


def test_valid_probability_log_does_not_require_unlabeled_lr(tmp_path):
    assert _audit(tmp_path, _rows())["errors"] == []


def test_status_compiler_rejects_zero_gradient_run(tmp_path):
    rows = _rows()
    rows[1]["student_unsupervised_gradient_norm"] = 0.0
    assert any("zero unlabeled gradient" in error for error in _audit(tmp_path, rows)["errors"])


@pytest.mark.parametrize("field,value", [
    ("objective_name", "hard_class_index_mse"), ("consistency_requires_grad", False),
    ("pas_joint_valid_pixels", 0), ("teacher_nonnull_gradient_count", 1),
    ("prototype_requires_grad", True), ("config_hash", "wrong"), ("git_commit", "wrong"),
    ("teacher_forward_no_grad", False), ("loss_consistency", float("nan")),
])
def test_compiler_rejects_invalid_evidence(tmp_path, field, value):
    rows = _rows()
    rows[1][field] = value
    assert _audit(tmp_path, rows)["errors"]


def test_supervised_row_requires_finite_lr(tmp_path):
    rows = _rows()
    rows[0].pop("lr")
    assert any("supervised lr" in error for error in _audit(tmp_path, rows)["errors"])


@pytest.mark.parametrize("name", [
    "UNIT_INTEGRATION_TEST_REPORT.json", "RESUME_EQUIVALENCE_REPORT.json", "PAS_GRADIENT_AUDIT.json",
    "LEAKAGE_AUDIT_REPORT.json", "EVAL_STOCHASTICITY_AUDIT.json",
])
def test_bare_pass_is_not_real_evidence(name):
    assert validate_report(name, {"status": "PASS"}, "sha", {"B0": "hash", "C0": "hash0"}, ["REFUGE"])


def test_wrong_source_real_gradient_report_is_rejected():
    report = {"status": "PASS", "git_commit": "wrong", "config_hashes": {"B0": "hash", "C0": "hash0"}}
    assert validate_report("PAS_GRADIENT_AUDIT.json", report, "sha", report["config_hashes"], ["REFUGE"])
