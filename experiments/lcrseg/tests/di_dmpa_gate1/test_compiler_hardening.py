import copy
import json

import pytest

from scripts.compile_gate0_reports import audit_log, validate_frozen_baseline
from tests.gate0.test_report_compiler import _rows


def audit(tmp_path, second):
    rows = _rows()
    other = copy.deepcopy(rows[-1])
    other.update(global_step=3, **second)
    rows.append(other)
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return audit_log(path, domains=["REFUGE"])


def test_one_zero_gradient_cannot_hide_behind_positive_max(tmp_path):
    result = audit(tmp_path, {"student_unsupervised_gradient_norm": 0.0})
    assert result["max_gradient_by_domain"]["REFUGE"] > 0
    assert result["zero_gradient_batch_count"] == 1
    assert result["min_gradient_by_domain"]["REFUGE"] == 0
    assert result["errors"]


def test_one_zero_coverage_cannot_hide_behind_positive_aggregate(tmp_path):
    result = audit(tmp_path, {"pas_joint_valid_pixels": 0, "pas_joint_coverage": 0.0})
    assert result["valid_pixels_by_domain"]["REFUGE"] > 0
    assert result["zero_coverage_batch_count"] == 1
    assert result["min_coverage_by_domain"]["REFUGE"] == 0
    assert result["errors"]


def test_per_domain_min_and_p01_are_actual_statistics(tmp_path):
    result = audit(tmp_path, {"student_unsupervised_gradient_norm": 0.3, "pas_joint_coverage": 0.75})
    assert result["errors"] == []
    assert result["zero_gradient_batch_count"] == result["zero_coverage_batch_count"] == 0
    assert result["min_gradient_by_domain"]["REFUGE"] == 0.1
    assert result["p01_gradient_by_domain"]["REFUGE"] == pytest.approx(0.102)
    assert result["min_coverage_by_domain"]["REFUGE"] == 0.5


def test_missing_gradient_is_not_treated_as_successful_audit(tmp_path):
    assert audit(tmp_path, {"student_unsupervised_gradient_norm": None})["errors"]


def test_frozen_baseline_bare_pass_is_rejected(tmp_path):
    path = tmp_path / "freeze.json"
    path.write_text('{"status":"PASS"}')
    with pytest.raises(RuntimeError):
        validate_frozen_baseline(path, tmp_path, {"C0":"a","B0":"b"})
