import math
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from care_hr_v0_7_1.conformal import aligned_vectors, calibrate_bounds, patient_quantile
from care_hr_v0_7_1.semantic_audit import BLOCKED, audit


def test_required_macro_counterexample_and_ignore_conflict():
    report = audit()
    assert report["status"] == BLOCKED and not report["new_scoring_rule_selected"]
    macro, ignored, empty, noop = report["witnesses"]
    assert macro["v0_7_targets"]["gain_fg"] == 0
    assert macro["v0_6b_gain_macro_fg"] == pytest.approx(4 / 15)
    assert ignored["v0_6b_gain_rim"] == 0
    assert ignored["v0_7_targets"]["harm"] == pytest.approx(1 / 3)
    assert empty["v0_6b_gain_rim"] == 0 and empty["v0_7_targets"]["gain_rim"] == 1
    assert set(noop["v0_7_targets"].values()) == {0}


def test_exact_eleventh_patient_order_statistic():
    bound, meta = patient_quantile(range(1, 12), [str(i) for i in range(11)])
    assert bound == 11 and meta["rank"] == 11 and meta["patients"] == 11


def test_duplicate_patients_use_maximum_and_do_not_inflate_n():
    bound, meta = patient_quantile([0, 10, 2, 3], ["same_patient", "same_patient", "b", "c"], alpha=0.5)
    assert bound == 3 and meta["patients"] == 3 and meta["rows"] == 4 and meta["rank"] == 2


@pytest.mark.parametrize("n", (0, 1, 2, 8))
def test_insufficient_patients_return_infinity_not_maximum(n):
    bound, meta = patient_quantile(range(n), [str(i) for i in range(n)])
    assert math.isinf(bound) and meta["status"] == "INSUFFICIENT_PATIENTS_INFINITE_BOUND"
    assert not meta["exchangeability_established"] and not meta["method_risk_guarantee"]


def test_decimal_rank_boundary_is_not_rounded_up_by_float_noise():
    bound, meta = patient_quantile(range(1, 100), [str(i) for i in range(99)], alpha=0.58)
    assert bound == meta["rank"] == 42


@pytest.mark.parametrize("alpha", (0, 1, -0.1, float("nan"), float("inf"), True))
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        patient_quantile([1], ["p"], alpha)


@pytest.mark.parametrize("values,patients", (([float("nan")], ["a"]), ([float("inf")], ["a"]),
                                             ([[1]], ["a"]), ([1, 2], ["a"]), ([1], [""])))
def test_invalid_residuals_and_patient_alignment(values, patients):
    with pytest.raises(ValueError):
        patient_quantile(values, patients)


def test_calibration_rejects_broadcasting_and_preserves_inputs():
    with pytest.raises(ValueError):
        calibrate_bounds([1, 2], [0], [0, 0], [0, 0], ["a", "b"])
    values = np.arange(9, dtype=float)
    original = values.tobytes()
    result = calibrate_bounds(values, np.zeros(9), np.zeros(9), values, [str(i) for i in range(9)])
    assert result[0][0] == result[1][0] == 8
    assert values.tobytes() == original


def test_all_vector_lengths_are_checked_before_arithmetic():
    with pytest.raises(ValueError):
        aligned_vectors([1, 2], [3])
    with pytest.raises(ValueError):
        aligned_vectors([1], [[2]])


def test_frozen_history_hashes_and_old_review_lock():
    root = Path(__file__).resolve().parents[4]
    docs = root / "experiments/lcrseg/docs"
    manifest = json.loads((docs / "care_hr_v0_7_1/HISTORY_PROTECTION.json").read_text())
    assert manifest["base_commit"] == "61c1e302fae515fe51adf4e777887a1a489859be"
    for entry in manifest["entries"]:
        assert hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    lock = json.loads((docs / "care_hr_v0_7_review/CARE_HR_V0_7_REVIEW_LOCK.json").read_text())
    assert not lock["training_authorized"] and not lock["real_data_access_authorized"]
    assert not lock["evaluation_authorized"]
