import numpy as np
import pytest

from care_hr_v0_7.conformal import bounds, calibrate_bounds, patient_quantile
from care_hr_v0_7.ridge import (bayesian_patient_weights, fit_dual_heads, fit_ridge,
                               grouped_folds, patient_proposal_weights,
                               prediction_accounting, select_lambda)


def synthetic():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 4)); patients = [f"p{i // 3}" for i in range(30)]
    y = 1.5 + x @ np.array([0.2, -0.3, 0.0, 0.1])
    return x, y, patients


def test_ridge_is_float64_and_intercept_is_unpenalized():
    x = np.zeros((8, 2)); y = np.full(8, 3.25)
    model = fit_ridge(x, y, np.ones(8), 100.0)
    assert model.coefficient.dtype == np.float64
    assert np.allclose(model.predict(x), y)


def test_nonfinite_ridge_input_hard_fails():
    with pytest.raises(ValueError, match="nonfinite"):
        fit_ridge([[np.nan]], [0], [1], 1)


def test_patient_proposal_weights_sum_to_one_per_patient():
    patient_ids = ["a", "a", "b"]
    weights = patient_proposal_weights(patient_ids)
    assert weights[:2].sum() == 1 and weights[2:].sum() == 1


def test_grouped_folds_never_split_a_patient():
    ids = ["a", "a", "b", "b", "c", "c"]
    folds = grouped_folds(ids, 3)
    assert all(len(set(folds[np.asarray(ids) == patient])) == 1 for patient in set(ids))


def test_exact_oof_tie_selects_larger_lambda():
    x = np.zeros((12, 2)); y = np.zeros(12); ids = [f"p{i}" for i in range(12)]
    assert select_lambda(x, y, ids) == 100.0


def test_dual_heads_are_independent_models():
    x, gain, ids = synthetic(); harm = np.maximum(0, -gain)
    gain_model, harm_model = fit_dual_heads(x, gain, harm, ids)
    assert gain_model is not harm_model


def test_bayesian_weights_are_positive_shared_and_domain_free():
    ids = ["a", "a", "b"]
    patient = bayesian_patient_weights(ids, 0)
    rows = patient_proposal_weights(ids, patient)
    assert all(value > 0 for value in patient.values())
    assert rows[:2].sum() == pytest.approx(patient["a"])


def test_bayesian_weights_are_deterministic():
    assert bayesian_patient_weights(["a", "b"], 9) == bayesian_patient_weights(["a", "b"], 9)


def test_conformal_uses_patient_maximum_before_higher_quantile():
    residuals = [0.1, 0.9, 0.2, 0.3]
    assert patient_quantile(residuals, ["a", "a", "b", "b"], coverage=0.5) == 0.9


def test_lcb_and_ucb_have_correct_direction():
    q_gain, q_harm = calibrate_bounds([0.4, 0.5], [0.3, 0.3], [0.1, 0.1], [0.2, 0.4], ["a", "b"], 1.0)
    lower, upper = bounds([0.5], [0.2], q_gain, q_harm)
    assert lower[0] <= 0.5 and upper[0] >= 0.2


def test_prediction_validity_is_separate_from_action_eligibility():
    predictions = [[1.0, 2.0], [3.0, 4.0]]
    validity, eligibility = prediction_accounting(predictions, [True, False])
    assert validity.tolist() == [2, 2] and eligibility.tolist() == [2, 0]
