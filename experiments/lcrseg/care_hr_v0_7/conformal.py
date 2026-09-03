"""Patient-level one-sided conformal bounds."""
from __future__ import annotations

import numpy as np


def _higher(values, coverage):
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if len(ordered) == 0 or not np.all(np.isfinite(ordered)):
        raise ValueError("invalid conformal residuals")
    return float(np.quantile(ordered, coverage, method="higher"))


def patient_quantile(residuals, patient_ids, coverage=0.90):
    residuals = np.asarray(residuals, dtype=np.float64)
    if len(residuals) != len(patient_ids) or not 0 < coverage <= 1:
        raise ValueError("invalid conformal inputs")
    maxima = [max(residuals[index] for index, observed in enumerate(patient_ids) if observed == patient)
              for patient in sorted(set(patient_ids), key=str)]
    return _higher(maxima, coverage)


def calibrate_bounds(predicted_gain, true_gain, predicted_harm, true_harm, patient_ids, coverage=0.90):
    predicted_gain = np.asarray(predicted_gain, dtype=np.float64)
    true_gain = np.asarray(true_gain, dtype=np.float64)
    predicted_harm = np.asarray(predicted_harm, dtype=np.float64)
    true_harm = np.asarray(true_harm, dtype=np.float64)
    q_gain = patient_quantile(predicted_gain - true_gain, patient_ids, coverage)
    q_harm = patient_quantile(true_harm - predicted_harm, patient_ids, coverage)
    return q_gain, q_harm


def bounds(predicted_gain, predicted_harm, q_gain, q_harm):
    return (np.asarray(predicted_gain, dtype=np.float64) - q_gain,
            np.asarray(predicted_harm, dtype=np.float64) + q_harm)
