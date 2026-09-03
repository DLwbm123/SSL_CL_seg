"""Small, closed-form, float64 ridge helpers."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .contracts import RIDGE_LAMBDAS


def _finite(value, name):
    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"nonfinite {name}")
    return value


def grouped_folds(patient_ids, n_folds=5, salt="care-hr-v0.7"):
    unique = sorted(set(patient_ids), key=lambda item: (hashlib.sha256((salt + "\0" + str(item)).encode()).hexdigest(), str(item)))
    mapping = {patient: index % n_folds for index, patient in enumerate(unique)}
    return np.asarray([mapping[patient] for patient in patient_ids], dtype=np.int64)


def patient_proposal_weights(patient_ids, patient_weights=None):
    patients, counts = np.unique(np.asarray(patient_ids, dtype=object), return_counts=True)
    totals = {patient: 1.0 for patient in patients} if patient_weights is None else patient_weights
    if set(totals) != set(patients) or any(not np.isfinite(totals[p]) or totals[p] <= 0 for p in patients):
        raise ValueError("patient weights must be finite and positive")
    count = dict(zip(patients, counts))
    return np.asarray([totals[p] / count[p] for p in patient_ids], dtype=np.float64)


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    ridge_lambda: float

    def predict(self, features):
        value = _finite(features, "features")
        return self.intercept + ((value - self.mean) / self.scale) @ self.coefficient


def fit_ridge(features, target, weights, ridge_lambda):
    x = _finite(features, "features")
    y = _finite(target, "target").reshape(-1)
    w = _finite(weights, "weights").reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(w) or np.any(w <= 0):
        raise ValueError("invalid ridge inputs")
    total = float(w.sum())
    mean = np.sum(x * w[:, None], axis=0) / total
    variance = np.sum((x - mean) ** 2 * w[:, None], axis=0) / total
    scale = np.where(variance > 0, np.sqrt(variance), 1.0)
    z = (x - mean) / scale
    intercept = float(np.sum(y * w) / total)
    centered = y - intercept
    gram = z.T @ (w[:, None] * z) + float(ridge_lambda) * np.eye(x.shape[1])
    coefficient = np.linalg.solve(gram, z.T @ (w * centered))
    return RidgeModel(mean, scale, coefficient, intercept, float(ridge_lambda))


def select_lambda(features, target, patient_ids, lambda_grid=RIDGE_LAMBDAS, n_folds=5):
    x = _finite(features, "features")
    y = _finite(target, "target")
    folds = grouped_folds(patient_ids, n_folds=min(n_folds, len(set(patient_ids))))
    scores = []
    for ridge_lambda in lambda_grid:
        residual = []
        for fold in sorted(set(folds.tolist())):
            train = folds != fold
            valid = ~train
            model = fit_ridge(x[train], y[train], patient_proposal_weights(np.asarray(patient_ids)[train]), ridge_lambda)
            residual.extend((model.predict(x[valid]) - y[valid]).tolist())
        scores.append((float(np.mean(np.square(residual))), float(ridge_lambda)))
    best_score = min(score for score, _ in scores)
    return max(value for score, value in scores if abs(score - best_score) <= 1e-15)


def fit_dual_heads(features, gain, harm, patient_ids):
    weights = patient_proposal_weights(patient_ids)
    gain_lambda = select_lambda(features, gain, patient_ids)
    harm_lambda = select_lambda(features, harm, patient_ids)
    return (fit_ridge(features, gain, weights, gain_lambda),
            fit_ridge(features, harm, weights, harm_lambda))


def bayesian_patient_weights(patient_ids, replicate, seed=2026090701):
    patients = sorted(set(patient_ids), key=str)
    rng = np.random.Generator(np.random.PCG64(int(seed) + int(replicate)))
    draw = rng.dirichlet(np.ones(len(patients))) * len(patients)
    return dict(zip(patients, draw))


def prediction_accounting(predictions, eligible):
    predictions = np.asarray(predictions, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    validity = np.sum(np.isfinite(predictions), axis=0, dtype=np.int64)
    return validity, np.where(eligible, validity, 0)
