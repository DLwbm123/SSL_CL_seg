"""Fixed CARe-HR inference feature schema."""
from __future__ import annotations

import math

import numpy as np

from .contracts import FEATURE_NAMES


def _validate_probability(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError("invalid probability array")
    if not np.allclose(value.sum(axis=0), 1.0, atol=1e-7):
        raise ValueError("probabilities must sum to one")
    return value


def _entropy(probability):
    return -np.sum(np.where(probability > 0, probability * np.log(probability + 1e-300), 0.0), axis=0)


def _margin(probability):
    ordered = np.sort(probability, axis=0)
    return ordered[-1] - ordered[-2]


def _compactness(mask):
    area = int(mask.sum())
    padded = np.pad(mask, 1)
    perimeter = sum(np.count_nonzero(padded[1:-1, 1:-1] & ~shift) for shift in (
        padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]))
    return 0.0 if perimeter == 0 else 4.0 * math.pi * area / (perimeter * perimeter)


def build_features(current_probability, historical_probability, proposals,
                   ppc_probability, ppc_consensus, shor_log_contrast, ridge_margin):
    current = _validate_probability(current_probability)
    historical = _validate_probability(historical_probability)
    if current.shape != historical.shape:
        raise ValueError("probability shapes differ")
    height, width = current.shape[1:]
    current_hard = np.argmax(current, axis=0)
    historical_hard = np.argmax(historical, axis=0)
    current_entropy = _entropy(current)
    historical_entropy = _entropy(historical)
    midpoint = 0.5 * (current + historical)
    js = 0.5 * np.sum(current * np.log((current + 1e-300) / (midpoint + 1e-300)), axis=0)
    js += 0.5 * np.sum(historical * np.log((historical + 1e-300) / (midpoint + 1e-300)), axis=0)
    current_margin = _margin(current)
    historical_margin = _margin(historical)
    disagreement = float(np.mean(current_hard != historical_hard))
    foreground = max(1, int(np.count_nonzero(current_hard)))
    center = np.asarray([(height - 1) / 2, (width - 1) / 2])
    radius = max(float(np.linalg.norm(center)), 1.0)
    shared = (float(ppc_probability), float(ppc_consensus),
              float(shor_log_contrast), float(ridge_margin))
    if not np.all(np.isfinite(shared)):
        raise ValueError("nonfinite external routing feature")
    rows = []
    for proposal in proposals:
        mask = np.asarray(proposal.mask, dtype=bool)
        if mask.shape != (height, width) or not mask.any():
            raise ValueError("invalid proposal mask")
        target = proposal.target_class
        delta = historical[target, mask] - current[target, mask]
        centroid = np.asarray([proposal.centroid_row, proposal.centroid_col])
        rows.append([
            float(target == 2),
            float(proposal.direction == "add"),
            math.log(proposal.area / (height * width)),
            math.log(proposal.area / foreground),
            _compactness(mask),
            float(np.linalg.norm(centroid - center) / radius),
            float(np.mean(current[target, mask])),
            float(np.mean(historical[target, mask])),
            float(np.mean(delta)),
            float(np.quantile(delta, 0.10)),
            float(np.mean(current_entropy[mask])),
            float(np.mean(historical_entropy[mask])),
            float(np.mean(js[mask])),
            float(np.mean(current_margin[mask])),
            float(np.mean(historical_margin[mask])),
            disagreement,
            *shared,
        ])
    output = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(FEATURE_NAMES))
    if output.shape[1] != 20 or not np.all(np.isfinite(output)):
        raise ValueError("invalid CARe-HR feature matrix")
    return output
