"""Numerical patient-maximum split-conformal rank, not a method guarantee.

Exchangeability is an external assumption. OOF pooling, two separate heads,
adaptive action selection and domain shift do not inherit a joint guarantee.
This component is tested on synthetic residuals only in this work package.
"""
from decimal import Decimal, ROUND_CEILING
import math

import numpy as np


def aligned_vectors(*values):
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
    if not arrays or any(a.ndim != 1 for a in arrays):
        raise ValueError("expected one-dimensional vectors")
    if len({len(a) for a in arrays}) != 1 or any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("length mismatch or nonfinite input")
    return arrays


def patient_quantile(residuals, patient_ids, alpha=0.1):
    """Return (upper residual bound, accounting); never cap an infinite bound."""
    residuals, = aligned_vectors(residuals)
    patients = tuple(patient_ids)
    if len(patients) != len(residuals) or any(not isinstance(p, str) or not p for p in patients):
        raise ValueError("patient IDs must be aligned nonempty strings")
    if isinstance(alpha, bool) or not np.isscalar(alpha) or not math.isfinite(float(alpha)) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one")
    maxima = {}
    for index, patient in enumerate(patients):
        maxima[patient] = max(maxima.get(patient, -math.inf), float(residuals[index]))
    n = len(maxima)
    # Decimal evaluates the declared decimal alpha without a floating ceil off-by-one.
    k = int((Decimal(n + 1) * (Decimal(1) - Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))
    bound = math.inf if k > n else sorted(maxima.values())[k - 1]
    return bound, {"patients": n, "rows": len(residuals), "rank": k,
                   "patient_score": "maximum_residual",
                   "status": "INSUFFICIENT_PATIENTS_INFINITE_BOUND" if k > n else "FINITE_ORDER_STATISTIC",
                   "exchangeability_established": False, "method_risk_guarantee": False}


def calibrate_bounds(predicted_gain, true_gain, predicted_harm, true_harm, patient_ids, alpha=0.1):
    pg, tg, ph, th = aligned_vectors(predicted_gain, true_gain, predicted_harm, true_harm)
    patients = tuple(patient_ids)
    return (patient_quantile(pg - tg, patients, alpha),
            patient_quantile(th - ph, patients, alpha))
