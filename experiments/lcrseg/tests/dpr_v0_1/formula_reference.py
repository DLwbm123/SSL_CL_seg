"""Local algebra checks for the proposed descent-preserving response correction.

This file does not train a model, read patient data, or connect to a server.
Run: python DPR_Formula_Checks.py --output DPR_Formula_Check_Results.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def correct_step(d0: np.ndarray, gradient: np.ndarray, jacobian: np.ndarray,
                 lam: float = 1.0, preserve_progress: bool = True) -> np.ndarray:
    """Solve the small linear system without constructing a parameter-square matrix.

    d0 is the actual proposed parameter increment, not a raw gradient.
    jacobian has shape (number_of_probes, number_of_trainable_parameters).
    Inputs are treated as FP64 constants; no differentiation through the solve.
    """
    d0 = np.asarray(d0, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    j = np.asarray(jacobian, dtype=np.float64)
    if d0.ndim != 1 or gradient.shape != d0.shape or j.ndim != 2 or j.shape[1] != d0.size:
        raise ValueError("Expected vector increments/gradient and a compatible 2D Jacobian")
    if lam < 0 or not np.isfinite(lam) or not all(np.isfinite(x).all() for x in (d0, gradient, j)):
        raise ValueError("Nonfinite inputs or invalid regularization strength")
    if lam == 0 or j.size == 0 or np.linalg.norm(j) == 0:
        return d0.copy()
    if preserve_progress:
        norm_g = np.linalg.norm(gradient)
        if norm_g <= 1e-30:  # Explicit proposed execution policy: no correction on zero task gradient.
            return d0.copy()
        e = gradient / norm_g
        a = j - (j @ e)[:, None] * e[None, :]
    else:
        a = j
    mat = np.eye(j.shape[0]) + lam * (a @ a.T)
    rhs = j @ d0
    return d0 - lam * a.T @ np.linalg.solve(mat, rhs)


def dense_kkt(d0: np.ndarray, g: np.ndarray, j: np.ndarray, lam: float) -> np.ndarray:
    p = d0.size
    h = np.eye(p) + lam * (j.T @ j)
    kkt = np.block([[h, g[:, None]], [g[None, :], np.zeros((1, 1))]])
    rhs = np.r_[d0, g @ d0]
    return np.linalg.solve(kkt, rhs)[:p]


def run_checks() -> dict:
    gen = np.random.default_rng(2026091203)
    n, p, m, lam = 60, 37, 2, 1.0
    kkt_errors, progress_errors, response_ratios, correction_ratios, violations = [], [], [], [], []
    for _ in range(n):
        g = gen.standard_normal(p)
        d0 = -0.01 * g + 0.003 * gen.standard_normal(p)
        j = gen.standard_normal((m, p))
        j /= np.linalg.norm(j)
        d = correct_step(d0, g, j, lam)
        expected = dense_kkt(d0, g, j, lam)
        scale = np.linalg.norm(g) * np.linalg.norm(d0) + 1e-30
        progress_errors.append(float(abs(g @ (d - d0)) / scale))
        kkt_errors.append(float(np.max(np.abs(d - expected))))
        before, after = np.linalg.norm(j @ d0), np.linalg.norm(j @ d)
        response_ratios.append(float(after / before))
        correction_ratios.append(float(np.linalg.norm(d - d0) / np.linalg.norm(d0)))
        f0 = 0.5 * lam * before**2
        f1 = 0.5 * np.linalg.norm(d - d0)**2 + 0.5 * lam * after**2
        violations.append(float(f1 - f0))
    # Degenerate alignment: preserving task progress can make the probe unchangeable.
    g = gen.standard_normal(p); e = g / np.linalg.norm(g)
    d0 = -0.01 * g + 0.003 * gen.standard_normal(p)
    aligned = correct_step(d0, g, e[None, :], lam)
    j = gen.standard_normal((m, p)); j /= np.linalg.norm(j)
    # A truly output-preserving null direction is left alone by the unconstrained solve.
    null_d = gen.standard_normal(p)
    null_d -= j.T @ np.linalg.solve(j @ j.T, j @ null_d)
    null_after = correct_step(null_d, g, j, lam, False)
    zeros = correct_step(d0, g, np.zeros_like(j), lam)
    disabled = correct_step(d0, g, j, 0.0)
    no_task_gradient = correct_step(d0, np.zeros_like(g), j, lam)
    assert max(progress_errors) < 1e-12
    assert max(kkt_errors) < 1e-12
    assert max(response_ratios) <= 1 + 1e-12
    assert max(violations) <= 1e-12
    assert np.max(np.abs(aligned - d0)) < 1e-12
    assert np.max(np.abs(null_after - null_d)) < 1e-12
    assert np.array_equal(zeros, d0) and np.array_equal(disabled, d0)
    assert np.array_equal(no_task_gradient, d0)
    return {
        "status": "ALGEBRA_CHECKS_PASSED",
        "scope": "CPU NumPy FP64 synthetic vectors only; no patient data, model training, or server access",
        "seed": 2026091203, "random_cases": n, "parameters": p, "probes": m,
        "lambda": lam,
        "max_normalized_supervised_progress_residual": max(progress_errors),
        "max_absolute_difference_from_dense_KKT": max(kkt_errors),
        "response_norm_ratio_after_over_before_mean": float(np.mean(response_ratios)),
        "response_norm_ratio_after_over_before_max": max(response_ratios),
        "correction_norm_over_proposal_norm_mean": float(np.mean(correction_ratios)),
        "max_objective_difference_from_feasible_proposal": max(violations),
        "aligned_probe_max_absolute_step_change": float(np.max(np.abs(aligned - d0))),
        "zero_probe_and_lambda_zero_exact_identity": True,
        "no_task_gradient_policy_identity": True,
        "formal_optimizer_updates": 0,
        "limitations": [
            "Does not test PyTorch VJP construction or Adam integration",
            "Does not establish a nonlinear loss-decrease guarantee",
            "Does not establish old-domain preservation from current-domain probes",
            "Does not verify FP32 applied-weight residuals or GPU numerical behavior",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("DPR_Formula_Check_Results.json"))
    args = parser.parse_args()
    result = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
