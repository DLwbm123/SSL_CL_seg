"""Algebra-only checks for the proposed finite-response DPR update.

No patient data, model checkpoints, network training, or optimizer steps are used.
NumPy is the only dependency. The counterexample is deliberately retained: a
sketched decrease is not a full-response decrease, even for a linear map.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def correction(g: np.ndarray, R: np.ndarray, projected_residual: np.ndarray,
               lam: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum correction with g.T @ correction == 0.

    R contains two response VJP rows evaluated at the native Adam proposal;
    projected_residual uses the SAME frozen output probes and unnormalized units.
    Both R and projected_residual are divided by the Frobenius norm of R.
    """
    g = np.asarray(g, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    q = np.asarray(projected_residual, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] != g.size or q.shape != (R.shape[0],):
        raise ValueError('incompatible vector/row shapes')
    if not (0.0 <= lam <= 1.0) or not all(np.isfinite(x).all() for x in (g, R, q)):
        raise ValueError('nonfinite input or invalid lambda')
    n = np.linalg.norm(R)
    if np.linalg.norm(g) <= 1e-30 or n <= 1e-30 or lam == 0:
        return np.zeros_like(g), np.zeros_like(R), np.zeros_like(q)
    J, b = R / n, q / n
    e = g / np.linalg.norm(g)
    A = J - np.outer(J @ e, e)
    eta = -lam * A.T @ np.linalg.solve(np.eye(R.shape[0]) + lam * A @ A.T, b)
    return eta, J, b


def main() -> dict:
    rng = np.random.default_rng(2026091206)
    errors, residuals, objective_errors = [], [], []
    for i in range(80):
        n = 3 + i % 17
        g, R, q = rng.normal(size=n), rng.normal(size=(2, n)), rng.normal(size=2)
        lam = (i % 10 + 1) / 10
        eta, J, b = correction(g, R, q, lam)
        K = np.block([[np.eye(n) + lam * J.T @ J, g[:, None]],
                      [g[None, :], np.zeros((1, 1))]])
        target = np.concatenate([-lam * J.T @ b, [0.]])
        eta_kkt = np.linalg.solve(K, target)[:n]
        errors.append(float(np.max(np.abs(eta - eta_kkt))))
        residuals.append(float(abs(g @ eta) / max(np.linalg.norm(g) * np.linalg.norm(eta), 1e-30)))
        o0 = .5 * lam * float(b @ b)
        o1 = .5 * float(eta @ eta) + .5 * lam * float((b + J @ eta) @ (b + J @ eta))
        objective_errors.append(o1 - o0)
    assert max(errors) < 1e-12
    assert max(residuals) < 1e-12
    assert max(objective_errors) <= 1e-12

    # A linear map with two contrast blocks; block 2 has zero drift.
    H = np.array([[1., 0., 0.], [0., 100., 0.], [0., 0., 0.]])
    d0 = np.array([1., .0001, -.1])
    g = np.array([0., 0., 1.])
    drift = H @ d0
    v = drift / np.linalg.norm(drift)
    R = np.stack([v[:2] @ H[:2], np.zeros(3)])
    q = np.array([v[:2] @ drift[:2], 0.])
    eta, J, b = correction(g, R, q)
    raw_energy = float(drift @ drift)
    candidate_drift = H @ (d0 + eta)
    candidate_energy = float(candidate_drift @ candidate_drift)
    accepted = candidate_energy <= raw_energy
    applied = d0 + eta if accepted else d0.copy()
    assert candidate_energy > raw_energy  # do not incorrectly assert full-field monotonicity
    assert not accepted and np.array_equal(applied, d0)
    assert abs(g @ (applied - d0)) < 1e-15
    assert np.linalg.norm(b + J @ eta) < np.linalg.norm(b)

    # The sign control changes alignment but not pointwise absolute coefficients.
    delta = rng.normal(size=(2, 37))
    aligned = delta / np.linalg.norm(delta)
    scrambled = aligned * rng.choice([-1., 1.], size=aligned.shape)
    assert np.array_equal(np.abs(aligned), np.abs(scrambled))
    assert abs(float(np.sum(aligned * delta)) - np.linalg.norm(delta)) < 1e-12

    result = {
        'scope': 'local NumPy float64 algebra only; no patient data, no neural-network or server training',
        'kkt_cases': 80,
        'max_correction_vs_kkt_absolute_error': max(errors),
        'max_normalized_progress_correction_residual': max(residuals),
        'max_objective_difference': max(objective_errors),
        'full_field_counterexample': {
            'linear_output_map': H.tolist(), 'native_increment': d0.tolist(),
            'correction': eta.tolist(), 'raw_squared_response': raw_energy,
            'candidate_squared_response': candidate_energy,
            'candidate_sampled_response_ratio': float(np.linalg.norm(b + J @ eta) / np.linalg.norm(b)),
            'candidate_full_response_ratio': float(np.sqrt(candidate_energy/raw_energy)),
            'finite_guard_rejects': not accepted, 'applied_equals_native': bool(np.array_equal(applied,d0)),
        },
        'sign_control_preserves_absolute_probe_weights': True,
        'algebra_tests_pass': True,
        'scientific_accuracy_tested': False,
        'formal_training_updates': 0,
    }
    output = Path(__file__).with_name('DPR_Finite_V0_2_Math_Check_Results.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
