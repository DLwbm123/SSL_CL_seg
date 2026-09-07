"""CPU-only mathematical witness for the published SCD bracket exhaustion.

NOT a production projector, NOT a training runner, and NOT a repair of the
private failed batch. Requires numpy/scipy; compares with mpmath when available.
Uses the *already max-abs-normalized* direction employed by the frozen solver.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, softmax


def run() -> dict:
    p = np.array([1., 1e-30, 8e-48], dtype=np.float64)
    q = np.array([.9999999933333333, 3.3333333333333334e-9,
                  3.3333333333333334e-9], dtype=np.float64)
    a = p - np.array([1., 0., 0.])
    scale = float(np.max(np.abs(a)))
    a = a / scale
    b = float(np.dot(a, p))
    logq = np.log(q)

    def target(eta: float) -> np.ndarray:
        return softmax(logq - eta * a)

    def gap(eta: float) -> float:
        return float(np.dot(target(eta), a) - b)

    old_hi = float(2**60)
    old_gap = gap(old_hi)
    assert old_gap > 0, 'Published exhaustion witness was not reproduced'

    # For d_c=a_c-min(a)>=0, v=b-min(a)>0, Q0=sum(q[d=0]):
    # E_eta[d] <= sum(q*d)*exp(-eta*delta)/Q0,
    # delta=min(d[d>0]). The +log(2) gives a genuine feasible-end margin,
    # not a relaxed residual threshold.
    d = a - a.min()
    v = float(b - a.min())
    pos = d > 0
    assert v > 0 and np.any(pos)
    delta = float(d[pos].min())
    log_q0 = float(logsumexp(logq[~pos]))
    log_k = float(logsumexp(logq[pos] + np.log(d[pos])))
    upper = (log_k - log_q0 - math.log(v) + math.log(2.)) / delta
    assert gap(upper) <= 0
    lo, hi = 0., upper
    for _ in range(100):
        mid = lo + (hi - lo) / 2
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    r = target(hi)
    assert gap(hi) <= 0
    assert abs(float(r.sum()) - 1.) <= 1e-14
    assert np.isfinite(r).all() and (r >= 0).all()

    mp_result = {'status': 'NOT_RUN_MPMATH_UNAVAILABLE'}
    try:
        import mpmath as mp
        with mp.workdps(100):
            # Interpret the frozen float direction/threshold values as input;
            # do not silently replace p_y by 1-sum(p_non_y).
            ma = [mp.mpf(repr(float(z))) for z in a]
            mq = [mp.mpf(repr(float(z))) for z in q]
            mb = mp.mpf(repr(b))
            def mg(z):
                w = [mq[i] * mp.exp(-z * ma[i]) for i in range(3)]
                return sum(w[i] * ma[i] for i in range(3)) / sum(w) - mb
            ml, mh = mp.mpf('0'), mp.mpf(repr(upper))
            assert mg(ml) > 0 and mg(mh) < 0
            for _ in range(400):
                mm = (ml + mh) / 2
                if mg(mm) > 0:
                    ml = mm
                else:
                    mh = mm
            rel = abs(mp.mpf(repr(hi)) - mh) / mh
            assert rel < mp.mpf('1e-12')
            mp_result = {'status': 'PASS_100_DIGIT_REFERENCE',
                         'root': mp.nstr(mh, 35),
                         'float64_root_relative_error': float(rel)}
    except ImportError:
        pass

    return dict(status='PASS_SYNTHETIC_BRACKET_WITNESS_ONLY',
                inputs=dict(p=p.tolist(), q=q.tolist(), y=0),
                normalized_direction=a.tolist(), normalized_b=b,
                old_scaled_eta_upper=old_hi, old_gap=old_gap,
                analytical_scaled_eta_upper=upper,
                feasible_scaled_eta=hi, eta_over_old_cap=hi/old_hi,
                target=r.tolist(), normalized_residual=gap(hi),
                original_unscaled_eta=hi/scale,
                high_precision=mp_result,
                real_data_reads=0, private_failure_vectors_read=0,
                model_forwards=0, optimizer_steps=0,
                limitation='Only this published synthetic witness was checked. '
                           'Boundary cases, GPU production parity and full training '
                           'remain Codex qualification tasks.')


if __name__ == '__main__':
    result = run()
    out = Path(__file__).with_name('scd_bracket_witness_result.json')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
