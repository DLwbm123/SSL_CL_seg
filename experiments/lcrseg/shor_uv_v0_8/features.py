"""Prediction-only feature and deployment boundary. No labels or identities."""
import numpy as np
from care_hr_v0_7_1.actions import probability_pair

NAMES = [
    'current_rim_area_fraction','current_cup_area_fraction',
    'historical_rim_area_fraction','historical_cup_area_fraction',
    'rim_log_area_ratio','cup_log_area_ratio',
    'current_entropy','historical_entropy','current_max','historical_max',
    'current_margin','historical_margin','JS','hard_disagreement',
    'rim_softDice','cup_softDice','alpha_log_ratio','alpha_margin']


def extract(current, historical, alpha, historical_expert):
    current, historical = probability_pair(current, historical)
    a = np.asarray(alpha, dtype=np.float64)
    if (a.shape != (3,) or not np.isfinite(a).all() or (a < 0).any()
            or (a > 1).any() or not np.isclose(a.sum(), 1, rtol=1e-5, atol=1e-7)
            or historical_expert not in (0, 1)):
        raise ValueError('invalid frozen alpha/expert')
    # Argmax before casting preserves the original input convention.
    ch, hh = current.argmax(0), historical.argmax(0)
    c, h = current.astype(np.float64), historical.astype(np.float64)
    u = (ch > 0) | (hh > 0)
    if not u.any(): u = np.ones(ch.shape, dtype=bool)
    ca = np.array([(ch == k).sum() for k in (1, 2)], dtype=np.float64)
    ha = np.array([(hh == k).sum() for k in (1, 2)], dtype=np.float64)
    def entropy(p):
        logp = np.zeros_like(p)
        np.log(p, out=logp, where=p > 0)
        return -(p * logp).sum(0)
    ce, he = entropy(c), entropy(h)
    cs, hs = np.sort(c, axis=0), np.sort(h, axis=0)
    sd = []
    for k in (1, 2):
        den = c[k].sum() + h[k].sum()
        sd.append(1.0 if den == 0 else 2 * (c[k] * h[k]).sum() / den)
    av = np.sort(a)
    result = np.array([
        * (ca/ch.size), * (ha/ch.size), *np.log((ha+1)/(ca+1)),
        ce[u].mean(), he[u].mean(), cs[-1][u].mean(), hs[-1][u].mean(),
        (cs[-1]-cs[-2])[u].mean(), (hs[-1]-hs[-2])[u].mean(),
        (entropy((c+h)/2)-(ce+he)/2)[u].mean(), (ch[u] != hh[u]).mean(),
        *sd, np.log(a[historical_expert]+1e-12)-np.log(a[2]+1e-12), av[-1]-av[-2]
    ], dtype=np.float64)
    if result.shape != (18,) or not np.isfinite(result).all():
        raise ValueError('invalid features')
    return result


def predict(model, x):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != 18 or not np.isfinite(x).all(): raise ValueError('invalid feature matrix')
    z = (x-np.asarray(model['mean']))/np.asarray(model['scale'])
    y = np.einsum('...i,ij->...j', z, np.asarray(model['coef']), optimize=False) + np.asarray(model['intercept'])
    if not np.isfinite(y).all(): raise ValueError('nonfinite head prediction')
    return np.clip(y, [-1, -1, 0], [1, 1, 1])


def choose(route, x, model, candidate):
    """Return only frozen route or current; accepts no GT/domain/identity."""
    if route not in (0, 1, 2): raise ValueError('invalid route')
    if route == 2 or candidate is None: return 2
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (18,) or not np.isfinite(x).all(): raise ValueError('invalid feature')
    if candidate['kind'] == 'confidence':
        accept = candidate['threshold'] is None or x[9]-x[8] >= candidate['threshold']
    else:
        y = predict(model, x)
        accept = (y[0]+y[1])/2 > candidate['epsilon']
        if candidate['kind'] == 'cf': accept = accept and y[2] <= candidate['harm_limit']
    return int(route if accept else 2)


def deploy(current, historical, alpha, route, model, candidate):
    """Returned object is the original probability array, never a mixture."""
    if route == 2: return current
    decision = choose(route, extract(current, historical, alpha, route), model, candidate)
    return historical if decision == route else current
