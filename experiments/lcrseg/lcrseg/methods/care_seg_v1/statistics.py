"""Patient-weighted labeled-component GMM, candidate risk and one temporal filter."""
from __future__ import annotations

import numpy as np
from care_hr_v0_7.ridge import fit_ridge, patient_proposal_weights


def weights(patients):
    return patient_proposal_weights(np.asarray(patients, dtype=str))


def weighted_median(x, w):
    order = np.argsort(x, kind='stable')
    return x[order[np.searchsorted(np.cumsum(w[order]), w.sum()*.5)]]


class StateGMM:
    """Each branch fails closed if any legal state lacks independent patient support."""
    def __init__(self, minimum_patients=2, iterations=25, shrinkage=.2):
        if minimum_patients < 2 or iterations < 1 or not 0 <= shrinkage <= 1:
            raise ValueError('invalid GMM settings')
        self.minimum_patients, self.iterations, self.shrinkage = minimum_patients, iterations, shrinkage
        self.branches = {}
        self.em_iterations = 0

    def fit(self, xb, state, agreement_b, patients_b, xu, agreement_u, patients_u, *, classes):
        xb, xu = np.asarray(xb, float), np.asarray(xu, float)
        state, agreement_b, agreement_u = np.asarray(state), np.asarray(agreement_b), np.asarray(agreement_u)
        if xb.ndim != 2 or xu.ndim != 2 or xb.shape[1] != 2 or xu.shape[1] != 2:
            raise ValueError('GMM needs N,2 evidence')
        if not np.isfinite(xb).all() or not np.isfinite(xu).all():
            raise ValueError('nonfinite GMM evidence')
        patients_b, patients_u = np.asarray(patients_b), np.asarray(patients_u)
        self.branches = {}
        # 0=neither, 1=semantic only, 2=predictive only, 3=both.
        for agree, legal in [(True, [0, 3]), (False, [1, 2] if classes == 2 else [0, 1, 2])]:
            b, u = agreement_b == agree, agreement_u == agree
            if np.any(~np.isin(state[b], legal)):
                raise ValueError('impossible correctness state')
            if any(len(set(patients_b[b & (state == z)])) < self.minimum_patients for z in legal):
                continue
            x = np.concatenate([xb[b], xu[u]])
            wb, wu = weights(patients_b[b]), weights(patients_u[u]) if u.any() else np.empty(0)
            wb /= wb.sum()
            if len(wu):
                wu /= wu.sum()
            w = np.concatenate([wb, wu])
            center = np.array([weighted_median(x[:, j], w) for j in range(2)])
            scale = np.array([max(1.4826*weighted_median(abs(x[:, j]-center[j]), w), .05) for j in range(2)])
            x = (x-center)/scale
            known = np.stack([state[b] == z for z in legal], axis=1).astype(float)
            means = np.stack([np.average(x[:len(wb)][known[:, k] > 0], axis=0, weights=wb[known[:, k] > 0]) for k in range(len(legal))])
            covariance = np.repeat(np.eye(2)[None], len(legal), axis=0)
            prior = np.full(len(legal), 1/len(legal))
            for _ in range(self.iterations):
                responsibility = self.posterior(x, means, covariance, prior)
                responsibility[:len(wb)] = known
                mass = (responsibility*w[:, None]).sum(0)
                if np.any(mass < 1e-10):
                    break
                prior = mass/mass.sum()
                means = ((responsibility*w[:, None]).T@x)/mass[:, None]
                for k in range(len(legal)):
                    delta = x-means[k]
                    cov = delta.T@((w*responsibility[:, k])[:, None]*delta)/mass[k]
                    covariance[k] = (1-self.shrinkage)*cov+self.shrinkage*np.diag(np.diag(cov))+np.eye(2)*1e-4
                self.em_iterations += 1
            if all(np.isfinite(t).all() for t in (means, covariance, prior)):
                self.branches[agree] = (legal, center, scale, means, covariance, prior)
        return self

    @staticmethod
    def posterior(x, means, covariance, prior):
        logp = []
        for mean, cov, pi in zip(means, covariance, prior):
            delta = x-mean
            logp.append(np.log(max(pi, 1e-12))-.5*(np.linalg.slogdet(cov)[1]+np.einsum('ni,ij,nj->n', delta, np.linalg.inv(cov), delta)))
        logp = np.stack(logp, 1)
        result = np.exp(logp-logp.max(1, keepdims=True))
        return result/result.sum(1, keepdims=True)

    def predict(self, x, agreement):
        x = np.asarray(x, float)
        if not np.isfinite(x).all():
            raise ValueError('nonfinite inference evidence')
        answer, valid = np.zeros((len(x), 4)), np.zeros(len(x), bool)
        for agree, (legal, center, scale, means, cov, prior) in self.branches.items():
            indices = np.flatnonzero(np.asarray(agreement) == agree)
            if not len(indices):
                continue
            normalized = (x[indices]-center)/scale
            post = self.posterior(normalized, means, cov, prior)
            # Reject far outside every fitted mode; do not turn OOD evidence into certainty.
            distance = np.stack([np.einsum('ni,ij,nj->n', normalized-m, np.linalg.inv(c), normalized-m) for m, c in zip(means, cov)])
            accepted = distance.min(0) <= 25
            answer[np.ix_(indices, legal)] = post
            valid[indices] = accepted
        return answer, valid


class CandidateRisk:
    """Shared per-candidate risks imply algebraically consistent pairwise differences."""
    def __init__(self, minimum_patients=2, ridge=1.):
        self.minimum_patients, self.ridge = minimum_patients, ridge
        self.models = [None]*3

    def fit(self, posterior, descriptors, losses, eligible, patients):
        for a in range(3):
            selected = eligible[:, a] & np.isfinite(losses[:, a])
            if len(set(np.asarray(patients)[selected])) < self.minimum_patients:
                continue
            x = np.concatenate([posterior[selected], descriptors[selected, a]], 1)
            self.models[a] = fit_ridge(x, losses[selected, a], weights(np.asarray(patients)[selected]), self.ridge)
        return self

    def predict(self, posterior, descriptors):
        output = np.full((len(posterior), 3), np.inf)
        for a, model in enumerate(self.models):
            if model is not None:
                output[:, a] = np.clip(model.predict(np.concatenate([posterior, descriptors[:, a]], 1)), 0, 1)
        return output


class TemporalRisk:
    """A single conservative risk filter, keyed by canonical location and target identity."""
    def __init__(self, momentum=.7):
        if not 0 <= momentum < 1:
            raise ValueError('invalid temporal momentum')
        self.momentum = momentum
        self.cache = {}

    def update(self, key, risks, identity):
        risks, identity = np.asarray(risks), np.asarray(identity)
        old = self.cache.get(key)
        result = risks.copy()
        if old is not None and old[0].shape == identity.shape:
            same = np.all(old[0] == identity, axis=-1)
            finite = np.isfinite(risks) & np.isfinite(old[1]) & same[:, None]
            result[finite] = np.maximum(risks[finite], self.momentum*old[1][finite]+(1-self.momentum)*risks[finite])
        self.cache[key] = (identity.copy(), result.copy())
        return result


def choose(risk, eligible, *, maximum, margin):
    """-1 is a fallback, never a zero-cost competing action. Order P,R,H."""
    risk = np.asarray(risk)
    allowed = np.asarray(eligible, bool) & np.isfinite(risk) & (risk <= maximum)
    result = np.full(len(risk), -1, dtype=np.int64)
    for i in range(len(risk)):
        p, r, h = allowed[i]
        incumbent = 2 if h else (0 if p else -1)
        if p and (incumbent < 0 or risk[i, 0]+margin < risk[i, incumbent]):
            incumbent = 0
        # Repair must demonstrate relative value even if P failed the absolute quality gate.
        if r and np.isfinite(risk[i, 0]) and risk[i, 1]+margin < risk[i, 0]:
            if incumbent < 0 or risk[i, 1]+margin < risk[i, incumbent]:
                incumbent = 1
        result[i] = incumbent
    return result
