from dataclasses import dataclass
import math
import torch
from .distributions import finite
from .local_inference import Refiner
from .stream_state import digest


@dataclass(frozen=True)
class InferenceBudget:
    S: int = 8
    L: int = 4
    K: int = 3
    repeats: int = 3
    seed: int = 8191
    local_ess_fraction: float = .1
    global_ess_fraction: float = .25
    max_probability_range: float = .1
    max_log_evidence_range: float = 1.
    max_local_weight: float = .5
    max_global_weight: float = .5
    integrator: str = 'factorized'

    def __post_init__(self):
        if min(self.S, self.L, self.repeats) < 1 or self.K < 0:
            raise ValueError('invalid MC budget')
        if self.integrator not in ('paired', 'factorized'):
            raise ValueError('invalid volume integrator')
        if any(not 0 < v <= 1 for v in (self.local_ess_fraction, self.global_ess_fraction,
                                        self.max_local_weight, self.max_global_weight)):
            raise ValueError('invalid inference gate')


def ess(logw, dim=-1):
    finite(logw, 'importance weights')
    weights = torch.softmax(logw, dim)
    return 1 / weights.square().sum(dim)


def nondegenerate(logw, fraction, max_weight, dim=-1):
    value = ess(logw, dim)
    largest = logw.softmax(dim).max(dim).values
    return (value > 1 + 1e-8) & (value / logw.shape[dim] >= fraction) & (largest <= max_weight)


def combine_slice_banks(logweights, probabilities, integrator='factorized'):
    """Per domain [S,V,L], [S,V,L,K,H,W]; all slices share each outer Theta/a draw."""
    if not logweights or len(logweights) != len(probabilities):
        raise ValueError('empty/mismatched domains')
    for w, p in zip(logweights, probabilities):
        if w.ndim != 3 or p.ndim != 6 or p.shape[:3] != w.shape or min(w.shape) < 1:
            raise ValueError('slice bank dimensions')
        finite(w, 'importance weights')
        finite(p, 'class probabilities')
    if integrator == 'paired':
        result, diag = combine_domains([w.sum(1) for w in logweights], [p.transpose(1, 2) for p in probabilities])
        volume_logz = [torch.logsumexp(w.sum(1), -1) - math.log(w.shape[-1]) for w in logweights]
    elif integrator == 'factorized':
        volume_logz = [(torch.logsumexp(w, -1) - math.log(w.shape[-1])).sum(1) for w in logweights]
        outer_logw = torch.cat([v - math.log(len(v)) - math.log(len(volume_logz)) for v in volume_logz])
        outer_w = outer_logw.softmax(0)
        local_p = torch.cat([torch.einsum('svl,svlkhw->svkhw', w.softmax(-1), p)
                             for w, p in zip(logweights, probabilities)])
        result = torch.einsum('s,svkhw->vkhw', outer_w, local_p)
        sizes = [len(v) for v in volume_logz]
        domain_p = torch.stack([v.sum() for v in outer_w.split(sizes)])
        diag = dict(log_evidence=torch.logsumexp(outer_logw, 0), domain_posterior=domain_p,
                    max_weight=outer_w.max(), combined_ess=1 / outer_w.square().sum())
    else:
        raise ValueError('unknown integrator')
    diag.update(local_ess=[ess(w) for w in logweights],
                local_max_weight=[w.softmax(-1).max(-1).values for w in logweights],
                global_ess=[ess(w) for w in volume_logz],
                global_max_weight=[w.softmax(-1).max() for w in volume_logz],
                volume_log_evidence=volume_logz, integrator=integrator,
                all_domains_retained=True)
    return result, diag


def combine_domains(logweights, probabilities):
    """Each domain has [S,L] joint-volume logweights; probabilities [S,L,V,K,H,W]."""
    if not logweights or len(logweights) != len(probabilities):
        raise ValueError('empty/mismatched domains')
    flatw = torch.cat([v.flatten() - math.log(v.numel()) - math.log(len(logweights)) for v in logweights])
    finite(flatw, 'domain weights')
    norm = flatw.softmax(0)
    flatp = torch.cat([v.flatten(0, 1) for v in probabilities])
    result = torch.einsum('n,nvkhw->vkhw', norm, flatp)
    domain = []; offset = 0
    for v in logweights:
        domain.append(norm[offset:offset + v.numel()].sum())
        offset += v.numel()
    return result, dict(log_evidence=torch.logsumexp(flatw, 0), domain_posterior=torch.stack(domain),
                        max_weight=norm.max(), combined_ess=1 / norm.square().sum(),
                        local_ess=[ess(v, 1) for v in logweights],
                        global_ess=[ess(torch.logsumexp(v, 1) - math.log(v.shape[1]), 0) for v in logweights])


class Predictor:
    def __init__(self, counter=None):
        from . import budgets
        self.counter = counter if counter is not None else budgets.ACTIVE_COUNTER

    def _one(self, x, state, budget, seed, slice_chunk):
        fit = torch.Generator(device=x.device).manual_seed(seed)
        estimate = torch.Generator(device=x.device).manual_seed(seed + 1000003)
        outer = torch.Generator(device=x.device).manual_seed(seed + 2000003)
        refiner = Refiner(state.model, state.phi, self.counter)
        memories = list(state.memories.values())
        if not memories:
            raise ValueError('prediction requires sealed domain memories')
        allweights = []; allprobs = []; delta_variances = []
        for memory in memories:
            ws = []; ps = []; variances = []
            for _ in range(budget.S):
                theta = state.q.rsample(outer).detach()
                a = memory.rsample(theta, outer).detach()
                if self.counter is not None:
                    self.counter.add('global_mc_samples', 1)
                    self.counter.add('domain_mc_samples', 1)
                slice_w = []; slice_p = []
                for start in range(0, len(x), slice_chunk):
                    for index in range(start, min(start + slice_chunk, len(x))):
                        image = x[index:index + 1]
                        ctx, offsets = refiner.refine(image, theta, a, K=budget.K, training=False, generator=fit)
                        weights = []; probs = []
                        for _ in range(budget.L):
                            noise = state.phi.noise(1, image, estimate)
                            logw, z, _, _ = refiner.draw(image, theta, a, ctx, offsets, noise)
                            weights.append(logw[0].detach())
                            probs.append(state.model.class_probabilities(z, theta)[0].detach())
                        slice_w.append(torch.stack(weights))
                        slice_p.append(torch.stack(probs))
                # l indexes a joint proposal sample across ALL slices.
                weights = torch.stack(slice_w)
                ws.append(weights)
                ps.append(torch.stack(slice_p))
                variances.append(weights.sum(0).var(unbiased=False))
            allweights.append(torch.stack(ws)); allprobs.append(torch.stack(ps))
            delta_variances.append(torch.stack(variances))
        p, diag = combine_slice_banks(allweights, allprobs, budget.integrator)
        diag.update(delta_variance_fixed_theta=delta_variances,
                    streams={'fit': seed, 'importance': seed + 1000003, 'outer': seed + 2000003})
        return p, diag

    def predict_image_only(self, x, state, budget=None, slice_chunk=None):
        budget = budget or InferenceBudget()
        if x.ndim != 4 or len(x) == 0:
            raise ValueError('expected nonempty volume [slices,C,H,W]')
        slice_chunk = len(x) if slice_chunk is None else slice_chunk
        if slice_chunk < 1:
            raise ValueError('slice_chunk')
        before = digest(state.payload())
        # autograd.grad targets only rho. Persistent grads/buffers/optimizer untouched.
        with torch.enable_grad():
            results = [self._one(x, state, budget, budget.seed + r * 10000319, slice_chunk)
                       for r in range(budget.repeats)]
        if digest(state.payload()) != before:
            raise RuntimeError('persistent state changed during evaluation')
        ps = torch.stack([p for p, _ in results])
        logz = torch.stack([d['log_evidence'] for _, d in results])
        spread = (ps.max(0).values - ps.min(0).values).max()
        zspread = logz.max() - logz.min()
        local_ok = all(bool(((v > 1 + 1e-8) & (v / budget.L >= budget.local_ess_fraction) &
                             (m <= budget.max_local_weight)).all())
                       for _, d in results for v, m in zip(d['local_ess'], d['local_max_weight']))
        global_ok = all(bool((v > 1 + 1e-8) & (v / budget.S >= budget.global_ess_fraction) &
                              (m <= budget.max_global_weight))
                        for _, d in results for v, m in zip(d['global_ess'], d['global_max_weight']))
        passed = budget.repeats >= 3 and local_ok and global_ok and spread <= budget.max_probability_range and zspread <= budget.max_log_evidence_range
        return dict(probability=ps[0], repeats=[d for _, d in results], probability_range=spread,
                    log_evidence_range=zspread, status='SYNTHETIC_INFERENCE_PASS' if passed else 'INFERENCE_UNQUALIFIED',
                    formal_qualification='NOT_QUALIFIED', state_unchanged=True, budget=vars(budget),
                    gates={'local':local_ok,'global':global_ok,'all_domains_retained':True})
