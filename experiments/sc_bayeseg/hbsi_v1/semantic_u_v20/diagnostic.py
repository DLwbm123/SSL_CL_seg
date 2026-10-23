"""Bounded training-only probes; donor checkpoints and original states are immutable."""
import copy
import math
import torch
from .method import components, posterior_probability, rng
from ..stream_state import StreamState, digest
from ..paired_semantic_v19.method import balanced_risk
from ..paired_semantic_v19.run import data


class ReverseBudget:
    def __init__(self):
        self.calls = 0

    def grad(self, loss, parameters, **kwargs):
        self.calls += 1
        if self.calls > 256:
            raise RuntimeError('diagnostic reverse cap')
        result = torch.autograd.grad(loss, parameters, allow_unused=True, **kwargs)
        if any(g is not None and not torch.isfinite(g).all() for g in result):
            raise FloatingPointError('diagnostic gradient')
        return result


def batch(d, role, seed):
    b = d.batch(role, 1, torch.Generator().manual_seed(seed))
    b['x'] = b['x'].cuda()
    if 'observations' in b:
        b['observations'] = {k: v.cuda() for k, v in b['observations'].items()}
    return b


def parameters(s):
    return [p for g in s.optimizer.param_groups for p in g['params']]


def groups(s, grads):
    out = {}; params = parameters(s)
    names = {id(p): name for name, p in s.phi.named_parameters()}
    cut = sum(math.prod(shape) for name, shape in s.q.layout if name.startswith('g.'))
    for p, grad in zip(params, grads):
        grad = torch.zeros_like(p) if grad is None else grad
        if p is s.q.mean or p is s.q.raw_scale:
            suffix = 'mean' if p is s.q.mean else 'scale'
            out['G_' + suffix] = [grad[:cut].flatten()]
            out['F_' + suffix] = [grad[cut:].flatten()]
        else:
            name = names.get(id(p))
            key = 'domain' if name is None else ('phi_shared' if name.startswith(('enc.', 'down.', 'up.')) else ('phi_cR' if name.startswith(('c_head.', 'r_head.')) else 'phi_Sb'))
            out.setdefault(key, []).append(grad.flatten())
    return {k: torch.cat(v) for k, v in out.items()}


def metrics(s, b):
    with torch.no_grad():
        p = posterior_probability(s, b['x'], tag=110)
        y = b['observations']['y_obs'].to(p.dtype)
        fg, gt = p.argmax(1) == 1, y[:, 1].bool()
        tp = (fg & gt).sum().double()
        soft = 1 - (2*(p[:, 1]*y[:, 1]).sum()+1e-6)/(p[:, 1].sum()+y[:, 1].sum()+1e-6)
        return dict(risk=float(balanced_risk(p, y).mean()), soft_dice_loss=float(soft), precision=float(tp/fg.sum().clamp_min(1)), recall=float(tp/gt.sum().clamp_min(1)))


def conditional_probe(s, L, budget):
    """Imposed Pi; optimize only S/b variational offsets with identical fixed noise."""
    with torch.no_grad():
        gen = rng(s, 150)
        theta = s.q.rsample(gen).detach()
        a = s.current.rsample(theta, gen).detach()
        context = s.phi.context(L['x'], theta, a).detach()
    noise = s.phi.noise(1, L['x'], rng(s, 151))
    truth = L['observations']['y_obs'].to(L['x'].dtype) * .98 + .01
    original = s.model.class_probabilities
    results = {}
    try:
        for name, prob in [('aligned', truth), ('shift32', truth.roll(32, -1))]:
            s.model.class_probabilities = lambda z, theta, value=prob: value
            offsets = {k: v.detach().clone().requires_grad_(k in ('S', 'b')) for k, v in s.phi.initial_offsets(1, L['x']).items()}
            params = [offsets[k] for k in ('S', 'b')]
            history = []
            for step in range(5):
                z, logq = s.phi.rsample_and_log_prob(context, noise, offsets, s.model, theta)
                parts = s.model.log_joint(L['x'], z, theta, a)
                loss = -(parts.total - logq).mean() / (s.config.height*s.config.width)
                history.append(float(loss.detach()))
                if step < 4:
                    grads = budget.grad(loss, params)
                    with torch.no_grad():
                        for p, grad in zip(params, grads):
                            p.add_(grad, alpha=-.01)
            results[name] = dict(full_negative_elbo_per_pixel=history, local_updates=4)
    finally:
        s.model.class_probabilities = original
    return results


def run(p, source, write_result):
    budget = ReverseBudget(); rows = []; candidates = 0
    for arm in ('L', 'LU'):
        for step in (400, 800, 1600):
            s = StreamState.load(source / f'results/s20261023_{arm}_A_attempt001/ACTIVE_STEP_{step:06}.pt').to_device('cuda')
            before = digest(s.payload())
            # Expand only the in-memory probe evidence to permit current-domain U.
            d = data(p, s.config, 'RUNMC', 'LU')
            saved_active = copy.deepcopy(s.active)
            s.active['evidence'] = copy.deepcopy(d.evidence)
            probe_before = digest(s.payload())
            for index in range(2):
                L, U = batch(d, 'L', 9100+index), batch(d, 'U', 9200+index)
                probe = batch(d, 'L', 9300+index)
                terms = components(s, L, U, d.counts)
                ps = parameters(s)
                gradients = {name: budget.grad(loss, ps, retain_graph=True) for name, loss in terms.items()}
                tg, ug = groups(s, gradients['task']), groups(s, gradients['image_U'])
                summary = {}
                for key in tg:
                    t, u = tg[key], ug[key]
                    nt, nu = t.norm(), u.norm()
                    summary[key] = dict(task_norm=float(nt), U_norm=float(nu), ratio=float(nu/(nt+1e-12)), cosine=float(torch.dot(t,u)/(nt*nu+1e-12)))
                # Discard graphs before cloning candidate states.
                scalars = {k: float(v.detach()) for k,v in terms.items()}
                del terms
                outcome = {'unchanged': metrics(s, probe)}
                for use_u in (False, True):
                    clone = copy.deepcopy(s)
                    clone.optimizer.zero_grad(set_to_none=True)
                    for j,param in enumerate(parameters(clone)):
                        pieces = [gradients[k][j] for k in ('task','image_L','kl') + (('image_U',) if use_u else ())]
                        param.grad = sum((g for g in pieces if g is not None), torch.zeros_like(param))
                    clone.optimizer.step(); candidates += 1
                    outcome['LU' if use_u else 'L'] = metrics(clone, probe)
                    del clone
                row = dict(arm=arm, snapshot_step=step, batch_index=index, weighted_terms=scalars, parameter_groups=summary, adam_candidates=outcome)
                if step in (400,1600) and index == 0:
                    row['conditional_semantic_probe'] = conditional_probe(s, L, budget)
                assert digest(s.payload()) == probe_before, 'probe mutated original in-memory state'
                assert all(v.grad is None for v in ps), 'probe leaked gradients'
                rows.append(row)
                write_result(dict(status='RUNNING',reverse_calls=budget.calls,completed_probes=len(rows)))
                del gradients
            s.active = saved_active
            assert digest(s.payload()) == before
            del s
    return dict(status='COMPLETE', rows=rows, reverse_calls=budget.calls, reverse_limit=256, donor_optimizer_updates=0, candidate_adam_steps_discarded=candidates, payload_unchanged=True, diagnostic_training_only=True, posterior_streams='v20 split common draws, not replayed v19 draws')
