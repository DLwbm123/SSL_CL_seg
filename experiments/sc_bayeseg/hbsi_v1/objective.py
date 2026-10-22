import torch
from .distributions import finite


def sum_estimate(values, population, sampling_weights=None):
    if population == 0:
        if values.numel():
            raise ValueError('nonempty batch from empty population')
        return values.sum()
    if not values.numel():
        raise ValueError('positive population requires a batch')
    if sampling_weights is None:
        return population * values.mean()
    # Explicit with-replacement draw probabilities: weights = 1/(b*p_i).
    if sampling_weights.shape != values.shape or torch.any(sampling_weights <= 0):
        raise ValueError('invalid inverse probability weights')
    return (sampling_weights * values).sum()


class SequentialObjective:
    def __init__(self, state, refiner):
        self.state, self.refiner = state, refiner

    def estimate(self, batchL, batchU, counts, generator, sampling_weights=None):
        s = self.state
        s.assert_active()
        cfg = s.config
        estimates = []
        named = {}
        for _ in range(cfg.global_mc):
            theta = s.q.rsample(generator)
            a = s.current.rsample(theta, generator)
            if self.refiner.counter is not None:
                self.refiner.counter.add('global_mc_samples', 1)
                self.refiner.counter.add('domain_mc_samples', 1)
            total = theta.sum() * 0
            for role, batch in (('L', batchL), ('U', batchU)):
                if counts[role] == 0:
                    if batch is not None:
                        raise ValueError('unexpected batch')
                    continue
                if batch is None:
                    raise ValueError('missing batch')
                s.check_batch(batch, role)
                x = batch['x']
                obs = batch.get('observations') if role == 'L' else None
                context, offsets = self.refiner.refine(x, theta, a, obs, generator=generator)
                values = []
                for _ in range(cfg.local_mc):
                    noise = self.refiner.recognition.noise(len(x), x, generator)
                    e, _, parts, logq = self.refiner.draw(x, theta, a, context, offsets, noise, obs)
                    values.append(e)
                    for key, val in dict(parts, logq=logq).items():
                        named.setdefault(role + '/' + key, []).append(val.detach().mean())
                v = torch.stack(values).mean(0)
                weights = None if sampling_weights is None else sampling_weights.get(role)
                total = total + sum_estimate(v, counts[role], weights)
            estimates.append(total)
        gkl = s.q.kl_to(s.prior)
        dkl = s.current.expected_kl(s.q)
        likelihood = torch.stack(estimates).mean()
        raw = finite(gkl + dkl - likelihood, 'objective')
        return raw * cfg.common_scale, dict(global_kl=gkl.detach(), domain_kl=dkl.detach(),
            local_elbo_sum=likelihood.detach(), unscaled=raw.detach(), scaled=(raw * cfg.common_scale).detach(),
            **{k: torch.stack(v).mean() for k, v in named.items()})
