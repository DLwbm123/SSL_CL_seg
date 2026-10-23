"""Generalized joint variational learning, with image-only amortized prediction."""
import torch
from ..objective import SequentialObjective


class UnifiedObjective(SequentialObjective):
    def estimate(self, batchL, batchU, counts, generator, sampling_weights=None):
        if sampling_weights is not None:
            raise ValueError('only uniform slice sampling is registered')
        s = self.state
        s.assert_active()
        pixels = s.config.height * s.config.width
        values = []
        for _ in range(s.config.global_mc):
            theta = s.q.rsample(generator)
            a = s.current.rsample(theta, generator)
            total = theta.sum() * 0
            for role, batch in [('L', batchL), ('U', batchU)]:
                if not counts[role]:
                    assert batch is None
                    continue
                s.check_batch(batch, role)
                x = batch['x']
                context = s.phi.context(x, theta, a)
                offsets = s.phi.initial_offsets(len(x), x)
                for _ in range(s.config.local_mc):
                    z, logq = s.phi.rsample_and_log_prob(context, s.phi.noise(len(x), x, generator), offsets, s.model, theta)
                    parts = s.model.log_joint(x, z, theta, a, **batch.get('observations', {}))
                    image_elbo = parts.total - parts['y'] - logq
                    total = total - (counts[role] / counts['L']) * (.01 * image_elbo + parts['y']).mean() / (pixels * s.config.local_mc)
            values.append(total)
        gkl, dkl = s.q.kl_to(s.prior), s.current.expected_kl(s.q)
        fit = torch.stack(values).mean()
        loss = fit + (gkl + dkl) / (counts['L'] * pixels)
        if not torch.isfinite(loss):
            raise FloatingPointError('joint variational objective')
        return loss, dict(scaled=loss.detach(), fit=fit.detach(), global_kl=gkl.detach(), domain_kl=dkl.detach())


@torch.no_grad()
def predict(x, state, domain, seed, outer=8, local=4):
    """Uniform variational integration, NOT self-normalized importance sampling."""
    rng = torch.Generator().manual_seed(seed)
    result = torch.zeros((len(x), state.config.classes, *x.shape[-2:]), device=x.device, dtype=x.dtype)
    memory = state.memories[domain]
    for _ in range(outer):
        theta = state.q.rsample(rng)
        a = memory.rsample(theta, rng)
        for index in range(len(x)):
            image = x[index:index+1]
            context = state.phi.context(image, theta, a)
            offsets = state.phi.initial_offsets(1, image)
            for _ in range(local):
                z, _ = state.phi.rsample_and_log_prob(context, state.phi.noise(1, image, rng), offsets, state.model, theta)
                result[index:index+1] += state.model.class_probabilities(z, theta) / (outer * local)
    return result
