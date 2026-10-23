"""Restore the historical segmentation risk; retain all v18 posterior terms."""
import torch
from ..objective import SequentialObjective


def balanced_risk(prob, truth):
    counts = truth.flatten(2).sum(2)
    terms = -(truth * prob.clamp_min(1e-12).log()).flatten(2).sum(2)
    present = counts > 0
    ce = ((terms / counts.clamp_min(1)) * present).sum(1) / present.sum(1)
    fg, y = prob[:, 1], truth[:, 1]
    dice = 1 - (2 * (fg*y).flatten(1).sum(1) + 1e-6) / (fg.flatten(1).sum(1) + y.flatten(1).sum(1) + 1e-6)
    return ce + dice


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
                    risk = image_elbo * 0
                    if role == 'L':
                        prob = s.model.class_probabilities(z, theta)
                        truth = batch['observations']['y_obs'].to(prob.dtype)
                        if not batch['observations']['obs_mask'].all() or not (truth.sum(1) == 1).all():
                            raise PermissionError('complete leaf labels required')
                        risk = balanced_risk(prob, truth)
                    total = total + (counts[role] / counts['L']) * (risk - .01 * image_elbo / pixels).mean() / s.config.local_mc
            values.append(total)
        gkl, dkl = s.q.kl_to(s.prior), s.current.expected_kl(s.q)
        fit = torch.stack(values).mean()
        loss = fit + (gkl + dkl) / (counts['L'] * pixels)
        if not torch.isfinite(loss):
            raise FloatingPointError('joint variational objective')
        return loss, dict(scaled=loss.detach(), fit=fit.detach(), global_kl=gkl.detach(), domain_kl=dkl.detach())


from ..unified_sslcl_v18.method import predict
