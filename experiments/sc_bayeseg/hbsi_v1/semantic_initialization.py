"""Prespecified semantic initialization, followed by the unchanged ELBO.

This is an optimization curriculum, not an extra Bayesian likelihood or a
qualified posterior claim. The original prior is retained throughout.
"""
import torch
from .objective import SequentialObjective


class SemanticCurriculum(SequentialObjective):
    def __init__(self, state, refiner, warm_steps=800, boundary=800):
        super().__init__(state, refiner)
        self.warm_steps, self.boundary = warm_steps, boundary

    def estimate(self, batchL, batchU, counts, generator, sampling_weights=None):
        s = self.state
        # The transition is replayable from the durable step-800 snapshot. Both
        # arms reset moments here; prior and evidence are never reset.
        if s.steps == self.boundary:
            s.optimizer.state.clear()
        if s.steps >= self.warm_steps:
            loss, parts = super().estimate(batchL, batchU, counts, generator, sampling_weights)
            return loss, dict(parts, semantic_initialization=0)
        s.assert_active(); s.check_batch(batchL, 'L')
        x, obs = batchL['x'], batchL['observations']
        if not obs['obs_mask'].all() or not (obs['y_obs'].sum(1) == 1).all():
            raise PermissionError('semantic initialization requires complete leaf labels')
        theta = s.q.mean
        a = s.current.mean(theta).detach()
        context = s.phi.context(x, theta.detach(), a)
        noise = s.phi.noise(len(x), x, generator)
        offsets = s.phi.initial_offsets(len(x), x)
        z, _ = s.phi.rsample_and_log_prob(context, noise, offsets, s.model, theta)
        prob = s.model.class_probabilities(z, theta)
        truth = obs['y_obs'].to(prob.dtype)
        ce = -(truth * prob.clamp_min(1e-12).log()).sum(1).mean()
        fg, y = prob[:, 1], truth[:, 1]
        dice_loss = (1 - (2*(fg*y).flatten(1).sum(1)+1e-6) /
                     (fg.flatten(1).sum(1)+y.flatten(1).sum(1)+1e-6)).mean()
        loss = ce + dice_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('semantic loss')
        if self.refiner.counter is not None:
            self.refiner.counter.add('semantic_updates')
            self.refiner.counter.add('semantic_labelled_images', len(x))
        return loss, dict(semantic_initialization=1, ce=ce.detach(),
                         foreground_dice_loss=dice_loss.detach(), scaled=loss.detach())
