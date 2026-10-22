import torch
from .distributions import finite


class Refiner:
    def __init__(self, model, recognition, counter=None):
        self.model = model
        self.recognition = recognition
        from . import budgets
        self.counter = counter if counter is not None else budgets.ACTIVE_COUNTER

    def draw(self, x, theta, a, context, offsets, noise, observations=None):
        z, logq = self.recognition.rsample_and_log_prob(context, noise, offsets, self.model, theta)
        obs = observations or {}
        parts = self.model.log_joint(x, z, theta, a, **obs)
        if self.counter is not None:
            self.counter.add('local_mc_samples', len(x))
            self.counter.add('joint_forwards', len(x))
        return finite(parts.total - logq, 'local ELBO'), z, parts, logq

    def refine(self, x, theta, a, observations=None, K=None, training=True, generator=None, noises=None):
        K = self.model.config.inner_steps if K is None else K
        if K < 0:
            raise ValueError('negative K')
        context = self.recognition.context(x, theta, a)
        offsets = self.recognition.initial_offsets(len(x), x)
        for step in range(K):
            noise = self.recognition.noise(len(x), x, generator) if noises is None else noises[step]
            elbo, _, _, _ = self.draw(x, theta, a, context, offsets, noise, observations)
            grad = torch.autograd.grad(elbo.sum(), tuple(offsets.values()), create_graph=training)
            offsets = {k: v + self.model.config.inner_lr * g for (k, v), g in zip(offsets.items(), grad)}
            for v in offsets.values():
                finite(v, 'local refinement')
            if not training:
                offsets = {k: v.detach().requires_grad_(True) for k, v in offsets.items()}
            if self.counter is not None:
                self.counter.add('refinement_sample_steps', len(x))
                self.counter.add('local_backward_calls', 1)
        return context, offsets
