"""Explicit composite objective; does not claim the original Bayesian posterior."""
import torch
from .semantic_initialization import SemanticCurriculum
from .objective import SequentialObjective


class SemanticJoint(SemanticCurriculum):
    def __init__(self, state, refiner, transition=1200, project_generator=False):
        super().__init__(state, refiner, warm_steps=10**9, boundary=transition, balanced_ce=True)
        self.transition = transition
        self.project_generator = project_generator

    def estimate(self, batchL, batchU, counts, generator, sampling_weights=None):
        semantic, parts = super().estimate(batchL, batchU, counts, generator, sampling_weights)
        if self.state.steps < self.transition:
            return semantic, parts
        elbo, probabilistic = SequentialObjective.estimate(self, batchL, batchU, counts, generator, sampling_weights)
        # One prespecified dataset-pixel denominator for the entire ELBO,
        # including KL. Preserve relative probabilistic terms, no tuning grid.
        denominator = sum(counts.values()) * self.state.config.height * self.state.config.width
        if denominator <= 0:
            raise ValueError('empty training population')
        normalized = elbo / denominator
        total = semantic + normalized
        projection = {}
        if self.project_generator:
            mean = self.state.q.mean
            gs, = torch.autograd.grad(semantic, mean, retain_graph=True)
            gp, = torch.autograd.grad(normalized, mean, retain_graph=True)
            # Semantic context is detached, so gs is zero on F. Only G's
            # opposing probabilistic component is removed; other gradients
            # (including recognition and global variance) are unchanged.
            dot = (gs * gp).sum()
            correction = -dot.clamp_max(0) / gs.square().sum().clamp_min(1e-30) * gs
            residual = (gs * (gp + correction)).sum()
            if residual < -1e-10 * (1 + gs.norm()*gp.norm()):
                raise FloatingPointError('generator projection algebra')
            # Zero-valued surrogate changes only the mean gradient.
            total = total + ((mean - mean.detach()) * correction.detach()).sum()
            projection = dict(generator_dot=dot.detach(), projected_dot=residual.detach(),
                              projection_applied=(dot < 0).to(dot.dtype))
        return total, dict(parts, scaled=total.detach(), normalized_elbo=normalized.detach(),
                           joint_phase=1, **projection, **{'prob/'+k:v for k,v in probabilistic.items()})
