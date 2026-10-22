"""Explicit composite objective; does not claim the original Bayesian posterior."""
from .semantic_initialization import SemanticCurriculum
from .objective import SequentialObjective


class SemanticJoint(SemanticCurriculum):
    def __init__(self, state, refiner, transition=1200):
        super().__init__(state, refiner, warm_steps=10**9, boundary=transition, balanced_ce=True)
        self.transition = transition

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
        return total, dict(parts, scaled=total.detach(), normalized_elbo=normalized.detach(),
                           joint_phase=1, **{'prob/'+k:v for k,v in probabilistic.items()})
