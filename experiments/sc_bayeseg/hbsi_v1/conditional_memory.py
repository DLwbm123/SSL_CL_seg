import torch
from torch import nn
from .distributions import positive, inverse_positive, normal_log_prob, randn


class ConditionalDomainMemory(nn.Module):
    def __init__(self, projection, channels, floor=1e-6):
        super().__init__()
        self.projection = projection
        self.floor = floor
        self.m = nn.Parameter(projection.coefficient.new_zeros(3 * channels))
        self.C = nn.Parameter(self.m.new_zeros(3 * channels, projection.dimension))
        self.raw_scale = nn.Parameter(inverse_positive(torch.ones_like(self.m) * .1, floor))

    @property
    def sd(self):
        return positive(self.raw_scale, self.floor)

    def mean(self, theta):
        return self.m + self.projection(theta) @ self.C.T

    def rsample(self, theta, generator=None, noise=None):
        mean = self.mean(theta)
        if noise is None:
            noise = randn(mean.shape, mean, generator)
        return mean + self.sd * noise

    def log_prob(self, a, theta):
        return normal_log_prob(a, self.mean(theta), self.sd).sum(-1)

    def marginal(self, q):
        v = self.projection.variance(q.sd.square())
        return self.mean(q.mean), torch.diag(self.sd.square()) + (self.C * v) @ self.C.T

    def expected_kl(self, q, prior_sd=1.):
        mean, covariance = self.marginal(q)
        return (.5 * (covariance.diag() + mean.square()) / prior_sd ** 2
                - .5 + torch.log(self.sd.new_tensor(prior_sd) / self.sd)).sum()
