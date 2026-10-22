import copy
import hashlib
import json
import math
import torch
from torch import nn
from .distributions import positive, inverse_positive, normal_log_prob, randn


class GlobalPosterior(nn.Module):
    def __init__(self, layout, config, generator):
        super().__init__()
        self.layout = [(name, tuple(shape)) for name, shape in layout]
        self.floor = config.scale_floor
        dtype = getattr(torch, config.dtype)
        scale = torch.cat([torch.full((math.prod(s),), config.bias_prior_sd if n.endswith('.bias')
                                     else math.prod(s[1:]) ** -.5, dtype=dtype) for n, s in self.layout])
        self.register_buffer('prior_sd', scale)
        self.mean = nn.Parameter(randn(scale.shape, scale, generator) * scale)
        self.raw_scale = nn.Parameter(inverse_positive(scale * config.initial_sd_ratio, self.floor))

    @property
    def sd(self):
        return positive(self.raw_scale, self.floor)

    def rsample(self, generator=None, noise=None):
        if noise is None:
            noise = randn(self.mean.shape, self.mean, generator)
        if noise.shape != self.mean.shape:
            raise ValueError('theta noise shape')
        return self.mean + self.sd * noise

    def unpack(self, theta):
        if theta.ndim != 1 or theta.numel() != self.mean.numel():
            raise ValueError('theta flatten shape')
        result = {}; offset = 0
        for name, shape in self.layout:
            size = math.prod(shape)
            result[name] = theta[offset:offset + size].reshape(shape)
            offset += size
        return result

    def log_prob(self, theta):
        return normal_log_prob(theta, self.mean, self.sd).sum(-1)

    def kl_to(self, prior=None):
        m, s = (torch.zeros_like(self.mean), self.prior_sd) if prior is None else (prior.mean, prior.sd)
        return (torch.log(s / self.sd) + (self.sd.square() + (self.mean - m).square()) / (2 * s.square()) - .5).sum()

    def frozen_copy(self):
        return copy.deepcopy(self).requires_grad_(False)


class FixedProjection(nn.Module):
    def __init__(self, q, dimension, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.dimension = dimension
        self.layout = q.layout
        self.seed = seed
        bucket = torch.randint(dimension, (q.mean.numel(),), generator=g)
        signs = torch.randint(2, bucket.shape, generator=g).to(q.mean.dtype) * 2 - 1
        counts = torch.bincount(bucket, minlength=dimension).clamp_min(1).to(q.mean.dtype)
        self.register_buffer('bucket', bucket)
        self.register_buffer('coefficient', signs / counts[bucket].sqrt() / q.prior_sd.detach().clone())

    def forward(self, theta):
        out = theta.new_zeros((*theta.shape[:-1], self.dimension))
        return out.scatter_add(-1, self.bucket.expand(theta.shape), theta * self.coefficient)

    def variance(self, variance):
        return variance.new_zeros(self.dimension).scatter_add(0, self.bucket, variance * self.coefficient.square())

    def dense(self):
        p = self.coefficient.new_zeros(self.dimension, self.bucket.numel())
        p[self.bucket, torch.arange(self.bucket.numel(), device=p.device)] = self.coefficient
        return p

    def fingerprint(self):
        h = hashlib.sha256(json.dumps([self.layout, self.seed, self.dimension]).encode())
        for v in self.state_dict().values():
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
