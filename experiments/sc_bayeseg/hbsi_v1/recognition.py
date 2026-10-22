import math
import torch
from torch import nn
from torch.nn import functional as F
from .distributions import positive, inverse_positive, normal_log_prob, randn
from .spatial_gaussian import sar, sar_solve, sar_logdet, sar_eigenvalues


class Recognition(nn.Module):
    """Small smooth U-shaped encoder, without normalization buffers."""
    def __init__(self, config, projection):
        super().__init__()
        self.config = config
        self.projection = projection
        h = config.hidden
        self.enc = nn.Conv2d(config.channels, h, 3, padding=1)
        self.down = nn.Conv2d(h, h, 3, padding=1)
        self.up = nn.Conv2d(2 * h, h, 3, padding=1)
        ctx = h + config.projection_dim + 3 * config.channels
        self.c_head = nn.Linear(ctx, 2 * config.c_dim)
        self.r_head = nn.Conv2d(ctx + config.c_dim, 2 * config.r_channels, 1)
        self.s_head = nn.Conv2d(ctx + config.c_dim + config.r_channels, 2 * config.channels, 1)
        self.b_head = nn.Linear(ctx + config.c_dim + config.r_channels, 2 * config.channels * config.b_dim)
        self.to(dtype=getattr(torch, config.dtype))
        if config.prior_aware_init:
            scales = {'c': 1., 'r': 1., 'b': config.tau_b}
            eig = sar_eigenvalues(config.height, config.width, self.enc.weight, config.alpha)
            scales['s'] = 1. if config.local_parameterization == 'sar_whitened' else float(config.sigma_s * eig.pow(-2).mean().sqrt())
            with torch.no_grad():
                for name, scale in scales.items():
                    head = getattr(self, name + '_head')
                    n = head.bias.numel() // 2
                    # b_head is reshaped [2*C,b_dim], so the second half is its scale output too.
                    head.weight[n:].zero_()
                    head.bias[n:].copy_(inverse_positive(head.bias.new_full((n,), scale), config.scale_floor))

    def context(self, x, theta, a):
        h = torch.tanh(self.enc(x))
        low = torch.tanh(self.down(F.avg_pool2d(h, 2)))
        feature = torch.tanh(self.up(torch.cat([h, F.interpolate(low, x.shape[-2:], mode='bilinear', align_corners=False)], 1)))
        extra = torch.cat([self.projection(theta), a])[None, :, None, None].expand(len(x), -1, *x.shape[-2:])
        return torch.cat([feature, extra], 1)

    def shapes(self, batch):
        c = self.config
        return {'c': (batch, c.c_dim), 'R': (batch, c.r_channels, c.r_height, c.r_width),
                'S': (batch, c.channels, c.height, c.width), 'b': (batch, c.channels, c.b_dim)}

    def noise(self, batch, like, generator):
        return {k: randn(s, like, generator) for k, s in self.shapes(batch).items()}

    def initial_offsets(self, batch, like):
        return {k: like.new_zeros((s[0], 2 * s[1], *s[2:]), requires_grad=True)
                for k, s in self.shapes(batch).items()}

    def parameters_given(self, name, context, z):
        if name == 'c':
            return self.c_head(context.mean((2, 3)))
        c = z['c'][:, :, None, None].expand(-1, -1, *context.shape[-2:])
        v = torch.cat([context, c], 1)
        if name == 'R':
            return self.r_head(F.interpolate(v, (self.config.r_height, self.config.r_width), mode='bilinear', align_corners=False))
        r = F.interpolate(z['R'], context.shape[-2:], mode='bilinear', align_corners=False)
        v = torch.cat([v, r], 1)
        if name == 'S':
            return self.s_head(v)
        return self.b_head(v.mean((2, 3))).reshape(len(v), 2 * self.config.channels, self.config.b_dim)

    def _transform(self, z, model, theta, inverse=False):
        if model is None or theta is None:
            raise ValueError('whitened proposal requires the sampled generative parameters')
        c = self.config
        mean = model.structure_mean(model.class_probabilities(z, theta), theta)
        result = dict(z)
        result['S'] = sar(z['S'] - mean, c.alpha) / c.sigma_s if inverse else mean + c.sigma_s * sar_solve(z['S'], c.alpha)
        logdet = c.channels * sar_logdet(c.height, c.width, z['S'], c.alpha) - c.channels * c.height * c.width * math.log(c.sigma_s)
        return result, logdet

    def rsample_and_log_prob(self, context, noise, offsets, model=None, theta=None):
        z = {}; logq = context.new_zeros(len(context))
        for name in ('c', 'R', 'S', 'b'):
            raw = self.parameters_given(name, context, z) + offsets[name]
            mean, raw_sd = raw.chunk(2, 1)
            sd = positive(raw_sd, self.config.scale_floor)
            if noise[name].shape != mean.shape:
                raise ValueError('local noise shape')
            z[name] = mean + sd * noise[name]
            logq = logq + normal_log_prob(z[name], mean, sd).flatten(1).sum(1)
        if self.config.local_parameterization == 'sar_whitened':
            z, logdet = self._transform(z, model, theta)
            logq = logq + logdet
        return z, logq

    def log_prob(self, z, context, offsets, model=None, theta=None):
        result = context.new_zeros(len(context))
        if self.config.local_parameterization == 'sar_whitened':
            z, logdet = self._transform(z, model, theta, inverse=True)
            result = result + logdet
        for name in ('c', 'R', 'S', 'b'):
            mean, raw_sd = (self.parameters_given(name, context, z) + offsets[name]).chunk(2, 1)
            result = result + normal_log_prob(z[name], mean, positive(raw_sd, self.config.scale_floor)).flatten(1).sum(1)
        return result
