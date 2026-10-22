import torch
from torch import nn
from torch.nn import functional as F
from .distributions import normal_log_prob, student_log_prob, categorical_log_prob, finite
from .spatial_gaussian import dct_basis, sar_log_prob


def layout(c):
    h = c.hidden
    return [('g.c.weight', (h, c.c_dim)), ('g.c.bias', (h,)),
            ('g.r.weight', (h, c.r_channels, 3, 3)), ('g.r.bias', (h,)),
            ('g.out.weight', (c.classes - 1, h, 3, 3)), ('g.out.bias', (c.classes - 1,)),
            ('f.in.weight', (h, c.classes, 3, 3)), ('f.in.bias', (h,)),
            ('f.out.weight', (c.channels, h, 3, 3)), ('f.out.bias', (c.channels,))]


class NamedParts(dict):
    @property
    def total(self):
        return sum(self.values())


class GenerativeModel(nn.Module):
    def __init__(self, config, posterior):
        super().__init__()
        self.config = config
        # Posterior is owned by StreamState, not duplicated as model parameters.
        object.__setattr__(self, 'posterior', posterior)
        self.register_buffer('basis', dct_basis(config.height, config.width, config.dct_order, getattr(torch, config.dtype)))

    def class_probabilities(self, z, theta):
        w = self.posterior.unpack(theta)
        c = F.linear(z['c'], w['g.c.weight'], w['g.c.bias'])[:, :, None, None]
        r = F.interpolate(z['R'], (self.config.height, self.config.width), mode='bilinear', align_corners=False)
        hidden = torch.tanh(c + F.conv2d(r, w['g.r.weight'], w['g.r.bias'], padding=1))
        logits = F.conv2d(hidden, w['g.out.weight'], w['g.out.bias'], padding=1)
        return torch.cat([torch.zeros_like(logits[:, :1]), logits], 1).softmax(1)

    def structure_mean(self, prob, theta):
        w = self.posterior.unpack(theta)
        hidden = torch.tanh(F.conv2d(prob, w['f.in.weight'], w['f.in.bias'], padding=1))
        return F.conv2d(hidden, w['f.out.weight'], w['f.out.bias'], padding=1)

    def log_joint(self, x, z, theta, a, y_obs=None, obs_mask=None):
        c = self.config
        if x.ndim != 4 or tuple(x.shape[1:]) != (c.channels, c.height, c.width) or a.shape != (3 * c.channels,):
            raise ValueError('image/domain shape')
        if x.dtype != theta.dtype or x.device != theta.device:
            raise TypeError('image/parameter dtype or device mismatch')
        prob = self.class_probabilities(z, theta)
        gamma, beta, xi = a.reshape(3, c.channels, 1, 1)
        bias = torch.einsum('bck,khw->bchw', z['b'], self.basis)
        loc = gamma.exp() * z['S'] + beta + bias
        zero = x.new_zeros(())
        one = x.new_ones(())
        parts = NamedParts(
            c=normal_log_prob(z['c'], zero, one).flatten(1).sum(1),
            R=normal_log_prob(z['R'], zero, one).flatten(1).sum(1),
            b=normal_log_prob(z['b'], zero, one * c.tau_b).flatten(1).sum(1),
            S=sar_log_prob(z['S'], self.structure_mean(prob, theta), c.sigma_s, c.alpha),
            x=student_log_prob(x, loc, xi.exp(), c.nu).flatten(1).sum(1),
            y=categorical_log_prob(prob, y_obs, obs_mask))
        finite(parts.total, 'nonfinite joint')
        return parts
