import math
import torch
from torch.nn import functional as F


def positive(raw, floor=1e-6):
    return F.softplus(raw) + floor


def inverse_positive(sd, floor=1e-6):
    v = sd - floor
    if not torch.all(v > 0):
        raise ValueError('scale must exceed floor')
    return v + torch.log(-torch.expm1(-v))


def normal_log_prob(x, mean, sd):
    return -.5 * math.log(2 * math.pi) - torch.log(sd) - .5 * ((x - mean) / sd).square()


def student_log_prob(x, loc, scale, nu=4.):
    n = x.new_tensor(nu)
    return (torch.lgamma((n + 1) / 2) - torch.lgamma(n / 2)
            - .5 * torch.log(n * math.pi) - torch.log(scale)
            - (n + 1) / 2 * torch.log1p(((x - loc) / scale).square() / n))


def categorical_log_prob(prob, allowed=None, observed=None):
    """One event per pixel; allowed is a bool set of leaves, B,K,H,W."""
    zero = prob.sum((1, 2, 3)) * 0
    if allowed is None:
        if observed is not None and observed.any():
            raise ValueError('observed labels require leaf sets')
        return zero
    if allowed.shape != prob.shape or allowed.dtype != torch.bool:
        raise ValueError('allowed leaf shape/type')
    if observed is None:
        observed = torch.ones_like(prob[:, 0], dtype=torch.bool)
    if observed.shape != prob[:, 0].shape or observed.dtype != torch.bool:
        raise ValueError('observation shape/type')
    if not observed.any():
        return zero
    mass = (prob * allowed).sum(1)
    if torch.any((mass <= 0) & observed):
        raise ValueError('observed event has no probability')
    return torch.where(observed, torch.log(torch.where(observed, mass, torch.ones_like(mass))), 0).sum((1, 2))


def finite(x, name):
    if not torch.isfinite(x).all():
        raise FloatingPointError(name)
    return x


def randn(shape, like, generator):
    if generator is None:
        raise ValueError('explicit generator required')
    return torch.randn(shape, dtype=like.dtype, device=like.device, generator=generator)
