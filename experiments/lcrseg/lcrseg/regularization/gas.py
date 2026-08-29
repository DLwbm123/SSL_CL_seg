"""Frozen classifier-only Gradient-Adaptive Stabilization primitives."""
from __future__ import annotations

import math

import torch


def unit_mean_source_normalize(sensitivity: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    detached = sensitivity.detach().float()
    if not bool(torch.isfinite(detached).all()) or bool(detached.lt(0).any()):
        raise ValueError("sensitivity must be finite and nonnegative")
    return (detached / (detached.mean() + float(eps))).detach()


def jascl_inverse_minmax_scale(sensitivity: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Return the exact registered inverse-minmax scale in ``(0, 1]``."""

    if eps <= 0:
        raise ValueError("epsilon must be positive")
    detached = sensitivity.detach().float()
    if not bool(torch.isfinite(detached).all()) or bool(detached.lt(0).any()):
        raise ValueError("sensitivity must be finite and nonnegative")
    if detached.numel() == 0:
        return detached
    if bool((detached.max() - detached.min()).abs().le(float(eps))):
        return torch.ones_like(detached)
    inverse = 1.0 / (detached + float(eps))
    minimum = inverse.min()
    maximum = inverse.max()
    scale = (1.0 + inverse - minimum) / (1.0 + maximum - minimum)
    return scale.clamp(min=torch.finfo(scale.dtype).tiny, max=1.0).detach()


def sample_perturbed_weight(
    weight: torch.Tensor,
    noise_scale: torch.Tensor,
    *,
    noise_sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample one detached perturbation without mutating the master weight."""

    if noise_sigma < 0:
        raise ValueError("noise sigma must be nonnegative")
    if noise_scale.shape != weight.shape:
        raise ValueError("noise scale and classifier weight shapes differ")
    scale = noise_scale.detach().to(device=weight.device, dtype=torch.float32)
    if not bool(torch.isfinite(scale).all()) or bool(scale.lt(0).any()):
        raise ValueError("noise scale must be finite and nonnegative")
    noise = torch.randn(weight.shape, device=weight.device, dtype=torch.float32, generator=generator).detach()
    perturbation = (float(noise_sigma) * scale * noise).to(dtype=weight.dtype).detach()
    return weight + perturbation


def linear_noise_warmup(
    *,
    successful_site_step: int,
    total_site_steps: int,
    warmup_fraction: float = 0.20,
) -> float:
    """Frozen V0.2 linear warm-start indexed by successful optimizer steps."""

    if successful_site_step < 0 or total_site_steps < 1:
        raise ValueError("successful step must be nonnegative and total steps positive")
    if warmup_fraction != 0.20:
        raise ValueError("SR-GAS V0.2 freezes the noise warm-up fraction at 0.20")
    warmup_steps = math.ceil(float(warmup_fraction) * int(total_site_steps))
    return min(1.0, float(successful_site_step) / float(warmup_steps))


__all__ = [
    "jascl_inverse_minmax_scale",
    "linear_noise_warmup",
    "sample_perturbed_weight",
    "unit_mean_source_normalize",
]
