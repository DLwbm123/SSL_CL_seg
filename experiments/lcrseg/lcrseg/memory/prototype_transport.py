"""Evidence-adaptive classwise prototype transport for ASPR."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TransportEstimate:
    case_count: int
    mean_displacement: torch.Tensor
    full_shift: torch.Tensor
    shrinkage: float
    delta: torch.Tensor
    variance: float
    signal: float
    valid: bool


@torch.no_grad()
def estimate_transport(old_case_prototypes: torch.Tensor, current_case_prototypes: torch.Tensor, *, eps: float = 1.0e-8) -> TransportEstimate:
    if old_case_prototypes.ndim != 2 or current_case_prototypes.shape != old_case_prototypes.shape:
        raise ValueError("paired case prototypes must share [N,D] shape")
    count = int(old_case_prototypes.shape[0])
    dimension = int(old_case_prototypes.shape[1])
    zero = torch.zeros(dimension, dtype=torch.float32, device=old_case_prototypes.device)
    if count < 2:
        return TransportEstimate(count, zero, zero, 0.0, zero, 0.0, 0.0, False)
    old_value = F.normalize(old_case_prototypes.detach().float(), p=2, dim=1, eps=eps)
    current_value = F.normalize(current_case_prototypes.detach().float(), p=2, dim=1, eps=eps)
    displacement = current_value - old_value
    mean = displacement.mean(dim=0)
    variance_tensor = (displacement - mean).square().sum(dim=1).sum() / float(count - 1)
    signal_tensor = mean.square().sum()
    finite = bool(torch.isfinite(displacement).all() and torch.isfinite(variance_tensor) and torch.isfinite(signal_tensor))
    signal = float(signal_tensor) if finite else 0.0
    variance = float(variance_tensor) if finite else 0.0
    if not finite or signal < eps:
        return TransportEstimate(count, zero, zero, 0.0, zero, variance, signal, False)
    shrinkage = max(0.0, min(1.0, signal / (signal + variance / float(count) + eps)))
    delta = (mean * shrinkage).detach()
    return TransportEstimate(count, mean.detach(), mean.detach(), shrinkage, delta, variance, signal, True)


@torch.no_grad()
def transport_prototypes(prototypes: torch.Tensor, delta: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    if prototypes.ndim != 2 or delta.ndim != 1 or prototypes.shape[1] != delta.shape[0]:
        raise ValueError("prototype transport expects [S,D] plus [D]")
    value = prototypes.detach().float() + delta.detach().float().unsqueeze(0)
    if not torch.isfinite(value).all() or bool(value.norm(dim=1).le(eps).any()):
        raise FloatingPointError("prototype transport produced invalid directions")
    return F.normalize(value, p=2, dim=1, eps=eps).detach()
