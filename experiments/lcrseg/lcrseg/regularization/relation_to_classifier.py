"""Class-space historical relation-to-classifier sensitivity proxy."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ..contracts import differentiable_zero
from ..data.transforms import downsample_valid_mask


@dataclass(frozen=True)
class RelationToClassifierOutput:
    loss: torch.Tensor
    kl_map: torch.Tensor
    valid_mask: torch.Tensor
    valid_count: int
    current_probability: torch.Tensor
    target_probability: torch.Tensor


def relation_to_classifier_loss(
    current_clean_logits: torch.Tensor,
    old_relation_probability: torch.Tensor,
    strong_valid_mask: torch.Tensor,
    *,
    historical_anchors_available: bool,
    temperature: float = 1.0,
) -> RelationToClassifierOutput:
    """KL(stopgrad(q_old) || downsampled clean classifier probability)."""

    if temperature != 1.0:
        raise ValueError("V0.1a freezes R2C temperature at 1.0")
    if current_clean_logits.ndim != 4 or old_relation_probability.ndim != 4:
        raise ValueError("R2C tensors must be [B,C,H,W]")
    if current_clean_logits.shape[:2] != old_relation_probability.shape[:2]:
        raise ValueError("R2C class count or batch count mismatch")
    target = old_relation_probability.detach().float()
    resized = F.interpolate(
        current_clean_logits.float(),
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    current = resized.softmax(dim=1)
    finite_target = torch.isfinite(target).all(dim=1, keepdim=True)
    valid = downsample_valid_mask(strong_valid_mask, target.shape[-2:]).bool()
    valid = valid & finite_target & bool(historical_anchors_available)
    safe_target = torch.where(torch.isfinite(target), target, torch.zeros_like(target)).clamp_min(1.0e-8)
    safe_target = safe_target / safe_target.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    safe_current = current.clamp_min(1.0e-8)
    kl = (safe_target * (safe_target.log() - safe_current.log())).sum(dim=1, keepdim=True)
    count = int(valid.sum().detach())
    loss = (kl * valid.float()).sum() / valid.float().sum().clamp_min(1.0) if count else differentiable_zero(current_clean_logits)
    return RelationToClassifierOutput(
        loss=loss,
        kl_map=kl.detach(),
        valid_mask=valid.detach(),
        valid_count=count,
        current_probability=current,
        target_probability=safe_target.detach(),
    )


__all__ = ["RelationToClassifierOutput", "relation_to_classifier_loss"]
