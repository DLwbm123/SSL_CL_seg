"""Losses and segmentation metrics shared by all LCR-Seg methods."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ..contracts import differentiable_zero


def _pixel_mask(valid_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if valid_mask.ndim != 4 or target.ndim != 3:
        raise ValueError("valid_mask must be [B,1,H,W] and target must be [B,H,W]")
    if valid_mask.shape[0] != target.shape[0] or valid_mask.shape[-2:] != target.shape[-2:]:
        raise ValueError("valid_mask and target geometry differ")
    return valid_mask[:, 0].bool()


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Cross entropy over a visible pixel set, with a differentiable empty zero."""

    mask = _pixel_mask(valid_mask, target)
    if not bool(mask.any()):
        return differentiable_zero(logits)
    per_pixel = F.cross_entropy(logits, target.long(), reduction="none")
    return (per_pixel * mask).sum() / mask.sum().clamp_min(1)


def multiclass_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    include_background: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-class soft Dice, returning NaN only for fully absent classes."""

    if logits.ndim != 4:
        raise ValueError("logits must be [B,C,H,W]")
    classes = logits.shape[1]
    mask = _pixel_mask(valid_mask, target).unsqueeze(1).float()
    probability = logits.softmax(dim=1)
    one_hot = F.one_hot(target.long(), num_classes=classes).permute(0, 3, 1, 2).float()
    probability = probability * mask
    one_hot = one_hot * mask
    numerator = 2.0 * (probability * one_hot).sum(dim=(0, 2, 3))
    denominator = probability.square().sum(dim=(0, 2, 3)) + one_hot.square().sum(dim=(0, 2, 3))
    dice = (numerator + eps) / (denominator + eps)
    if not include_background:
        dice = dice[1:]
    return dice


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    dice = multiclass_dice(logits, target, valid_mask, include_background=False)
    if dice.numel() == 0:
        return differentiable_zero(logits)
    return 1.0 - dice.mean()
