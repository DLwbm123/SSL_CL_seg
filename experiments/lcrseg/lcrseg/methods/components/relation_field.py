"""Cosine pixel-to-semantic-anchor relation distributions."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .anchor_bank import AnchorBank


@dataclass(frozen=True)
class RelationOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_class: torch.Tensor
    top1: torch.Tensor
    top2: torch.Tensor
    margin: torch.Tensor
    valid_class_mask: torch.Tensor


def relation_field(features: torch.Tensor, anchors: AnchorBank, *, temperature: float) -> RelationOutput:
    """Compute a numerically stable relation distribution in class space."""

    if temperature <= 0:
        raise ValueError("relation temperature must be positive")
    if features.ndim != 4 or features.shape[1] != anchors.relation_dim:
        raise ValueError("relation feature shape is incompatible with anchor bank")
    valid = anchors.valid_class_mask
    if not bool(valid.any()):
        raise RuntimeError("relation field has no valid semantic anchor class")
    feature_unit = F.normalize(features.float(), p=2, dim=1, eps=anchors.eps)
    anchor_unit = F.normalize(anchors.anchors[:, 0].float(), p=2, dim=1, eps=anchors.eps)
    logits = torch.einsum("bdhw,cd->bchw", feature_unit, anchor_unit) / float(temperature)
    # Finite negative logits are intentional: they preserve stable softmax and
    # underflow to exact zero for invalid classes in float32.
    logits = logits.masked_fill(~valid.view(1, -1, 1, 1), -1.0e4)
    probabilities = logits.softmax(dim=1)
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("non-finite relation probabilities")
    top_values, top_indices = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1)
    top1 = top_values[:, :1]
    top2 = top_values[:, 1:2] if probabilities.shape[1] > 1 else torch.zeros_like(top1)
    predicted = top_indices[:, 0]
    return RelationOutput(
        logits=logits,
        probabilities=probabilities,
        predicted_class=predicted,
        top1=top1,
        top2=top2,
        margin=top1 - top2,
        valid_class_mask=valid.detach().clone(),
    )
