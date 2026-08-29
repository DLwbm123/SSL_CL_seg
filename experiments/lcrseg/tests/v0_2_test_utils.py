from __future__ import annotations

import torch

from lcrseg.methods.components.pseudo_label import PseudoLabelOutput
from lcrseg.methods.components.relation_field import RelationOutput


def pseudo(labels: torch.Tensor, *, valid: torch.Tensor | None = None) -> PseudoLabelOutput:
    if valid is None:
        valid = labels.ne(-100).unsqueeze(1)
    return PseudoLabelOutput(
        labels=labels.long(),
        valid=valid.bool(),
        source=torch.where(labels.eq(-100), torch.zeros_like(labels), torch.ones_like(labels)),
        source_weight=valid.float(),
        spatial_weight=valid.float(),
        spatial_agreement=valid.float(),
    )


def relation(probabilities: torch.Tensor) -> RelationOutput:
    normalized = probabilities / probabilities.sum(dim=1, keepdim=True)
    top, indexes = normalized.topk(k=2, dim=1)
    return RelationOutput(
        logits=normalized.log(),
        probabilities=normalized,
        predicted_class=indexes[:, 0],
        top1=top[:, :1],
        top2=top[:, 1:2],
        margin=top[:, :1] - top[:, 1:2],
        valid_class_mask=torch.ones(normalized.shape[1], dtype=torch.bool),
    )
