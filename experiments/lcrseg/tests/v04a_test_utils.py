from __future__ import annotations

import torch

from lcrseg.methods.components.pseudo_label import PseudoLabelOutput
from lcrseg.methods.components.relation_field import RelationOutput


def pseudo(labels: torch.Tensor, valid: torch.Tensor | None = None) -> PseudoLabelOutput:
    valid = torch.ones((labels.shape[0], 1, *labels.shape[-2:]), dtype=torch.bool) if valid is None else valid
    weight = valid.float()
    return PseudoLabelOutput(
        labels=labels.long(),
        valid=valid.bool(),
        source=torch.ones_like(labels),
        source_weight=weight,
        spatial_weight=weight,
        spatial_agreement=weight,
    )


def relation(logits: torch.Tensor) -> RelationOutput:
    probability = logits.softmax(dim=1)
    values, indices = probability.topk(2, dim=1)
    return RelationOutput(
        logits=logits,
        probabilities=probability,
        predicted_class=indices[:, 0],
        top1=values[:, :1],
        top2=values[:, 1:2],
        margin=values[:, :1] - values[:, 1:2],
        valid_class_mask=torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device),
    )
