"""Frozen current/previous prototype-anchored validation for SPARC V0.1."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .session_prototypes import SessionPrototypeSet


@dataclass(frozen=True)
class AnchoredValidation:
    predicted_class: torch.Tensor
    confidence: torch.Tensor
    prototype_similarity: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class StablePlasticPartition:
    stable: torch.Tensor
    plastic: torch.Tensor
    rejected: torch.Tensor
    current_valid: torch.Tensor
    previous_valid: torch.Tensor
    current_class: torch.Tensor
    previous_class: torch.Tensor


def _valid_grid(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("relation valid mask must be [B,1,H,W] or [B,H,W]")
    if mask.shape[-2:] != size:
        raise ValueError("existing relation valid mask must already match the relation grid")
    return mask[:, 0].detach().bool()


def anchored_validation(
    logits: torch.Tensor,
    relation_features: torch.Tensor,
    prototypes: SessionPrototypeSet,
    relation_valid_mask: torch.Tensor,
    *,
    confidence_threshold: float = 0.70,
    prototype_similarity_threshold: float = 0.70,
) -> AnchoredValidation:
    if confidence_threshold != 0.70 or prototype_similarity_threshold != 0.70:
        raise ValueError("SPARC V0.1 freezes confidence/similarity thresholds at 0.70/0.70")
    if logits.ndim != 4 or relation_features.ndim != 4:
        raise ValueError("logits and relation_features must be BCHW")
    if logits.shape[0] != relation_features.shape[0]:
        raise ValueError("batch dimensions differ")
    classes = logits.shape[1]
    if prototypes.prototypes.shape != (classes, relation_features.shape[1]):
        raise ValueError("prototype class/feature dimensions do not match model outputs")
    size = tuple(relation_features.shape[-2:])
    logits_r = F.interpolate(logits.float(), size=size, mode="bilinear", align_corners=False)
    probabilities = logits_r.softmax(dim=1)
    confidence, predicted = probabilities.max(dim=1)
    normalized_features = F.normalize(relation_features.float(), p=2, dim=1, eps=1.0e-8)
    prototype_table = prototypes.prototypes.to(normalized_features.device).detach()
    prototype_valid = prototypes.valid_classes.to(normalized_features.device).detach()
    selected_prototypes = prototype_table[predicted].permute(0, 3, 1, 2)
    similarity = (normalized_features * selected_prototypes).sum(dim=1)
    class_valid = prototype_valid[predicted]
    existing_valid = _valid_grid(relation_valid_mask, size)
    valid = (
        existing_valid
        & class_valid
        & confidence.gt(confidence_threshold)
        & similarity.gt(prototype_similarity_threshold)
    )
    return AnchoredValidation(
        predicted_class=predicted.detach(),
        confidence=confidence.detach(),
        prototype_similarity=similarity.detach(),
        valid=valid.detach(),
    )


def partition_stable_plastic(
    current: AnchoredValidation,
    previous: AnchoredValidation,
    relation_valid_mask: torch.Tensor,
) -> StablePlasticPartition:
    if current.valid.shape != previous.valid.shape:
        raise ValueError("current/previous validation grids differ")
    valid_grid = _valid_grid(relation_valid_mask, tuple(current.valid.shape[-2:]))
    stable = current.valid & previous.valid & current.predicted_class.eq(previous.predicted_class)
    plastic = current.valid & ~stable
    rejected = valid_grid & ~current.valid
    if bool((stable & plastic).any() or (stable & rejected).any() or (plastic & rejected).any()):
        raise AssertionError("SPARC partition masks overlap")
    if not torch.equal(stable | plastic, current.valid):
        raise AssertionError("SPARC partition does not cover current-valid cells")
    return StablePlasticPartition(
        stable=stable.detach(),
        plastic=plastic.detach(),
        rejected=rejected.detach(),
        current_valid=current.valid.detach(),
        previous_valid=previous.valid.detach(),
        current_class=current.predicted_class.detach(),
        previous_class=previous.predicted_class.detach(),
    )
