"""Detached spatial divide-and-conquer pseudo-label construction."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .relation_field import RelationOutput


IGNORE_INDEX = -100
SOURCE_DEFERRED = 0
SOURCE_CLASSIFIER = 1
SOURCE_ANCHOR = 2


@dataclass(frozen=True)
class PseudoLabelOutput:
    labels: torch.Tensor
    valid: torch.Tensor
    source: torch.Tensor
    source_weight: torch.Tensor
    spatial_weight: torch.Tensor
    spatial_agreement: torch.Tensor


def _validate_grid(labels: torch.Tensor, valid: torch.Tensor) -> None:
    if labels.ndim != 3 or valid.shape != (labels.shape[0], 1, *labels.shape[-2:]):
        raise ValueError("labels must be [B,H,W] and valid must be [B,1,H,W]")


@torch.no_grad()
def spatial_agreement(labels: torch.Tensor, valid: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    """3x3 same-class agreement, safely zeroing invalid positions."""

    _validate_grid(labels, valid)
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    valid_bool = valid.bool()
    valid_float = valid_bool.float()
    denominator = F.conv2d(valid_float, torch.ones((1, 1, 3, 3), device=labels.device), padding=1)
    numerator_for_label = torch.zeros_like(denominator)
    for class_index in range(num_classes):
        same = labels.eq(class_index).unsqueeze(1) & valid_bool
        count = F.conv2d(same.float(), torch.ones((1, 1, 3, 3), device=labels.device), padding=1)
        numerator_for_label = torch.where(labels.eq(class_index).unsqueeze(1), count, numerator_for_label)
    agreement = numerator_for_label / denominator.clamp_min(1.0)
    return torch.where(valid_bool, agreement, torch.zeros_like(agreement)).detach()


@torch.no_grad()
def spatial_weight(
    labels: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_classes: int,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= floor <= 1.0:
        raise ValueError("spatial floor must be in [0, 1]")
    agreement = spatial_agreement(labels, valid, num_classes=num_classes)
    weights = float(floor) + (1.0 - float(floor)) * agreement
    return torch.where(valid.bool(), weights, torch.zeros_like(weights)).detach(), agreement


def _resized_segmentation_probabilities(probabilities: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if probabilities.ndim != 4:
        raise ValueError("segmentation probabilities must be [B,C,H,W]")
    resized = F.interpolate(probabilities, size=size, mode="bilinear", align_corners=False)
    return resized / resized.sum(dim=1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def build_pseudo_labels(
    weak_segmentation_probabilities: torch.Tensor,
    relation: RelationOutput,
    *,
    tau_cls: float,
    tau_anchor: float,
    delta_anchor: float,
    tau_spatial: float,
    temperature_cls: float,
    temperature_anchor: float,
    spatial_floor: float,
) -> PseudoLabelOutput:
    """Build classifier-easy, anchor-recoverable, or deferred labels.

    Every input is converted to a detached calculation.  The returned labels
    live on the relation grid and are only later nearest-upsampled for the
    strong segmentation loss.
    """

    for name, value in {
        "tau_cls": tau_cls,
        "tau_anchor": tau_anchor,
        "delta_anchor": delta_anchor,
        "tau_spatial": tau_spatial,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if temperature_cls <= 0 or temperature_anchor <= 0:
        raise ValueError("pseudo-label temperatures must be positive")
    relation_probabilities = relation.probabilities.detach()
    grid_size = relation_probabilities.shape[-2:]
    seg_probabilities = _resized_segmentation_probabilities(weak_segmentation_probabilities.detach(), grid_size)
    if seg_probabilities.shape[:2] != relation_probabilities.shape[:2]:
        raise ValueError("segmentation and relation class counts differ")
    seg_confidence, seg_class = seg_probabilities.max(dim=1)
    relation_class = relation.predicted_class.detach()
    relation_confidence = relation.top1[:, 0].detach()
    relation_margin = relation.margin[:, 0].detach()
    classifier = seg_confidence.ge(float(tau_cls)) & seg_class.eq(relation_class)
    anchor_candidate = (~classifier) & relation_confidence.ge(float(tau_anchor)) & relation_margin.ge(float(delta_anchor))
    preliminary_valid = classifier | anchor_candidate
    preliminary_labels = torch.full_like(seg_class, IGNORE_INDEX)
    preliminary_labels = torch.where(classifier, seg_class, preliminary_labels)
    preliminary_labels = torch.where(anchor_candidate, relation_class, preliminary_labels)
    preliminary_agreement = spatial_agreement(
        preliminary_labels,
        preliminary_valid.unsqueeze(1),
        num_classes=seg_probabilities.shape[1],
    )[:, 0]
    anchor = anchor_candidate & preliminary_agreement.ge(float(tau_spatial))
    valid = (classifier | anchor).unsqueeze(1)
    labels = torch.full_like(seg_class, IGNORE_INDEX)
    labels = torch.where(classifier, seg_class, labels)
    labels = torch.where(anchor, relation_class, labels)
    source = torch.full_like(seg_class, SOURCE_DEFERRED)
    source = torch.where(classifier, torch.full_like(source, SOURCE_CLASSIFIER), source)
    source = torch.where(anchor, torch.full_like(source, SOURCE_ANCHOR), source)
    classifier_weight = torch.sigmoid((seg_confidence - float(tau_cls)) / float(temperature_cls))
    anchor_weight = torch.sigmoid((relation_confidence - float(tau_anchor)) / float(temperature_anchor))
    source_weight = torch.where(classifier, classifier_weight, torch.where(anchor, anchor_weight, torch.zeros_like(anchor_weight)))
    final_spatial_weight, final_agreement = spatial_weight(
        labels,
        valid,
        num_classes=seg_probabilities.shape[1],
        floor=spatial_floor,
    )
    return PseudoLabelOutput(
        labels=labels.detach().long(),
        valid=valid.detach().bool(),
        source=source.detach().long(),
        source_weight=source_weight.unsqueeze(1).detach(),
        spatial_weight=final_spatial_weight.detach(),
        spatial_agreement=final_agreement.detach(),
    )
