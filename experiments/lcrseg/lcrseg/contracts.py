"""Stable types shared by LCR-Seg models, data loaders, and methods."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SegModelOutput:
    """Segmentation logits plus L2-normalized relation features."""

    logits: torch.Tensor
    relation_features: torch.Tensor
    # Optional references to tensors already produced by the same forward pass.
    # This is an output-contract extension only: callers that do not request or
    # consume decoder features retain the exact historical computation path.
    decoder_features: dict[str, torch.Tensor] | None = None


@dataclass
class LabeledBatch:
    image: torch.Tensor
    label: torch.Tensor
    valid_mask: torch.Tensor
    case_id: list[str]
    patient_id: list[str]
    site: list[str]
    slice_index: list[int | None]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "LabeledBatch":
        return LabeledBatch(
            image=self.image.to(device, non_blocking=non_blocking),
            label=self.label.to(device, non_blocking=non_blocking),
            valid_mask=self.valid_mask.to(device, non_blocking=non_blocking),
            case_id=self.case_id,
            patient_id=self.patient_id,
            site=self.site,
            slice_index=self.slice_index,
        )


@dataclass
class UnlabeledBatch:
    """Training-time unlabeled batch; intentionally has no label field."""

    weak_image: torch.Tensor
    strong_image: torch.Tensor
    strong_valid_mask: torch.Tensor
    case_id: list[str]
    patient_id: list[str]
    site: list[str]
    slice_index: list[int | None]
    geometry_record: list[dict[str, Any]]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "UnlabeledBatch":
        return UnlabeledBatch(
            weak_image=self.weak_image.to(device, non_blocking=non_blocking),
            strong_image=self.strong_image.to(device, non_blocking=non_blocking),
            strong_valid_mask=self.strong_valid_mask.to(device, non_blocking=non_blocking),
            case_id=self.case_id,
            patient_id=self.patient_id,
            site=self.site,
            slice_index=self.slice_index,
            geometry_record=self.geometry_record,
        )


@dataclass
class MethodStepOutput:
    total_loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    # V0.2 adds structured, non-patient branch statistics (for example
    # per-class fractions and an explicit calibrator status) to the same
    # provenance row. V0.1 continues to provide floats only.
    scalars: dict[str, Any]
    maps: dict[str, torch.Tensor] | None = None


def differentiable_zero(reference: torch.Tensor) -> torch.Tensor:
    """Return a scalar zero that remains valid for `backward()`."""

    return reference.sum() * 0.0
