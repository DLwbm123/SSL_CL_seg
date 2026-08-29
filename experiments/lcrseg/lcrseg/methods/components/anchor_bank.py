"""Non-parametric class semantic anchors used by LCR-Seg V0.1.

The bank intentionally stores only normalized class directions and aggregate
support counters.  It never stores a patient image, feature map, or a
trainable parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AnchorUpdate:
    """Auditable result of one no-gradient anchor update."""

    updated_pixels: dict[int, int]
    skipped_classes: list[int]


def background_boundary_mask(labels: torch.Tensor, width: int) -> torch.Tensor:
    """Return background locations sufficiently far from foreground boundaries.

    ``labels`` is on the relation grid.  The operation is deliberately based
    only on visible/pseudo labels and uses a Chebyshev-radius dilation so that
    a background anchor cannot be dominated by boundary pixels.
    """

    if labels.ndim != 3:
        raise ValueError(f"labels must be [B,H,W], got {tuple(labels.shape)}")
    if width < 0:
        raise ValueError("background boundary exclusion must be non-negative")
    background = labels.eq(0)
    if width == 0:
        return background
    foreground = labels.ne(0).unsqueeze(1).float()
    dilated_foreground = F.max_pool2d(
        foreground,
        kernel_size=2 * width + 1,
        stride=1,
        padding=width,
    ).bool()[:, 0]
    return background & ~dilated_foreground


class AnchorBank(nn.Module):
    """One fixed-memory semantic anchor per class (``K=1`` in V0.1)."""

    def __init__(
        self,
        num_classes: int,
        relation_dim: int,
        *,
        k: int = 1,
        momentum: float = 0.99,
        max_pixels_per_class: int = 2048,
        min_support_pixels: int = 64,
        background_boundary_exclusion: int = 3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("anchor bank requires at least background and one foreground class")
        if relation_dim < 1:
            raise ValueError("relation_dim must be positive")
        if k != 1:
            raise ValueError("LCR-Seg V0.1 fixes one semantic anchor per class (K=1)")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if max_pixels_per_class < 1 or min_support_pixels < 1:
            raise ValueError("anchor support limits must be positive")
        self.num_classes = int(num_classes)
        self.relation_dim = int(relation_dim)
        self.k = int(k)
        self.momentum = float(momentum)
        self.max_pixels_per_class = int(max_pixels_per_class)
        self.min_support_pixels = int(min_support_pixels)
        self.background_boundary_exclusion = int(background_boundary_exclusion)
        self.eps = float(eps)
        shape = (self.num_classes, self.k)
        self.register_buffer("anchors", torch.zeros((*shape, self.relation_dim), dtype=torch.float32))
        self.register_buffer("valid", torch.zeros(shape, dtype=torch.bool))
        self.register_buffer("counts_total", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("counts_labeled", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("counts_unlabeled", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("last_update_step", torch.full(shape, -1, dtype=torch.int64))

    @property
    def valid_class_mask(self) -> torch.Tensor:
        return self.valid.any(dim=1)

    @property
    def all_classes_valid(self) -> bool:
        return bool(self.valid_class_mask.all())

    def assert_no_parameters(self) -> None:
        if list(self.parameters()):
            raise AssertionError("semantic anchors must not be trainable parameters")

    def clone(self) -> "AnchorBank":
        clone = AnchorBank(
            self.num_classes,
            self.relation_dim,
            k=self.k,
            momentum=self.momentum,
            max_pixels_per_class=self.max_pixels_per_class,
            min_support_pixels=self.min_support_pixels,
            background_boundary_exclusion=self.background_boundary_exclusion,
            eps=self.eps,
        ).to(device=self.anchors.device)
        clone.load_state_dict({key: value.detach().clone() for key, value in self.state_dict().items()})
        return clone

    def exported_state(self) -> dict[str, torch.Tensor]:
        """Return detached copies suitable for a checkpoint payload."""

        return {key: value.detach().clone() for key, value in self.state_dict().items()}

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        *,
        source: str,
        step: int,
    ) -> AnchorUpdate:
        """Update valid class centers using deterministic capped samples.

        Features and the resulting anchor both remain L2-normalized.  The
        update is intentionally deterministic given the passed tensors: the
        first valid flattened locations are selected when capping support.
        """

        if source not in {"labeled", "unlabeled"}:
            raise ValueError(f"unknown anchor source: {source}")
        if features.ndim != 4 or features.shape[1] != self.relation_dim:
            raise ValueError(
                f"features must be [B,{self.relation_dim},H,W], got {tuple(features.shape)}"
            )
        if labels.shape != features.shape[:1] + features.shape[-2:]:
            raise ValueError("anchor labels and features have incompatible shapes")
        if weights.shape != (features.shape[0], 1, *features.shape[-2:]):
            raise ValueError("anchor weights and features have incompatible shapes")
        if not torch.isfinite(features).all():
            raise ValueError("cannot update anchors with non-finite features")
        if not torch.isfinite(weights).all():
            raise ValueError("cannot update anchors with non-finite weights")

        normalized = F.normalize(features.detach().float(), p=2, dim=1, eps=self.eps)
        flat_features = normalized.permute(0, 2, 3, 1).reshape(-1, self.relation_dim)
        flat_labels = labels.detach().reshape(-1).long()
        flat_weights = weights.detach()[:, 0].reshape(-1).float()
        background_safe = background_boundary_mask(labels.detach(), self.background_boundary_exclusion).reshape(-1)
        updated: dict[int, int] = {}
        skipped: list[int] = []

        for class_index in range(self.num_classes):
            eligible = flat_labels.eq(class_index) & flat_weights.gt(0) & torch.isfinite(flat_weights)
            if class_index == 0:
                eligible &= background_safe
            indices = torch.nonzero(eligible, as_tuple=False).flatten()
            if indices.numel() < self.min_support_pixels:
                skipped.append(class_index)
                continue
            indices = indices[: self.max_pixels_per_class]
            selected_features = flat_features.index_select(0, indices)
            selected_weights = flat_weights.index_select(0, indices)
            center = (selected_features * selected_weights.unsqueeze(1)).sum(dim=0)
            weight_sum = selected_weights.sum()
            if not torch.isfinite(center).all() or float(weight_sum) <= self.eps:
                skipped.append(class_index)
                continue
            center = F.normalize((center / weight_sum).unsqueeze(0), p=2, dim=1, eps=self.eps)[0]
            old_anchor = self.anchors[class_index, 0]
            candidate = center if not bool(self.valid[class_index, 0]) else self.momentum * old_anchor + (1.0 - self.momentum) * center
            if float(candidate.norm()) <= self.eps or not torch.isfinite(candidate).all():
                skipped.append(class_index)
                continue
            self.anchors[class_index, 0].copy_(F.normalize(candidate.unsqueeze(0), p=2, dim=1, eps=self.eps)[0])
            self.valid[class_index, 0] = True
            support = int(indices.numel())
            self.counts_total[class_index, 0] += support
            if source == "labeled":
                self.counts_labeled[class_index, 0] += support
            else:
                self.counts_unlabeled[class_index, 0] += support
            self.last_update_step[class_index, 0] = int(step)
            updated[class_index] = support
        return AnchorUpdate(updated_pixels=updated, skipped_classes=skipped)

    def diagnostics(self, *, start_anchors: torch.Tensor | None = None) -> dict[str, Any]:
        norms = self.anchors.norm(dim=-1)
        payload: dict[str, Any] = {
            "valid": self.valid.detach().cpu().tolist(),
            "norms": norms.detach().cpu().tolist(),
            "counts_total": self.counts_total.detach().cpu().tolist(),
            "counts_labeled": self.counts_labeled.detach().cpu().tolist(),
            "counts_unlabeled": self.counts_unlabeled.detach().cpu().tolist(),
            "last_update_step": self.last_update_step.detach().cpu().tolist(),
        }
        if start_anchors is not None:
            if start_anchors.shape != self.anchors.shape:
                raise ValueError("start anchors have wrong shape")
            payload["drift"] = (self.anchors - start_anchors.to(self.anchors)).norm(dim=-1).detach().cpu().tolist()
        return payload
