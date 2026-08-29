"""All-class, case-balanced anchor transport used by TARC feasibility audits."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ..memory.prototype_transport import TransportEstimate, estimate_transport


@dataclass(frozen=True)
class CasePrototypeBatch:
    """One normalized prototype per case and class where support is sufficient."""

    prototypes: torch.Tensor  # [N,C,D]
    valid: torch.Tensor  # [N,C]
    pixel_counts: torch.Tensor  # [N,C]


@dataclass(frozen=True)
class AllClassTransport:
    """Detached class-specific and class-agnostic transport estimates."""

    class_estimates: tuple[TransportEstimate, ...]
    global_estimate: TransportEstimate
    class_deltas: torch.Tensor  # [C,D]
    global_delta: torch.Tensor  # [D]
    paired_case_counts: torch.Tensor  # [C]
    global_case_count: int


@torch.no_grad()
def build_case_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    minimum_pixels: int = 32,
    eps: float = 1.0e-8,
) -> CasePrototypeBatch:
    """Build case-equal class prototypes from one relation-feature batch.

    Each batch item is a case. Pixel counts are used only as an eligibility
    threshold; every eligible case contributes exactly one normalized vector.
    """

    if features.ndim != 4:
        raise ValueError("features must have [N,D,H,W] shape")
    if labels.shape != (features.shape[0], *features.shape[-2:]):
        raise ValueError("labels must match the feature grid")
    if num_classes < 2 or minimum_pixels < 1:
        raise ValueError("invalid all-class transport contract")
    if not torch.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError("labels contain an out-of-range class id")

    normalized = F.normalize(features.detach().float(), p=2, dim=1, eps=eps)
    prototypes = torch.zeros(
        (features.shape[0], num_classes, features.shape[1]),
        dtype=torch.float32,
        device=features.device,
    )
    valid = torch.zeros((features.shape[0], num_classes), dtype=torch.bool, device=features.device)
    counts = torch.zeros((features.shape[0], num_classes), dtype=torch.int64, device=features.device)
    for case_index in range(features.shape[0]):
        case_features = normalized[case_index].permute(1, 2, 0)
        for class_id in range(num_classes):
            mask = labels[case_index].eq(class_id)
            count = int(mask.sum())
            counts[case_index, class_id] = count
            if count < minimum_pixels:
                continue
            center = case_features[mask].mean(dim=0)
            if not torch.isfinite(center).all() or float(center.norm()) <= eps:
                continue
            prototypes[case_index, class_id] = F.normalize(center[None], p=2, dim=1, eps=eps)[0]
            valid[case_index, class_id] = True
    return CasePrototypeBatch(prototypes=prototypes.detach(), valid=valid.detach(), pixel_counts=counts.detach())


@torch.no_grad()
def estimate_all_class_transport(
    old: CasePrototypeBatch,
    current: CasePrototypeBatch,
    *,
    eps: float = 1.0e-8,
) -> AllClassTransport:
    """Estimate per-class and global shrinkage without a strength parameter."""

    if old.prototypes.shape != current.prototypes.shape or old.valid.shape != current.valid.shape:
        raise ValueError("old/current case prototype batches must be aligned")
    if old.prototypes.ndim != 3 or old.valid.shape != old.prototypes.shape[:2]:
        raise ValueError("case prototypes must have [N,C,D] plus [N,C] validity")
    if not torch.isfinite(old.prototypes).all() or not torch.isfinite(current.prototypes).all():
        raise ValueError("case prototypes contain non-finite values")

    paired = old.valid & current.valid
    estimates: list[TransportEstimate] = []
    deltas: list[torch.Tensor] = []
    case_counts: list[int] = []
    for class_id in range(old.prototypes.shape[1]):
        mask = paired[:, class_id]
        estimate = estimate_transport(old.prototypes[mask, class_id], current.prototypes[mask, class_id], eps=eps)
        estimates.append(estimate)
        deltas.append(estimate.delta)
        case_counts.append(int(mask.sum()))

    # The global displacement is class-balanced within each case. A case is
    # eligible only when all classes, including background, are paired.
    global_mask = paired.all(dim=1)
    old_global = old.prototypes[global_mask]
    current_global = current.prototypes[global_mask]
    if int(global_mask.sum()):
        old_displacement_base = old_global
        current_displacement_base = current_global
        old_case = torch.zeros((old_global.shape[0], old_global.shape[2]), device=old_global.device)
        current_case = torch.zeros_like(old_case)
        # estimate_transport consumes normalized points. Construct paired
        # points whose difference is exactly the class-mean displacement by
        # adding that displacement to a fixed unit origin; the direct formula
        # below then preserves the ASPR shrinkage semantics exactly.
        displacement = (current_displacement_base - old_displacement_base).mean(dim=1)
        origin = F.normalize(old_displacement_base.mean(dim=1), p=2, dim=1, eps=eps)
        old_case.copy_(origin)
        current_case.copy_(origin + displacement)
        normalized_current = F.normalize(current_case, p=2, dim=1, eps=eps)
        # Correct for normalization so the generic estimator sees the direct
        # class-balanced displacement specified by the protocol.
        direct = displacement
        count = int(direct.shape[0])
        dimension = int(direct.shape[1])
        zero = torch.zeros(dimension, dtype=torch.float32, device=direct.device)
        if count < 2:
            global_estimate = TransportEstimate(count, zero, zero, 0.0, zero, 0.0, 0.0, False)
        else:
            mean = direct.mean(dim=0)
            variance_tensor = (direct - mean).square().sum(dim=1).sum() / float(count - 1)
            signal_tensor = mean.square().sum()
            finite = bool(torch.isfinite(direct).all() and torch.isfinite(variance_tensor) and torch.isfinite(signal_tensor))
            variance = float(variance_tensor) if finite else 0.0
            signal = float(signal_tensor) if finite else 0.0
            if not finite or signal < eps:
                global_estimate = TransportEstimate(count, zero, zero, 0.0, zero, variance, signal, False)
            else:
                rho = max(0.0, min(1.0, signal / (signal + variance / float(count) + eps)))
                delta = (rho * mean).detach()
                global_estimate = TransportEstimate(count, mean.detach(), mean.detach(), rho, delta, variance, signal, True)
        del normalized_current  # documents that no model-space point is persisted
    else:
        dimension = int(old.prototypes.shape[2])
        zero = torch.zeros(dimension, dtype=torch.float32, device=old.prototypes.device)
        global_estimate = TransportEstimate(0, zero, zero, 0.0, zero, 0.0, 0.0, False)

    return AllClassTransport(
        class_estimates=tuple(estimates),
        global_estimate=global_estimate,
        class_deltas=torch.stack(deltas).detach(),
        global_delta=global_estimate.delta.detach(),
        paired_case_counts=torch.tensor(case_counts, dtype=torch.int64, device=old.prototypes.device),
        global_case_count=int(global_mask.sum()),
    )


@torch.no_grad()
def transport_anchors(anchors: torch.Tensor, delta: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    """Return a normalized detached view; never mutate the historical bank."""

    if anchors.ndim not in {2, 3}:
        raise ValueError("anchors must be [C,D] or [C,K,D]")
    view = anchors[:, 0] if anchors.ndim == 3 else anchors
    if delta.ndim == 1:
        delta = delta[None].expand(view.shape[0], -1)
    if delta.shape != view.shape:
        raise ValueError("transport delta has incompatible all-class shape")
    candidate = view.detach().float() + delta.detach().float()
    if not torch.isfinite(candidate).all() or bool(candidate.norm(dim=1).le(eps).any()):
        raise FloatingPointError("transport produced an invalid anchor direction")
    transported = F.normalize(candidate, p=2, dim=1, eps=eps).detach()
    return transported[:, None] if anchors.ndim == 3 else transported


@torch.no_grad()
def swap_fundus_foreground_deltas(class_deltas: torch.Tensor) -> torch.Tensor:
    """Keep background fixed and swap only disc-rim/cup displacements."""

    if class_deltas.ndim != 2 or class_deltas.shape[0] != 3:
        raise ValueError("Fundus shift-swap expects exactly three class deltas")
    swapped = class_deltas.detach().clone()
    swapped[1] = class_deltas[2]
    swapped[2] = class_deltas[1]
    return swapped.detach()
