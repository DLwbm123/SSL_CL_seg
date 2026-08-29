"""CRISP audit-only IFC/PFC losses on frozen decoder feature paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ..representation.channel_roles import FEATURE_LAYERS, ROLE_EPSILON, centered_channel_unit


@dataclass(frozen=True)
class ChannelRoleConsistencyOutput:
    loss: torch.Tensor
    layer_losses: dict[str, torch.Tensor]
    channel_distances: dict[str, torch.Tensor]


def _channel_cosine_distance(
    first: torch.Tensor,
    second: torch.Tensor,
    valid_mask: torch.Tensor | None,
    *,
    stop_second: bool,
    epsilon: float,
) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError("paired CRISP features must have identical shapes")
    second_value = second.detach() if stop_second else second
    first_unit, first_valid = centered_channel_unit(first, valid_mask, epsilon=epsilon)
    second_unit, second_valid = centered_channel_unit(second_value, valid_mask, epsilon=epsilon)
    usable = first_valid & second_valid
    if not bool(usable.any()):
        return first.sum(dim=(0, 2, 3)) * 0.0
    distance = 1.0 - (first_unit * second_unit).sum(dim=(-2, -1))
    return distance[usable].mean(dim=0)


def _role_loss(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor],
    valid_mask: torch.Tensor | None,
    *,
    stop_second: bool,
    epsilon: float,
) -> ChannelRoleConsistencyOutput:
    if tuple(first) != FEATURE_LAYERS or tuple(second) != FEATURE_LAYERS or tuple(weights) != FEATURE_LAYERS:
        raise ValueError(f"CRISP loss requires layers in order {FEATURE_LAYERS}")
    layer_losses: dict[str, torch.Tensor] = {}
    distances: dict[str, torch.Tensor] = {}
    for layer in FEATURE_LAYERS:
        distance = _channel_cosine_distance(
            first[layer], second[layer], valid_mask, stop_second=stop_second, epsilon=epsilon
        )
        weight = weights[layer].detach().to(device=distance.device, dtype=distance.dtype)
        if weight.ndim != 1 or weight.numel() != distance.numel():
            raise ValueError(f"role/channel mismatch for {layer}")
        if bool((weight < 0).any()) or not bool(torch.isfinite(weight).all()):
            raise ValueError(f"invalid role weights for {layer}")
        layer_losses[layer] = (weight * distance).sum() / (weight.sum() + float(epsilon))
        distances[layer] = distance
    loss = 0.5 * (layer_losses["dec3"] + layer_losses["dec1"])
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite channel-role consistency loss")
    return ChannelRoleConsistencyOutput(loss=loss, layer_losses=layer_losses, channel_distances=distances)


def invariant_feature_consolidation(
    current_features: Mapping[str, torch.Tensor],
    previous_features: Mapping[str, torch.Tensor],
    invariant_weights: Mapping[str, torch.Tensor],
    valid_mask: torch.Tensor | None = None,
    *,
    epsilon: float = ROLE_EPSILON,
) -> ChannelRoleConsistencyOutput:
    """Content-invariant consolidation with a stop-gradient previous branch."""

    return _role_loss(
        current_features,
        previous_features,
        invariant_weights,
        valid_mask,
        stop_second=True,
        epsilon=epsilon,
    )


def plastic_feature_consistency(
    weak_features: Mapping[str, torch.Tensor],
    strong_features: Mapping[str, torch.Tensor],
    plastic_weights: Mapping[str, torch.Tensor],
    valid_mask: torch.Tensor,
    *,
    epsilon: float = ROLE_EPSILON,
) -> ChannelRoleConsistencyOutput:
    """Style-plastic consistency with gradients through both current branches."""

    return _role_loss(
        weak_features,
        strong_features,
        plastic_weights,
        valid_mask,
        stop_second=False,
        epsilon=epsilon,
    )


__all__ = [
    "ChannelRoleConsistencyOutput",
    "invariant_feature_consolidation",
    "plastic_feature_consistency",
]
