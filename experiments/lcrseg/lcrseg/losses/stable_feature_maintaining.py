"""Targeted same-layer feature maintaining primitives for SPARC V0.1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class StableFeatureMaintainingOutput:
    loss: torch.Tensor
    layer_losses: dict[str, torch.Tensor]
    class_layer_losses: dict[str, dict[int, torch.Tensor]]
    kappa: torch.Tensor
    present_classes: tuple[int, ...]


def evidence_coefficient(
    stable_mask: torch.Tensor,
    current_valid: torch.Tensor,
    current_class: torch.Tensor,
    *,
    foreground_class_ids: Iterable[int] = (1, 2),
) -> tuple[torch.Tensor, tuple[int, ...]]:
    stable = stable_mask.detach().bool()
    valid = current_valid.detach().bool()
    classes = current_class.detach().long()
    if stable.shape != valid.shape or stable.shape != classes.shape:
        raise ValueError("stable/current-valid/class grids must match")
    ratios: list[torch.Tensor] = []
    present: list[int] = []
    for raw_class_id in foreground_class_ids:
        class_id = int(raw_class_id)
        denominator_mask = valid & classes.eq(class_id)
        denominator = denominator_mask.sum()
        if int(denominator) == 0:
            continue
        numerator = (stable & denominator_mask).sum()
        ratios.append(numerator.float() / denominator.float())
        present.append(class_id)
    if not ratios:
        return torch.zeros((), dtype=torch.float32, device=stable.device), ()
    kappa = torch.stack(ratios).mean().clamp(0.0, 1.0).detach()
    return kappa, tuple(present)


def stable_feature_maintaining(
    current_features: Mapping[str, torch.Tensor],
    previous_features: Mapping[str, torch.Tensor],
    feature_mask: torch.Tensor,
    class_map: torch.Tensor,
    current_valid: torch.Tensor,
    *,
    foreground_class_ids: Iterable[int] = (1, 2),
    layer_names: tuple[str, str] = ("dec3", "dec1"),
    class_balanced: bool = True,
    foreground_only: bool = True,
) -> StableFeatureMaintainingOutput:
    if layer_names != ("dec3", "dec1"):
        raise ValueError("SPARC V0.1 freezes feature layers at dec3 and dec1")
    if feature_mask.shape != class_map.shape or feature_mask.shape != current_valid.shape:
        raise ValueError("feature mask, class map, and current-valid grid must match")
    mask = feature_mask.detach().bool()
    labels = class_map.detach().long()
    valid = current_valid.detach().bool()
    foreground = tuple(int(value) for value in foreground_class_ids)
    kappa, present_valid = evidence_coefficient(mask, valid, labels, foreground_class_ids=foreground)
    per_layer: dict[str, torch.Tensor] = {}
    per_class_layer: dict[str, dict[int, torch.Tensor]] = {}
    selected_present: set[int] = set()
    for layer_name in layer_names:
        if layer_name not in current_features or layer_name not in previous_features:
            raise KeyError(f"missing same-name feature layer: {layer_name}")
        current = current_features[layer_name]
        previous = previous_features[layer_name].detach()
        if current.shape != previous.shape or current.ndim != 4:
            raise ValueError(f"old/current {layer_name} feature shapes differ")
        layer_size = tuple(current.shape[-2:])
        layer_mask = F.interpolate(mask[:, None].float(), size=layer_size, mode="nearest")[:, 0].bool()
        layer_labels = F.interpolate(labels[:, None].float(), size=layer_size, mode="nearest")[:, 0].long()
        current_normalized = F.normalize(current.float(), p=2, dim=1, eps=1.0e-8)
        previous_normalized = F.normalize(previous.float(), p=2, dim=1, eps=1.0e-8)
        distance = 1.0 - (current_normalized * previous_normalized).sum(dim=1)
        class_losses: dict[int, torch.Tensor] = {}
        if class_balanced:
            class_ids = foreground if foreground_only else tuple(int(value) for value in torch.unique(layer_labels).tolist())
            for class_id in class_ids:
                selected = layer_mask & layer_labels.eq(class_id)
                if not bool(selected.any()):
                    continue
                class_losses[class_id] = distance[selected].mean()
                selected_present.add(class_id)
            per_layer[layer_name] = torch.stack(list(class_losses.values())).mean() if class_losses else current.sum() * 0.0
        else:
            selected = layer_mask
            if foreground_only:
                foreground_mask = torch.zeros_like(selected)
                for class_id in foreground:
                    foreground_mask |= layer_labels.eq(class_id)
                selected &= foreground_mask
            per_layer[layer_name] = distance[selected].mean() if bool(selected.any()) else current.sum() * 0.0
        per_class_layer[layer_name] = class_losses
    loss = torch.stack([per_layer[name] for name in layer_names]).mean()
    return StableFeatureMaintainingOutput(
        loss=loss,
        layer_losses=per_layer,
        class_layer_losses=per_class_layer,
        kappa=kappa,
        present_classes=tuple(sorted(selected_present or set(present_valid))),
    )
