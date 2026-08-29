"""Shared lifecycle and loss utilities for every continual segmentation method."""
from __future__ import annotations

import copy
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import LabeledBatch, MethodStepOutput, UnlabeledBatch, differentiable_zero
from ..engine.checkpoint import load_checkpoint
from ..engine.metrics import masked_cross_entropy, multiclass_dice_loss
from ..models import UNet2D


DEFAULT_METHOD_CONFIG: dict[str, Any] = {
    "lambda_dice": 1.0,
    "lambda_anchor_sup": 0.1,
    "lambda_assim": 1.0,
    "lambda_relation": 1.0,
    "assim_ramp_steps": 1000,
    "relation_ramp_steps": 1000,
    "tau_cls": 0.95,
    "ssl_temperature": 0.5,
    "uniform_kd_temperature": 0.5,
    "uniform_kd_lambda": 1.0,
}


def merged_method_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_METHOD_CONFIG)
    if config:
        merged.update(dict(config))
    return merged


def clone_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    """Deep clone tensors without aliasing a current/old model or anchor state."""

    cloned: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().clone()
        elif isinstance(value, dict):
            cloned[key] = clone_state_dict(value)
        elif isinstance(value, list):
            cloned[key] = [item.detach().clone() if isinstance(item, torch.Tensor) else copy.deepcopy(item) for item in value]
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def model_checksum(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    model.requires_grad_(False)
    return model


class ContinualSegMethod(nn.Module, ABC):
    """The sole lifecycle interface consumed by the shared continual runner."""

    method_name = "base"
    method_version = "0.1"

    def __init__(self, model: UNet2D, *, config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.model = model
        self.config = merged_method_config(config)
        self.old_model: UNet2D | None = None
        self.site_id: str | None = None
        self.site_index = -1
        self.total_steps = 0
        self._old_model_checksum: str | None = None

    @property
    def num_classes(self) -> int:
        return self.model.num_classes

    def set_site_index(self, site_index: int) -> None:
        self.site_index = int(site_index)

    def _load_previous_model(self, previous_checkpoint: Path) -> dict[str, Any]:
        payload = load_checkpoint(previous_checkpoint, map_location="cpu")
        self.model.load_state_dict(payload["current_model_state"], strict=True)
        return payload

    def _make_old_model(self) -> None:
        self.old_model = freeze_model(copy.deepcopy(self.model))
        for current, old in zip(self.model.parameters(), self.old_model.parameters(), strict=True):
            if current.data_ptr() == old.data_ptr():
                raise AssertionError("current and old model parameters share storage")
        self._old_model_checksum = model_checksum(self.old_model)

    def _restore_old_model(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            self.old_model = None
            self._old_model_checksum = None
            return
        restored = copy.deepcopy(self.model)
        restored.load_state_dict(state, strict=True)
        self.old_model = freeze_model(restored)
        self._old_model_checksum = model_checksum(restored)

    def assert_old_state_unchanged(self) -> None:
        if self.old_model is None:
            return
        if any(parameter.grad is not None for parameter in self.old_model.parameters()):
            raise AssertionError("old model received a gradient")
        if self._old_model_checksum != model_checksum(self.old_model):
            raise AssertionError("old model parameters changed during the current site")

    def begin_site(self, site_id: str, previous_checkpoint: Path | None, total_steps: int) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.site_id = str(site_id)
        self.total_steps = int(total_steps)
        self.old_model = None
        self._old_model_checksum = None
        if previous_checkpoint is not None:
            self._load_previous_model(previous_checkpoint)

    @abstractmethod
    def training_step(
        self,
        labeled_batch: LabeledBatch,
        unlabeled_batch: UnlabeledBatch | None,
        global_step: int,
        site_step: int,
    ) -> MethodStepOutput:
        raise NotImplementedError

    def after_optimizer_step(self) -> None:
        """Hook for no-gradient anchor updates after an optimizer step."""

    def end_site(self, site_id: str) -> dict[str, Any]:
        if self.site_id != site_id:
            raise ValueError(f"attempted to end {site_id}, but active site is {self.site_id}")
        self.assert_old_state_unchanged()
        return {"site_id": site_id, "method_name": self.method_name}

    def method_state_dict(self) -> dict[str, Any]:
        return {
            "current_anchor_state": {},
            "historical_anchor_state": {},
            "bootstrap_state": {},
            "method_statistics": {
                "old_model_state": clone_state_dict(self.old_model.state_dict()) if self.old_model is not None else {},
                "old_model_checksum": self._old_model_checksum or "",
                "active_site_total_steps": self.total_steps,
                "active_site_id": self.site_id or "",
                "active_site_index": self.site_index,
            },
        }

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        statistics = dict(state.get("method_statistics", {}))
        self._restore_old_model(statistics.get("old_model_state") or None)

    def _supervised_losses(self, output: Any, batch: LabeledBatch, *, relation_anchor_loss: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        ce = masked_cross_entropy(output.logits, batch.label, batch.valid_mask)
        dice = multiclass_dice_loss(output.logits, batch.label, batch.valid_mask)
        anchor = relation_anchor_loss if relation_anchor_loss is not None else differentiable_zero(output.logits)
        sup = ce + float(self.config["lambda_dice"]) * dice + float(self.config["lambda_anchor_sup"]) * anchor
        return {
            "loss_sup": sup,
            "loss_seg_ce": ce,
            "loss_seg_dice": dice,
            "loss_anchor_sup": anchor,
        }

    @staticmethod
    def _all_loss_keys(reference: torch.Tensor, partial: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        losses = {
            "loss_sup": differentiable_zero(reference),
            "loss_seg_ce": differentiable_zero(reference),
            "loss_seg_dice": differentiable_zero(reference),
            "loss_anchor_sup": differentiable_zero(reference),
            "loss_assim": differentiable_zero(reference),
            "loss_relation": differentiable_zero(reference),
        }
        losses.update(partial)
        return losses

    def _assimilation_ramp(self, site_step: int, *, bootstrap_complete_at: int = 0) -> float:
        elapsed = max(0, int(site_step) - int(bootstrap_complete_at) + 1)
        return min(1.0, elapsed / max(1, int(self.config["assim_ramp_steps"])))

    def _relation_ramp(self, site_step: int) -> float:
        return min(1.0, float(site_step + 1) / max(1, int(self.config["relation_ramp_steps"])))


def relation_supervision_loss(
    relation_logits: torch.Tensor,
    label: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Visible-GT relation CE on a nearest-downsampled relation grid."""

    if relation_logits.ndim != 4:
        raise ValueError("relation logits must be [B,C,H,W]")
    target = F.interpolate(label.unsqueeze(1).float(), size=relation_logits.shape[-2:], mode="nearest")[:, 0].long()
    valid = F.interpolate(valid_mask.float(), size=relation_logits.shape[-2:], mode="nearest").bool()
    return masked_cross_entropy(relation_logits, target, valid)
