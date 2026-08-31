"""Shared weak-to-strong SSL and uniform-KD baseline implementations."""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from ..contracts import LabeledBatch, MethodStepOutput, UnlabeledBatch, differentiable_zero
from .base import ContinualSegMethod, freeze_model
from .components.routing import weighted_mean


class SequentialSSLMethod(ContinualSegMethod):
    """Current-model confidence-thresholded weak-to-strong SSL baseline."""

    method_name = "sequential_ssl"

    def __init__(self, *args: Any, static: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.static = bool(static)
        self.method_name = "static_ssl" if self.static else type(self).method_name

    @torch.no_grad()
    def _weak_pseudo(self, unlabeled_batch: UnlabeledBatch) -> tuple[torch.Tensor, torch.Tensor]:
        weak_output = self.model(unlabeled_batch.weak_image)
        probabilities = weak_output.logits.detach().softmax(dim=1)
        confidence, labels = probabilities.max(dim=1)
        valid = confidence.ge(float(self.config["tau_cls"])).unsqueeze(1)
        return labels.detach(), valid.detach()

    def _ssl_loss(self, strong_logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor, strong_valid_mask: torch.Tensor) -> torch.Tensor:
        weight = valid.float() * strong_valid_mask.bool().float()
        per_pixel = F.cross_entropy(strong_logits, labels, reduction="none").unsqueeze(1)
        return weighted_mean(per_pixel, weight, reference=strong_logits)

    def training_step(
        self,
        labeled_batch: LabeledBatch,
        unlabeled_batch: UnlabeledBatch | None,
        global_step: int,
        site_step: int,
    ) -> MethodStepOutput:
        if unlabeled_batch is None:
            raise ValueError(f"{self.method_name} requires an unlabeled batch")
        labeled_output = self.model(labeled_batch.image)
        losses = self._all_loss_keys(labeled_output.logits, self._supervised_losses(labeled_output, labeled_batch))
        labels, valid = self._weak_pseudo(unlabeled_batch)
        strong_output = self.model(unlabeled_batch.strong_image)
        assimilation = self._ssl_loss(strong_output.logits, labels, valid, unlabeled_batch.strong_valid_mask)
        effective_lambda = float(self.config["lambda_assim"]) * self._assimilation_ramp(site_step)
        losses["loss_assim"] = assimilation
        total = losses["loss_sup"] + effective_lambda * assimilation
        return MethodStepOutput(
            total_loss=total,
            losses=losses,
            scalars={
                "pseudo_valid_ratio": float(valid.float().mean().detach()),
                "pseudo_classifier_ratio": float(valid.float().mean().detach()),
                "pseudo_anchor_ratio": 0.0,
                "pseudo_deferred_ratio": float((~valid).float().mean().detach()),
                "learnability_mean": float(valid.float().mean().detach()),
                "compatibility_mean": 0.0,
                "lambda_assim_effective": effective_lambda,
            },
        )


class UniformKDMethod(SequentialSSLMethod):
    """Frozen-previous-model LwF control with unconditioned current-input KD."""

    method_name = "uniform_kd"

    def begin_site(self, site_id: str, previous_checkpoint, total_steps: int) -> None:  # type: ignore[override]
        super().begin_site(site_id, previous_checkpoint, total_steps)
        if previous_checkpoint is not None:
            self._make_old_model()

    def _uniform_kd_loss(self, strong_logits: torch.Tensor, old_weak_logits: torch.Tensor, strong_valid_mask: torch.Tensor) -> torch.Tensor:
        temperature = float(self.config["uniform_kd_temperature"])
        if temperature <= 0:
            raise ValueError("uniform KD temperature must be positive")
        old_probability = old_weak_logits.detach().float().div(temperature).softmax(dim=1)
        current_log_probability = strong_logits.float().div(temperature).log_softmax(dim=1)
        per_pixel = (old_probability * (old_probability.clamp_min(1e-8).log() - current_log_probability)).sum(dim=1, keepdim=True)
        return temperature**2 * weighted_mean(per_pixel, strong_valid_mask.float(), reference=strong_logits)

    def training_step(
        self,
        labeled_batch: LabeledBatch,
        unlabeled_batch: UnlabeledBatch | None,
        global_step: int,
        site_step: int,
    ) -> MethodStepOutput:
        result = super().training_step(labeled_batch, unlabeled_batch, global_step, site_step)
        if self.old_model is None or unlabeled_batch is None:
            return result
        with torch.no_grad():
            old_output = self.old_model(unlabeled_batch.weak_image)
        strong_output = self.model(unlabeled_batch.strong_image)
        kd = self._uniform_kd_loss(strong_output.logits, old_output.logits, unlabeled_batch.strong_valid_mask)
        losses = dict(result.losses)
        losses["loss_relation"] = kd
        total = result.total_loss + float(self.config["uniform_kd_lambda"]) * self._relation_ramp(site_step) * kd
        scalars = dict(result.scalars)
        scalars.update({"uniform_kd_loss": float(kd.detach()), "lambda_relation_effective": float(self.config["uniform_kd_lambda"]) * self._relation_ramp(site_step)})
        return MethodStepOutput(total_loss=total, losses=losses, scalars=scalars)

    def end_site(self, site_id: str) -> dict[str, Any]:
        result = super().end_site(site_id)
        self.assert_old_state_unchanged()
        return result
