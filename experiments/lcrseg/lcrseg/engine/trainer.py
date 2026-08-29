"""Shared optimizer step implementation for every baseline and LCR-Seg."""
from __future__ import annotations

import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..common import write_json, write_text
from ..contracts import LabeledBatch, MethodStepOutput, UnlabeledBatch
from ..methods.base import ContinualSegMethod


@dataclass
class TrainerState:
    global_step: int = 0
    site_step: int = 0
    epoch: int = 0


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_optimizer(method: ContinualSegMethod, *, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in method.model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("method has no trainable current-model parameters")
    return torch.optim.Adam(parameters, lr=float(lr), weight_decay=float(weight_decay))


def build_scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int) -> torch.optim.lr_scheduler.LRScheduler:
    if total_steps < 1:
        raise ValueError("total steps must be positive")
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(total_steps)))


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all())


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach().float()
    return {
        "shape": list(detached.shape),
        "dtype": str(value.dtype),
        "min": float(detached.min()) if detached.numel() else 0.0,
        "max": float(detached.max()) if detached.numel() else 0.0,
        "mean": float(detached.mean()) if detached.numel() else 0.0,
        "finite": bool(torch.isfinite(detached).all()),
    }


class Trainer:
    """AMP-aware train step with explicit old-model and finite-loss gates."""

    def __init__(
        self,
        method: ContinualSegMethod,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device | str,
        amp: bool = True,
        amp_init_scale: float = 1024.0,
        grad_clip_norm: float | None = None,
    ) -> None:
        self.method = method
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")
        if amp_init_scale <= 0:
            raise ValueError("AMP initial scale must be positive")
        self.amp_init_scale = float(amp_init_scale)
        self.grad_clip_norm = grad_clip_norm
        # The server's frozen PyTorch 2.2 runtime predates the unified
        # ``torch.amp.GradScaler`` API; keep a narrow compatibility fallback.
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp, init_scale=self.amp_init_scale)
        else:  # PyTorch 2.2
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp, init_scale=self.amp_init_scale)

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        if state.get("scaler_state"):
            self.scaler.load_state_dict(state["scaler_state"])

    def failure_bundle(
        self,
        output_dir: Path,
        *,
        error: BaseException,
        labeled_batch: LabeledBatch,
        unlabeled_batch: UnlabeledBatch | None,
        state: TrainerState,
        result: MethodStepOutput | None,
    ) -> Path:
        target = Path(output_dir) / "failure_bundle"
        target.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "error_type": type(error).__name__,
            "error": str(error),
            "site_id": self.method.site_id,
            "epoch": state.epoch,
            "site_step": state.site_step,
            "global_step": state.global_step,
            "labeled_case_ids": labeled_batch.case_id,
            "unlabeled_case_ids": unlabeled_batch.case_id if unlabeled_batch is not None else [],
            "labeled_image": _tensor_summary(labeled_batch.image),
            "labeled_label": _tensor_summary(labeled_batch.label),
        }
        if unlabeled_batch is not None:
            payload["unlabeled_weak_image"] = _tensor_summary(unlabeled_batch.weak_image)
            payload["unlabeled_strong_image"] = _tensor_summary(unlabeled_batch.strong_image)
            payload["unlabeled_valid_mask"] = _tensor_summary(unlabeled_batch.strong_valid_mask)
        if result is not None:
            payload["losses"] = {name: float(value.detach()) for name, value in result.losses.items()}
            payload["scalars"] = result.scalars
            if result.maps:
                payload["maps"] = {name: _tensor_summary(value) for name, value in result.maps.items()}
        write_json(target / "failure.json", payload)
        write_text(target / "error.txt", f"{type(error).__name__}: {error}\n")
        return target

    def train_step(
        self,
        labeled_batch: LabeledBatch,
        unlabeled_batch: UnlabeledBatch | None,
        *,
        state: TrainerState,
        collect_gradient_cosine: bool = False,
    ) -> MethodStepOutput:
        self.method.model.train()
        if self.method.old_model is not None:
            self.method.old_model.eval()
        labeled = labeled_batch.to(self.device, non_blocking=True)
        unlabeled = unlabeled_batch.to(self.device, non_blocking=True) if unlabeled_batch is not None else None
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        step_started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        stochastic_snapshot = (
            self.method.snapshot_stochastic_state()  # type: ignore[attr-defined]
            if hasattr(self.method, "snapshot_stochastic_state")
            else None
        )
        if self.amp:
            context = torch.amp.autocast("cuda", enabled=True) if hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=True)
        else:
            context = nullcontext()
        with context:
            result = self.method.training_step(labeled, unlabeled, state.global_step, state.site_step)
        if not _finite(result.total_loss):
            raise FloatingPointError(f"non-finite total loss at global step {state.global_step}")
        if not all(_finite(value) for value in result.losses.values()):
            raise FloatingPointError(f"non-finite component loss at global step {state.global_step}")
        if collect_gradient_cosine:
            diagnostics = self._assim_relation_gradient_diagnostics(result)
            if diagnostics is not None:
                result.scalars.update(diagnostics)
        self.scaler.scale(result.total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.method.model.parameters(), float(self.grad_clip_norm))
        scale_before = float(self.scaler.get_scale())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        optimizer_step_skipped = bool(self.amp and float(self.scaler.get_scale()) < scale_before)
        if not optimizer_step_skipped:
            self.scheduler.step()
            self.method.after_optimizer_step()
        elif stochastic_snapshot is not None:
            self.method.restore_stochastic_state(stochastic_snapshot)  # type: ignore[attr-defined]
            if bool(getattr(self.method, "fail_on_optimizer_skip", False)):
                raise FloatingPointError("SR-GAS AMP optimizer step skipped; stochastic state restored")
        self.method.assert_old_state_unchanged()
        result.scalars["lr"] = learning_rate
        result.scalars["optimizer_step_skipped"] = float(optimizer_step_skipped)
        result.scalars["training_step_seconds"] = float(time.perf_counter() - step_started)
        result.scalars["peak_memory_bytes"] = float(
            torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        )
        return result

    def _assim_relation_gradient_diagnostics(self, result: MethodStepOutput) -> dict[str, float] | None:
        """Sample unmodified assimilation/relation gradient norms and angle."""

        assimilation = result.losses["loss_assim"]
        relation = result.losses["loss_relation"]
        if not assimilation.requires_grad or not relation.requires_grad:
            return None
        parameters = [parameter for parameter in self.method.model.parameters() if parameter.requires_grad]
        if not parameters:
            return None
        assimilation_gradients = torch.autograd.grad(assimilation, parameters, retain_graph=True, allow_unused=True)
        relation_gradients = torch.autograd.grad(relation, parameters, retain_graph=True, allow_unused=True)
        dot = torch.zeros((), device=self.device)
        norm_a = torch.zeros((), device=self.device)
        norm_r = torch.zeros((), device=self.device)
        for parameter, first, second in zip(parameters, assimilation_gradients, relation_gradients, strict=True):
            first = first.detach().float() if first is not None else torch.zeros_like(parameter, dtype=torch.float32)
            second = second.detach().float() if second is not None else torch.zeros_like(parameter, dtype=torch.float32)
            dot += (first * second).sum()
            norm_a += first.square().sum()
            norm_r += second.square().sum()
        denominator = norm_a.sqrt() * norm_r.sqrt()
        if not bool(denominator.gt(1e-12)):
            return None
        return {
            "gradient_norm_assim": float(norm_a.sqrt().cpu()),
            "gradient_norm_relation": float(norm_r.sqrt().cpu()),
            "gradient_cosine_assim_relation": float((dot / denominator).clamp(-1.0, 1.0).cpu()),
        }
