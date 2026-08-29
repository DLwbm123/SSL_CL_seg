"""Standalone online EWC control baseline; never part of LCR-Seg internals."""
from __future__ import annotations

from typing import Any, Mapping

import torch

from ..data.continual_sampler import DeterministicBatcher
from ..engine.checkpoint import load_checkpoint
from .base import clone_state_dict
from .sequential_ssl import SequentialSSLMethod


EWC_DEFAULTS: dict[str, Any] = {
    "ewc_lambda": 0.1,
    "ewc_gamma": 1.0,
    "ewc_fisher_batches": 8,
}


class EWCSegMethod(SequentialSSLMethod):
    """Sequential SSL plus an online diagonal-Fisher EWC penalty."""

    method_name = "ss_ewc"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        supplied = dict(kwargs.pop("config", None) or {})
        config = dict(EWC_DEFAULTS)
        config.update(supplied)
        super().__init__(*args, config=config, **kwargs)
        self.method_name = "ss_ewc"
        self.ewc_lambda = float(self.config["ewc_lambda"])
        self.ewc_gamma = float(self.config["ewc_gamma"])
        self.fisher_batches = int(self.config["ewc_fisher_batches"])
        self.reference_parameters: dict[str, torch.Tensor] = {}
        self.fisher_diagonal: dict[str, torch.Tensor] = {}

    def _load_ewc_state(self, state: Mapping[str, Any] | None) -> None:
        state = state or {}
        reference = state.get("reference_parameters") or {}
        fisher = state.get("fisher_diagonal") or {}
        self.reference_parameters = {name: value.detach().clone().to(next(self.model.parameters()).device) for name, value in reference.items()}
        self.fisher_diagonal = {name: value.detach().clone().to(next(self.model.parameters()).device) for name, value in fisher.items()}

    def begin_site(self, site_id: str, previous_checkpoint, total_steps: int) -> None:  # type: ignore[override]
        super().begin_site(site_id, previous_checkpoint, total_steps)
        if previous_checkpoint is None:
            self.reference_parameters = {}
            self.fisher_diagonal = {}
            return
        payload = load_checkpoint(previous_checkpoint, map_location="cpu")
        self._load_ewc_state((payload.get("method_statistics") or {}).get("ewc_state"))

    def _ewc_loss(self, reference: torch.Tensor) -> torch.Tensor:
        if not self.reference_parameters or not self.fisher_diagonal:
            return reference.sum() * 0.0
        penalty = reference.sum() * 0.0
        for name, parameter in self.model.named_parameters():
            fisher = self.fisher_diagonal.get(name)
            previous = self.reference_parameters.get(name)
            if fisher is None or previous is None:
                continue
            penalty = penalty + (fisher.to(parameter) * (parameter - previous.to(parameter)).square()).sum()
        return penalty

    def training_step(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        result = super().training_step(*args, **kwargs)
        ewc = self._ewc_loss(result.total_loss)
        losses = dict(result.losses)
        losses["loss_relation"] = ewc
        total = result.total_loss + self.ewc_lambda * ewc
        scalars = dict(result.scalars)
        scalars["loss_ewc"] = float(ewc.detach())
        return type(result)(total_loss=total, losses=losses, scalars=scalars, maps=result.maps)

    @torch.no_grad()
    def _snapshot_reference(self) -> None:
        self.reference_parameters = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}

    def estimate_fisher(self, labeled_batcher: DeterministicBatcher, *, device: torch.device | str) -> dict[str, float]:
        """Estimate diagonal Fisher from visible labeled data after a site."""

        device = torch.device(device)
        estimates = {name: torch.zeros_like(parameter, dtype=torch.float32, device=device) for name, parameter in self.model.named_parameters()}
        count = min(max(1, self.fisher_batches), labeled_batcher.steps_per_epoch)
        was_training = self.model.training
        self.model.eval()
        for index in range(count):
            self.model.zero_grad(set_to_none=True)
            batch = labeled_batcher.batch_at(index).to(device)
            output = self.model(batch.image)
            losses = self._supervised_losses(output, batch)
            losses["loss_sup"].backward()
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    estimates[name].add_(parameter.grad.detach().float().square())
        if was_training:
            self.model.train()
        updated: dict[str, torch.Tensor] = {}
        for name, estimate in estimates.items():
            estimate.div_(float(count))
            previous = self.fisher_diagonal.get(name)
            updated[name] = estimate if previous is None else self.ewc_gamma * previous.to(estimate) + estimate
        self.fisher_diagonal = updated
        self._snapshot_reference()
        return {"fisher_batches": float(count), "fisher_mean": float(torch.cat([value.flatten() for value in updated.values()]).mean())}

    def method_state_dict(self) -> dict[str, Any]:
        state = super().method_state_dict()
        statistics = dict(state["method_statistics"])
        statistics["ewc_state"] = {
            "reference_parameters": clone_state_dict(self.reference_parameters),
            "fisher_diagonal": clone_state_dict(self.fisher_diagonal),
        }
        state["method_statistics"] = statistics
        return state

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_method_state_dict(state)
        self._load_ewc_state((state.get("method_statistics") or {}).get("ewc_state"))
