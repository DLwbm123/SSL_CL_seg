"""Successful-step lagged sensitivity state for SR-GAS V0.2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass
class LaggedSensitivityState:
    buffer: torch.Tensor
    valid: bool = False
    site_id: str | None = None
    successful_site_step: int = 0

    @classmethod
    def empty(cls) -> "LaggedSensitivityState":
        return cls(buffer=torch.empty(0), valid=False, site_id=None, successful_site_step=0)

    def reset_for_site(self, *, site_id: str, reference: torch.Tensor) -> None:
        self.buffer = torch.ones_like(reference, dtype=torch.float32).detach()
        self.valid = False
        self.site_id = str(site_id)
        self.successful_site_step = 0

    def current_or_ones(self, reference: torch.Tensor) -> torch.Tensor:
        if self.valid:
            if self.buffer.shape != reference.shape:
                raise ValueError("lagged sensitivity buffer shape differs from classifier weight")
            value = self.buffer.to(device=reference.device, dtype=torch.float32)
        else:
            value = torch.ones_like(reference, dtype=torch.float32)
        if value.requires_grad or not bool(torch.isfinite(value).all()) or bool(value.lt(0).any()):
            raise ValueError("lagged sensitivity must be detached, finite, and nonnegative")
        return value.detach()

    def commit_after_success(self, pending: torch.Tensor) -> None:
        value = pending.detach().float()
        if self.site_id is None:
            raise RuntimeError("lagged sensitivity state has not been reset for a site")
        if value.shape != self.buffer.shape:
            raise ValueError("pending sensitivity shape differs from lag buffer")
        if not bool(torch.isfinite(value).all()) or bool(value.lt(0).any()):
            raise ValueError("pending sensitivity must be finite and nonnegative")
        self.buffer = value.cpu().clone()
        self.valid = True
        self.successful_site_step += 1

    def advance_without_commit(self) -> None:
        if self.site_id is None:
            raise RuntimeError("lagged sensitivity state has not been reset for a site")
        self.successful_site_step += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "buffer": self.buffer.detach().cpu().clone(),
            "valid": bool(self.valid),
            "site_id": self.site_id,
            "successful_site_step": int(self.successful_site_step),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        buffer = state.get("buffer", torch.empty(0))
        if not isinstance(buffer, torch.Tensor):
            raise TypeError("lagged sensitivity checkpoint buffer must be a tensor")
        self.buffer = buffer.detach().float().cpu().clone()
        self.valid = bool(state.get("valid", False))
        self.site_id = str(state["site_id"]) if state.get("site_id") is not None else None
        self.successful_site_step = int(state.get("successful_site_step", 0))
        if self.successful_site_step < 0:
            raise ValueError("successful site step cannot be negative")


__all__ = ["LaggedSensitivityState"]
