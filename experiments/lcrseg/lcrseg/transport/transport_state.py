"""Serializable immutable transport view for one site epoch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class TransportState:
    site_id: str
    site_index: int
    epoch: int
    variant: str
    transported_anchors: torch.Tensor
    class_deltas: torch.Tensor
    global_delta: torch.Tensor
    paired_case_counts: torch.Tensor

    def __post_init__(self) -> None:
        if self.variant not in {"T1", "T2", "T3", "T3_SHIFT_SWAP"}:
            raise ValueError("invalid transport variant")
        tensors = (self.transported_anchors, self.class_deltas, self.global_delta)
        if any(value.requires_grad for value in tensors):
            raise ValueError("transport state must be detached")
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("transport state contains non-finite values")

    def state_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "site_index": int(self.site_index),
            "epoch": int(self.epoch),
            "variant": self.variant,
            "transported_anchors": self.transported_anchors.detach().clone(),
            "class_deltas": self.class_deltas.detach().clone(),
            "global_delta": self.global_delta.detach().clone(),
            "paired_case_counts": self.paired_case_counts.detach().clone(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "TransportState":
        return cls(
            site_id=str(state["site_id"]),
            site_index=int(state["site_index"]),
            epoch=int(state["epoch"]),
            variant=str(state["variant"]),
            transported_anchors=state["transported_anchors"].detach().clone(),
            class_deltas=state["class_deltas"].detach().clone(),
            global_delta=state["global_delta"].detach().clone(),
            paired_case_counts=state["paired_case_counts"].detach().clone(),
        )
