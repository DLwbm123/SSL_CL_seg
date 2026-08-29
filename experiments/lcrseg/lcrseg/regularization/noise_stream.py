"""Stateless shared standard-normal stream for SR-GAS V0.2 variants."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch


@dataclass
class SharedNoiseStream:
    protocol_seed: int
    split_seed: int
    hashes: dict[str, str] = field(default_factory=dict)

    def _key(self, *, site_id: str, successful_site_step: int, weight_shape: Iterable[int]) -> tuple[str, int]:
        shape = [int(value) for value in weight_shape]
        payload = {
            "protocol_seed": int(self.protocol_seed),
            "split_seed": int(self.split_seed),
            "site_id": str(site_id),
            "successful_site_step": int(successful_site_step),
            "weight_shape": shape,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        seed = int(digest[:16], 16) % (2**63 - 1)
        return digest, seed

    def sample(
        self,
        *,
        site_id: str,
        successful_site_step: int,
        weight_shape: Iterable[int],
        device: torch.device | str,
    ) -> tuple[torch.Tensor, str]:
        if successful_site_step < 0:
            raise ValueError("successful site step cannot be negative")
        shape = tuple(int(value) for value in weight_shape)
        key_hash, seed = self._key(
            site_id=site_id,
            successful_site_step=successful_site_step,
            weight_shape=shape,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        cpu_noise = torch.randn(shape, dtype=torch.float32, device="cpu", generator=generator)
        tensor_hash = hashlib.sha256(cpu_noise.numpy().tobytes()).hexdigest()
        existing = self.hashes.get(key_hash)
        if existing is not None and existing != tensor_hash:
            raise AssertionError("shared noise stream changed for an existing key")
        self.hashes[key_hash] = tensor_hash
        return cpu_noise.to(device=torch.device(device)), tensor_hash

    def state_dict(self) -> dict[str, Any]:
        return {
            "protocol_seed": int(self.protocol_seed),
            "split_seed": int(self.split_seed),
            "hashes": dict(self.hashes),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        if int(state["protocol_seed"]) != int(self.protocol_seed) or int(state["split_seed"]) != int(self.split_seed):
            raise ValueError("shared noise stream seed differs from resolved protocol")
        self.hashes = {str(key): str(value) for key, value in dict(state.get("hashes") or {}).items()}


__all__ = ["SharedNoiseStream"]
