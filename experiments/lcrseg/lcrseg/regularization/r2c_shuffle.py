"""Deterministic, checkpointable spatial shuffle for the V0.1a control."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import torch


class SpatialRelationShuffler:
    def __init__(self, *, protocol_seed: int) -> None:
        self.protocol_seed = int(protocol_seed)
        self._permutations: dict[str, torch.Tensor] = {}

    def _key(self, site_id: str, height: int, width: int) -> str:
        return f"{self.protocol_seed}:{site_id}:{height}:{width}"

    def permutation(self, *, site_id: str, height: int, width: int) -> torch.Tensor:
        if height < 1 or width < 1:
            raise ValueError("shuffle grid must be positive")
        key = self._key(site_id, height, width)
        if key not in self._permutations:
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
            self._permutations[key] = torch.randperm(height * width, generator=generator)
        return self._permutations[key].clone()

    def shuffle(self, probability: torch.Tensor, *, site_id: str) -> torch.Tensor:
        if probability.ndim != 4:
            raise ValueError("relation probability must be [B,C,H,W]")
        height, width = probability.shape[-2:]
        permutation = self.permutation(site_id=site_id, height=height, width=width).to(probability.device)
        return probability.flatten(2).index_select(2, permutation).view_as(probability)

    def state_dict(self) -> dict[str, Any]:
        permutations = {key: value.clone() for key, value in self._permutations.items()}
        hashes = {
            key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
            for key, value in sorted(permutations.items())
        }
        return {"protocol_seed": self.protocol_seed, "permutations": permutations, "hashes": hashes}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        if int(state["protocol_seed"]) != self.protocol_seed:
            raise ValueError("shuffle checkpoint protocol seed mismatch")
        restored = {str(key): value.detach().cpu().long().clone() for key, value in dict(state["permutations"]).items()}
        expected = dict(state.get("hashes") or {})
        for key, value in restored.items():
            actual = hashlib.sha256(value.numpy().tobytes()).hexdigest()
            if expected and expected.get(key) != actual:
                raise ValueError("shuffle permutation checkpoint hash mismatch")
        self._permutations = restored


__all__ = ["SpatialRelationShuffler"]
