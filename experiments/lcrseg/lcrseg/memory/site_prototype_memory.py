"""Append-only foreground site-prototype memory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .prototype_transport import transport_prototypes


@dataclass(frozen=True)
class SitePrototypeRecord:
    site_id: str
    class_id: int
    prototype: torch.Tensor
    dispersion: torch.Tensor
    labeled_case_count: int
    unlabeled_case_count: int
    labeled_pixel_weight: float
    unlabeled_pixel_weight: float
    source_checkpoint_sha256: str
    class_semantics_sha256: str
    feature_dim: int


class SitePrototypeMemory(nn.Module):
    def __init__(self, feature_dim: int, foreground_ids: tuple[int, ...] = (1, 2), *, background_id: int = 0, eps: float = 1.0e-8) -> None:
        super().__init__()
        if feature_dim < 1 or background_id in foreground_ids:
            raise ValueError("invalid ASPR memory class/dimension contract")
        self.feature_dim = int(feature_dim)
        self.foreground_ids = tuple(int(value) for value in foreground_ids)
        self.background_id = int(background_id)
        self.eps = float(eps)
        self.register_buffer("prototypes", torch.empty((0, len(self.foreground_ids), self.feature_dim), dtype=torch.float32))
        self.register_buffer("valid", torch.empty((0, len(self.foreground_ids)), dtype=torch.bool))
        self.register_buffer("dispersion", torch.empty((0, len(self.foreground_ids)), dtype=torch.float32))
        self._metadata: list[dict[str, Any]] = []

    def append_site(
        self,
        site_id: str,
        records: Mapping[int, Mapping[str, Any]],
        *,
        source_checkpoint_sha256: str,
        class_semantics_sha256: str,
        manifest_sha256: str,
        split_sha256: str,
    ) -> None:
        if site_id in self.historical_sites():
            raise RuntimeError(f"refusing to overwrite ASPR site memory record: {site_id}")
        if set(records) != set(self.foreground_ids):
            raise ValueError("site append must contain every and only foreground class")
        vectors: list[torch.Tensor] = []
        spreads: list[float] = []
        class_metadata: dict[str, Any] = {}
        for class_id in self.foreground_ids:
            if class_id == self.background_id:
                raise AssertionError("background cannot enter site prototype memory")
            record = dict(records[class_id])
            prototype = record.pop("prototype").detach().float().reshape(-1)
            if prototype.shape != (self.feature_dim,) or not torch.isfinite(prototype).all() or float(prototype.norm()) <= self.eps:
                raise ValueError("invalid ASPR site prototype")
            vectors.append(F.normalize(prototype.unsqueeze(0), p=2, dim=1, eps=self.eps)[0])
            spreads.append(float(record["dispersion"]))
            class_metadata[str(class_id)] = record
        bank = torch.stack(vectors).to(self.prototypes.device)
        self.prototypes = torch.cat((self.prototypes, bank.unsqueeze(0)), dim=0)
        self.valid = torch.cat((self.valid, torch.ones((1, len(self.foreground_ids)), dtype=torch.bool, device=self.valid.device)), dim=0)
        self.dispersion = torch.cat((self.dispersion, torch.tensor(spreads, dtype=torch.float32, device=self.dispersion.device).unsqueeze(0)), dim=0)
        self._metadata.append(
            {
                "site_id": str(site_id),
                "source_checkpoint_sha256": str(source_checkpoint_sha256),
                "class_semantics_sha256": str(class_semantics_sha256),
                "manifest_sha256": str(manifest_sha256),
                "split_sha256": str(split_sha256),
                "feature_dim": self.feature_dim,
                "classes": class_metadata,
            }
        )
        self.validate()

    def historical_sites(self) -> tuple[str, ...]:
        return tuple(str(record["site_id"]) for record in self._metadata)

    def get_old_frame_bank(self) -> torch.Tensor:
        self.validate()
        return self.prototypes.detach().clone()

    def get_transported_view(self, deltas: Mapping[int, torch.Tensor]) -> torch.Tensor:
        self.validate()
        result = self.prototypes.detach().clone()
        for class_offset, class_id in enumerate(self.foreground_ids):
            if class_id not in deltas:
                raise ValueError(f"missing transport delta for foreground class {class_id}")
            result[:, class_offset] = transport_prototypes(result[:, class_offset], deltas[class_id])
        return result.detach()

    def commit_transport(self, deltas: Mapping[int, torch.Tensor], *, end_site: bool) -> None:
        if not end_site:
            raise RuntimeError("ASPR transport may be committed only at end_site")
        self.prototypes = self.get_transported_view(deltas)
        self.validate()

    def validate(self) -> None:
        expected = (len(self._metadata), len(self.foreground_ids), self.feature_dim)
        if tuple(self.prototypes.shape) != expected or tuple(self.valid.shape) != expected[:2]:
            raise ValueError("ASPR memory tensor/metadata shape mismatch")
        if len(set(self.historical_sites())) != len(self._metadata):
            raise ValueError("duplicate ASPR site memory record")
        if self.prototypes.numel():
            if not bool(self.valid.all()) or not torch.isfinite(self.prototypes).all():
                raise ValueError("invalid ASPR memory values")
            norm = self.prototypes.norm(dim=-1)
            if not torch.allclose(norm, torch.ones_like(norm), atol=1.0e-5, rtol=0.0):
                raise ValueError("ASPR prototypes must remain normalized")
        if list(self.parameters()):
            raise AssertionError("ASPR site memory must contain buffers, not parameters")

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "foreground_ids": list(self.foreground_ids),
            "background_id": self.background_id,
            "metadata": self._metadata,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if int(state["feature_dim"]) != self.feature_dim or tuple(state["foreground_ids"]) != self.foreground_ids:
            raise ValueError("ASPR memory checkpoint contract mismatch")
        self._metadata = list(state["metadata"])

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for name in ("prototypes", "valid", "dispersion"):
            key = prefix + name
            if key in state_dict:
                self._buffers[name] = torch.empty_like(state_dict[key], device=self._buffers[name].device)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
