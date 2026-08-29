"""Case-balanced, session-local semantic prototypes for SPARC feasibility.

Only visible current-site labeled batches are accepted by the builder. This
module has no dependency on diagnostic-label resolvers or training methods.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SessionPrototypeSet:
    model_role: Literal["current", "previous"]
    site_id: str
    epoch_id: int
    prototypes: torch.Tensor
    valid_classes: torch.Tensor
    case_counts: torch.Tensor
    pixel_counts: torch.Tensor
    class_semantics_sha256: str
    source_checkpoint_sha256: str
    source_case_ids_sha256: str

    def __post_init__(self) -> None:
        if self.model_role not in {"current", "previous"}:
            raise ValueError(f"invalid model role: {self.model_role}")
        if self.prototypes.ndim != 2:
            raise ValueError("prototypes must be [C,D]")
        classes = self.prototypes.shape[0]
        if self.valid_classes.shape != (classes,) or self.valid_classes.dtype != torch.bool:
            raise ValueError("valid_classes must be bool [C]")
        if self.case_counts.shape != (classes,) or self.pixel_counts.shape != (classes,):
            raise ValueError("case_counts and pixel_counts must be [C]")
        if any(tensor.requires_grad for tensor in (self.prototypes, self.valid_classes, self.case_counts, self.pixel_counts)):
            raise ValueError("prototype state must be detached")
        if not bool(torch.isfinite(self.prototypes).all()):
            raise FloatingPointError("prototype state contains non-finite values")
        if bool(self.valid_classes.any()):
            norms = self.prototypes[self.valid_classes].float().norm(dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1.0e-5, rtol=0.0):
                raise ValueError("valid prototypes must be unit-normalized")
        if bool((self.prototypes[~self.valid_classes] != 0).any()):
            raise ValueError("invalid prototypes must be exact zero")

    def state_dict(self) -> dict[str, Any]:
        return {
            "model_role": self.model_role,
            "site_id": self.site_id,
            "epoch_id": int(self.epoch_id),
            "prototypes": self.prototypes.detach().clone(),
            "valid_classes": self.valid_classes.detach().clone(),
            "case_counts": self.case_counts.detach().clone(),
            "pixel_counts": self.pixel_counts.detach().clone(),
            "class_semantics_sha256": self.class_semantics_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_case_ids_sha256": self.source_case_ids_sha256,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SessionPrototypeSet":
        return cls(
            model_role=state["model_role"],
            site_id=str(state["site_id"]),
            epoch_id=int(state["epoch_id"]),
            prototypes=state["prototypes"].detach(),
            valid_classes=state["valid_classes"].detach().bool(),
            case_counts=state["case_counts"].detach().long(),
            pixel_counts=state["pixel_counts"].detach().long(),
            class_semantics_sha256=str(state["class_semantics_sha256"]),
            source_checkpoint_sha256=str(state["source_checkpoint_sha256"]),
            source_case_ids_sha256=str(state["source_case_ids_sha256"]),
        )


def _case_id_hash(case_ids: Iterable[str]) -> str:
    payload = json.dumps(sorted(set(case_ids)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_batch(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device, non_blocking=True)
    raise TypeError("prototype loader must yield a labeled batch with .to(device)")


def build_session_prototypes(
    model: torch.nn.Module,
    loader: Iterable[Any],
    *,
    model_role: Literal["current", "previous"],
    site_id: str,
    epoch_id: int,
    num_classes: int,
    class_semantics_sha256: str,
    source_checkpoint_sha256: str,
    minimum_relation_pixels_per_case_class: int = 32,
    device: torch.device | str | None = None,
) -> SessionPrototypeSet:
    """Build per-case-normalized then case-equal prototypes."""

    if minimum_relation_pixels_per_case_class != 32:
        raise ValueError("SPARC V0.1 freezes the minimum at 32 relation pixels")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    target_device = torch.device(device) if device is not None else _model_device(model)
    was_training = bool(model.training)
    model.eval()
    case_sums: dict[tuple[str, int], torch.Tensor] = {}
    case_pixels: dict[tuple[str, int], int] = {}
    seen_case_ids: list[str] = []
    relation_dim: int | None = None
    try:
        with torch.inference_mode():
            for raw_batch in loader:
                batch = _move_batch(raw_batch, target_device)
                for forbidden in ("hidden_label", "diagnostic_label", "unlabeled_label"):
                    if hasattr(batch, forbidden):
                        raise RuntimeError(f"forbidden prototype field present: {forbidden}")
                if not all(hasattr(batch, field) for field in ("image", "label", "valid_mask", "case_id")):
                    raise TypeError("prototype builder requires visible labeled image/label/valid_mask/case_id")
                output = model(batch.image)
                features = F.normalize(output.relation_features.float(), p=2, dim=1, eps=1.0e-8)
                relation_dim = int(features.shape[1]) if relation_dim is None else relation_dim
                if features.shape[1] != relation_dim:
                    raise ValueError("relation dimension changed inside prototype loader")
                labels = F.interpolate(batch.label[:, None].float(), size=features.shape[-2:], mode="nearest")[:, 0].long()
                valid = F.interpolate(batch.valid_mask.float(), size=features.shape[-2:], mode="nearest")[:, 0].gt(0.5)
                if len(batch.case_id) != features.shape[0]:
                    raise ValueError("case_id count does not match prototype batch")
                spatial = features.permute(0, 2, 3, 1)
                for sample_index, raw_case_id in enumerate(batch.case_id):
                    case_id = str(raw_case_id)
                    seen_case_ids.append(case_id)
                    for class_id in range(num_classes):
                        mask = labels[sample_index].eq(class_id) & valid[sample_index]
                        count = int(mask.sum())
                        if not count:
                            continue
                        key = (case_id, class_id)
                        contribution = spatial[sample_index][mask].double().sum(dim=0).cpu()
                        case_sums[key] = case_sums.get(key, torch.zeros_like(contribution)) + contribution
                        case_pixels[key] = case_pixels.get(key, 0) + count
    finally:
        model.train(was_training)

    if relation_dim is None:
        raise RuntimeError("prototype loader was empty")
    prototypes = torch.zeros((num_classes, relation_dim), dtype=torch.float32, device=target_device)
    valid_classes = torch.zeros(num_classes, dtype=torch.bool, device=target_device)
    case_counts = torch.zeros(num_classes, dtype=torch.long, device=target_device)
    pixel_counts = torch.zeros(num_classes, dtype=torch.long, device=target_device)
    for class_id in range(num_classes):
        case_vectors: list[torch.Tensor] = []
        included_pixels = 0
        for key in sorted(case_sums):
            if key[1] != class_id:
                continue
            count = case_pixels[key]
            if count < minimum_relation_pixels_per_case_class:
                continue
            mean = case_sums[key] / float(count)
            if not bool(torch.isfinite(mean).all()) or float(mean.norm()) <= 1.0e-12:
                continue
            case_vectors.append(F.normalize(mean.float(), p=2, dim=0, eps=1.0e-8))
            included_pixels += count
        if not case_vectors:
            continue
        case_mean = torch.stack(case_vectors, dim=0).mean(dim=0)
        if float(case_mean.norm()) <= 1.0e-12:
            continue
        prototypes[class_id] = F.normalize(case_mean, p=2, dim=0, eps=1.0e-8).to(target_device)
        valid_classes[class_id] = True
        case_counts[class_id] = len(case_vectors)
        pixel_counts[class_id] = included_pixels
    return SessionPrototypeSet(
        model_role=model_role,
        site_id=str(site_id),
        epoch_id=int(epoch_id),
        prototypes=prototypes.detach(),
        valid_classes=valid_classes.detach(),
        case_counts=case_counts.detach(),
        pixel_counts=pixel_counts.detach(),
        class_semantics_sha256=str(class_semantics_sha256),
        source_checkpoint_sha256=str(source_checkpoint_sha256),
        source_case_ids_sha256=_case_id_hash(seen_case_ids),
    )
