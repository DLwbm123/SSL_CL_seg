"""Deterministic case-to-site prototype aggregation for ASPR."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CasePrototype:
    case_id: str
    class_id: int
    prototype: torch.Tensor
    pixel_weight: float
    confidence: float
    source: str


class SitePrototypeBuilder:
    def __init__(self, feature_dim: int, foreground_ids: tuple[int, ...] = (1, 2), *, minimum_pixels: int = 32, eps: float = 1.0e-8) -> None:
        if feature_dim < 1 or minimum_pixels != 32:
            raise ValueError("ASPR requires a positive feature dimension and exactly 32 minimum relation pixels")
        self.feature_dim = int(feature_dim)
        self.foreground_ids = tuple(int(value) for value in foreground_ids)
        if 0 in self.foreground_ids:
            raise ValueError("background cannot enter ASPR site memory")
        self.minimum_pixels = int(minimum_pixels)
        self.eps = float(eps)
        self.labeled: dict[int, list[CasePrototype]] = {class_id: [] for class_id in self.foreground_ids}
        self.unlabeled: dict[int, list[CasePrototype]] = {class_id: [] for class_id in self.foreground_ids}
        self._pixel_vector: dict[str, dict[int, torch.Tensor]] = {
            source: {
                class_id: torch.zeros(self.feature_dim, dtype=torch.float64) for class_id in self.foreground_ids
            }
            for source in ("labeled", "unlabeled")
        }
        self._pixel_weight: dict[str, dict[int, float]] = {
            source: {class_id: 0.0 for class_id in self.foreground_ids}
            for source in ("labeled", "unlabeled")
        }

    def _add(self, *, case_id: str, class_id: int, features: torch.Tensor, weights: torch.Tensor, source: str) -> CasePrototype | None:
        if class_id not in self.foreground_ids or source not in {"labeled", "unlabeled"}:
            raise ValueError("invalid ASPR case prototype source or class")
        if features.ndim != 2 or features.shape[1] != self.feature_dim or weights.shape != (features.shape[0],):
            raise ValueError("case prototype expects [N,D] features and [N] weights")
        value = F.normalize(features.detach().float(), p=2, dim=1, eps=self.eps)
        weight = weights.detach().float().clamp_min(0.0)
        weight_sum = float(weight.sum())
        threshold = float(self.minimum_pixels)
        if weight_sum < threshold:
            return None
        center = (value * weight.unsqueeze(1)).sum(dim=0) / max(weight_sum, self.eps)
        if not torch.isfinite(center).all() or float(center.norm()) <= self.eps:
            return None
        prototype = F.normalize(center.unsqueeze(0), p=2, dim=1, eps=self.eps)[0].cpu().detach()
        confidence = 1.0 if source == "labeled" else weight_sum / max(1, int(weight.gt(0).sum()))
        record = CasePrototype(str(case_id), int(class_id), prototype, weight_sum, float(confidence), source)
        holder = self.labeled if source == "labeled" else self.unlabeled
        holder[class_id].append(record)
        self._pixel_vector[source][class_id] += (value.double() * weight.double().unsqueeze(1)).sum(dim=0).cpu()
        self._pixel_weight[source][class_id] += weight_sum
        return record

    def add_labeled(self, case_id: str, features: torch.Tensor, labels: torch.Tensor) -> list[CasePrototype]:
        if features.ndim != 3 or features.shape[0] != self.feature_dim or labels.shape != features.shape[-2:]:
            raise ValueError("labeled case tensors must be [D,H,W] and [H,W]")
        flat_features = features.permute(1, 2, 0).reshape(-1, self.feature_dim)
        flat_labels = labels.reshape(-1)
        result: list[CasePrototype] = []
        for class_id in self.foreground_ids:
            selected = flat_labels.eq(class_id)
            record = self._add(
                case_id=case_id,
                class_id=class_id,
                features=flat_features[selected],
                weights=torch.ones(int(selected.sum()), device=features.device),
                source="labeled",
            )
            if record is not None:
                result.append(record)
        return result

    def add_unlabeled(self, case_id: str, features: torch.Tensor, predicted_class: torch.Tensor, reliable_weight: torch.Tensor) -> list[CasePrototype]:
        if features.ndim != 3 or features.shape[0] != self.feature_dim:
            raise ValueError("unlabeled relation feature must be [D,H,W]")
        if predicted_class.shape != features.shape[-2:] or reliable_weight.shape != features.shape[-2:]:
            raise ValueError("unlabeled masks must match the relation grid")
        flat_features = features.permute(1, 2, 0).reshape(-1, self.feature_dim)
        flat_class = predicted_class.reshape(-1)
        flat_weight = reliable_weight.reshape(-1).float()
        result: list[CasePrototype] = []
        for class_id in self.foreground_ids:
            selected = flat_class.eq(class_id) & flat_weight.gt(0)
            record = self._add(
                case_id=case_id,
                class_id=class_id,
                features=flat_features[selected],
                weights=flat_weight[selected],
                source="unlabeled",
            )
            if record is not None:
                result.append(record)
        return result

    def build(self, *, include_unlabeled: bool) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for class_id in self.foreground_ids:
            labeled = self.labeled[class_id]
            unlabeled = self.unlabeled[class_id] if include_unlabeled else []
            if not labeled:
                raise RuntimeError(f"site memory has no labeled case prototype for class {class_id}")
            numerator = torch.stack([record.prototype for record in labeled]).sum(dim=0)
            denominator = float(len(labeled))
            for record in unlabeled:
                numerator += record.prototype * float(record.confidence)
                denominator += float(record.confidence)
            mean = numerator / max(denominator, self.eps)
            prototype = F.normalize(mean.unsqueeze(0), p=2, dim=1, eps=self.eps)[0].detach()
            pixel_weight = self._pixel_weight["labeled"][class_id]
            pixel_vector = self._pixel_vector["labeled"][class_id].clone()
            if include_unlabeled:
                pixel_weight += self._pixel_weight["unlabeled"][class_id]
                pixel_vector += self._pixel_vector["unlabeled"][class_id]
            pixel_mean = pixel_vector.float() / max(pixel_weight, self.eps)
            dispersion = max(0.0, float(1.0 - 2.0 * torch.dot(prototype, pixel_mean) + prototype.square().sum()))
            result[class_id] = {
                "prototype": prototype,
                "dispersion": dispersion,
                "labeled_case_count": len(labeled),
                "unlabeled_case_count": len(unlabeled),
                "labeled_pixel_weight": float(sum(record.pixel_weight for record in labeled)),
                "unlabeled_pixel_weight": float(sum(record.pixel_weight for record in unlabeled)),
            }
        return result
