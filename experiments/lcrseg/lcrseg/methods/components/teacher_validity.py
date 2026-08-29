"""Old-model-only teacher validity and deterministic monotonic calibration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .pseudo_label import spatial_weight
from .relation_field import RelationOutput


@dataclass(frozen=True)
class TeacherValidityOutput:
    raw_score: torch.Tensor
    margin_validity: torch.Tensor
    certainty_validity: torch.Tensor
    spatial_validity: torch.Tensor
    old_predicted_class: torch.Tensor


@torch.no_grad()
def compute_teacher_validity(
    old_segmentation_logits: torch.Tensor,
    old_relation: RelationOutput,
    *,
    margin_temperature: float,
    spatial_floor: float,
    eps: float = 1.0e-8,
) -> TeacherValidityOutput:
    """Compute V_raw without accepting any current-model tensor."""

    if margin_temperature <= 0:
        raise ValueError("teacher-validity margin temperature must be positive")
    if old_segmentation_logits.ndim != 4 or old_segmentation_logits.shape[1] < 2:
        raise ValueError("old segmentation logits must be [B,C,H,W] with C >= 2")
    resized = F.interpolate(
        old_segmentation_logits.detach().float(),
        size=old_relation.probabilities.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    probability = resized.softmax(dim=1).clamp_min(float(eps))
    entropy = -(probability * probability.log()).sum(dim=1, keepdim=True)
    normalizer = float(np.log(probability.shape[1]))
    certainty = (1.0 - entropy / normalizer).clamp(0.0, 1.0)
    margin = torch.sigmoid(old_relation.margin.detach().float() / float(margin_temperature))
    all_valid = torch.ones_like(old_relation.predicted_class, dtype=torch.bool).unsqueeze(1)
    spatial, _ = spatial_weight(
        old_relation.predicted_class.detach(),
        all_valid,
        num_classes=probability.shape[1],
        floor=float(spatial_floor),
    )
    raw = (margin * certainty * spatial).clamp_min(0.0).pow(1.0 / 3.0).clamp(0.0, 1.0)
    return TeacherValidityOutput(
        raw_score=raw.detach(),
        margin_validity=margin.detach(),
        certainty_validity=certainty.detach(),
        spatial_validity=spatial.detach(),
        old_predicted_class=old_relation.predicted_class.detach(),
    )


def _pava(successes: list[float], totals: list[float]) -> list[float]:
    blocks: list[dict[str, float | int]] = []
    for index, (success, total) in enumerate(zip(successes, totals, strict=True)):
        if total <= 0:
            raise ValueError("PAVA totals must be positive")
        blocks.append({"start": index, "end": index, "success": float(success), "total": float(total)})
        while len(blocks) > 1:
            left, right = blocks[-2], blocks[-1]
            left_value = float(left["success"]) / float(left["total"])
            right_value = float(right["success"]) / float(right["total"])
            if left_value <= right_value + 1.0e-12:
                break
            blocks[-2:] = [
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "success": float(left["success"]) + float(right["success"]),
                    "total": float(left["total"]) + float(right["total"]),
                }
            ]
    values = [0.0] * len(successes)
    for block in blocks:
        value = float(block["success"]) / float(block["total"])
        for index in range(int(block["start"]), int(block["end"]) + 1):
            values[index] = value
    return values


def _evenly_spaced_indices(size: int, limit: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size, dtype=np.int64)
    # Midpoints of equal-width index intervals give deterministic uniform
    # coverage without consuming or mutating any RNG state.
    return np.floor((np.arange(limit, dtype=np.float64) + 0.5) * size / limit).astype(np.int64)


def _ece(probability: np.ndarray, target: np.ndarray, *, bins: int) -> float:
    if probability.size == 0:
        return 0.0
    order = np.argsort(probability, kind="stable")
    total = float(probability.size)
    value = 0.0
    for selected in (chunk for chunk in np.array_split(order, bins) if chunk.size):
        value += (selected.size / total) * abs(float(target[selected].mean()) - float(probability[selected].mean()))
    return float(value)


@dataclass(frozen=True)
class TeacherValidityMapping:
    bin_left: tuple[float, ...]
    bin_right: tuple[float, ...]
    probabilities: tuple[float, ...]
    sample_count: int
    table: tuple[dict[str, Any], ...]
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        size = len(self.bin_right)
        if size < 1 or len(self.bin_left) != size or len(self.probabilities) != size:
            raise ValueError("teacher-validity mapping fields must be nonempty and aligned")
        if any(right < left for left, right in zip(self.bin_left, self.bin_right, strict=True)):
            raise ValueError("teacher-validity bins have invalid bounds")
        if any(second + 1.0e-12 < first for first, second in zip(self.probabilities, self.probabilities[1:])):
            raise ValueError("teacher-validity probabilities must be monotonic")

    @torch.no_grad()
    def evaluate(self, raw_score: torch.Tensor) -> torch.Tensor:
        edges = torch.as_tensor(self.bin_right, device=raw_score.device, dtype=torch.float32)
        values = torch.as_tensor(self.probabilities, device=raw_score.device, dtype=torch.float32)
        indices = torch.bucketize(raw_score.detach().float().reshape(-1), edges, right=False)
        indices = indices.clamp_max(len(self.probabilities) - 1)
        return values.index_select(0, indices).reshape_as(raw_score).detach()

    def state_dict(self) -> dict[str, Any]:
        return {
            "bin_left": list(self.bin_left),
            "bin_right": list(self.bin_right),
            "probabilities": list(self.probabilities),
            "sample_count": self.sample_count,
            "table": [dict(row) for row in self.table],
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "TeacherValidityMapping":
        return cls(
            bin_left=tuple(float(value) for value in state["bin_left"]),
            bin_right=tuple(float(value) for value in state["bin_right"]),
            probabilities=tuple(float(value) for value in state["probabilities"]),
            sample_count=int(state["sample_count"]),
            table=tuple(dict(row) for row in state.get("table", [])),
            metrics={str(key): float(value) for key, value in dict(state.get("metrics") or {}).items()},
        )


def fit_teacher_validity_mapping(
    raw_score: np.ndarray,
    correct: np.ndarray,
    *,
    bins: int,
    fallback_scope: str,
    class_id: int,
) -> TeacherValidityMapping:
    score = np.asarray(raw_score, dtype=np.float64).reshape(-1)
    target = np.asarray(correct, dtype=np.float64).reshape(-1)
    finite = np.isfinite(score) & np.isfinite(target)
    score, target = score[finite], target[finite]
    if score.size == 0 or bins < 1:
        raise ValueError("cannot fit an empty teacher-validity mapping")
    order = np.argsort(score, kind="stable")
    provisional: list[dict[str, Any]] = []
    for selected in (chunk for chunk in np.array_split(order, bins) if chunk.size):
        item = {
            "left": float(score[selected].min()),
            "right": float(score[selected].max()),
            "count": int(selected.size),
            "correct": int(target[selected].sum()),
            "indices": selected,
        }
        if provisional and item["right"] == provisional[-1]["right"]:
            provisional[-1]["right"] = item["right"]
            provisional[-1]["count"] += item["count"]
            provisional[-1]["correct"] += item["correct"]
            provisional[-1]["indices"] = np.concatenate((provisional[-1]["indices"], selected))
        else:
            provisional.append(item)
    successes = [float(item["correct"] + 1) for item in provisional]
    totals = [float(item["count"] + 2) for item in provisional]
    calibrated_bins = _pava(successes, totals)
    edges = np.asarray([item["right"] for item in provisional], dtype=np.float64)
    calibrated = np.asarray(calibrated_bins, dtype=np.float64)[
        np.searchsorted(edges, score, side="left").clip(max=len(edges) - 1)
    ]
    metrics = {
        "brier_raw": float(np.mean((score - target) ** 2)),
        "brier_calibrated": float(np.mean((calibrated - target) ** 2)),
        "ece_raw": _ece(score, target, bins=bins),
        "ece_calibrated": _ece(calibrated, target, bins=bins),
    }
    rows: list[dict[str, Any]] = []
    for index, (item, pava_value) in enumerate(zip(provisional, calibrated_bins, strict=True)):
        raw_accuracy = float(item["correct"] / item["count"])
        rows.append(
            {
                "class_id": class_id,
                "fallback_scope": fallback_scope,
                "bin": index,
                "bin_left": item["left"],
                "bin_right": item["right"],
                "bin_count": item["count"],
                "raw_accuracy": raw_accuracy,
                "smoothed_accuracy": float((item["correct"] + 1) / (item["count"] + 2)),
                "pava_accuracy": float(pava_value),
                **metrics,
            }
        )
    return TeacherValidityMapping(
        bin_left=tuple(float(item["left"]) for item in provisional),
        bin_right=tuple(float(item["right"]) for item in provisional),
        probabilities=tuple(float(value) for value in calibrated_bins),
        sample_count=int(score.size),
        table=tuple(rows),
        metrics=metrics,
    )


class TeacherValidityCalibrator:
    """Frozen, serializable, class-conditional PAVA lookup tables."""

    version = "teacher_validity_pava_v0_2a"

    def __init__(
        self,
        *,
        num_classes: int,
        bins: int = 20,
        minimum_pixels_per_class: int = 2048,
        maximum_pixels_per_class: int = 100000,
    ) -> None:
        if num_classes < 1 or bins < 1 or minimum_pixels_per_class < 1 or maximum_pixels_per_class < 1:
            raise ValueError("invalid teacher-validity calibrator configuration")
        self.num_classes = int(num_classes)
        self.bins = int(bins)
        self.minimum_pixels_per_class = int(minimum_pixels_per_class)
        self.maximum_pixels_per_class = int(maximum_pixels_per_class)
        self.global_mapping: TeacherValidityMapping | None = None
        self.class_mappings: dict[int, TeacherValidityMapping] = {}
        self.status = "unavailable"
        self.fit_count = 0
        self.site_id = ""
        self.rows: list[dict[str, Any]] = []
        self.sample_counts_by_class: list[int] = [0] * self.num_classes

    @property
    def available(self) -> bool:
        return self.global_mapping is not None

    def fit(
        self,
        raw_score: torch.Tensor,
        old_predicted_class: torch.Tensor,
        correct: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        site_id: str,
    ) -> list[dict[str, Any]]:
        if raw_score.numel() != old_predicted_class.numel():
            raise ValueError("raw validity and old class geometry must match")
        if raw_score.numel() != correct.numel() or raw_score.numel() != valid_mask.numel():
            raise ValueError("teacher-validity calibration tensors must have matching geometry")
        score = raw_score.detach().float().reshape(-1).cpu().numpy()
        predicted = old_predicted_class.detach().long().reshape(-1).cpu().numpy()
        target = correct.detach().bool().reshape(-1).cpu().numpy()
        valid = valid_mask.detach().bool().reshape(-1).cpu().numpy() & np.isfinite(score)
        sampled_by_class: dict[int, np.ndarray] = {}
        self.sample_counts_by_class = []
        for class_id in range(self.num_classes):
            indices = np.flatnonzero(valid & (predicted == class_id))
            chosen = indices[_evenly_spaced_indices(int(indices.size), self.maximum_pixels_per_class)]
            sampled_by_class[class_id] = chosen
            self.sample_counts_by_class.append(int(chosen.size))
        pooled = np.concatenate([value for value in sampled_by_class.values() if value.size]) if any(
            value.size for value in sampled_by_class.values()
        ) else np.empty(0, dtype=np.int64)
        if pooled.size == 0:
            raise RuntimeError("site-start labeled calibration has no valid pixels")
        self.global_mapping = fit_teacher_validity_mapping(
            score[pooled], target[pooled], bins=self.bins, fallback_scope="global", class_id=-1
        )
        self.class_mappings = {}
        rows = [dict(row) for row in self.global_mapping.table]
        for class_id in range(self.num_classes):
            selected = sampled_by_class[class_id]
            if selected.size < self.minimum_pixels_per_class:
                rows.append(
                    {
                        "class_id": class_id,
                        "fallback_scope": "global_fallback",
                        "bin": "",
                        "bin_left": "",
                        "bin_right": "",
                        "bin_count": int(selected.size),
                        "raw_accuracy": float(target[selected].mean()) if selected.size else "",
                        "smoothed_accuracy": "",
                        "pava_accuracy": "",
                        **self.global_mapping.metrics,
                    }
                )
                continue
            mapping = fit_teacher_validity_mapping(
                score[selected], target[selected], bins=self.bins, fallback_scope="class", class_id=class_id
            )
            self.class_mappings[class_id] = mapping
            rows.extend(dict(row) for row in mapping.table)
        self.status = "available_frozen_site_start"
        self.fit_count += 1
        self.site_id = str(site_id)
        self.rows = rows
        return [dict(row) for row in rows]

    @torch.no_grad()
    def calibrate(self, raw_score: torch.Tensor, old_predicted_class: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if raw_score.numel() != old_predicted_class.numel():
            raise ValueError("raw validity and old class geometry must match")
        raw = raw_score.detach().float()
        if self.global_mapping is None:
            return raw, False
        result = self.global_mapping.evaluate(raw)
        flat_result = result.reshape(-1)
        flat_raw = raw.reshape(-1)
        flat_class = old_predicted_class.detach().reshape(-1)
        for class_id, mapping in self.class_mappings.items():
            selected = flat_class.eq(class_id)
            if bool(selected.any()):
                flat_result[selected] = mapping.evaluate(flat_raw[selected])
        return result.clamp(0.0, 1.0).detach(), True

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "num_classes": self.num_classes,
            "bins": self.bins,
            "minimum_pixels_per_class": self.minimum_pixels_per_class,
            "maximum_pixels_per_class": self.maximum_pixels_per_class,
            "status": self.status,
            "fit_count": self.fit_count,
            "site_id": self.site_id,
            "sample_counts_by_class": list(self.sample_counts_by_class),
            "global_mapping": self.global_mapping.state_dict() if self.global_mapping is not None else {},
            "class_mappings": {str(key): value.state_dict() for key, value in self.class_mappings.items()},
            "rows": [dict(row) for row in self.rows],
            "provenance": {
                "fit_scope": "current_site_train_labeled_only",
                "old_model_only": True,
                "hidden_gt_usage": 0,
                "frozen_during_site_training": True,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        expected = (
            self.num_classes,
            self.bins,
            self.minimum_pixels_per_class,
            self.maximum_pixels_per_class,
        )
        actual = (
            int(state["num_classes"]),
            int(state["bins"]),
            int(state["minimum_pixels_per_class"]),
            int(state["maximum_pixels_per_class"]),
        )
        if actual != expected:
            raise ValueError("teacher-validity calibrator configuration differs from checkpoint")
        global_state = dict(state.get("global_mapping") or {})
        self.global_mapping = TeacherValidityMapping.from_state_dict(global_state) if global_state else None
        self.class_mappings = {
            int(key): TeacherValidityMapping.from_state_dict(dict(value))
            for key, value in dict(state.get("class_mappings") or {}).items()
        }
        self.status = str(state.get("status", "unavailable"))
        self.fit_count = int(state.get("fit_count", 0))
        self.site_id = str(state.get("site_id", ""))
        self.sample_counts_by_class = [int(value) for value in state.get("sample_counts_by_class", [])]
        self.rows = [dict(row) for row in state.get("rows", [])]
