"""Labeled-only monotonic compatibility calibration for LCR-Seg V0.2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PiecewiseMonotonicMapping:
    """A detached, PAVA-fitted probability map indexed by raw-score upper edges."""

    upper_edges: tuple[float, ...]
    probabilities: tuple[float, ...]
    sample_count: int
    table: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.upper_edges or len(self.upper_edges) != len(self.probabilities):
            raise ValueError("mapping needs equally sized, nonempty edges and probabilities")
        if any(not np.isfinite(value) for value in self.upper_edges + self.probabilities):
            raise ValueError("mapping cannot contain non-finite values")
        if any(second < first for first, second in zip(self.upper_edges, self.upper_edges[1:])):
            raise ValueError("mapping upper edges must be nondecreasing")
        if any(second + 1.0e-12 < first for first, second in zip(self.probabilities, self.probabilities[1:])):
            raise ValueError("PAVA probabilities must be nondecreasing")

    def evaluate(self, raw_score: torch.Tensor) -> torch.Tensor:
        edges = torch.as_tensor(self.upper_edges, device=raw_score.device, dtype=torch.float32)
        probabilities = torch.as_tensor(self.probabilities, device=raw_score.device, dtype=torch.float32)
        index = torch.bucketize(raw_score.detach().float().reshape(-1), edges, right=False).clamp_max(len(self.probabilities) - 1)
        return probabilities.index_select(0, index).reshape_as(raw_score).detach()

    def state_dict(self) -> dict[str, Any]:
        return {
            "upper_edges": list(self.upper_edges),
            "probabilities": list(self.probabilities),
            "sample_count": int(self.sample_count),
            "table": [dict(row) for row in self.table],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "PiecewiseMonotonicMapping":
        return cls(
            upper_edges=tuple(float(value) for value in state["upper_edges"]),
            probabilities=tuple(float(value) for value in state["probabilities"]),
            sample_count=int(state["sample_count"]),
            table=tuple(dict(row) for row in state.get("table", [])),
        )


def _pava(successes: list[float], totals: list[float]) -> list[float]:
    """Weighted pool-adjacent-violators for a nondecreasing bin accuracy curve."""

    blocks: list[dict[str, Any]] = []
    for index, (success, total) in enumerate(zip(successes, totals, strict=True)):
        if total <= 0:
            raise ValueError("PAVA total must be positive")
        blocks.append({"start": index, "end": index, "success": float(success), "total": float(total)})
        while len(blocks) >= 2:
            previous, current = blocks[-2], blocks[-1]
            if previous["success"] / previous["total"] <= current["success"] / current["total"] + 1.0e-12:
                break
            merged = {
                "start": previous["start"],
                "end": current["end"],
                "success": previous["success"] + current["success"],
                "total": previous["total"] + current["total"],
            }
            blocks[-2:] = [merged]
    result = [0.0] * len(successes)
    for block in blocks:
        probability = block["success"] / block["total"]
        for index in range(int(block["start"]), int(block["end"]) + 1):
            result[index] = probability
    return result


def fit_pava_mapping(
    raw_score: np.ndarray,
    correct: np.ndarray,
    *,
    bins: int,
    scope: str,
    class_id: int | None,
) -> PiecewiseMonotonicMapping:
    """Fit equal-frequency, Laplace-smoothed, weighted-PAVA calibration."""

    score = np.asarray(raw_score, dtype=np.float64).reshape(-1)
    target = np.asarray(correct, dtype=bool).reshape(-1)
    finite = np.isfinite(score)
    score, target = score[finite], target[finite]
    if score.size == 0:
        raise ValueError("cannot calibrate an empty score set")
    if bins < 1:
        raise ValueError("bins must be positive")
    order = np.argsort(score, kind="stable")
    chunks = [selected for selected in np.array_split(order, int(bins)) if selected.size]
    # Scores can be tied at an equal-frequency boundary.  Coalescing those
    # bins keeps the resulting score-to-probability function well-defined
    # while preserving the required initial equal-frequency partition.
    provisional: list[dict[str, Any]] = []
    for selected in chunks:
        upper = float(score[selected].max())
        item = {
            "upper_edge": upper,
            "count": int(selected.size),
            "correct": int(target[selected].sum()),
        }
        if provisional and np.isclose(provisional[-1]["upper_edge"], upper, rtol=0.0, atol=0.0):
            provisional[-1]["count"] += item["count"]
            provisional[-1]["correct"] += item["correct"]
        else:
            provisional.append(item)
    successes = [float(item["correct"] + 1) for item in provisional]
    totals = [float(item["count"] + 2) for item in provisional]
    calibrated = _pava(successes, totals)
    rows: list[dict[str, Any]] = []
    for index, (item, probability, total) in enumerate(zip(provisional, calibrated, totals, strict=True)):
        rows.append(
            {
                "scope": scope,
                "class_id": "" if class_id is None else int(class_id),
                "bin": index,
                "upper_edge": float(item["upper_edge"]),
                "pixel_count": int(item["count"]),
                "correct_count": int(item["correct"]),
                "laplace_accuracy": float((item["correct"] + 1) / (item["count"] + 2)),
                "pava_probability": float(probability),
                "pava_weight": float(total),
            }
        )
    return PiecewiseMonotonicMapping(
        upper_edges=tuple(float(item["upper_edge"]) for item in provisional),
        probabilities=tuple(float(value) for value in calibrated),
        sample_count=int(score.size),
        table=tuple(rows),
    )


class LabeledOnlyCompatibilityCalibrator:
    """Serializable non-parameter calibrator built only from visible labels."""

    def __init__(self, *, num_classes: int, bins: int = 10, min_pixels: int = 500) -> None:
        if num_classes < 1 or bins < 1 or min_pixels < 1:
            raise ValueError("invalid calibrator dimensions")
        self.num_classes = int(num_classes)
        self.bins = int(bins)
        self.min_pixels = int(min_pixels)
        self.global_mapping: PiecewiseMonotonicMapping | None = None
        self.class_mappings: dict[int, PiecewiseMonotonicMapping] = {}
        self.status = "unavailable"
        self.last_update_epoch = -1
        self.fit_count = 0
        self.last_table: list[dict[str, Any]] = []

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
        epoch: int,
    ) -> list[dict[str, Any]]:
        """Fit global plus eligible per-old-predicted-class mappings."""

        if raw_score.shape[0] != old_predicted_class.shape[0] or raw_score.numel() != old_predicted_class.numel():
            raise ValueError("raw score and old class geometry must match")
        if raw_score.numel() != correct.numel() or raw_score.numel() != valid_mask.numel():
            raise ValueError("calibration inputs must have matching geometry")
        score = raw_score.detach().float().reshape(-1).cpu().numpy()
        predicted = old_predicted_class.detach().long().reshape(-1).cpu().numpy()
        target = correct.detach().bool().reshape(-1).cpu().numpy()
        valid = valid_mask.detach().bool().reshape(-1).cpu().numpy() & np.isfinite(score)
        table: list[dict[str, Any]] = []
        self.class_mappings = {}
        if int(valid.sum()) < self.min_pixels:
            self.global_mapping = None
            self.status = "unavailable_global_insufficient_pixels"
            self.last_update_epoch = int(epoch)
            self.fit_count += 1
            self.last_table = table
            return table
        self.global_mapping = fit_pava_mapping(score[valid], target[valid], bins=self.bins, scope="global", class_id=None)
        table.extend(self.global_mapping.table)
        for class_id in range(self.num_classes):
            selected = valid & (predicted == class_id)
            if int(selected.sum()) < self.min_pixels:
                table.append(
                    {
                        "scope": "class_fallback_global",
                        "class_id": int(class_id),
                        "bin": "",
                        "upper_edge": "",
                        "pixel_count": int(selected.sum()),
                        "correct_count": int(target[selected].sum()),
                        "laplace_accuracy": "",
                        "pava_probability": "",
                        "pava_weight": "",
                    }
                )
                continue
            mapping = fit_pava_mapping(score[selected], target[selected], bins=self.bins, scope="class", class_id=class_id)
            self.class_mappings[class_id] = mapping
            table.extend(mapping.table)
        self.status = "available"
        self.last_update_epoch = int(epoch)
        self.fit_count += 1
        self.last_table = [dict(row) for row in table]
        return [dict(row) for row in table]

    @torch.no_grad()
    def calibrate(self, raw_score: torch.Tensor, old_predicted_class: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Return a detached calibrated probability or raw score when unavailable."""

        if raw_score.numel() != old_predicted_class.numel():
            raise ValueError("raw score and old-predicted class geometry must match")
        raw = raw_score.detach().float()
        if self.global_mapping is None:
            return raw, False
        predicted = old_predicted_class.detach().reshape(-1)
        flat_raw = raw.reshape(-1)
        result = self.global_mapping.evaluate(flat_raw)
        for class_id, mapping in self.class_mappings.items():
            selected = predicted.eq(int(class_id))
            if bool(selected.any()):
                result[selected] = mapping.evaluate(flat_raw[selected])
        return result.reshape_as(raw).clamp(0.0, 1.0).detach(), True

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "bins": self.bins,
            "min_pixels": self.min_pixels,
            "status": self.status,
            "last_update_epoch": self.last_update_epoch,
            "fit_count": self.fit_count,
            "global_mapping": self.global_mapping.state_dict() if self.global_mapping is not None else {},
            "class_mappings": {str(key): value.state_dict() for key, value in self.class_mappings.items()},
            "last_table": [dict(row) for row in self.last_table],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        if int(state["num_classes"]) != self.num_classes or int(state["bins"]) != self.bins or int(state["min_pixels"]) != self.min_pixels:
            raise ValueError("calibrator configuration differs from checkpoint state")
        global_state = dict(state.get("global_mapping") or {})
        self.global_mapping = PiecewiseMonotonicMapping.from_state_dict(global_state) if global_state else None
        self.class_mappings = {
            int(key): PiecewiseMonotonicMapping.from_state_dict(dict(value))
            for key, value in dict(state.get("class_mappings") or {}).items()
        }
        self.status = str(state.get("status", "unavailable"))
        self.last_update_epoch = int(state.get("last_update_epoch", -1))
        self.fit_count = int(state.get("fit_count", 0))
        self.last_table = [dict(row) for row in state.get("last_table", [])]
