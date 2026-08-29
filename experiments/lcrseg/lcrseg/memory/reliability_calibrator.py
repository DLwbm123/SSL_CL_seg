"""Frozen PAVA reliability calibration for ASPR memory selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class _Curve:
    upper: np.ndarray
    value: np.ndarray
    count: np.ndarray

    def predict(self, score: np.ndarray) -> np.ndarray:
        if not len(self.upper):
            raise RuntimeError("empty reliability curve")
        index = np.searchsorted(self.upper, score, side="left").clip(0, len(self.upper) - 1)
        return self.value[index]

    def state_dict(self) -> dict[str, Any]:
        return {"upper": self.upper.tolist(), "value": self.value.tolist(), "count": self.count.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "_Curve":
        return cls(
            upper=np.asarray(state["upper"], dtype=np.float64),
            value=np.asarray(state["value"], dtype=np.float64),
            count=np.asarray(state["count"], dtype=np.int64),
        )


def _pava(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Return the weighted nondecreasing isotonic projection."""

    if value.ndim != 1 or weight.shape != value.shape or np.any(weight <= 0):
        raise ValueError("PAVA expects positive weighted one-dimensional inputs")
    means: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (mean, mass) in enumerate(zip(value.tolist(), weight.tolist())):
        means.append(float(mean))
        weights.append(float(mass))
        starts.append(index)
        ends.append(index + 1)
        while len(means) >= 2 and means[-2] > means[-1]:
            total = weights[-2] + weights[-1]
            merged = (means[-2] * weights[-2] + means[-1] * weights[-1]) / total
            means[-2:] = [merged]
            weights[-2:] = [total]
            ends[-2:] = [ends[-1]]
            starts.pop()
    result = np.empty_like(value, dtype=np.float64)
    for mean, start, end in zip(means, starts, ends):
        result[start:end] = mean
    return result


def _fit_curve(score: np.ndarray, correctness: np.ndarray, bins: int) -> _Curve:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    correctness = np.asarray(correctness, dtype=np.float64).reshape(-1)
    if score.shape != correctness.shape or not len(score):
        raise ValueError("calibration arrays must be non-empty and shape compatible")
    if not np.isfinite(score).all() or not np.isfinite(correctness).all():
        raise ValueError("calibration arrays contain non-finite values")
    order = np.argsort(score, kind="stable")
    groups = [group for group in np.array_split(order, min(int(bins), len(order))) if len(group)]
    upper = np.asarray([float(score[group].max()) for group in groups], dtype=np.float64)
    count = np.asarray([len(group) for group in groups], dtype=np.int64)
    smoothed = np.asarray(
        [(float(correctness[group].sum()) + 1.0) / (float(len(group)) + 2.0) for group in groups],
        dtype=np.float64,
    )
    value = _pava(smoothed, count.astype(np.float64)).clip(0.0, 1.0)
    upper[-1] = np.inf
    return _Curve(upper=upper, value=value, count=count)


class MonotonicReliabilityCalibrator:
    """Classwise 20-bin calibrator with a deterministic global fallback."""

    def __init__(
        self,
        foreground_ids: tuple[int, ...] = (1, 2),
        *,
        bins: int = 20,
        minimum_class_pixels: int = 2048,
        maximum_pixels_per_class: int = 100_000,
    ) -> None:
        if bins != 20 or minimum_class_pixels != 2048 or maximum_pixels_per_class != 100_000:
            raise ValueError("ASPR V0.1 freezes calibrator bins/support/cap at 20/2048/100000")
        self.foreground_ids = tuple(int(value) for value in foreground_ids)
        self.bins = int(bins)
        self.minimum_class_pixels = int(minimum_class_pixels)
        self.maximum_pixels_per_class = int(maximum_pixels_per_class)
        self.global_curve: _Curve | None = None
        self.class_curves: dict[int, _Curve] = {}
        self.support: dict[int, int] = {}

    @property
    def fitted(self) -> bool:
        return self.global_curve is not None

    def fit(
        self,
        score: np.ndarray | torch.Tensor,
        correctness: np.ndarray | torch.Tensor,
        predicted_class: np.ndarray | torch.Tensor,
        valid: np.ndarray | torch.Tensor,
    ) -> "MonotonicReliabilityCalibrator":
        arrays = [
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            for value in (score, correctness, predicted_class, valid)
        ]
        score_array, correct_array, class_array, valid_array = (value.reshape(-1) for value in arrays)
        if not (score_array.shape == correct_array.shape == class_array.shape == valid_array.shape):
            raise ValueError("calibrator inputs must share a flattened shape")
        foreground = np.isin(class_array, np.asarray(self.foreground_ids))
        eligible = valid_array.astype(bool) & foreground & np.isfinite(score_array)
        if not eligible.any():
            raise RuntimeError("no valid foreground calibration pixels")
        selected_global: list[np.ndarray] = []
        self.class_curves = {}
        self.support = {}
        for class_id in self.foreground_ids:
            indices = np.flatnonzero(eligible & (class_array == class_id))[: self.maximum_pixels_per_class]
            self.support[class_id] = int(len(indices))
            if len(indices):
                selected_global.append(indices)
            if len(indices) >= self.minimum_class_pixels:
                self.class_curves[class_id] = _fit_curve(score_array[indices], correct_array[indices], self.bins)
        if not selected_global:
            raise RuntimeError("no deterministic calibration sample was retained")
        global_indices = np.concatenate(selected_global)
        self.global_curve = _fit_curve(score_array[global_indices], correct_array[global_indices], self.bins)
        return self

    def predict(self, score: torch.Tensor, predicted_class: torch.Tensor) -> torch.Tensor:
        if self.global_curve is None:
            raise RuntimeError("reliability calibrator has not been fitted")
        if score.shape != predicted_class.shape:
            raise ValueError("reliability score and class tensors must have equal shapes")
        score_np = score.detach().float().cpu().numpy()
        class_np = predicted_class.detach().cpu().numpy()
        output = self.global_curve.predict(score_np)
        for class_id, curve in self.class_curves.items():
            mask = class_np == class_id
            output[mask] = curve.predict(score_np[mask])
        return torch.from_numpy(output).to(device=score.device, dtype=torch.float32).detach()

    def state_dict(self) -> dict[str, Any]:
        if self.global_curve is None:
            raise RuntimeError("cannot serialize an unfitted calibrator")
        return {
            "protocol_id": "asprseg_v0_1",
            "bins": self.bins,
            "minimum_class_pixels": self.minimum_class_pixels,
            "maximum_pixels_per_class": self.maximum_pixels_per_class,
            "foreground_ids": list(self.foreground_ids),
            "support": {str(key): value for key, value in self.support.items()},
            "global_curve": self.global_curve.state_dict(),
            "class_curves": {str(key): curve.state_dict() for key, curve in self.class_curves.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "MonotonicReliabilityCalibrator":
        instance = cls(
            tuple(int(value) for value in state["foreground_ids"]),
            bins=int(state["bins"]),
            minimum_class_pixels=int(state["minimum_class_pixels"]),
            maximum_pixels_per_class=int(state["maximum_pixels_per_class"]),
        )
        instance.support = {int(key): int(value) for key, value in state["support"].items()}
        instance.global_curve = _Curve.from_state_dict(state["global_curve"])
        instance.class_curves = {
            int(key): _Curve.from_state_dict(value) for key, value in state["class_curves"].items()
        }
        return instance
