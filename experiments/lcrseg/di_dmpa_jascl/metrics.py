from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


class ConfusionMetrics:
    def __init__(self, num_classes: int, ignore_label: int = 255) -> None:
        self.num_classes = int(num_classes)
        self.ignore_label = int(ignore_label)
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().cpu().reshape(-1).long()
        target = target.detach().cpu().reshape(-1).long()
        valid = (target != self.ignore_label) & (target >= 0) & (target < self.num_classes)
        if valid.any():
            encoded = target[valid] * self.num_classes + prediction[valid]
            self.matrix += torch.bincount(encoded, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def summary(self) -> dict[str, Any]:
        matrix = self.matrix.double()
        true_positive = matrix.diag()
        target_total = matrix.sum(dim=1)
        prediction_total = matrix.sum(dim=0)
        union = target_total + prediction_total - true_positive
        denominator_dice = target_total + prediction_total
        iou = torch.where(union > 0, true_positive / union, torch.nan)
        dice = torch.where(denominator_dice > 0, 2.0 * true_positive / denominator_dice, torch.nan)
        foreground = dice[1:] if self.num_classes > 1 else dice
        return {
            "confusion_matrix": self.matrix.tolist(),
            "per_class_iou": [None if torch.isnan(value) else float(value) for value in iou],
            "per_class_dice": [None if torch.isnan(value) else float(value) for value in dice],
            "mean_iou": float(torch.nanmean(iou)),
            "mean_dice": float(torch.nanmean(dice)),
            "mean_foreground_dice": float(torch.nanmean(foreground)),
        }


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_lower_triangular_csv(path: str | Path, domain_order: list[str], matrix: dict[str, dict[str, float]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trained_domain", *domain_order], lineterminator="\n")
        writer.writeheader()
        for trained_domain in domain_order:
            row: dict[str, Any] = {"trained_domain": trained_domain}
            for evaluation_domain in domain_order:
                row[evaluation_domain] = matrix.get(trained_domain, {}).get(evaluation_domain, "")
            writer.writerow(row)
