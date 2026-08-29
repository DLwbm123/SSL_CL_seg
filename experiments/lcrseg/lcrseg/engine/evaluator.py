"""One evaluator shared by all baseline and proposed continual methods."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import H5LabeledDataset, collate_labeled


@dataclass(frozen=True)
class EvaluationResult:
    per_case: list[dict[str, Any]]
    per_site: list[dict[str, Any]]


def _surface_distances(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """ASD/HD95 in processed-pixel units, with explicit empty-set handling."""

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return 0.0, 0.0
    if not prediction.any() or not target.any():
        return float("nan"), float("nan")
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - environment contract check
        raise RuntimeError("scipy is required for ASD/HD95 evaluation") from exc
    structure = ndimage.generate_binary_structure(prediction.ndim, 1)
    pred_surface = prediction ^ ndimage.binary_erosion(prediction, structure=structure, border_value=0)
    target_surface = target ^ ndimage.binary_erosion(target, structure=structure, border_value=0)
    if not pred_surface.any() or not target_surface.any():
        return float("nan"), float("nan")
    to_target = ndimage.distance_transform_edt(~target_surface)[pred_surface]
    to_prediction = ndimage.distance_transform_edt(~pred_surface)[target_surface]
    all_distances = np.concatenate((to_target, to_prediction))
    return float(all_distances.mean()), float(np.percentile(all_distances, 95.0))


def _case_metrics(prediction: np.ndarray, target: np.ndarray, *, num_classes: int) -> dict[str, float]:
    result: dict[str, float] = {}
    foreground_dice: list[float] = []
    foreground_asd: list[float] = []
    foreground_hd95: list[float] = []
    for class_index in range(1, num_classes):
        predicted = prediction == class_index
        truth = target == class_index
        denominator = int(predicted.sum()) + int(truth.sum())
        dice = 1.0 if denominator == 0 else (2.0 * float((predicted & truth).sum())) / float(denominator)
        asd, hd95 = _surface_distances(predicted, truth)
        result[f"dice_class_{class_index}"] = dice
        result[f"asd_class_{class_index}"] = asd
        result[f"hd95_class_{class_index}"] = hd95
        foreground_dice.append(dice)
        if np.isfinite(asd):
            foreground_asd.append(asd)
        if np.isfinite(hd95):
            foreground_hd95.append(hd95)
    result["mean_foreground_dice"] = float(np.mean(foreground_dice)) if foreground_dice else 0.0
    result["mean_foreground_asd"] = float(np.mean(foreground_asd)) if foreground_asd else float("nan")
    result["mean_foreground_hd95"] = float(np.mean(foreground_hd95)) if foreground_hd95 else float("nan")
    return result


def _site_summary(site: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"site": site, "patients": len(rows)}
    metric_names = sorted({name for row in rows for name in row if name.startswith(("dice_", "asd_", "hd95_", "mean_"))})
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in rows if name in row], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[name] = float(finite.mean()) if finite.size else float("nan")
    return result


@torch.no_grad()
def evaluate_site(
    model: torch.nn.Module,
    *,
    data_root,
    seed: int,
    dataset: str,
    site: str,
    num_classes: int,
    role: str = "test",
    device: torch.device | str = "cpu",
    batch_size: int = 4,
    num_workers: int = 0,
) -> EvaluationResult:
    """Aggregate ED/ES or MRI slices at the patient level before metrics."""

    source = H5LabeledDataset(data_root, seed=seed, dataset=dataset, sites=(site,), roles=(role,), transform=None)
    loader = DataLoader(source, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_labeled)
    model_device = torch.device(device)
    old_mode = model.training
    model.eval()
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"prediction": [], "target": [], "site": site, "case_ids": set()})
    for batch in loader:
        batch = batch.to(model_device, non_blocking=True)
        prediction = model(batch.image).logits.argmax(dim=1).detach().cpu().numpy()
        target = batch.label.detach().cpu().numpy()
        for index, patient_id in enumerate(batch.patient_id):
            group = grouped[patient_id]
            group["prediction"].append(prediction[index])
            group["target"].append(target[index])
            group["case_ids"].add(batch.case_id[index])
    if old_mode:
        model.train()
    per_case: list[dict[str, Any]] = []
    for patient_id in sorted(grouped):
        group = grouped[patient_id]
        prediction = np.stack(group["prediction"], axis=0)
        target = np.stack(group["target"], axis=0)
        row: dict[str, Any] = {
            "site": site,
            "patient_id": patient_id,
            "case_ids": ";".join(sorted(group["case_ids"])),
            "slices_or_images": int(prediction.shape[0]),
        }
        row.update(_case_metrics(prediction, target, num_classes=num_classes))
        per_case.append(row)
    return EvaluationResult(per_case=per_case, per_site=[_site_summary(site, per_case)])


def evaluate_sites(
    model: torch.nn.Module,
    *,
    data_root,
    seed: int,
    dataset: str,
    sites: Iterable[str],
    num_classes: int,
    role: str,
    device: torch.device | str,
    batch_size: int,
    num_workers: int = 0,
) -> EvaluationResult:
    all_case: list[dict[str, Any]] = []
    all_site: list[dict[str, Any]] = []
    for site in sites:
        result = evaluate_site(
            model,
            data_root=data_root,
            seed=seed,
            dataset=dataset,
            site=site,
            num_classes=num_classes,
            role=role,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        all_case.extend(result.per_case)
        all_site.extend(result.per_site)
    return EvaluationResult(per_case=all_case, per_site=all_site)
