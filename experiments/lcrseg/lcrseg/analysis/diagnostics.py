"""Post-training hidden-GT analysis only.

This module is intentionally never imported by ``lcrseg.data``, methods, or
the training runner.  It may read the diagnostics manifest solely after a
checkpoint is frozen, to calibrate reliability scores against held-out labels.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from ..common import read_csv, write_csv, write_json
from ..engine.checkpoint import load_checkpoint
from ..methods.components.compatibility import compute_compatibility, zero_compatibility
from ..methods.components.learnability import compute_learnability
from ..methods.components.pseudo_label import build_pseudo_labels
from ..methods.components.relation_field import relation_field
from ..methods.lcrseg_v0_1 import LCRSegV01Method
from ..models import UNet2D


@dataclass(frozen=True)
class DiagnosticRecord:
    case_id: str
    patient_id: str
    site: str
    image_path: Path
    label_path: Path


def _safe_path(root: Path, value: str, field: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe diagnostics {field}: {value!r}")
    result = root / "h5" / "v1" / path
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def diagnostic_records(root: Path, *, seed: int, dataset: str, site: str) -> list[DiagnosticRecord]:
    """Read only analysis-eligible diagnostic labels after training has ended."""

    manifest = Path(root) / "manifests" / "diagnostics" / f"lcrseg_v1_seed{seed}.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"diagnostics manifest is missing: {manifest}")
    records: list[DiagnosticRecord] = []
    for row in read_csv(manifest):
        if row.get("dataset") != dataset or (row.get("site_or_vendor") or row.get("site")) != site:
            continue
        if row.get("primary_20pct_split") != "train_unlabeled":
            continue
        if row.get("evaluation_eligible", "").strip().lower() != "true":
            continue
        records.append(
            DiagnosticRecord(
                case_id=str(row["case_id"]),
                patient_id=str(row.get("patient_id") or row["case_id"]),
                site=site,
                image_path=_safe_path(root, row.get("image_h5_relpath", ""), "image_h5_relpath"),
                label_path=_safe_path(root, row.get("label_h5_relpath", ""), "label_h5_relpath"),
            )
        )
    if not records:
        raise ValueError("no analysis-eligible diagnostic unlabeled records matched")
    return records


def _images_and_labels(record: DiagnosticRecord, dataset: str):
    with h5py.File(record.image_path, "r") as image_file, h5py.File(record.label_path, "r") as label_file:
        image = np.asarray(image_file["image"])
        label = np.asarray(label_file["label"])
    if dataset == "fundus":
        if image.shape[0] != 3 or label.ndim != 2:
            raise ValueError(f"invalid fundus diagnostic payload for {record.case_id}")
        yield image.astype(np.float32) / 255.0, label.astype(np.int64)
        return
    if image.shape != label.shape or image.ndim != 3:
        raise ValueError(f"invalid MRI diagnostic payload for {record.case_id}")
    for index in range(image.shape[0]):
        yield image[index][None].astype(np.float32), label[index].astype(np.int64)


def _method_from_checkpoint(checkpoint: Path, device: torch.device) -> tuple[LCRSegV01Method, dict[str, Any]]:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    if payload["method_name"] != "lcrseg_v0_1":
        raise ValueError("reliability calibration is defined for an LCR-Seg V0.1 checkpoint")
    config = dict(payload["config_resolved"])
    model_config = dict(config["model"])
    method_config = dict(config.get("method", {}))
    model = UNet2D(
        int(model_config["in_channels"]),
        int(model_config["num_classes"]),
        base_channels=int(model_config.get("base_channels", 16)),
        relation_dim=int(model_config.get("relation_dim", 128)),
    ).to(device)
    method = LCRSegV01Method(model, config=method_config).to(device)
    method.model.load_state_dict(payload["current_model_state"], strict=True)
    method.load_method_state_dict(payload)
    method.model.eval()
    if method.old_model is not None:
        method.old_model.eval()
    if not method.current_anchor_bank.all_classes_valid:
        raise RuntimeError("checkpoint current anchors are incomplete")
    return method, payload


def _equal_frequency_bins(
    score: np.ndarray,
    correctness: np.ndarray,
    valid: np.ndarray,
    *,
    bins: int = 10,
    extra_correctness: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    if not (score.shape == correctness.shape == valid.shape):
        raise ValueError("calibration arrays must have equal shape")
    order = np.argsort(score, kind="stable")
    rows: list[dict[str, Any]] = []
    for index, selected in enumerate(np.array_split(order, bins)):
        if selected.size == 0:
            row = {"bin": index, "count": 0, "score_mean": float("nan"), "coverage": float("nan"), "accuracy": float("nan")}
            if extra_correctness:
                row.update({f"{name}_accuracy": float("nan") for name in extra_correctness})
            rows.append(row)
            continue
        chosen_valid = valid[selected]
        correct_valid = correctness[selected][chosen_valid]
        row = {
            "bin": index,
            "count": int(selected.size),
            "score_mean": float(score[selected].mean()),
            "coverage": float(chosen_valid.mean()),
            "accuracy": float(correct_valid.mean()) if correct_valid.size else float("nan"),
        }
        for name, extra in (extra_correctness or {}).items():
            if extra.shape != score.shape:
                raise ValueError(f"extra calibration array {name} has a mismatched shape")
            selected_extra = extra[selected][chosen_valid]
            row[f"{name}_accuracy"] = float(selected_extra.mean()) if selected_extra.size else float("nan")
        rows.append(row)
    return rows


def _plot(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    """Dependency-light calibration bar chart; contains no patient imagery."""

    width, height, margin = 720, 360, 55
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 16), title, fill="black")
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    usable = width - 2 * margin
    bar_width = max(2, usable // max(1, len(rows)) - 8)
    for index, row in enumerate(rows):
        accuracy = row.get("accuracy")
        if accuracy is None or not np.isfinite(accuracy):
            accuracy = 0.0
        left = margin + index * (usable / max(1, len(rows))) + 4
        top = height - margin - float(accuracy) * (height - 2 * margin)
        draw.rectangle((left, top, left + bar_width, height - margin), fill=(58, 123, 213))
        draw.text((left, height - margin + 7), str(index), fill="black")
    canvas.save(path)


@torch.no_grad()
def analyze_reliability(
    *,
    root: Path,
    checkpoint: Path,
    dataset: str,
    site: str,
    seed: int,
    output_dir: Path,
    device: torch.device | str,
) -> dict[str, Any]:
    """Write post-hoc L/C calibration artifacts without touching training code."""

    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    frozen = (root / "h5" / "v1").resolve()
    if output_dir == frozen or frozen in output_dir.parents:
        raise ValueError("diagnostic analysis output may not be written into frozen HDF5")
    device = torch.device(device)
    method, payload = _method_from_checkpoint(checkpoint, device)
    config = method.config
    total_steps = int((payload.get("method_statistics") or {}).get("active_site_total_steps") or max(1, int(payload["site_step"])))
    records = diagnostic_records(root, seed=seed, dataset=dataset, site=site)
    learnability_scores: list[np.ndarray] = []
    pseudo_correct: list[np.ndarray] = []
    pseudo_valid: list[np.ndarray] = []
    pseudo_source: list[np.ndarray] = []
    compatibility_scores: list[np.ndarray] = []
    old_segmentation_correct: list[np.ndarray] = []
    old_relation_correct: list[np.ndarray] = []
    relation_js: list[np.ndarray] = []
    for record in records:
        for image, label in _images_and_labels(record, dataset):
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
            label_tensor = torch.from_numpy(label).unsqueeze(0).to(device)
            current_output = method.model(image_tensor)
            current_relation = relation_field(current_output.relation_features, method.current_anchor_bank, temperature=float(config["relation_temperature"]))
            pseudo = build_pseudo_labels(
                current_output.logits.softmax(dim=1), current_relation,
                tau_cls=float(config["tau_cls"]), tau_anchor=float(config["tau_anchor"]), delta_anchor=float(config["delta_anchor"]),
                tau_spatial=float(config["tau_spatial"]), temperature_cls=float(config["temperature_cls"]),
                temperature_anchor=float(config["temperature_anchor"]), spatial_floor=float(config["spatial_floor"]),
            )
            learnability = compute_learnability(
                current_output.logits, current_relation, pseudo,
                site_step=max(0, total_steps - 1), total_steps=total_steps,
                rank_start=float(config["rank_start"]), rank_end=float(config["rank_end"]), rank_temperature=float(config["rank_temperature"]),
                relation_margin_center=float(config["relation_margin_center"]), relation_margin_temperature=float(config["relation_margin_temperature"]),
                min_rank_pixels=int(config["min_rank_pixels"]),
            )
            grid_label = F.interpolate(label_tensor.unsqueeze(1).float(), size=current_relation.probabilities.shape[-2:], mode="nearest")[:, 0].long()
            valid = pseudo.valid[:, 0]
            correct = pseudo.labels.eq(grid_label)
            learnability_scores.append(learnability.score[:, 0].cpu().numpy().reshape(-1))
            pseudo_correct.append(correct.cpu().numpy().reshape(-1))
            pseudo_valid.append(valid.cpu().numpy().reshape(-1))
            pseudo_source.append(pseudo.source.cpu().numpy().reshape(-1))
            if method.old_model is not None and method.old_anchor_bank is not None:
                old_output = method.old_model(image_tensor)
                old_relation = relation_field(old_output.relation_features, method.old_anchor_bank, temperature=float(config["relation_temperature"]))
                compatibility = compute_compatibility(
                    current_relation, old_relation,
                    old_margin_center=float(config["old_margin_center"]), old_margin_temperature=float(config["old_margin_temperature"]),
                    js_temperature=float(config["js_temperature"]), spatial_floor=float(config["spatial_floor"]),
                )
                old_grid_prediction = F.interpolate(old_output.logits, size=grid_label.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
                compatibility_scores.append(compatibility.score[:, 0].cpu().numpy().reshape(-1))
                old_segmentation_correct.append(old_grid_prediction.eq(grid_label).cpu().numpy().reshape(-1))
                old_relation_correct.append(old_relation.predicted_class.eq(grid_label).cpu().numpy().reshape(-1))
                relation_js.append(compatibility.js_divergence[:, 0].cpu().numpy().reshape(-1))
    output_dir.mkdir(parents=True, exist_ok=True)
    l_score = np.concatenate(learnability_scores)
    l_correct = np.concatenate(pseudo_correct)
    l_valid = np.concatenate(pseudo_valid)
    l_rows = _equal_frequency_bins(l_score, l_correct, l_valid)
    write_csv(output_dir / "pseudo_label_accuracy_by_learnability_bin.csv", l_rows)
    write_csv(output_dir / "pseudo_label_coverage_by_bin.csv", l_rows)
    source_values = np.concatenate(pseudo_source)
    source_rows: list[dict[str, Any]] = []
    for code, name in ((1, "classifier"), (2, "anchor"), (0, "deferred")):
        mask = source_values == code
        source_rows.append(
            {
                "source": name,
                "count": int(mask.sum()),
                "accuracy": float(l_correct[mask & l_valid].mean()) if bool((mask & l_valid).any()) else float("nan"),
            }
        )
    write_csv(output_dir / "source_branch_accuracy.csv", source_rows)
    _plot(l_rows, output_dir / "learnability_calibration.png", title="Pseudo-label accuracy by learnability bin")
    summary: dict[str, Any] = {
        "dataset": dataset,
        "site": site,
        "seed": seed,
        "records": len(records),
        "pixels": int(l_score.size),
        "learnability_valid_coverage": float(l_valid.mean()),
        "has_historical_model": method.old_model is not None,
    }
    if compatibility_scores:
        c_score = np.concatenate(compatibility_scores)
        c_seg_correct = np.concatenate(old_segmentation_correct)
        c_relation_correct = np.concatenate(old_relation_correct)
        c_valid = np.ones_like(c_relation_correct, dtype=bool)
        c_rows = _equal_frequency_bins(
            c_score,
            c_relation_correct,
            c_valid,
            extra_correctness={"old_segmentation": c_seg_correct, "old_relation": c_relation_correct},
        )
        write_csv(output_dir / "old_model_accuracy_by_compatibility_bin.csv", c_rows)
        write_csv(output_dir / "compatibility_bins.csv", c_rows)
        _plot(c_rows, output_dir / "compatibility_calibration.png", title="Old-model accuracy by compatibility bin")
        js = np.concatenate(relation_js)
        write_csv(output_dir / "current_old_relation_js.csv", [{"mean": float(js.mean()), "p10": float(np.quantile(js, 0.1)), "p50": float(np.quantile(js, 0.5)), "p90": float(np.quantile(js, 0.9))}])
        high_l = l_score >= np.median(l_score)
        high_c = c_score >= np.median(c_score)
        quadrants: list[dict[str, Any]] = []
        for name, mask in (
            ("high_l_high_c", high_l & high_c),
            ("high_l_low_c", high_l & ~high_c),
            ("low_l_high_c", ~high_l & high_c),
            ("low_l_low_c", ~high_l & ~high_c),
        ):
            quadrants.append(
                {
                    "quadrant": name,
                    "count": int(mask.sum()),
                    "pseudo_accuracy": float(l_correct[mask & l_valid].mean()) if bool((mask & l_valid).any()) else float("nan"),
                    "old_relation_accuracy": float(c_relation_correct[mask].mean()) if bool(mask.any()) else float("nan"),
                    "old_segmentation_accuracy": float(c_seg_correct[mask].mean()) if bool(mask.any()) else float("nan"),
                }
            )
        write_csv(output_dir / "quadrant_stats.csv", quadrants)
        summary["compatibility_pixels"] = int(c_score.size)
    else:
        write_csv(output_dir / "old_model_accuracy_by_compatibility_bin.csv", [], fieldnames=["bin", "count", "score_mean", "coverage", "accuracy"])
        write_csv(output_dir / "compatibility_bins.csv", [], fieldnames=["bin", "count", "score_mean", "coverage", "accuracy"])
        write_csv(output_dir / "current_old_relation_js.csv", [], fieldnames=["mean", "p10", "p50", "p90"])
        write_csv(output_dir / "quadrant_stats.csv", [], fieldnames=["quadrant", "count", "pseudo_accuracy", "old_relation_accuracy", "old_segmentation_accuracy"])
    write_json(output_dir / "reliability_analysis_summary.json", summary)
    return summary
