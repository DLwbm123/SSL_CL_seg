#!/usr/bin/env python3
"""Run hidden-GT V0.2a diagnostics only after all formal training is frozen."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.diagnostics import _images_and_labels, diagnostic_records
from lcrseg.analysis.v0_1_routing import _interior_grid, effective_sample_size
from lcrseg.common import write_csv, write_json
from lcrseg.engine.checkpoint import load_checkpoint
from lcrseg.methods.components.compatibility import compute_compatibility
from lcrseg.methods.components.learnability import compute_learnability
from lcrseg.methods.components.pseudo_label import build_pseudo_labels
from lcrseg.methods.components.rejection_only_routing import rejection_only_weights
from lcrseg.methods.components.teacher_validity import compute_teacher_validity
from lcrseg.methods.lcrseg_v0_2a import (
    ConsolidationMode,
    LCRSegV02AMethod,
    _uniform_consolidation,
)
from lcrseg.models import UNet2D


RUN_NAMES = {
    "R0": "fundus_seed0_lcrseg_uniform_relation_kd_full200e",
    "R1": "fundus_seed0_lcrseg_v0_2a_r1_progressive_uniform_full200e",
    "R2": "fundus_seed0_lcrseg_v0_2a_r2_legacy_teacherreject_full200e",
    "R3": "fundus_seed0_lcrseg_v0_2a_r3_progressive_teacherreject_full200e",
}


def _load(checkpoint: Path, device: torch.device) -> tuple[LCRSegV02AMethod, dict[str, Any]]:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    if payload["method_name"] != "lcrseg_v0_2a":
        raise ValueError(f"not a V0.2a checkpoint: {checkpoint}")
    config = dict(payload["config_resolved"])
    model_config = dict(config["model"])
    method = LCRSegV02AMethod(
        UNet2D(
            int(model_config["in_channels"]),
            int(model_config["num_classes"]),
            base_channels=int(model_config.get("base_channels", 16)),
            relation_dim=int(model_config.get("relation_dim", 128)),
        ).to(device),
        config=dict(config["method"]),
    ).to(device)
    method.model.load_state_dict(payload["current_model_state"], strict=True)
    method.load_method_state_dict(payload)
    method.site_id = str(payload["site_id"])
    method.site_index = int(payload["site_index"])
    method.total_steps = int((payload.get("method_statistics") or {}).get("active_site_total_steps") or payload["site_step"])
    method.model.eval()
    if method.old_model is not None:
        method.old_model.eval()
    return method, payload


def _ece(score: np.ndarray, correct: np.ndarray, bins: int = 20) -> float:
    if not score.size:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(score, edges[1:-1], right=False), 0, bins - 1)
    result = 0.0
    for index in range(bins):
        selected = indices == index
        if bool(selected.any()):
            result += float(selected.mean()) * abs(float(score[selected].mean()) - float(correct[selected].mean()))
    return result


def _histogram_rows(
    *,
    variant: str,
    site: str,
    class_id: int,
    raw: np.ndarray,
    calibrated: np.ndarray,
    correct: np.ndarray,
    rejected: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, 21)
    for score_kind, score in (("raw_teacher_validity", raw), ("calibrated_teacher_validity", calibrated)):
        indices = np.clip(np.digitize(score, edges[1:-1], right=False), 0, 19)
        brier = float(np.mean((score - correct.astype(np.float64)) ** 2)) if score.size else float("nan")
        ece = _ece(score, correct)
        for bin_index in range(20):
            selected = indices == bin_index
            rows.append(
                {
                    "variant": variant,
                    "site": site,
                    "class_id": class_id,
                    "score_kind": score_kind,
                    "bin": bin_index,
                    "bin_left": edges[bin_index],
                    "bin_right": edges[bin_index + 1],
                    "pixel_count": int(selected.sum()),
                    "score_mean": float(score[selected].mean()) if bool(selected.any()) else "",
                    "empirical_old_correctness": float(correct[selected].mean()) if bool(selected.any()) else "",
                    "rejected_fraction": float(rejected[selected].mean()) if bool(selected.any()) else "",
                    "brier": brier,
                    "ece": ece,
                    "hidden_gt_scope": "post_hoc_diagnostics_only",
                }
            )
    return rows


@torch.no_grad()
def _checkpoint_metrics(
    *,
    root: Path,
    variant: str,
    checkpoint: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[np.ndarray]]]:
    method, payload = _load(checkpoint, device)
    site = str(payload["site_id"])
    holders: dict[int, dict[str, list[np.ndarray]]] = {
        class_id: {
            key: []
            for key in (
                "pseudo_valid",
                "pseudo_correct",
                "admitted",
                "boundary",
                "teacher_raw",
                "teacher_calibrated",
                "old_correct",
                "rejected",
                "relation_weight",
                "relation_valid",
            )
        }
        for class_id in range(method.num_classes)
    }
    region_holders: dict[tuple[int, str], dict[str, list[np.ndarray]]] = {}
    global_teacher = {key: [] for key in ("correct", "rejected")}
    for record in diagnostic_records(root, seed=0, dataset="fundus", site=site):
        for image, label in _images_and_labels(record, "fundus"):
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
            output = method.model(image_tensor)
            relation = method._relation(output.relation_features, method.current_anchor_bank)
            pseudo = build_pseudo_labels(
                output.logits.softmax(dim=1),
                relation,
                tau_cls=float(method.config["tau_cls"]),
                tau_anchor=float(method.config["tau_anchor"]),
                delta_anchor=float(method.config["delta_anchor"]),
                tau_spatial=float(method.config["tau_spatial"]),
                temperature_cls=float(method.config["temperature_cls"]),
                temperature_anchor=float(method.config["temperature_anchor"]),
                spatial_floor=float(method.config["spatial_floor"]),
            )
            learnability = compute_learnability(
                output.logits,
                relation,
                pseudo,
                site_step=max(0, int(payload["site_step"]) - 1),
                total_steps=max(1, method.total_steps),
                rank_start=float(method.config["rank_start"]),
                rank_end=float(method.config["rank_end"]),
                rank_temperature=float(method.config["rank_temperature"]),
                relation_margin_center=float(method.config["relation_margin_center"]),
                relation_margin_temperature=float(method.config["relation_margin_temperature"]),
                min_rank_pixels=int(method.config["min_rank_pixels"]),
            )
            strong_valid = torch.ones((1, 1, image.shape[-2], image.shape[-1]), dtype=torch.bool, device=device)
            admission = method._compute_admission(
                pseudo,
                learnability,
                strong_valid,
                site_step=max(0, int(payload["site_step"]) - 1),
            )
            grid_shape = tuple(relation.probabilities.shape[-2:])
            grid_label = F.interpolate(
                torch.from_numpy(label).to(device)[None, None].float(), size=grid_shape, mode="nearest"
            )[0, 0].long()
            boundary = (~torch.from_numpy(_interior_grid(label, grid_shape))).numpy().reshape(-1)
            pseudo_class = pseudo.labels[0].cpu().numpy().reshape(-1)
            pseudo_valid = pseudo.valid[0, 0].cpu().numpy().astype(bool).reshape(-1)
            pseudo_correct = pseudo.labels[0].eq(grid_label).cpu().numpy().astype(bool).reshape(-1)
            admitted = admission.mask[0, 0].cpu().numpy().astype(bool).reshape(-1)
            teacher_raw = teacher_calibrated = old_correct = rejected = relation_weight = relation_valid = old_class = None
            if method.old_model is not None and method.old_anchor_bank is not None:
                old_output = method.old_model(image_tensor)
                old_relation = method._relation(old_output.relation_features, method.old_anchor_bank)
                old_class = old_relation.predicted_class[0].cpu().numpy().reshape(-1)
                old_correct = old_relation.predicted_class[0].eq(grid_label).cpu().numpy().astype(bool).reshape(-1)
                validity = compute_teacher_validity(
                    old_output.logits,
                    old_relation,
                    margin_temperature=float(method.config["teacher_validity_margin_temperature"]),
                    spatial_floor=float(method.config["teacher_validity_spatial_floor"]),
                )
                relation_valid_tensor = torch.ones_like(validity.raw_score, dtype=torch.bool)
                if method.config["consolidation_mode"] == ConsolidationMode.CALIBRATED_TEACHER_REJECTION.value:
                    calibrated_tensor, available = method.teacher_validity_calibrator.calibrate(
                        validity.raw_score, validity.old_predicted_class
                    )
                    if not available:
                        raise RuntimeError(f"missing frozen calibrator in {checkpoint}")
                    routing = rejection_only_weights(
                        calibrated_tensor,
                        validity.old_predicted_class,
                        relation_valid_tensor,
                        num_classes=method.num_classes,
                        calibrator_available=True,
                        probability_threshold=float(method.config["rejection_threshold"]),
                        max_reject_fraction_per_class=float(method.config["rejection_cap"]),
                        rejected_weight_floor=float(method.config["rejection_floor"]),
                    )
                else:
                    compatibility = compute_compatibility(
                        relation,
                        old_relation,
                        old_margin_center=float(method.config["old_margin_center"]),
                        old_margin_temperature=float(method.config["old_margin_temperature"]),
                        js_temperature=float(method.config["js_temperature"]),
                        spatial_floor=float(method.config["spatial_floor"]),
                    )
                    routing = _uniform_consolidation(
                        compatibility,
                        relation_valid_tensor,
                        num_classes=method.num_classes,
                        old_predicted_class=old_relation.predicted_class,
                    )
                    calibrated_tensor = validity.raw_score
                teacher_raw = validity.raw_score[0, 0].cpu().numpy().reshape(-1)
                teacher_calibrated = calibrated_tensor[0, 0].cpu().numpy().reshape(-1)
                rejected = routing.rejection_mask[0, 0].cpu().numpy().astype(bool).reshape(-1)
                relation_weight = routing.weights[0, 0].cpu().numpy().reshape(-1)
                relation_valid = routing.relation_valid_mask[0, 0].cpu().numpy().astype(bool).reshape(-1)
                global_teacher["correct"].append(old_correct[relation_valid])
                global_teacher["rejected"].append(rejected[relation_valid])
            for class_id in range(method.num_classes):
                selected = pseudo_class == class_id
                holder = holders[class_id]
                holder["pseudo_valid"].append(pseudo_valid[selected])
                holder["pseudo_correct"].append(pseudo_correct[selected])
                holder["admitted"].append(admitted[selected])
                holder["boundary"].append(boundary[selected])
                for region_name, region_mask in (("boundary", boundary), ("interior", ~boundary)):
                    chosen = selected & region_mask
                    region = region_holders.setdefault(
                        (class_id, region_name),
                        {key: [] for key in ("pseudo_valid", "pseudo_correct", "admitted", "old_correct", "rejected", "relation_valid")},
                    )
                    region["pseudo_valid"].append(pseudo_valid[chosen])
                    region["pseudo_correct"].append(pseudo_correct[chosen])
                    region["admitted"].append(admitted[chosen])
                    if old_class is not None and old_correct is not None and rejected is not None and relation_valid is not None:
                        old_chosen = (old_class == class_id) & region_mask
                        region["old_correct"].append(old_correct[old_chosen])
                        region["rejected"].append(rejected[old_chosen])
                        region["relation_valid"].append(relation_valid[old_chosen])
                if old_class is not None and teacher_raw is not None and teacher_calibrated is not None and old_correct is not None and rejected is not None and relation_weight is not None and relation_valid is not None:
                    old_selected = old_class == class_id
                    holder["teacher_raw"].append(teacher_raw[old_selected])
                    holder["teacher_calibrated"].append(teacher_calibrated[old_selected])
                    holder["old_correct"].append(old_correct[old_selected])
                    holder["rejected"].append(rejected[old_selected])
                    holder["relation_weight"].append(relation_weight[old_selected])
                    holder["relation_valid"].append(relation_valid[old_selected])
    class_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for class_id, values in holders.items():
        valid = np.concatenate(values["pseudo_valid"])
        correct = np.concatenate(values["pseudo_correct"])
        admitted_values = np.concatenate(values["admitted"])
        boundary_values = np.concatenate(values["boundary"])
        candidate = valid
        admitted_candidate = admitted_values & candidate
        interior_admitted = int((admitted_candidate & ~boundary_values).sum())
        boundary_admitted = int((admitted_candidate & boundary_values).sum())
        row: dict[str, Any] = {
            "variant": variant,
            "site": site,
            "site_index": int(payload["site_index"]),
            "class_id": class_id,
            "pseudo_valid_count": int(candidate.sum()),
            "admitted_count": int(admitted_candidate.sum()),
            "deferred_count": int((candidate & ~admitted_values).sum()),
            "pseudo_label_accuracy": float(correct[candidate].mean()) if bool(candidate.any()) else "",
            "admitted_pseudo_label_accuracy": float(correct[admitted_candidate].mean()) if bool(admitted_candidate.any()) else "",
            "boundary_interior_admitted_ratio": boundary_admitted / interior_admitted if interior_admitted else "",
            "hidden_gt_scope": "post_hoc_diagnostics_only",
        }
        if values["teacher_raw"]:
            raw = np.concatenate(values["teacher_raw"])
            calibrated = np.concatenate(values["teacher_calibrated"])
            old_is_correct = np.concatenate(values["old_correct"])
            rejected_values = np.concatenate(values["rejected"])
            weights = np.concatenate(values["relation_weight"])
            relation_is_valid = np.concatenate(values["relation_valid"])
            selected = relation_is_valid
            retained = selected & ~rejected_values
            rejected_valid = selected & rejected_values
            row.update(
                {
                    "raw_validity_mean": float(raw[selected].mean()) if bool(selected.any()) else "",
                    "calibrated_validity_mean": float(calibrated[selected].mean()) if bool(selected.any()) else "",
                    "raw_brier": float(np.mean((raw[selected] - old_is_correct[selected]) ** 2)) if bool(selected.any()) else "",
                    "calibrated_brier": float(np.mean((calibrated[selected] - old_is_correct[selected]) ** 2)) if bool(selected.any()) else "",
                    "raw_ece": _ece(raw[selected], old_is_correct[selected]),
                    "calibrated_ece": _ece(calibrated[selected], old_is_correct[selected]),
                    "rejected_fraction": float(rejected_valid.sum() / selected.sum()) if bool(selected.any()) else "",
                    "rejected_old_correctness": float(old_is_correct[rejected_valid].mean()) if bool(rejected_valid.any()) else "",
                    "retained_old_correctness": float(old_is_correct[retained].mean()) if bool(retained.any()) else "",
                    "relation_ess": effective_sample_size(weights * selected.astype(np.float64)),
                    "relation_ess_ratio": effective_sample_size(weights * selected.astype(np.float64)) / int(selected.sum()) if bool(selected.any()) else "",
                }
            )
            hist_rows.extend(
                _histogram_rows(
                    variant=variant,
                    site=site,
                    class_id=class_id,
                    raw=raw[selected],
                    calibrated=calibrated[selected],
                    correct=old_is_correct[selected],
                    rejected=rejected_values[selected],
                )
            )
        class_rows.append(row)
    for (class_id, region_name), values in sorted(region_holders.items()):
        valid = np.concatenate(values["pseudo_valid"])
        correct = np.concatenate(values["pseudo_correct"])
        admission = np.concatenate(values["admitted"])
        row = {
            "variant": variant,
            "site": site,
            "site_index": int(payload["site_index"]),
            "class_id": class_id,
            "region": region_name,
            "pseudo_valid_count": int(valid.sum()),
            "pseudo_label_accuracy": float(correct[valid].mean()) if bool(valid.any()) else "",
            "admission_fraction": float(admission[valid].mean()) if bool(valid.any()) else "",
            "hidden_gt_scope": "post_hoc_diagnostics_only",
        }
        if values["old_correct"]:
            old_is_correct = np.concatenate(values["old_correct"])
            rejected_values = np.concatenate(values["rejected"])
            relation_is_valid = np.concatenate(values["relation_valid"])
            row.update(
                {
                    "old_correctness": float(old_is_correct[relation_is_valid].mean()) if bool(relation_is_valid.any()) else "",
                    "rejected_fraction": float(rejected_values[relation_is_valid].mean()) if bool(relation_is_valid.any()) else "",
                }
            )
        region_rows.append(row)
    return class_rows, hist_rows, region_rows, global_teacher


def _final_class_means(run_dir: Path) -> dict[int, float]:
    import csv

    rows = list(csv.DictReader((run_dir / "site_matrix_long.csv").open()))
    final_index = max(int(row["trained_site_index"]) for row in rows)
    final = [row for row in rows if int(row["trained_site_index"]) == final_index]
    return {
        class_id: float(np.mean([float(row[f"dice_class_{class_id}"]) for row in final]))
        for class_id in (1, 2)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/analysis/v0_2a")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    class_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    r3_correct: list[np.ndarray] = []
    r3_rejected: list[np.ndarray] = []
    for variant in ("R1", "R2", "R3"):
        run_dir = (args.run_root / RUN_NAMES[variant]).resolve()
        summary = json.loads((run_dir / "run_summary.json").read_text())
        if summary.get("status") != "complete" or int(summary.get("completed_global_steps", -1)) != 13400:
            raise RuntimeError(f"post-hoc analysis requires a frozen complete run: {run_dir}")
        for checkpoint in sorted(run_dir.glob("checkpoint_final_site*_*.pt")):
            class_part, hist_part, region_part, teacher = _checkpoint_metrics(
                root=args.root.resolve(), variant=variant, checkpoint=checkpoint, device=device
            )
            class_rows.extend(class_part)
            hist_rows.extend(hist_part)
            region_rows.extend(region_part)
            if variant == "R3":
                r3_correct.extend(teacher["correct"])
                r3_rejected.extend(teacher["rejected"])
    write_csv(output_dir / "posthoc_classwise_metrics.csv", class_rows)
    write_csv(output_dir / "teacher_validity_posthoc.csv", hist_rows)
    write_csv(output_dir / "regionwise_results.csv", region_rows)
    if not r3_correct or not r3_rejected:
        raise RuntimeError("R3 post-hoc analysis did not produce incremental teacher-validity pixels")
    correct = np.concatenate(r3_correct)
    rejected = np.concatenate(r3_rejected)
    retained_correctness = float(correct[~rejected].mean()) if bool((~rejected).any()) else float("nan")
    rejected_correctness = float(correct[rejected].mean()) if bool(rejected.any()) else float("nan")
    r0_class = _final_class_means((args.run_root / RUN_NAMES["R0"]).resolve())
    r3_class = _final_class_means((args.run_root / RUN_NAMES["R3"]).resolve())
    class_deltas = {str(key): r3_class[key] - r0_class[key] for key in r0_class}
    summary = {
        "hidden_gt_scope": "post_hoc_diagnostics_process_only",
        "training_imports_this_module": False,
        "R3_retained_old_correctness": retained_correctness,
        "R3_rejected_old_correctness": rejected_correctness,
        "R3_retained_old_correctness_gt_rejected": retained_correctness > rejected_correctness,
        "R3_final_foreground_class_dice_deltas_vs_R0": class_deltas,
        "R3_no_foreground_class_drop_over_0_01": min(class_deltas.values()) >= -0.01,
        "foreground_material_drop_definition": "absolute final mean class Dice decrease greater than 0.01",
        "classwise_rows": len(class_rows),
        "teacher_histogram_rows": len(hist_rows),
        "regionwise_rows": len(region_rows),
    }
    write_json(output_dir / "posthoc_teacher_metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
