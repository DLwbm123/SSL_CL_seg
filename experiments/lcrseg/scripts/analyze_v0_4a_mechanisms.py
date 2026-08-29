#!/usr/bin/env python3
"""Run/merge frozen post-hoc mechanism checks for the V0.4a internal gate."""
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
from lcrseg.analysis.v0_4 import SITE_ORDER, diagnostic_snapshot, load_frozen_method, stable_seed
from lcrseg.common import read_csv, sha256_path, write_csv, write_json
from lcrseg.methods.components.learnability import compute_learnability
from lcrseg.methods.components.progressive_admission import strict_relation_valid_mask
from lcrseg.methods.components.pseudo_label import build_pseudo_labels
from scripts.analyze_v0_4_admission_stability import _augment, _jaccard
from scripts.analyze_v0_4_anchor_relation_drift import _cosine, _labeled_centroids
from scripts.analyze_v0_4_gradient_utility import _batchers, _gradient, _norm


MECHANISM_FIELDS = [
    "analysis", "seed", "site_index", "site", "case_id", "rng",
    "candidate_jaccard", "sra_weighted_jaccard", "sra_hard_dominant_jaccard",
    "class", "anchor_to_labeled_centroid_cosine",
    "low_percentile_max", "low_candidate_count", "soft_loss", "soft_gradient_norm",
    "gradient_finite", "old_model_gradient_detected", "optimizer_step_called",
    "hidden_gt_usage",
]


def _run_name(seed: int) -> str:
    return f"fundus_seed{seed}_lcrseg_v0_4a_sra_uniform_full200e"


def _checkpoint(root: Path, seed: int, site_index: int) -> Path:
    site = SITE_ORDER[site_index]
    return root / "runs" / _run_name(seed) / f"checkpoint_final_site{site_index}_{site}.pt"


def _weighted_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.maximum(first, second).sum(dtype=np.float64))
    return float(np.minimum(first, second).sum(dtype=np.float64) / denominator) if denominator > 0 else 1.0


def _stability_rows(
    *, root: Path, seed: int, checkpoint: Path, site_index: int, device: torch.device
) -> list[dict[str, Any]]:
    site = SITE_ORDER[site_index]
    method, payload = load_frozen_method(checkpoint, device)
    records = sorted(
        diagnostic_records(root, seed=seed, dataset="fundus", site=site),
        key=lambda record: stable_seed("v0.4-stability-case", seed, site, record.case_id),
    )[:32]
    if len(records) != 32:
        raise RuntimeError(f"V0.4a stability requires exactly 32 cases: seed={seed} site={site}")
    rows: list[dict[str, Any]] = []
    for record in records:
        image, _ = next(iter(_images_and_labels(record, "fundus")))
        snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for rng_index in range(8):
            augmented = _augment(
                image,
                seed=stable_seed("v0.4-stability-augmentation", seed, site, record.case_id, rng_index),
            )
            snapshot = diagnostic_snapshot(method, payload, torch.from_numpy(augmented)[None].to(device))
            candidate = snapshot.pseudo.valid[0, 0].detach().cpu().numpy().astype(bool)
            weight = snapshot.assimilation_weight[0, 0].detach().cpu().numpy().astype(np.float64)
            weight = np.where(candidate, weight, 0.0)
            hard = candidate & (weight >= 0.5)
            snapshots.append((candidate, weight, hard))
        reference = snapshots[0]
        for rng_index in range(1, 8):
            current = snapshots[rng_index]
            rows.append(
                {
                    "analysis": "stability",
                    "seed": seed,
                    "site_index": site_index,
                    "site": site,
                    "case_id": record.case_id,
                    "rng": rng_index,
                    "candidate_jaccard": _jaccard(reference[0], current[0]),
                    "sra_weighted_jaccard": _weighted_jaccard(reference[1], current[1]),
                    "sra_hard_dominant_jaccard": _jaccard(reference[2], current[2]),
                    "hidden_gt_usage": "images_only_labels_not_used_for_stability",
                }
            )
    return rows


def _anchor_rows(
    *, root: Path, seed: int, checkpoint: Path, site_index: int, device: torch.device
) -> list[dict[str, Any]]:
    site = SITE_ORDER[site_index]
    method, _ = load_frozen_method(checkpoint, device)
    centroids = _labeled_centroids(root=root, seed=seed, site=site, method=method, device=device)
    rows: list[dict[str, Any]] = []
    for class_id in (1, 2):
        anchor = method.current_anchor_bank.anchors[class_id, 0].detach().cpu().numpy().astype(np.float64)
        rows.append(
            {
                "analysis": "anchor",
                "seed": seed,
                "site_index": site_index,
                "site": site,
                "class": class_id,
                "anchor_to_labeled_centroid_cosine": _cosine(anchor, centroids[class_id]),
                "hidden_gt_usage": "none_train_labeled_only",
            }
        )
    return rows


def _low_soft_gradient_rows(
    *, root: Path, seed: int, checkpoint: Path, site_index: int, device: torch.device
) -> list[dict[str, Any]]:
    site = SITE_ORDER[site_index]
    rows: list[dict[str, Any]] = []
    for rng_index in range(8):
        method, payload = load_frozen_method(checkpoint, device)
        method.train()
        if method.old_model is not None:
            method.old_model.eval()
        parameters = [
            parameter for name, parameter in method.model.named_parameters()
            if name.startswith("projection_head.") or name.startswith("dec1.")
        ]
        batch_seed = stable_seed("v0.4-gradient-batch", seed, site, rng_index)
        torch.manual_seed(batch_seed)
        _, unlabeled_batcher = _batchers(root, seed, site, batch_seed)
        unlabeled = unlabeled_batcher.batch_at(0).to(device)
        with torch.no_grad():
            weak_output = method.model(unlabeled.weak_image)
            weak_relation = method._relation(weak_output.relation_features, method.current_anchor_bank)
            pseudo = build_pseudo_labels(
                weak_output.logits.softmax(dim=1),
                weak_relation,
                tau_cls=float(method.config["tau_cls"]),
                tau_anchor=float(method.config["tau_anchor"]),
                delta_anchor=float(method.config["delta_anchor"]),
                tau_spatial=float(method.config["tau_spatial"]),
                temperature_cls=float(method.config["temperature_cls"]),
                temperature_anchor=float(method.config["temperature_anchor"]),
                spatial_floor=float(method.config["spatial_floor"]),
            )
            learnability = compute_learnability(
                weak_output.logits,
                weak_relation,
                pseudo,
                site_step=max(0, int(payload["site_step"]) - 1),
                total_steps=max(1, int(method.total_steps)),
                rank_start=float(method.config["rank_start"]),
                rank_end=float(method.config["rank_end"]),
                rank_temperature=float(method.config["rank_temperature"]),
                relation_margin_center=float(method.config["relation_margin_center"]),
                relation_margin_temperature=float(method.config["relation_margin_temperature"]),
                min_rank_pixels=int(method.config["min_rank_pixels"]),
            )
            method._compute_admission(
                pseudo,
                learnability,
                unlabeled.strong_valid_mask,
                site_step=max(0, int(payload["site_step"]) - 1),
            )
            allocation = getattr(method, "_sra_allocation", None)
            if allocation is None:
                raise AssertionError("missing frozen SRA allocation")
        strong_output = method.model(unlabeled.strong_image)
        strong_relation = method._relation(strong_output.relation_features, method.current_anchor_bank)
        temperature = float(method.config["soft_allocation"]["current_relation_temperature"])
        weak_probability = F.softmax(weak_relation.logits.detach().float() / temperature, dim=1).clamp_min(1.0e-8)
        strong_log_probability = F.log_softmax(strong_relation.logits.float() / temperature, dim=1)
        soft = temperature**2 * (
            weak_probability * (weak_probability.log() - strong_log_probability)
        ).sum(dim=1, keepdim=True)
        strict_valid = strict_relation_valid_mask(unlabeled.strong_valid_mask, tuple(soft.shape[-2:]))
        low = allocation.candidate_mask & strict_valid & allocation.percentile.le(0.20)
        count = int(low.sum())
        if count == 0:
            raise RuntimeError(f"empty low-learnability decile mask: seed={seed} site={site} rng={rng_index}")
        loss = ((1.0 - allocation.alpha.detach()) * soft * low.float()).sum() / low.float().sum()
        gradient = _gradient(loss, parameters)
        old_grad_detected = bool(method.old_model is not None and any(
            parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
            for parameter in method.old_model.parameters()
        ))
        rows.append(
            {
                "analysis": "low_learnability_soft_gradient",
                "seed": seed,
                "site_index": site_index,
                "site": site,
                "rng": rng_index,
                "low_percentile_max": 0.20,
                "low_candidate_count": count,
                "soft_loss": float(loss.detach()),
                "soft_gradient_norm": _norm(gradient),
                "gradient_finite": bool(np.isfinite(gradient).all()),
                "old_model_gradient_detected": old_grad_detected,
                "optimizer_step_called": False,
                "hidden_gt_usage": "none",
            }
        )
    return rows


def _run_worker(root: Path, output_dir: Path, seed: int, device: torch.device) -> None:
    part_dir = output_dir / "_mechanism_parts"
    csv_path = part_dir / f"worker_{seed}.csv"
    json_path = part_dir / f"worker_{seed}.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(f"refusing to overwrite V0.4a mechanism worker {seed}")
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for site_index in range(3):
        checkpoint = _checkpoint(root, seed, site_index)
        hashes[str(checkpoint)] = sha256_path(checkpoint)
        rows.extend(_stability_rows(root=root, seed=seed, checkpoint=checkpoint, site_index=site_index, device=device))
        rows.extend(_anchor_rows(root=root, seed=seed, checkpoint=checkpoint, site_index=site_index, device=device))
        if site_index > 0:
            rows.extend(
                _low_soft_gradient_rows(
                    root=root, seed=seed, checkpoint=checkpoint, site_index=site_index, device=device
                )
            )
    write_csv(csv_path, rows, fieldnames=MECHANISM_FIELDS)
    write_json(
        json_path,
        {
            "protocol_id": "lcrseg_v0_4a",
            "status": "complete",
            "worker_seed": seed,
            "rows": len(rows),
            "checkpoint_sha256": hashes,
            "optimizer_steps": 0,
            "hidden_gt_scope": "independent_post_hoc_only",
        },
    )
    print(json.dumps({"status": "complete", "seed": seed, "rows": len(rows)}, sort_keys=True))


def _merge(root: Path, features_dir: Path, output_dir: Path) -> None:
    raw_csv = output_dir / "v04a_mechanism_raw.csv"
    summary_json = output_dir / "v04a_mechanism_summary.json"
    if raw_csv.exists() or summary_json.exists():
        raise FileExistsError("refusing to overwrite V0.4a mechanism analysis")
    rows: list[dict[str, str]] = []
    checkpoint_hashes: dict[str, str] = {}
    for seed in (0, 1, 2):
        part = output_dir / "_mechanism_parts" / f"worker_{seed}"
        metadata = json.loads(part.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            raise RuntimeError(f"incomplete V0.4a mechanism worker {seed}")
        rows.extend(read_csv(part.with_suffix(".csv")))
        checkpoint_hashes.update(metadata["checkpoint_sha256"])
    write_csv(raw_csv, rows, fieldnames=MECHANISM_FIELDS)

    stability = [row for row in rows if row["analysis"] == "stability"]
    anchors = [row for row in rows if row["analysis"] == "anchor" and int(row["site_index"]) > 0]
    gradients = [row for row in rows if row["analysis"] == "low_learnability_soft_gradient"]
    r1_stability = [
        row for row in read_csv(PROJECT_ROOT / "reports/analysis/v0_4/admission_stability.csv")
        if row["audit_type"] == "augmentation" and row["variant"] == "R1"
    ]
    r1_anchors = [
        row for row in read_csv(PROJECT_ROOT / "reports/analysis/v0_4/anchor_drift.csv")
        if row["variant"] == "R1" and int(row["site_index"]) > 0
    ]
    weighted_jaccard = float(np.mean([float(row["sra_weighted_jaccard"]) for row in stability]))
    r1_hard_jaccard = float(np.mean([float(row["admission_mask_jaccard"]) for row in r1_stability]))
    sra_anchor = float(np.mean([float(row["anchor_to_labeled_centroid_cosine"]) for row in anchors]))
    r1_anchor = float(np.mean([float(row["anchor_to_labeled_centroid_cosine"]) for row in r1_anchors]))

    boundary_sum = 0.0
    boundary_count = 0
    interior_sum = 0.0
    interior_count = 0
    shard_hashes: dict[str, str] = {}
    paths = sorted(features_dir.glob("seed*_SRA_*_max200000.npz"))
    if len(paths) != 27:
        raise RuntimeError(f"expected 27 V0.4a feature shards, found {len(paths)}")
    for path in paths:
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata.get("output_sha256") != sha256_path(path) or metadata.get("variant") != "SRA":
            raise RuntimeError(f"invalid V0.4a feature shard: {path}")
        shard_hashes[path.name] = metadata["output_sha256"]
        with np.load(path) as shard:
            distance = shard["boundary_distance"].astype(np.float64)
            alpha = shard["assimilation_weight"].astype(np.float64)
            effective = alpha + (1.0 - alpha)
            boundary = np.abs(distance) <= 3.0
            interior = distance > 3.0
            boundary_sum += float(effective[boundary].sum(dtype=np.float64))
            boundary_count += int(boundary.sum())
            interior_sum += float(effective[interior].sum(dtype=np.float64))
            interior_count += int(interior.sum())
    if not boundary_count or not interior_count:
        raise RuntimeError("V0.4a boundary/interior mechanism groups are empty")
    boundary_mean = boundary_sum / boundary_count
    interior_mean = interior_sum / interior_count
    boundary_ratio = boundary_mean / interior_mean if interior_mean > 0 else float("nan")
    gradient_norms = [float(row["soft_gradient_norm"]) for row in gradients]
    gradient_passed = bool(
        gradients
        and all(math.isfinite(value) and value > 0 for value in gradient_norms)
        and all(row["gradient_finite"] == "True" for row in gradients)
        and not any(row["old_model_gradient_detected"] == "True" for row in gradients)
    )
    checks = {
        "boundary_effective_assimilation_mass_ratio_ge_0_75": boundary_ratio >= 0.75,
        "low_learnability_soft_gradient_nonzero": gradient_passed,
        "anchor_to_labeled_centroid_not_below_r1": sra_anchor >= r1_anchor,
        "sra_weight_stability_above_r1_hard_mask": weighted_jaccard > r1_hard_jaccard,
    }
    summary = {
        "protocol_id": "lcrseg_v0_4a",
        "status": "complete",
        "hidden_gt_scope": "independent_post_hoc_only",
        "optimizer_steps": 0,
        "checkpoint_sha256": checkpoint_hashes,
        "feature_sha256": shard_hashes,
        "raw": {
            "boundary_effective_mass_mean": boundary_mean,
            "interior_effective_mass_mean": interior_mean,
            "boundary_to_interior_effective_mass_ratio": boundary_ratio,
            "low_learnability_soft_gradient_norm_min": min(gradient_norms),
            "low_learnability_soft_gradient_norm_mean": float(np.mean(gradient_norms)),
            "sra_anchor_to_labeled_centroid_cosine_mean": sra_anchor,
            "r1_anchor_to_labeled_centroid_cosine_mean": r1_anchor,
            "sra_weighted_augmentation_jaccard_mean": weighted_jaccard,
            "r1_hard_mask_augmentation_jaccard_mean": r1_hard_jaccard,
        },
        "checks": checks,
        "mechanism_gate_passed": all(checks.values()),
        "definitions": {
            "effective_assimilation_mass": "alpha hard allocation plus (1-alpha) soft allocation for every valid candidate",
            "low_learnability": "classwise empirical-CDF percentile <= 0.20",
            "weight_stability": "continuous weighted Jaccard under the same fixed 32 cases and 8 photometric RNGs used by the R1 hard-mask audit",
            "anchor_comparison": "mean over sites 1-2, seeds 0-2, foreground classes 1-2",
        },
    }
    write_json(summary_json, summary)
    print(json.dumps({"status": "complete", "checks": checks}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        if args.worker_index is not None or args.features_dir is None:
            raise ValueError("merge requires --features-dir and no --worker-index")
        _merge(args.root.resolve(), args.features_dir.resolve(), args.output_dir.resolve())
    else:
        if args.worker_index not in (0, 1, 2):
            raise ValueError("worker-index must be 0, 1, or 2")
        _run_worker(args.root.resolve(), args.output_dir.resolve(), int(args.worker_index), torch.device(args.device))


if __name__ == "__main__":
    main()
