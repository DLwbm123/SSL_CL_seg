#!/usr/bin/env python3
"""Run/merge the preregistered V0.4 hard-admission stability audit."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.diagnostics import _images_and_labels, diagnostic_records
from lcrseg.analysis.v0_4 import RUN_NAMES, SITE_ORDER, diagnostic_snapshot, load_frozen_method, stable_seed
from lcrseg.common import read_csv, sha256_path, write_csv, write_json


def _jobs(root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        for variant in ("R0", "R1"):
            for site_index, site in enumerate(SITE_ORDER):
                checkpoint = root / "runs" / RUN_NAMES[(seed, variant)] / f"checkpoint_final_site{site_index}_{site}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                jobs.append({"seed": seed, "variant": variant, "site_index": site_index, "site": site, "checkpoint": checkpoint})
    return jobs


def _augment(image: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    contrast = float(rng.uniform(0.90, 1.10))
    brightness = float(rng.uniform(-0.10, 0.10))
    noise = rng.normal(0.0, 0.03, size=image.shape).astype(np.float32)
    centered = (image - image.mean(axis=(1, 2), keepdims=True)) * contrast + image.mean(axis=(1, 2), keepdims=True)
    return np.clip(centered + brightness + noise, 0.0, 1.0).astype(np.float32)


def _jaccard(first: np.ndarray, second: np.ndarray) -> float:
    union = first | second
    return float(np.sum(first & second) / np.sum(union)) if union.any() else 1.0


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return float("nan")
    return float(spearmanr(first, second).statistic)


def _run_worker(root: Path, output_dir: Path, worker_index: int, workers: int, device: torch.device) -> None:
    part_dir = output_dir / "_stability_parts"
    part_csv = part_dir / f"worker_{worker_index}.csv"
    part_json = part_dir / f"worker_{worker_index}.json"
    if part_csv.exists() or part_json.exists():
        raise FileExistsError(f"refusing to overwrite stability worker {worker_index}")
    assigned = [job for index, job in enumerate(_jobs(root)) if index % workers == worker_index]
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for ordinal, job in enumerate(assigned, start=1):
        checkpoint = Path(job["checkpoint"])
        hashes[str(checkpoint)] = sha256_path(checkpoint)
        method, payload = load_frozen_method(checkpoint, device)
        records = diagnostic_records(root, seed=int(job["seed"]), dataset="fundus", site=str(job["site"]))
        records = sorted(
            records,
            key=lambda record: stable_seed("v0.4-stability-case", job["seed"], job["site"], record.case_id),
        )[:32]
        if len(records) != 32:
            raise RuntimeError(f"stability audit requires exactly 32 fixed cases: {job}")
        print(f"RUN {ordinal}/{len(assigned)} seed={job['seed']} {job['variant']} {job['site']}", flush=True)
        for record in records:
            image, _ = next(iter(_images_and_labels(record, "fundus")))
            snapshots: dict[int, dict[int, dict[str, Any]]] = {}
            for rng_index in range(8):
                augmented = _augment(
                    image,
                    seed=stable_seed("v0.4-stability-augmentation", job["seed"], job["site"], record.case_id, rng_index),
                )
                snapshot = diagnostic_snapshot(method, payload, torch.from_numpy(augmented)[None].to(device))
                snapshots[rng_index] = {}
                labels = snapshot.pseudo.labels[0].cpu().numpy()
                candidate = snapshot.pseudo.valid[0, 0].cpu().numpy().astype(bool)
                admitted = snapshot.admission.mask[0, 0].cpu().numpy().astype(bool)
                scores = snapshot.learnability[0, 0].cpu().numpy()
                for class_id in (1, 2):
                    class_candidate = candidate & (labels == class_id)
                    class_admitted = admitted & (labels == class_id)
                    threshold = float(snapshot.admission.learnability_thresholds[class_id])
                    near = (
                        float(np.mean(np.abs(scores[class_candidate] - threshold) <= 0.02))
                        if class_candidate.any() and np.isfinite(threshold)
                        else float("nan")
                    )
                    snapshots[rng_index][class_id] = {
                        "candidate": class_candidate,
                        "admitted": class_admitted,
                        "scores": scores,
                        "threshold": threshold,
                        "threshold_near_fraction": near,
                    }
            for rng_index in range(1, 8):
                for class_id in (1, 2):
                    reference = snapshots[0][class_id]
                    current = snapshots[rng_index][class_id]
                    common = reference["candidate"] & current["candidate"]
                    rows.append(
                        {
                            "audit_type": "augmentation",
                            "status": "complete",
                            "seed": job["seed"],
                            "variant": job["variant"],
                            "site_index": job["site_index"],
                            "site": job["site"],
                            "case_id": record.case_id,
                            "patient_id": record.patient_id,
                            "class": class_id,
                            "reference_rng": 0,
                            "comparison_rng": rng_index,
                            "common_candidate_count": int(common.sum()),
                            "rank_spearman": _spearman(reference["scores"][common], current["scores"][common]),
                            "admission_mask_jaccard": _jaccard(reference["admitted"], current["admitted"]),
                            "threshold_near_fraction": float(np.nanmean([
                                reference["threshold_near_fraction"], current["threshold_near_fraction"]
                            ])),
                            "geometry": "identity_fixed",
                            "photometric_rng_varied": True,
                            "threshold_near_definition": "abs(learnability - class threshold) <= 0.02",
                        }
                    )
    write_csv(part_csv, rows)
    write_json(
        part_json,
        {
            "status": "complete",
            "worker_index": worker_index,
            "workers": workers,
            "jobs": len(assigned),
            "rows": len(rows),
            "checkpoint_sha256": hashes,
        },
    )
    print(json.dumps({"status": "complete", "worker": worker_index, "rows": len(rows)}, sort_keys=True))


def _merge(root: Path, output_dir: Path, workers: int) -> None:
    final_csv = output_dir / "admission_stability.csv"
    final_json = output_dir / "admission_stability_summary.json"
    if final_csv.exists() or final_json.exists():
        raise FileExistsError("refusing to overwrite completed V0.4 admission stability outputs")
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for worker in range(workers):
        part_dir = output_dir / "_stability_parts"
        metadata = json.loads((part_dir / f"worker_{worker}.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "complete" or int(metadata.get("workers", -1)) != workers:
            raise RuntimeError(f"invalid admission stability worker {worker}")
        rows.extend(read_csv(part_dir / f"worker_{worker}.csv"))
        hashes.update(metadata["checkpoint_sha256"])
    expected_augmentation_rows = 3 * 2 * 3 * 32 * 2 * 7
    if len(rows) != expected_augmentation_rows:
        raise RuntimeError(f"expected {expected_augmentation_rows} augmentation rows, found {len(rows)}")
    temporal_rows: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        for variant in ("R0", "R1"):
            run_dir = root / "runs" / RUN_NAMES[(seed, variant)]
            for site_index, site in enumerate(SITE_ORDER):
                for first, second in ((50, 100), (100, 150), (150, 200)):
                    first_matches = list(run_dir.glob(f"checkpoint*site{site_index}*epoch{first}*.pt"))
                    second_matches = list(run_dir.glob(f"checkpoint*site{site_index}*epoch{second}*.pt"))
                    temporal_rows.append(
                        {
                            "audit_type": "temporal",
                            "status": "unavailable",
                            "seed": seed,
                            "variant": variant,
                            "site_index": site_index,
                            "site": site,
                            "epoch_pair": f"{first}-{second}",
                            "first_checkpoint_count": len(first_matches),
                            "second_checkpoint_count": len(second_matches),
                            "reason": "intermediate epoch checkpoint unavailable; protocol forbids retraining to create it",
                        }
                    )
    all_rows = rows + temporal_rows
    write_csv(final_csv, all_rows)
    seed_raw: dict[str, Any] = {}
    seed_support: dict[str, bool] = {}
    for seed in (0, 1, 2):
        selected = [row for row in rows if int(row["seed"]) == seed and row["variant"] == "R1"]
        jaccard = np.asarray([float(row["admission_mask_jaccard"]) for row in selected], dtype=np.float64)
        near = np.asarray([float(row["threshold_near_fraction"]) for row in selected], dtype=np.float64)
        raw = {
            "augmentation_jaccard_mean": float(np.nanmean(jaccard)),
            "augmentation_jaccard_median": float(np.nanmedian(jaccard)),
            "threshold_near_fraction_mean": float(np.nanmean(near)),
            "threshold_near_fraction_median": float(np.nanmedian(near)),
            "augmentation_jaccard_below_0_75": bool(np.nanmean(jaccard) < 0.75),
            "threshold_near_fraction_above_0_20": bool(np.nanmean(near) > 0.20),
        }
        seed_raw[str(seed)] = raw
        seed_support[str(seed)] = bool(raw["augmentation_jaccard_below_0_75"] or raw["threshold_near_fraction_above_0_20"])
    summary = {
        "protocol_id": "lcrseg_v0_4_failure_audit",
        "status": "complete",
        "hidden_gt_usage": "post_hoc_images_only_labels_not_read_for_stability",
        "augmentation_rng_count": 8,
        "fixed_cases_per_seed_site": 32,
        "augmentation_rows": len(rows),
        "temporal_rows": len(temporal_rows),
        "temporal_status": "unavailable",
        "temporal_retraining_performed": False,
        "checkpoint_sha256": hashes,
        "seed_raw": seed_raw,
        "seed_support": seed_support,
        "support_count": int(sum(seed_support.values())),
        "hard_selection_instability_supported_2_of_3": bool(sum(seed_support.values()) >= 2),
        "definitions": {
            "augmentation": "identity geometry; deterministic brightness, contrast, and Gaussian-noise RNG varied",
            "threshold_near": "candidate pixel with abs(learnability - class admission threshold) <= 0.02",
            "seed_decision": "mean R1 augmentation Jaccard < 0.75 or mean R1 threshold-near fraction > 0.20",
        },
    }
    write_json(final_json, summary)
    print(json.dumps({"status": "complete", "seed_support": seed_support}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.workers != 3:
        raise ValueError("V0.4 admission stability is partitioned over exactly three workers")
    if args.merge:
        if args.worker_index is not None:
            raise ValueError("--merge and --worker-index are mutually exclusive")
        _merge(args.root.resolve(), args.output_dir.resolve(), args.workers)
    else:
        if args.worker_index is None or not 0 <= args.worker_index < args.workers:
            raise ValueError("worker-index must be in [0, workers)")
        _run_worker(
            args.root.resolve(), args.output_dir.resolve(), args.worker_index, args.workers, torch.device(args.device)
        )


if __name__ == "__main__":
    main()
