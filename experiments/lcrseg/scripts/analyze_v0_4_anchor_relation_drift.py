#!/usr/bin/env python3
"""Audit V0.4 R0/R1 anchor and relation drift from frozen checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.diagnostics import _images_and_labels, diagnostic_records
from lcrseg.analysis.v0_4 import RUN_NAMES, SITE_ORDER, load_frozen_method, stable_seed
from lcrseg.common import read_csv, sha256_path, write_csv, write_json


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 0 else float("nan")


def _checkpoint(root: Path, seed: int, variant: str, site_index: int, site: str) -> Path:
    return root / "runs" / RUN_NAMES[(seed, variant)] / f"checkpoint_final_site{site_index}_{site}.pt"


@torch.no_grad()
def _labeled_centroids(
    *, root: Path, seed: int, site: str, method: Any, device: torch.device
) -> dict[int, np.ndarray]:
    rows = [
        row for row in read_csv(root / "manifests/training" / f"lcrseg_v1_seed{seed}.csv")
        if row.get("dataset") == "fundus"
        and row.get("site_or_vendor") == site
        and row.get("primary_20pct_split") == "train_labeled"
    ]
    if not rows:
        raise RuntimeError(f"no labeled rows for seed={seed}, site={site}")
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = defaultdict(int)
    for row in rows:
        with h5py.File(root / "h5/v1" / row["image_h5_relpath"], "r") as handle:
            image = np.asarray(handle["image"], dtype=np.float32) / 255.0
        with h5py.File(root / "h5/v1" / row["label_h5_relpath"], "r") as handle:
            label = np.asarray(handle["label"], dtype=np.int64)
        output = method.model(torch.from_numpy(image)[None].to(device))
        feature = F.normalize(output.relation_features.float(), p=2, dim=1, eps=1.0e-8)[0]
        grid_label = F.interpolate(
            torch.from_numpy(label)[None, None].float(), size=feature.shape[-2:], mode="nearest"
        )[0, 0].long()
        value = feature.permute(1, 2, 0)
        for class_id in (1, 2):
            selected = value[grid_label == class_id]
            if not selected.numel():
                continue
            part = selected.sum(dim=0).cpu().numpy().astype(np.float64)
            sums[class_id] = sums.get(class_id, np.zeros_like(part)) + part
            counts[class_id] += int(len(selected))
    result: dict[int, np.ndarray] = {}
    for class_id in (1, 2):
        if counts[class_id] == 0:
            raise RuntimeError(f"empty labeled centroid for seed={seed}, site={site}, class={class_id}")
        centroid = sums[class_id] / counts[class_id]
        result[class_id] = centroid / max(1.0e-12, float(np.linalg.norm(centroid)))
    return result


@torch.no_grad()
def _previous_relation_kl(
    *, root: Path, seed: int, previous_site: str, method: Any, device: torch.device
) -> float:
    if method.old_model is None or method.old_anchor_bank is None:
        return float("nan")
    records = diagnostic_records(root, seed=seed, dataset="fundus", site=previous_site)
    records = sorted(records, key=lambda record: stable_seed("v0.4-relation-kl", seed, previous_site, record.case_id))[:32]
    values: list[float] = []
    for record in records:
        for image, _ in _images_and_labels(record, "fundus"):
            tensor = torch.from_numpy(image)[None].to(device)
            current_output = method.model(tensor)
            old_output = method.old_model(tensor)
            current = method._relation(current_output.relation_features, method.current_anchor_bank).probabilities.float()
            old = method._relation(old_output.relation_features, method.old_anchor_bank).probabilities.float()
            kl = torch.sum(old.clamp_min(1.0e-8) * (old.clamp_min(1.0e-8).log() - current.clamp_min(1.0e-8).log()), dim=1)
            values.append(float(kl.mean().cpu()))
    return float(np.mean(values)) if values else float("nan")


def _run_metrics(run_dir: Path) -> dict[str, float]:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))["summary"]
    rows = list(csv.DictReader((run_dir / "site_matrix_long.csv").open(encoding="utf-8")))
    refuge_final = [
        row for row in rows
        if row["trained_site"] == "Drishti_GS" and row["evaluation_site"] == "REFUGE"
    ]
    if len(refuge_final) != 1:
        raise RuntimeError(f"missing unique final REFUGE row: {run_dir}")
    return {
        "final": float(summary["final_average_dice"]),
        "bwt": float(summary["bwt"]),
        "previous": float(summary["previous_site_dice"]),
        "refuge_class1": float(refuge_final[0]["dice_class_1"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    features_dir = args.features_dir.resolve()
    output_dir = args.output_dir.resolve()
    outputs = [
        output_dir / "anchor_drift.csv",
        output_dir / "relation_drift.csv",
        output_dir / "anchor_relation_summary.json",
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite completed V0.4 anchor/relation audit")
    device = torch.device(args.device)
    anchor_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    drift_vectors: dict[tuple[int, str, int, int], np.ndarray] = {}
    for seed in (0, 1, 2):
        for variant in ("R0", "R1"):
            for site_index, site in enumerate(SITE_ORDER):
                checkpoint = _checkpoint(root, seed, variant, site_index, site)
                method, _ = load_frozen_method(checkpoint, device)
                checkpoint_hashes[str(checkpoint)] = sha256_path(checkpoint)
                centroids = _labeled_centroids(root=root, seed=seed, site=site, method=method, device=device)
                shard_path = features_dir / (
                    f"seed{seed}_{variant}_through{site_index}-{site}_eval-{site}_max200000.npz"
                )
                if not shard_path.is_file():
                    raise FileNotFoundError(shard_path)
                with np.load(shard_path) as shard:
                    anchors = shard["anchors"].astype(np.float64)
                    predicted = shard["predicted_class"].astype(np.int64)
                    admitted = shard["admitted"].astype(bool)
                    for class_id in (1, 2):
                        class_mask = predicted == class_id
                        candidate_features = shard["features"][class_mask].astype(np.float64)
                        admitted_features = shard["features"][class_mask & admitted].astype(np.float64)
                        candidate_centroid = candidate_features.mean(axis=0)
                        admitted_centroid = admitted_features.mean(axis=0)
                        end_anchor = anchors[class_id]
                        old_anchor = (
                            method.old_anchor_bank.anchors[class_id, 0].detach().cpu().numpy().astype(np.float64)
                            if method.old_anchor_bank is not None else np.full_like(end_anchor, np.nan)
                        )
                        if site_index == 0:
                            start_anchor = np.full_like(end_anchor, np.nan)
                            drift = np.full_like(end_anchor, np.nan)
                        else:
                            previous_checkpoint = _checkpoint(
                                root, seed, variant, site_index - 1, SITE_ORDER[site_index - 1]
                            )
                            previous_method, _ = load_frozen_method(previous_checkpoint, "cpu")
                            start_anchor = previous_method.current_anchor_bank.anchors[class_id, 0].detach().cpu().numpy().astype(np.float64)
                            drift = end_anchor - start_anchor
                        drift_vectors[(seed, variant, site_index, class_id)] = drift
                        anchor_rows.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "site_index": site_index,
                                "site": site,
                                "class": class_id,
                                "checkpoint": str(checkpoint),
                                "checkpoint_sha256": checkpoint_hashes[str(checkpoint)],
                                "anchor_to_labeled_centroid_cosine": _cosine(end_anchor, centroids[class_id]),
                                "anchor_to_candidate_centroid_cosine": _cosine(end_anchor, candidate_centroid),
                                "anchor_to_admitted_centroid_cosine": _cosine(end_anchor, admitted_centroid),
                                "historical_current_anchor_cosine": _cosine(old_anchor, end_anchor),
                                "anchor_drift_norm": float(np.linalg.norm(drift)) if np.isfinite(drift).all() else float("nan"),
                                "start_anchor_available": bool(site_index > 0),
                            }
                        )
                    for class_id in (1, 2):
                        class_mask = predicted == class_id
                        for state, state_mask in (("candidate", class_mask), ("admitted", class_mask & admitted)):
                            relation_rows.append(
                                {
                                    "seed": seed,
                                    "variant": variant,
                                    "site_index": site_index,
                                    "site": site,
                                    "class": class_id,
                                    "state": state,
                                    "count": int(state_mask.sum()),
                                    "relation_entropy_mean": float(shard["relation_entropy"][state_mask].mean()),
                                    "relation_margin_mean": float(shard["anchor_margin"][state_mask].mean()),
                                    "previous_site_relation_kl": float("nan"),
                                }
                            )
                if site_index > 0:
                    previous_kl = _previous_relation_kl(
                        root=root,
                        seed=seed,
                        previous_site=SITE_ORDER[site_index - 1],
                        method=method,
                        device=device,
                    )
                    for row in relation_rows:
                        if (
                            row["seed"] == seed
                            and row["variant"] == variant
                            and row["site_index"] == site_index
                        ):
                            row["previous_site_relation_kl"] = previous_kl
    for row in anchor_rows:
        site_index = int(row["site_index"])
        if site_index == 2:
            previous = drift_vectors[(int(row["seed"]), str(row["variant"]), 1, int(row["class"]))]
            current = drift_vectors[(int(row["seed"]), str(row["variant"]), 2, int(row["class"]))]
            row["site_to_site_drift_direction_cosine"] = _cosine(previous, current)
        else:
            row["site_to_site_drift_direction_cosine"] = float("nan")
    write_csv(output_dir / "anchor_drift.csv", anchor_rows)
    write_csv(output_dir / "relation_drift.csv", relation_rows)

    deltas: list[dict[str, Any]] = []
    correlations: dict[str, Any] = {}
    for seed in (0, 1, 2):
        metrics = {
            variant: _run_metrics(root / "runs" / RUN_NAMES[(seed, variant)])
            for variant in ("R0", "R1")
        }
        r0 = [row for row in anchor_rows if row["seed"] == seed and row["variant"] == "R0" and row["site_index"] > 0]
        r1 = [row for row in anchor_rows if row["seed"] == seed and row["variant"] == "R1" and row["site_index"] > 0]
        delta = {
            "seed": seed,
            "anchor_drift": float(np.mean([row["anchor_drift_norm"] for row in r1]) - np.mean([row["anchor_drift_norm"] for row in r0])),
            **{name: metrics["R1"][name] - metrics["R0"][name] for name in metrics["R0"]},
        }
        deltas.append(delta)
    for metric in ("final", "bwt", "previous", "refuge_class1"):
        result = spearmanr([row["anchor_drift"] for row in deltas], [row[metric] for row in deltas])
        correlations[metric] = {"spearman": float(result.statistic), "pvalue_descriptive_only": float(result.pvalue)}
    failed_seed_checks: dict[str, Any] = {}
    for seed in (1, 2):
        labeled = {
            variant: float(np.mean([
                row["anchor_to_labeled_centroid_cosine"] for row in anchor_rows
                if row["seed"] == seed and row["variant"] == variant and row["site_index"] > 0
            ])) for variant in ("R0", "R1")
        }
        relation_kl = {
            variant: float(np.mean([
                row["previous_site_relation_kl"] for row in relation_rows
                if row["seed"] == seed and row["variant"] == variant and row["site_index"] > 0 and row["state"] == "candidate"
            ])) for variant in ("R0", "R1")
        }
        previous_delta = next(row["previous"] for row in deltas if row["seed"] == seed)
        failed_seed_checks[str(seed)] = {
            "r1_labeled_cosine_below_r0": labeled["R1"] < labeled["R0"],
            "r1_previous_relation_kl_above_r0": relation_kl["R1"] > relation_kl["R0"],
            "previous_site_dice_delta_negative": previous_delta < 0,
            "raw": {"labeled_cosine": labeled, "previous_relation_kl": relation_kl, "previous_dice_delta": previous_delta},
        }
    biased = all(all(value[key] for key in (
        "r1_labeled_cosine_below_r0",
        "r1_previous_relation_kl_above_r0",
        "previous_site_dice_delta_negative",
    )) for value in failed_seed_checks.values())
    summary = {
        "protocol_id": "lcrseg_v0_4_failure_audit",
        "status": "complete",
        "hidden_gt_usage": "post_hoc_previous_site_relation_kl_only",
        "checkpoint_sha256": checkpoint_hashes,
        "seedwise_deltas_r1_minus_r0": deltas,
        "spearman_descriptive_only_n3": correlations,
        "failed_seed_checks": failed_seed_checks,
        "biased_memory_update_supported": bool(biased),
        "definitions": {
            "labeled_centroid": "L2-normalized centroid of current-model projection features over frozen train_labeled pixels at the current site",
            "candidate_admitted_centroid": "centroid of deterministic sampled foreground pseudo-label features on the current evaluation site",
            "previous_site_relation_kl": "mean KL(old historical relation || current relation) on 32 fixed post-hoc diagnostic cases",
        },
    }
    write_json(output_dir / "anchor_relation_summary.json", summary)
    print(json.dumps({"status": "complete", "anchor_rows": len(anchor_rows), "relation_rows": len(relation_rows), "biased": biased}, sort_keys=True))


if __name__ == "__main__":
    main()
