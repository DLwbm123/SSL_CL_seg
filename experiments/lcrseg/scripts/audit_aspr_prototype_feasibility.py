#!/usr/bin/env python3
"""Hidden-GT-only ASPR feasibility audit for one completed seed bundle."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.analysis.diagnostics import _images_and_labels, diagnostic_records  # noqa: E402
from lcrseg.analysis.v0_4 import diagnostic_snapshot, load_frozen_method  # noqa: E402
from lcrseg.common import sha256_path, write_csv, write_json  # noqa: E402
from lcrseg.data import H5LabeledDataset, collate_labeled  # noqa: E402
from lcrseg.memory import MonotonicReliabilityCalibrator  # noqa: E402


SITE_ORDER = ("REFUGE", "RIM_ONE_r3", "Drishti_GS")
FOREGROUND_IDS = (1, 2)


def _loader(dataset: Any, *, workers: int) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_labeled,
        generator=torch.Generator().manual_seed(0),
        persistent_workers=workers > 0,
    )


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(F.cosine_similarity(first.float().reshape(1, -1), second.float().reshape(1, -1), dim=1)[0])


def _normalized(vector: torch.Tensor) -> torch.Tensor:
    value = vector.detach().float().reshape(-1)
    if not torch.isfinite(value).all() or float(value.norm()) <= 1.0e-8:
        raise FloatingPointError("cannot normalize an invalid feasibility vector")
    return F.normalize(value.unsqueeze(0), p=2, dim=1, eps=1.0e-8)[0]


def _selection(
    method: Any,
    payload: dict[str, Any],
    calibrator: MonotonicReliabilityCalibrator,
    image: torch.Tensor,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    snapshot = diagnostic_snapshot(method, payload, image)
    grid_logits = F.interpolate(
        snapshot.logits.float(), size=snapshot.features.shape[-2:], mode="bilinear", align_corners=False
    )
    classifier = grid_logits.argmax(dim=1)
    relation = snapshot.relation_probabilities.argmax(dim=1)
    reliability = calibrator.predict(snapshot.learnability[:, 0], classifier)
    foreground = torch.zeros_like(classifier, dtype=torch.bool)
    for class_id in FOREGROUND_IDS:
        foreground |= classifier.eq(class_id)
    selected = (
        reliability.ge(0.90)
        & classifier.eq(relation)
        & snapshot.pseudo.spatial_agreement[:, 0].ge(0.50)
        & snapshot.pseudo.valid[:, 0]
        & foreground
    )
    return snapshot, classifier, selected


def _visible_feature_sums(
    *,
    method: Any,
    data_root: Path,
    seed: int,
    site_id: str,
    roles: tuple[str, ...],
    device: torch.device,
    workers: int,
) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    dataset = H5LabeledDataset(
        data_root,
        seed=seed,
        dataset="fundus",
        sites=(site_id,),
        roles=roles,
        transform=None,
    )
    sums = {class_id: torch.zeros(method.model.relation_dim, dtype=torch.float64) for class_id in FOREGROUND_IDS}
    counts = {class_id: 0 for class_id in FOREGROUND_IDS}
    with torch.inference_mode():
        for batch in _loader(dataset, workers=workers):
            batch = batch.to(device, non_blocking=True)
            features = F.normalize(method.model(batch.image).relation_features.float(), p=2, dim=1, eps=1.0e-8)
            label = F.interpolate(batch.label.unsqueeze(1).float(), size=features.shape[-2:], mode="nearest")[:, 0].long()
            for class_id in FOREGROUND_IDS:
                mask = label.eq(class_id)
                counts[class_id] += int(mask.sum())
                if bool(mask.any()):
                    sums[class_id] += features.permute(0, 2, 3, 1)[mask].double().sum(dim=0).cpu()
    return sums, counts


def _hidden_training_audit(
    *,
    method: Any,
    payload: dict[str, Any],
    calibrator: MonotonicReliabilityCalibrator,
    data_root: Path,
    seed: int,
    site_id: str,
    device: torch.device,
) -> tuple[dict[int, dict[str, Any]], dict[int, torch.Tensor], dict[int, int]]:
    statistics: dict[int, dict[str, Any]] = {
        class_id: {
            "selected": 0,
            "correct_selected": 0,
            "true_pixels": 0,
            "candidate": 0,
            "case_coverages": [],
            "selected_cases": 0,
            "cases": 0,
        }
        for class_id in FOREGROUND_IDS
    }
    sums = {class_id: torch.zeros(method.model.relation_dim, dtype=torch.float64) for class_id in FOREGROUND_IDS}
    counts = {class_id: 0 for class_id in FOREGROUND_IDS}
    with torch.inference_mode():
        for record in diagnostic_records(data_root, seed=seed, dataset="fundus", site=site_id):
            for image, label in _images_and_labels(record, "fundus"):
                image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
                label_tensor = torch.from_numpy(label).unsqueeze(0).to(device)
                snapshot, classifier, selected = _selection(method, payload, calibrator, image_tensor)
                grid_label = F.interpolate(
                    label_tensor.unsqueeze(1).float(), size=snapshot.features.shape[-2:], mode="nearest"
                )[:, 0].long()
                for class_id in FOREGROUND_IDS:
                    truth = grid_label.eq(class_id)
                    chosen = selected & classifier.eq(class_id)
                    correct = chosen & truth
                    candidate = snapshot.pseudo.valid[:, 0] & classifier.eq(class_id)
                    values = statistics[class_id]
                    values["selected"] += int(chosen.sum())
                    values["correct_selected"] += int(correct.sum())
                    values["true_pixels"] += int(truth.sum())
                    values["candidate"] += int(candidate.sum())
                    values["cases"] += 1
                    case_coverage = float(correct.sum()) / max(1, int(truth.sum()))
                    values["case_coverages"].append(case_coverage)
                    values["selected_cases"] += int(bool(chosen.any()))
                    counts[class_id] += int(truth.sum())
                    if bool(truth.any()):
                        sums[class_id] += snapshot.features.permute(0, 2, 3, 1)[truth].double().sum(dim=0).cpu()
    return statistics, sums, counts


def _oracle_from_sums(sums: dict[int, torch.Tensor], counts: dict[int, int]) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for class_id in FOREGROUND_IDS:
        if counts[class_id] < 1:
            raise RuntimeError(f"oracle prototype has no class {class_id} pixels")
        result[class_id] = _normalized(sums[class_id] / float(counts[class_id]))
    return result


def _memory_selection_rows(
    *,
    bundle: dict[str, Any],
    data_root: Path,
    seed: int,
    device: torch.device,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, torch.Tensor]]]:
    rows: list[dict[str, Any]] = []
    oracle_by_site: dict[str, dict[int, torch.Tensor]] = {}
    aggregate: dict[int, dict[str, Any]] = {
        class_id: defaultdict(float, case_coverages=[]) for class_id in FOREGROUND_IDS
    }
    for site_id in SITE_ORDER:
        site = bundle["sites"][site_id]
        method, payload = load_frozen_method(Path(site["checkpoint"]), device)
        calibrator = MonotonicReliabilityCalibrator.from_state_dict(site["calibrator"])
        hidden_stats, hidden_sums, hidden_counts = _hidden_training_audit(
            method=method,
            payload=payload,
            calibrator=calibrator,
            data_root=data_root,
            seed=seed,
            site_id=site_id,
            device=device,
        )
        labeled_sums, labeled_counts = _visible_feature_sums(
            method=method,
            data_root=data_root,
            seed=seed,
            site_id=site_id,
            roles=("train_labeled",),
            device=device,
            workers=workers,
        )
        total_sums = {class_id: labeled_sums[class_id] + hidden_sums[class_id] for class_id in FOREGROUND_IDS}
        total_counts = {class_id: labeled_counts[class_id] + hidden_counts[class_id] for class_id in FOREGROUND_IDS}
        oracle = _oracle_from_sums(total_sums, total_counts)
        oracle_by_site[site_id] = oracle
        for class_id in FOREGROUND_IDS:
            stats = hidden_stats[class_id]
            labeled = site["labeled_records"][class_id]["prototype"]
            combined = site["combined_records"][class_id]["prototype"]
            cosine_l = _cosine(labeled, oracle[class_id])
            cosine_lu = _cosine(combined, oracle[class_id])
            row = {
                "seed": seed,
                "site_id": site_id,
                "class_id": class_id,
                "selected_pixels": stats["selected"],
                "correct_selected_pixels": stats["correct_selected"],
                "true_foreground_pixels": stats["true_pixels"],
                "candidate_pixels": stats["candidate"],
                "selected_precision": stats["correct_selected"] / max(1, stats["selected"]),
                "selected_foreground_coverage": stats["correct_selected"] / max(1, stats["true_pixels"]),
                "selected_candidate_fraction": stats["selected"] / max(1, stats["candidate"]),
                "median_per_case_coverage": float(np.median(stats["case_coverages"])),
                "selected_case_fraction": stats["selected_cases"] / max(1, stats["cases"]),
                "labeled_case_count": int(site["labeled_records"][class_id]["labeled_case_count"]),
                "unlabeled_case_count": int(site["combined_records"][class_id]["unlabeled_case_count"]),
                "labeled_oracle_cosine": cosine_l,
                "labeled_unlabeled_oracle_cosine": cosine_lu,
                "delta_lu": cosine_lu - cosine_l,
                "hidden_gt_usage": "post_hoc_only",
            }
            rows.append(row)
            agg = aggregate[class_id]
            for key in ("selected", "correct_selected", "true_pixels", "candidate", "selected_cases", "cases"):
                agg[key] += stats[key]
            agg["case_coverages"].extend(stats["case_coverages"])
        del method
        torch.cuda.empty_cache()
    for class_id in FOREGROUND_IDS:
        agg = aggregate[class_id]
        selected = int(agg["selected"])
        correct = int(agg["correct_selected"])
        true_pixels = int(agg["true_pixels"])
        candidate = int(agg["candidate"])
        rows.append(
            {
                "seed": seed,
                "site_id": "ALL",
                "class_id": class_id,
                "selected_pixels": selected,
                "correct_selected_pixels": correct,
                "true_foreground_pixels": true_pixels,
                "candidate_pixels": candidate,
                "selected_precision": correct / max(1, selected),
                "selected_foreground_coverage": correct / max(1, true_pixels),
                "selected_candidate_fraction": selected / max(1, candidate),
                "median_per_case_coverage": float(np.median(agg["case_coverages"])),
                "selected_case_fraction": int(agg["selected_cases"]) / max(1, int(agg["cases"])),
                "labeled_case_count": sum(
                    int(bundle["sites"][site_id]["labeled_records"][class_id]["labeled_case_count"])
                    for site_id in SITE_ORDER
                ),
                "unlabeled_case_count": sum(
                    int(bundle["sites"][site_id]["combined_records"][class_id]["unlabeled_case_count"])
                    for site_id in SITE_ORDER
                ),
                "labeled_oracle_cosine": float("nan"),
                "labeled_unlabeled_oracle_cosine": float("nan"),
                "delta_lu": float("nan"),
                "hidden_gt_usage": "post_hoc_only",
            }
        )
    return rows, oracle_by_site


def _validation_oracle(
    *,
    checkpoint: Path,
    data_root: Path,
    seed: int,
    site_id: str,
    device: torch.device,
    workers: int,
) -> dict[int, torch.Tensor]:
    method, _ = load_frozen_method(checkpoint, device)
    sums, counts = _visible_feature_sums(
        method=method,
        data_root=data_root,
        seed=seed,
        site_id=site_id,
        roles=("val",),
        device=device,
        workers=workers,
    )
    result = _oracle_from_sums(sums, counts)
    del method
    torch.cuda.empty_cache()
    return result


def _all_validation_oracles(
    *, bundle: dict[str, Any], data_root: Path, seed: int, device: torch.device, workers: int
) -> dict[tuple[int, str], dict[int, torch.Tensor]]:
    result: dict[tuple[int, str], dict[int, torch.Tensor]] = {}
    for model_index, model_site in enumerate(SITE_ORDER):
        checkpoint = Path(bundle["sites"][model_site]["checkpoint"])
        for eval_site in SITE_ORDER[: model_index + 1]:
            result[(model_index, eval_site)] = _validation_oracle(
                checkpoint=checkpoint,
                data_root=data_root,
                seed=seed,
                site_id=eval_site,
                device=device,
                workers=workers,
            )
    return result


def _drift_rows(
    *, bundle: dict[str, Any], seed: int, validation: dict[tuple[int, str], dict[int, torch.Tensor]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name, record_name in (("labeled", "labeled_records"), ("combined", "combined_records")):
        for source_index, source_site in enumerate(SITE_ORDER[:-1]):
            for later_index in range(source_index + 1, len(SITE_ORDER)):
                for class_id in FOREGROUND_IDS:
                    prototype = bundle["sites"][source_site][record_name][class_id]["prototype"]
                    self_cosine = _cosine(prototype, validation[(source_index, source_site)][class_id])
                    static_cosine = _cosine(prototype, validation[(later_index, source_site)][class_id])
                    rows.append(
                        {
                            "seed": seed,
                            "memory_source": source_name,
                            "source_site": source_site,
                            "source_site_index": source_index,
                            "later_site": SITE_ORDER[later_index],
                            "later_site_index": later_index,
                            "class_id": class_id,
                            "source_model_self_cosine": self_cosine,
                            "later_model_static_cosine": static_cosine,
                            "cosine_degradation": self_cosine - static_cosine,
                        }
                    )
    return rows


def _transport_rows(
    *, bundle: dict[str, Any], seed: int, validation: dict[tuple[int, str], dict[int, torch.Tensor]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name, snapshot_name in (("labeled", "labeled_memory_snapshots"), ("combined", "combined_memory_snapshots")):
        snapshots = bundle[snapshot_name]
        for current_index in (1, 2):
            transition = f"{current_index - 1}_to_{current_index}"
            before = snapshots[f"after_site{current_index - 1}"]
            for historical_index, historical_site in enumerate(SITE_ORDER[:current_index]):
                for class_offset, class_id in enumerate(FOREGROUND_IDS):
                    transport = bundle["transports"][transition][class_id]
                    static = before[historical_index, class_offset]
                    full = _normalized(static + transport["full_shift"])
                    shrink = _normalized(static + transport["delta"])
                    oracle = validation[(current_index, historical_site)][class_id]
                    static_cosine = _cosine(static, oracle)
                    full_cosine = _cosine(full, oracle)
                    shrink_cosine = _cosine(shrink, oracle)
                    rows.append(
                        {
                            "seed": seed,
                            "memory_source": source_name,
                            "transition": transition,
                            "current_site": SITE_ORDER[current_index],
                            "historical_site": historical_site,
                            "historical_site_index": historical_index,
                            "class_id": class_id,
                            "case_count": int(transport["case_count"]),
                            "shrinkage": float(transport["shrinkage"]),
                            "static_oracle_cosine": static_cosine,
                            "full_shift_oracle_cosine": full_cosine,
                            "shrinkage_oracle_cosine": shrink_cosine,
                            "shrinkage_minus_static": shrink_cosine - static_cosine,
                            "full_shift_minus_static": full_cosine - static_cosine,
                        }
                    )
    return rows


def _site_mode_rows(
    *,
    bundle: dict[str, Any],
    data_root: Path,
    seed: int,
    device: torch.device,
    workers: int,
) -> list[dict[str, Any]]:
    method, _ = load_frozen_method(Path(bundle["sites"][SITE_ORDER[2]]["checkpoint"]), device)
    transition = bundle["transports"]["1_to_2"]
    rows: list[dict[str, Any]] = []
    for source_name, snapshot_name in (("labeled", "labeled_memory_snapshots"), ("combined", "combined_memory_snapshots")):
        bank_site1 = bundle[snapshot_name]["after_site1"].float()
        bank = bank_site1.clone()
        for class_offset, class_id in enumerate(FOREGROUND_IDS):
            delta = transition[class_id]["delta"].float()
            bank[:, class_offset] = F.normalize(bank[:, class_offset] + delta.unsqueeze(0), p=2, dim=1)
        global_bank = F.normalize(bank.mean(dim=0), p=2, dim=1)
        features_by_class: dict[int, list[torch.Tensor]] = {class_id: [] for class_id in FOREGROUND_IDS}
        sites_by_class: dict[int, list[torch.Tensor]] = {class_id: [] for class_id in FOREGROUND_IDS}
        with torch.inference_mode():
            for own_site_index, site_id in enumerate(SITE_ORDER[:2]):
                dataset = H5LabeledDataset(
                    data_root,
                    seed=seed,
                    dataset="fundus",
                    sites=(site_id,),
                    roles=("val",),
                    transform=None,
                )
                for batch in _loader(dataset, workers=workers):
                    batch = batch.to(device, non_blocking=True)
                    features = F.normalize(method.model(batch.image).relation_features.float(), p=2, dim=1, eps=1.0e-8)
                    label = F.interpolate(batch.label.unsqueeze(1).float(), size=features.shape[-2:], mode="nearest")[:, 0].long()
                    flat_features = features.permute(0, 2, 3, 1)
                    for class_id in FOREGROUND_IDS:
                        mask = label.eq(class_id)
                        if bool(mask.any()):
                            selected = flat_features[mask].cpu()
                            features_by_class[class_id].append(selected)
                            sites_by_class[class_id].append(
                                torch.full((len(selected),), own_site_index, dtype=torch.long)
                            )
        all_feature: list[torch.Tensor] = []
        all_truth: list[torch.Tensor] = []
        per_class_cache: dict[int, dict[str, Any]] = {}
        for class_offset, class_id in enumerate(FOREGROUND_IDS):
            feature = torch.cat(features_by_class[class_id]).float()
            own_site = torch.cat(sites_by_class[class_id])
            within_scores = feature @ bank[:, class_offset].T
            top_site = within_scores.argmax(dim=1)
            nearest_distance = 1.0 - within_scores.max(dim=1).values
            global_distance = 1.0 - feature @ global_bank[class_offset]
            median_nearest = float(nearest_distance.median())
            median_global = float(global_distance.median())
            reduction = 1.0 - median_nearest / max(median_global, 1.0e-8)
            occupancy = torch.bincount(top_site, minlength=2)
            probability = occupancy.float() / max(1, int(occupancy.sum()))
            entropy = float(-(probability[probability > 0] * probability[probability > 0].log()).sum())
            per_class_cache[class_id] = {
                "feature": feature,
                "own_site": own_site,
                "top_site": top_site,
                "median_nearest": median_nearest,
                "median_global": median_global,
                "reduction": reduction,
                "occupancy": occupancy,
                "entropy": entropy,
            }
            all_feature.append(feature)
            all_truth.append(torch.full((len(feature),), class_offset, dtype=torch.long))
        combined_feature = torch.cat(all_feature)
        combined_truth = torch.cat(all_truth)
        global_scores = combined_feature @ global_bank.T
        site_scores = torch.stack(
            [combined_feature @ bank[:, class_offset].T for class_offset in range(len(FOREGROUND_IDS))], dim=1
        ).max(dim=2).values
        global_accuracy = float(global_scores.argmax(dim=1).eq(combined_truth).float().mean())
        max_site_accuracy = float(site_scores.argmax(dim=1).eq(combined_truth).float().mean())
        for class_id in FOREGROUND_IDS:
            cache = per_class_cache[class_id]
            rows.append(
                {
                    "seed": seed,
                    "memory_source": source_name,
                    "class_id": class_id,
                    "pixels": len(cache["feature"]),
                    "median_nearest_prototype_distance": cache["median_nearest"],
                    "median_global_prototype_distance": cache["median_global"],
                    "nearest_distance_reduction": cache["reduction"],
                    "global_ncm_accuracy_all_foreground": global_accuracy,
                    "max_over_site_ncm_accuracy_all_foreground": max_site_accuracy,
                    "own_site_top_mode_rate": float(cache["top_site"].eq(cache["own_site"]).float().mean()),
                    "site0_occupancy": int(cache["occupancy"][0]),
                    "site1_occupancy": int(cache["occupancy"][1]),
                    "site_mode_entropy": cache["entropy"],
                    "site_conditioned_global_reconstruction_ratio": cache["median_nearest"]
                    / max(cache["median_global"], 1.0e-8),
                }
            )
    del method
    torch.cuda.empty_cache()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    bundle_path = args.bundle.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    outputs = {
        "memory": output_dir / "memory_selection_quality.csv",
        "drift": output_dir / "prototype_drift.csv",
        "transport": output_dir / "transport_quality.csv",
        "mode": output_dir / "site_mode_utility.csv",
        "summary": output_dir / "feasibility_seed_summary.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite ASPR feasibility seed artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if bundle.get("protocol_id") != "asprseg_v0_1" or bundle.get("hidden_gt_usage") != "none":
        raise ValueError("invalid ASPR reconstruction bundle")
    seed = int(bundle["seed"])
    device = torch.device(args.device)
    memory_rows, _ = _memory_selection_rows(
        bundle=bundle,
        data_root=data_root,
        seed=seed,
        device=device,
        workers=args.workers,
    )
    validation = _all_validation_oracles(
        bundle=bundle,
        data_root=data_root,
        seed=seed,
        device=device,
        workers=args.workers,
    )
    drift_rows = _drift_rows(bundle=bundle, seed=seed, validation=validation)
    transport_rows = _transport_rows(bundle=bundle, seed=seed, validation=validation)
    mode_rows = _site_mode_rows(
        bundle=bundle,
        data_root=data_root,
        seed=seed,
        device=device,
        workers=args.workers,
    )
    write_csv(outputs["memory"], memory_rows)
    write_csv(outputs["drift"], drift_rows)
    write_csv(outputs["transport"], transport_rows)
    write_csv(outputs["mode"], mode_rows)
    summary = {
        "protocol_id": "asprseg_v0_1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ASPR_FEASIBILITY_SEED_AUDIT_COMPLETE",
        "seed": seed,
        "hidden_gt_usage": "post_hoc_only",
        "optimizer_steps": 0,
        "bundle": str(bundle_path),
        "bundle_sha256": sha256_path(bundle_path),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "rows": {
            "memory_selection_quality": len(memory_rows),
            "prototype_drift": len(drift_rows),
            "transport_quality": len(transport_rows),
            "site_mode_utility": len(mode_rows),
        },
        "artifacts": {name: {"path": str(path), "sha256": sha256_path(path)} for name, path in outputs.items() if name != "summary"},
    }
    write_json(outputs["summary"], summary)
    print(json.dumps({"status": summary["status"], "seed": seed, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
