#!/usr/bin/env python3
"""Reconstruct one seed of ASPR memory without consulting hidden labels."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.analysis.v0_4 import diagnostic_snapshot, load_frozen_method  # noqa: E402
from lcrseg.common import sha256_path, write_json  # noqa: E402
from lcrseg.data import H5LabeledDataset, H5UnlabeledDataset, collate_labeled, collate_unlabeled  # noqa: E402
from lcrseg.memory import (  # noqa: E402
    MonotonicReliabilityCalibrator,
    SitePrototypeBuilder,
    SitePrototypeMemory,
    estimate_transport,
)


SITE_ORDER = ("REFUGE", "RIM_ONE_r3", "Drishti_GS")
RUNS = {
    0: "fundus_seed0_lcrseg_uniform_relation_kd_full200e",
    1: "fundus_seed1_lcrseg_v0_3_r0_legacy_uniform_full200e",
    2: "fundus_seed2_lcrseg_v0_3_r0_legacy_uniform_full200e",
}
FOREGROUND_IDS = (1, 2)


def _loader(dataset: Any, *, batch_size: int, collate_fn: Any, workers: int) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        generator=generator,
        persistent_workers=workers > 0,
    )


def _checkpoint_path(run_root: Path, seed: int, site_index: int, site_id: str) -> Path:
    return run_root / RUNS[seed] / f"checkpoint_final_site{site_index}_{site_id}.pt"


def _gpu_record(physical_gpu: int, device: torch.device) -> dict[str, Any]:
    record: dict[str, Any] = {
        "physical_index": physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    uuid, name, driver = (item.strip() for item in result.stdout.strip().split(",", maxsplit=2))
    record.update(uuid=uuid, name=name, driver=driver)
    return record


def _collect_site(
    *,
    data_root: Path,
    seed: int,
    site_id: str,
    checkpoint: Path,
    device: torch.device,
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    method, payload = load_frozen_method(checkpoint, device)
    batch_size = int(payload["config_resolved"]["training"].get("evaluation_batch_size", 4))
    if batch_size != 4:
        raise RuntimeError("ASPR feasibility expected the frozen Fundus evaluation batch size 4")
    feature_dim = int(method.model.relation_dim)
    builder = SitePrototypeBuilder(feature_dim, FOREGROUND_IDS, minimum_pixels=32)
    labeled_dataset = H5LabeledDataset(
        data_root,
        seed=seed,
        dataset="fundus",
        sites=(site_id,),
        roles=("train_labeled",),
        transform=None,
    )
    score_chunks: list[np.ndarray] = []
    correct_chunks: list[np.ndarray] = []
    class_chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []
    labeled_cases: set[str] = set()
    with torch.inference_mode():
        for batch in _loader(labeled_dataset, batch_size=batch_size, collate_fn=collate_labeled, workers=workers):
            batch = batch.to(device, non_blocking=True)
            snapshot = diagnostic_snapshot(method, payload, batch.image)
            grid_label = F.interpolate(
                batch.label.unsqueeze(1).float(), size=snapshot.features.shape[-2:], mode="nearest"
            )[:, 0].long()
            pseudo_class = snapshot.pseudo.labels
            pseudo_valid = snapshot.pseudo.valid[:, 0]
            score_chunks.append(snapshot.learnability[:, 0].cpu().numpy().reshape(-1))
            correct_chunks.append(pseudo_class.eq(grid_label).cpu().numpy().reshape(-1))
            class_chunks.append(pseudo_class.cpu().numpy().reshape(-1))
            valid_chunks.append(pseudo_valid.cpu().numpy().reshape(-1))
            for index, case_id in enumerate(batch.case_id):
                labeled_cases.add(case_id)
                builder.add_labeled(case_id, snapshot.features[index], grid_label[index])
    calibrator = MonotonicReliabilityCalibrator(FOREGROUND_IDS).fit(
        np.concatenate(score_chunks),
        np.concatenate(correct_chunks),
        np.concatenate(class_chunks),
        np.concatenate(valid_chunks),
    )

    unlabeled_dataset = H5UnlabeledDataset(
        data_root,
        seed=seed,
        dataset="fundus",
        sites=(site_id,),
        transform=None,
    )
    unlabeled_cases: set[str] = set()
    candidate_by_class = {class_id: 0 for class_id in FOREGROUND_IDS}
    reliable_by_class = {class_id: 0 for class_id in FOREGROUND_IDS}
    reliable_weight_by_class = {class_id: 0.0 for class_id in FOREGROUND_IDS}
    with torch.inference_mode():
        for batch in _loader(unlabeled_dataset, batch_size=batch_size, collate_fn=collate_unlabeled, workers=workers):
            batch = batch.to(device, non_blocking=True)
            snapshot = diagnostic_snapshot(method, payload, batch.weak_image)
            grid_logits = F.interpolate(
                snapshot.logits.float(), size=snapshot.features.shape[-2:], mode="bilinear", align_corners=False
            )
            classifier_class = grid_logits.argmax(dim=1)
            relation_class = snapshot.relation_probabilities.argmax(dim=1)
            reliability = calibrator.predict(snapshot.learnability[:, 0], classifier_class)
            foreground = torch.zeros_like(classifier_class, dtype=torch.bool)
            for class_id in FOREGROUND_IDS:
                foreground |= classifier_class.eq(class_id)
            reliable = (
                reliability.ge(0.90)
                & classifier_class.eq(relation_class)
                & snapshot.pseudo.spatial_agreement[:, 0].ge(0.50)
                & snapshot.pseudo.valid[:, 0]
                & foreground
            )
            reliable_weight = torch.where(reliable, reliability, torch.zeros_like(reliability)).detach()
            for class_id in FOREGROUND_IDS:
                candidates = snapshot.pseudo.valid[:, 0] & classifier_class.eq(class_id)
                selected = reliable & classifier_class.eq(class_id)
                candidate_by_class[class_id] += int(candidates.sum())
                reliable_by_class[class_id] += int(selected.sum())
                reliable_weight_by_class[class_id] += float(reliable_weight[selected].sum())
            for index, case_id in enumerate(batch.case_id):
                unlabeled_cases.add(case_id)
                builder.add_unlabeled(
                    case_id,
                    snapshot.features[index],
                    classifier_class[index],
                    reliable_weight[index],
                )

    labeled_records = builder.build(include_unlabeled=False)
    combined_records = builder.build(include_unlabeled=True)
    site_payload: dict[str, Any] = {
        "site_id": site_id,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "manifest_sha256": str(payload["manifest_hash"]),
        "split_sha256": str(payload["data_split_hash"]),
        "feature_dim": feature_dim,
        "batch_size": batch_size,
        "calibrator": calibrator.state_dict(),
        "labeled_records": labeled_records,
        "combined_records": combined_records,
        "labeled_case_ids": sorted(labeled_cases),
        "unlabeled_case_ids": sorted(unlabeled_cases),
        "selection_without_hidden_gt": {
            str(class_id): {
                "candidate_pixels": candidate_by_class[class_id],
                "reliable_pixels": reliable_by_class[class_id],
                "reliable_weight": reliable_weight_by_class[class_id],
            }
            for class_id in FOREGROUND_IDS
        },
    }
    summary = {
        "site_id": site_id,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": site_payload["checkpoint_sha256"],
        "feature_dim": feature_dim,
        "batch_size": batch_size,
        "labeled_cases": len(labeled_cases),
        "unlabeled_cases": len(unlabeled_cases),
        "calibrator_support": calibrator.support,
        "classwise_calibrators": sorted(calibrator.class_curves),
        "selection_without_hidden_gt": site_payload["selection_without_hidden_gt"],
        "labeled_memory_counts": {
            str(class_id): int(labeled_records[class_id]["labeled_case_count"]) for class_id in FOREGROUND_IDS
        },
        "combined_memory_counts": {
            str(class_id): {
                "labeled": int(combined_records[class_id]["labeled_case_count"]),
                "unlabeled": int(combined_records[class_id]["unlabeled_case_count"]),
            }
            for class_id in FOREGROUND_IDS
        },
    }
    del method
    torch.cuda.empty_cache()
    return site_payload, summary


def _paired_transport(
    *,
    data_root: Path,
    seed: int,
    site_id: str,
    old_checkpoint: Path,
    current_checkpoint: Path,
    device: torch.device,
    workers: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    old_method, old_payload = load_frozen_method(old_checkpoint, device)
    current_method, current_payload = load_frozen_method(current_checkpoint, device)
    batch_size = int(current_payload["config_resolved"]["training"].get("evaluation_batch_size", 4))
    dataset = H5LabeledDataset(
        data_root,
        seed=seed,
        dataset="fundus",
        sites=(site_id,),
        roles=("train_labeled",),
        transform=None,
    )
    old_cases: dict[int, list[torch.Tensor]] = {class_id: [] for class_id in FOREGROUND_IDS}
    current_cases: dict[int, list[torch.Tensor]] = {class_id: [] for class_id in FOREGROUND_IDS}
    case_ids: dict[int, list[str]] = {class_id: [] for class_id in FOREGROUND_IDS}
    with torch.inference_mode():
        for batch in _loader(dataset, batch_size=batch_size, collate_fn=collate_labeled, workers=workers):
            batch = batch.to(device, non_blocking=True)
            old_features = F.normalize(old_method.model(batch.image).relation_features.float(), p=2, dim=1, eps=1.0e-8)
            current_features = F.normalize(current_method.model(batch.image).relation_features.float(), p=2, dim=1, eps=1.0e-8)
            grid_label = F.interpolate(batch.label.unsqueeze(1).float(), size=old_features.shape[-2:], mode="nearest")[:, 0].long()
            for index, case_id in enumerate(batch.case_id):
                for class_id in FOREGROUND_IDS:
                    mask = grid_label[index].eq(class_id)
                    if int(mask.sum()) < 32:
                        continue
                    old_center = F.normalize(old_features[index, :, mask].mean(dim=1).unsqueeze(0), p=2, dim=1)[0]
                    current_center = F.normalize(current_features[index, :, mask].mean(dim=1).unsqueeze(0), p=2, dim=1)[0]
                    old_cases[class_id].append(old_center.cpu())
                    current_cases[class_id].append(current_center.cpu())
                    case_ids[class_id].append(case_id)
    estimates: dict[int, dict[str, Any]] = {}
    summary: dict[str, Any] = {
        "site_id": site_id,
        "old_checkpoint": str(old_checkpoint),
        "old_checkpoint_sha256": sha256_path(old_checkpoint),
        "current_checkpoint": str(current_checkpoint),
        "current_checkpoint_sha256": sha256_path(current_checkpoint),
        "classes": {},
    }
    for class_id in FOREGROUND_IDS:
        old_tensor = torch.stack(old_cases[class_id]) if old_cases[class_id] else torch.empty((0, old_method.model.relation_dim))
        current_tensor = torch.stack(current_cases[class_id]) if current_cases[class_id] else torch.empty_like(old_tensor)
        estimate = estimate_transport(old_tensor, current_tensor)
        estimates[class_id] = {
            "case_ids": case_ids[class_id],
            "case_count": estimate.case_count,
            "mean_displacement": estimate.mean_displacement,
            "full_shift": estimate.full_shift,
            "shrinkage": estimate.shrinkage,
            "delta": estimate.delta,
            "variance": estimate.variance,
            "signal": estimate.signal,
            "valid": estimate.valid,
        }
        summary["classes"][str(class_id)] = {
            "case_count": estimate.case_count,
            "shrinkage": estimate.shrinkage,
            "variance": estimate.variance,
            "signal": estimate.signal,
            "delta_norm": float(estimate.delta.norm()),
            "full_shift_norm": float(estimate.full_shift.norm()),
            "valid": estimate.valid,
        }
    del old_method, current_method
    torch.cuda.empty_cache()
    return estimates, summary


def _records_for_memory(records: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(class_id): dict(record) for class_id, record in records.items()}


def _build_sequential_memory(
    sites: dict[str, dict[str, Any]],
    transports: dict[str, dict[int, dict[str, Any]]],
    *,
    source: str,
    class_semantics_sha256: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    feature_dim = int(sites[SITE_ORDER[0]]["feature_dim"])
    memory = SitePrototypeMemory(feature_dim, FOREGROUND_IDS)
    snapshots: dict[str, torch.Tensor] = {}
    for site_index, site_id in enumerate(SITE_ORDER):
        if site_index:
            transition = f"{site_index - 1}_to_{site_index}"
            memory.commit_transport(
                {class_id: transports[transition][class_id]["delta"] for class_id in FOREGROUND_IDS},
                end_site=True,
            )
        site = sites[site_id]
        memory.append_site(
            site_id,
            _records_for_memory(site[source]),
            source_checkpoint_sha256=site["checkpoint_sha256"],
            class_semantics_sha256=class_semantics_sha256,
            manifest_sha256=site["manifest_sha256"],
            split_sha256=site["split_sha256"],
        )
        snapshots[f"after_site{site_index}"] = memory.get_old_frame_bank().cpu()
    return memory.state_dict(), snapshots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    bundle_path = output_dir / f"seed{args.seed}_memory_reconstruction.pt"
    summary_path = output_dir / f"seed{args.seed}_memory_reconstruction.json"
    if bundle_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite ASPR memory reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reconstruction requested but unavailable")
    class_semantics_path = ROOT / "reports" / "experiment_status" / "class_semantics.json"
    class_semantics_sha = sha256_path(class_semantics_path)
    sites: dict[str, dict[str, Any]] = {}
    site_summaries: list[dict[str, Any]] = []
    for site_index, site_id in enumerate(SITE_ORDER):
        checkpoint = _checkpoint_path(run_root, args.seed, site_index, site_id)
        site_payload, site_summary = _collect_site(
            data_root=data_root,
            seed=args.seed,
            site_id=site_id,
            checkpoint=checkpoint,
            device=device,
            workers=args.workers,
        )
        sites[site_id] = site_payload
        site_summaries.append(site_summary)
    transports: dict[str, dict[int, dict[str, Any]]] = {}
    transport_summaries: list[dict[str, Any]] = []
    for current_index in (1, 2):
        transition = f"{current_index - 1}_to_{current_index}"
        estimates, summary = _paired_transport(
            data_root=data_root,
            seed=args.seed,
            site_id=SITE_ORDER[current_index],
            old_checkpoint=_checkpoint_path(run_root, args.seed, current_index - 1, SITE_ORDER[current_index - 1]),
            current_checkpoint=_checkpoint_path(run_root, args.seed, current_index, SITE_ORDER[current_index]),
            device=device,
            workers=args.workers,
        )
        transports[transition] = estimates
        summary["transition"] = transition
        transport_summaries.append(summary)
    labeled_state, labeled_snapshots = _build_sequential_memory(
        sites,
        transports,
        source="labeled_records",
        class_semantics_sha256=class_semantics_sha,
    )
    combined_state, combined_snapshots = _build_sequential_memory(
        sites,
        transports,
        source="combined_records",
        class_semantics_sha256=class_semantics_sha,
    )
    bundle: dict[str, Any] = {
        "protocol_id": "asprseg_v0_1",
        "stage": "post_hoc_memory_reconstruction",
        "seed": args.seed,
        "hidden_gt_usage": "none",
        "optimizer_steps": 0,
        "sites": sites,
        "transports": transports,
        "labeled_memory_state": labeled_state,
        "combined_memory_state": combined_state,
        "labeled_memory_snapshots": labeled_snapshots,
        "combined_memory_snapshots": combined_snapshots,
        "class_semantics_sha256": class_semantics_sha,
        "environment": _gpu_record(args.physical_gpu, device),
    }
    temporary = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(bundle, temporary)
    os.replace(temporary, bundle_path)
    summary = {
        "protocol_id": "asprseg_v0_1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ASPR_MEMORY_RECONSTRUCTION_COMPLETE",
        "stage": "post_hoc_memory_reconstruction",
        "seed": args.seed,
        "hidden_gt_usage": "none",
        "optimizer_steps": 0,
        "bundle": str(bundle_path),
        "bundle_sha256": sha256_path(bundle_path),
        "class_semantics_sha256": class_semantics_sha,
        "manifest_sha256": sites[SITE_ORDER[0]]["manifest_sha256"],
        "split_sha256": sites[SITE_ORDER[0]]["split_sha256"],
        "sites": site_summaries,
        "transports": transport_summaries,
        "environment": bundle["environment"],
    }
    write_json(summary_path, summary)
    print(json.dumps({"status": summary["status"], "seed": args.seed, "bundle": str(bundle_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
