#!/usr/bin/env python3
"""Export deterministic, sampled V0.4 post-hoc features from one frozen checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.diagnostics import _images_and_labels, diagnostic_records
from lcrseg.analysis.v0_4 import (
    checkpoint_variant,
    diagnostic_snapshot,
    load_frozen_method,
    resize_numpy,
    signed_distance_and_component_size,
    stable_seed,
)
from lcrseg.common import canonical_json, sha256_path, write_json


class CandidateCollector:
    def __init__(self, *, class_id: int, maximum: int, seed: int) -> None:
        self.class_id = int(class_id)
        self.maximum = int(maximum)
        self.rng = np.random.default_rng(seed)
        self.chunks: dict[str, list[np.ndarray]] = {}
        self.priorities: list[np.ndarray] = []

    def add(self, fields: dict[str, np.ndarray]) -> None:
        lengths = {len(value) for value in fields.values()}
        if not lengths or len(lengths) != 1:
            raise ValueError("candidate export fields have inconsistent lengths")
        count = lengths.pop()
        if count == 0:
            return
        self.priorities.append(self.rng.random(count))
        for key, value in fields.items():
            self.chunks.setdefault(key, []).append(np.asarray(value))

    def finalize(self) -> dict[str, np.ndarray]:
        if not self.priorities:
            raise RuntimeError(f"no candidate pixels found for class {self.class_id}")
        priorities = np.concatenate(self.priorities)
        count = min(self.maximum, len(priorities))
        selected = (
            np.arange(len(priorities), dtype=np.int64)
            if count == len(priorities)
            else np.sort(np.argpartition(priorities, count - 1)[:count])
        )
        return {key: np.concatenate(chunks, axis=0)[selected] for key, chunks in self.chunks.items()}


def _flatten(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=("fundus",), default="fundus")
    parser.add_argument("--site", required=True, help="Evaluation site from the diagnostics manifest")
    parser.add_argument("--split", choices=("train_unlabeled",), default="train_unlabeled")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--max-pixels-per-class", type=int, default=200_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 1 <= args.max_pixels_per_class <= 200_000:
        raise ValueError("V0.4 fixes max_pixels_per_class in [1, 200000]")
    output = args.output.resolve()
    if output.suffix != ".npz":
        raise ValueError("diagnostic feature output must be a compressed .npz shard")
    companion = output.with_suffix(".json")
    if output.exists() or companion.exists():
        raise FileExistsError("refusing to overwrite V0.4 diagnostic features")
    frozen = (args.root / "h5" / "v1").resolve()
    if output == frozen or frozen in output.parents:
        raise ValueError("diagnostic output may not be written inside frozen HDF5")
    device = torch.device(args.device)
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha = sha256_path(checkpoint)
    method, payload = load_frozen_method(checkpoint, device)
    variant = checkpoint_variant(payload)
    trained_site = str(payload["site_id"])
    collectors = {
        class_id: CandidateCollector(
            class_id=class_id,
            maximum=args.max_pixels_per_class,
            seed=stable_seed("v0.4-export", checkpoint_sha, args.site, class_id),
        )
        for class_id in (1, 2)
    }
    cases = 0
    images = 0
    for record in diagnostic_records(args.root.resolve(), seed=args.seed, dataset=args.dataset, site=args.site):
        cases += 1
        for image, label in _images_and_labels(record, args.dataset):
            images += 1
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
            snapshot = diagnostic_snapshot(method, payload, image_tensor)
            grid_shape = tuple(snapshot.features.shape[-2:])
            true_class = F.interpolate(
                torch.from_numpy(label)[None, None].float(), size=grid_shape, mode="nearest"
            )[0, 0].long().cpu().numpy().reshape(-1)
            predicted = snapshot.pseudo.labels[0].cpu().numpy().reshape(-1)
            candidate = _flatten(snapshot.pseudo.valid[0, 0]).astype(bool)
            admitted = _flatten(snapshot.admission.mask[0, 0]).astype(bool)
            assimilation_weight = _flatten(snapshot.assimilation_weight[0, 0]).astype(np.float32)
            features = (
                snapshot.features[0].permute(1, 2, 0).cpu().numpy().reshape(-1, snapshot.features.shape[1]).astype(np.float16)
            )
            learnability = _flatten(snapshot.learnability[0, 0]).astype(np.float32)
            anchor_margin = _flatten(snapshot.relation_margin[0, 0]).astype(np.float32)
            relation_entropy = _flatten(snapshot.relation_entropy[0, 0]).astype(np.float32)
            logit_margin = _flatten(snapshot.logit_margin[0, 0]).astype(np.float32)
            spatial_weight = _flatten(snapshot.spatial_weight[0, 0]).astype(np.float32)
            source = _flatten(snapshot.pseudo.source[0]).astype(np.int8)
            for class_id, collector in collectors.items():
                selected = candidate & (predicted == class_id)
                if not selected.any():
                    continue
                signed_distance, component_size = signed_distance_and_component_size(label, class_id)
                distance_grid = resize_numpy(signed_distance, grid_shape, mode="bilinear").reshape(-1)
                component_grid = resize_numpy(component_size, grid_shape, mode="nearest").reshape(-1).astype(np.int64)
                count = int(selected.sum())
                collector.add(
                    {
                        "features": features[selected],
                        "predicted_class": np.full(count, class_id, dtype=np.int8),
                        "true_class": true_class[selected].astype(np.int8),
                        "correct": (true_class[selected] == class_id).astype(np.int8),
                        "candidate": np.ones(count, dtype=np.int8),
                        "admitted": admitted[selected].astype(np.int8),
                        "deferred": (~admitted[selected]).astype(np.int8),
                        "assimilation_weight": assimilation_weight[selected],
                        "patient_id": np.full(count, record.patient_id),
                        "case_id": np.full(count, record.case_id),
                        "boundary_distance": distance_grid[selected].astype(np.float32),
                        "component_size": component_grid[selected],
                        "learnability": learnability[selected],
                        "anchor_margin": anchor_margin[selected],
                        "relation_entropy": relation_entropy[selected],
                        "logit_margin": logit_margin[selected],
                        "spatial_weight": spatial_weight[selected],
                        "pseudo_source": source[selected],
                    }
                )
    finalized = [collectors[class_id].finalize() for class_id in (1, 2)]
    combined = {
        key: np.concatenate([group[key] for group in finalized], axis=0)
        for key in finalized[0]
    }
    feature_norm_error = float(
        np.max(np.abs(np.linalg.norm(combined["features"].astype(np.float32), axis=1) - 1.0))
    )
    if feature_norm_error > 2.0e-3:
        raise AssertionError("exported projection features are not L2 normalized")
    metadata: dict[str, Any] = {
        "protocol_id": "lcrseg_v0_4_failure_audit",
        "status": "complete",
        "hidden_gt_usage": "post_hoc_only",
        "training_imports_this_script": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "method_name": payload["method_name"],
        "variant": variant,
        "seed": args.seed,
        "trained_through_site": trained_site,
        "trained_through_site_index": int(payload["site_index"]),
        "evaluation_site": args.site,
        "split": args.split,
        "data_split_hash": payload["data_split_hash"],
        "manifest_hash": payload["manifest_hash"],
        "max_pixels_per_class": args.max_pixels_per_class,
        "sampled_pixels_by_class": {
            str(class_id): int((combined["predicted_class"] == class_id).sum()) for class_id in (1, 2)
        },
        "cases": cases,
        "images": images,
        "feature_dim": int(combined["features"].shape[1]),
        "feature_norm_max_abs_error": feature_norm_error,
        "requested_device": str(args.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "boundary_definition": "signed processed-pixel distance; boundary abs(distance)<=3",
        "component_definition": "connected component in hidden GT for predicted foreground class",
    }
    anchors = method.current_anchor_bank.anchors[:, 0].detach().cpu().numpy().astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **combined,
        anchors=anchors,
        metadata=np.asarray(canonical_json(metadata)),
    )
    metadata["output"] = str(output)
    metadata["output_sha256"] = sha256_path(output)
    write_json(companion, metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
