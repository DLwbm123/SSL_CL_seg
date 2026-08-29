#!/usr/bin/env python3
"""Run/merge the frozen-checkpoint V0.4 gradient-utility audit without optimizer steps."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.diagnostics import _images_and_labels, diagnostic_records
from lcrseg.analysis.v0_4 import RUN_NAMES, SITE_ORDER, load_frozen_method, signed_distance_and_component_size, stable_seed
from lcrseg.common import read_csv, sha256_path, write_csv, write_json
from lcrseg.data import DeterministicBatcher, H5LabeledDataset, H5UnlabeledDataset, LabeledTransform, WeakStrongTransform, collate_labeled, collate_unlabeled
from lcrseg.methods.base import relation_supervision_loss
from lcrseg.methods.components.compatibility import zero_compatibility
from lcrseg.methods.components.learnability import compute_learnability
from lcrseg.methods.components.progressive_admission import admission_assimilation_loss
from lcrseg.methods.components.pseudo_label import build_pseudo_labels
from lcrseg.methods.components.routing import assimilation_loss, relation_consolidation_loss
from lcrseg.methods.lcrseg_v0_1 import _uniform_compatibility


def _jobs(root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        for variant in ("R0", "R1"):
            for site_index in (1, 2):
                site = SITE_ORDER[site_index]
                checkpoint = root / "runs" / RUN_NAMES[(seed, variant)] / f"checkpoint_final_site{site_index}_{site}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                jobs.append({"seed": seed, "variant": variant, "site_index": site_index, "site": site, "checkpoint": checkpoint})
    return jobs


def _gradient(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> np.ndarray:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    values = [
        (torch.zeros_like(parameter) if gradient is None else gradient).detach().reshape(-1).float().cpu()
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]
    return torch.cat(values).numpy()


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(value.astype(np.float64)))


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = _norm(first) * _norm(second)
    return float(np.dot(first.astype(np.float64), second.astype(np.float64)) / denominator) if denominator > 0 else float("nan")


def _diagnostic_labels(root: Path, seed: int, site: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for record in diagnostic_records(root, seed=seed, dataset="fundus", site=site):
        _, label = next(iter(_images_and_labels(record, "fundus")))
        result[record.case_id] = label
    return result


def _batchers(root: Path, seed: int, site: str, rng: int):
    labeled = H5LabeledDataset(
        root, seed=seed, dataset="fundus", sites=(site,), transform=LabeledTransform(flip_probability=0.5)
    )
    unlabeled = H5UnlabeledDataset(
        root,
        seed=seed,
        dataset="fundus",
        sites=(site,),
        transform=WeakStrongTransform(
            flip_probability=0.5,
            strong_noise_std=0.03,
            brightness_delta=0.10,
            contrast_delta=0.10,
            cutout_probability=0.5,
            cutout_fraction=0.20,
        ),
    )
    return (
        DeterministicBatcher(labeled, batch_size=2, seed=rng, namespace=f"v04-gradient:{seed}:{site}:labeled", collate=collate_labeled),
        DeterministicBatcher(unlabeled, batch_size=4, seed=rng, namespace=f"v04-gradient:{seed}:{site}:unlabeled", collate=collate_unlabeled),
    )


def _region_masks(labels: dict[str, np.ndarray], batch: Any, grid_shape: tuple[int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    boundary: list[torch.Tensor] = []
    interior: list[torch.Tensor] = []
    for case_id, geometry in zip(batch.case_id, batch.geometry_record, strict=True):
        label = labels[case_id]
        union = np.zeros_like(label, dtype=bool)
        inside = np.zeros_like(label, dtype=bool)
        for class_id in (1, 2):
            distance, _ = signed_distance_and_component_size(label, class_id)
            union |= np.abs(distance) <= 3.0
            inside |= distance > 3.0
        boundary_tensor = torch.from_numpy(union)
        interior_tensor = torch.from_numpy(inside)
        dimensions: list[int] = []
        if bool(geometry.get("hflip")):
            dimensions.append(-1)
        if bool(geometry.get("vflip")):
            dimensions.append(-2)
        if dimensions:
            boundary_tensor = torch.flip(boundary_tensor, dimensions)
            interior_tensor = torch.flip(interior_tensor, dimensions)
        boundary.append(boundary_tensor)
        interior.append(interior_tensor)
    boundary_grid = torch.nn.functional.interpolate(torch.stack(boundary)[:, None].float(), size=grid_shape, mode="nearest").bool().to(device)
    interior_grid = torch.nn.functional.interpolate(torch.stack(interior)[:, None].float(), size=grid_shape, mode="nearest").bool().to(device)
    return boundary_grid, interior_grid


def _masked_assimilation(
    *, variant: str, strong_logits: torch.Tensor, pseudo: Any, learnability: Any, admission: Any, valid: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if variant == "R1":
        return admission_assimilation_loss(strong_logits, pseudo, replace(admission, mask=admission.mask & mask), valid)
    return assimilation_loss(strong_logits, pseudo, replace(learnability, score=learnability.score * mask.float()), valid)


def _run_worker(root: Path, output_dir: Path, worker_index: int, workers: int, device: torch.device) -> None:
    part_dir = output_dir / "_gradient_parts"
    part_csv = part_dir / f"worker_{worker_index}.csv"
    part_npz = part_dir / f"worker_{worker_index}.npz"
    part_json = part_dir / f"worker_{worker_index}.json"
    if any(path.exists() for path in (part_csv, part_npz, part_json)):
        raise FileExistsError(f"refusing to overwrite gradient worker {worker_index}")
    assigned = [job for index, job in enumerate(_jobs(root)) if index % workers == worker_index]
    rows: list[dict[str, Any]] = []
    matrices: dict[str, list[np.ndarray]] = {}
    hashes: dict[str, str] = {}
    for ordinal, job in enumerate(assigned, start=1):
        checkpoint = Path(job["checkpoint"])
        hashes[str(checkpoint)] = sha256_path(checkpoint)
        labels = _diagnostic_labels(root, int(job["seed"]), str(job["site"]))
        print(f"RUN {ordinal}/{len(assigned)} seed={job['seed']} {job['variant']} {job['site']}", flush=True)
        for rng_index in range(8):
            method, payload = load_frozen_method(checkpoint, device)
            method.train()
            if method.old_model is not None:
                method.old_model.eval()
            parameters = [
                parameter for name, parameter in method.model.named_parameters()
                if name.startswith("projection_head.") or name.startswith("dec1.")
            ]
            if not parameters:
                raise RuntimeError("fixed gradient parameter scope is empty")
            batch_seed = stable_seed("v0.4-gradient-batch", job["seed"], job["site"], rng_index)
            torch.manual_seed(batch_seed)
            labeled_batcher, unlabeled_batcher = _batchers(root, int(job["seed"]), str(job["site"]), batch_seed)
            labeled = labeled_batcher.batch_at(0).to(device)
            unlabeled = unlabeled_batcher.batch_at(0).to(device)
            labeled_output = method.model(labeled.image)
            labeled_relation = method._relation(labeled_output.relation_features, method.current_anchor_bank)
            sup = method._supervised_losses(
                labeled_output,
                labeled,
                relation_anchor_loss=relation_supervision_loss(labeled_relation.logits, labeled.label, labeled.valid_mask),
            )["loss_sup"]
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
                if job["variant"] == "R1":
                    admission = method._compute_admission(
                        pseudo, learnability, unlabeled.strong_valid_mask, site_step=max(0, int(payload["site_step"]) - 1)
                    )
                else:
                    from lcrseg.analysis.v0_4 import _uniform_admission
                    admission = _uniform_admission(pseudo, num_classes=method.model.num_classes)
            strong_output = method.model(unlabeled.strong_image)
            strong_relation = method._relation(strong_output.relation_features, method.current_anchor_bank)
            if job["variant"] == "R1":
                assim = admission_assimilation_loss(strong_output.logits, pseudo, admission, unlabeled.strong_valid_mask)
            else:
                assim = assimilation_loss(strong_output.logits, pseudo, learnability, unlabeled.strong_valid_mask)
            if method.old_model is None or method.old_anchor_bank is None:
                raise RuntimeError("gradient utility requires an incremental checkpoint with historical teacher")
            with torch.no_grad():
                old_output = method.old_model(unlabeled.weak_image)
                old_relation = method._relation(old_output.relation_features, method.old_anchor_bank)
                compatibility = _uniform_compatibility(zero_compatibility(old_relation.probabilities))
            relation = relation_consolidation_loss(
                strong_relation,
                old_relation,
                compatibility,
                unlabeled.strong_valid_mask,
                distill_temperature=float(method.config["distill_temperature"]),
            )
            gradients = {"sup": _gradient(sup, parameters), "assim": _gradient(assim, parameters), "rel": _gradient(relation, parameters)}
            class_norms: dict[str, float] = {}
            for class_id in (1, 2):
                class_mask = pseudo.labels.eq(class_id).unsqueeze(1) & pseudo.valid
                class_loss = _masked_assimilation(
                    variant=str(job["variant"]), strong_logits=strong_output.logits, pseudo=pseudo,
                    learnability=learnability, admission=admission, valid=unlabeled.strong_valid_mask, mask=class_mask,
                )
                class_norms[f"assim_class{class_id}_norm"] = _norm(_gradient(class_loss, parameters))
            boundary, interior = _region_masks(labels, unlabeled, tuple(pseudo.valid.shape[-2:]), device)
            boundary_loss = _masked_assimilation(
                variant=str(job["variant"]), strong_logits=strong_output.logits, pseudo=pseudo,
                learnability=learnability, admission=admission, valid=unlabeled.strong_valid_mask, mask=boundary,
            )
            interior_loss = _masked_assimilation(
                variant=str(job["variant"]), strong_logits=strong_output.logits, pseudo=pseudo,
                learnability=learnability, admission=admission, valid=unlabeled.strong_valid_mask, mask=interior,
            )
            key_prefix = f"seed{job['seed']}_{job['variant']}_site{job['site_index']}"
            for loss_name, gradient in gradients.items():
                matrices.setdefault(f"{key_prefix}_{loss_name}", []).append(gradient)
            old_grad_detected = bool(method.old_model is not None and any(
                parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)) for parameter in method.old_model.parameters()
            ))
            row = {
                "seed": job["seed"],
                "variant": job["variant"],
                "site_index": job["site_index"],
                "site": job["site"],
                "rng": rng_index,
                "parameter_scope": "projection_head+dec1",
                "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
                "g_sup_norm": _norm(gradients["sup"]),
                "g_assim_norm": _norm(gradients["assim"]),
                "g_rel_norm": _norm(gradients["rel"]),
                "cos_sup_assim": _cosine(gradients["sup"], gradients["assim"]),
                "cos_sup_rel": _cosine(gradients["sup"], gradients["rel"]),
                "cos_assim_rel": _cosine(gradients["assim"], gradients["rel"]),
                **class_norms,
                "assim_boundary_norm": _norm(_gradient(boundary_loss, parameters)),
                "assim_interior_norm": _norm(_gradient(interior_loss, parameters)),
                "admitted_gradient_norm": _norm(gradients["assim"]),
                "deferred_actual_gradient_norm": 0.0 if job["variant"] == "R1" else float("nan"),
                "all_gradients_finite": bool(all(np.isfinite(value).all() for value in gradients.values())),
                "old_model_gradient_detected": old_grad_detected,
                "optimizer_step_called": False,
                "hidden_gt_usage": "post_hoc_boundary_grouping_only",
            }
            rows.append(row)
    stacked = {key: np.stack(value).astype(np.float32) for key, value in matrices.items()}
    part_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(part_npz, **stacked)
    write_csv(part_csv, rows)
    write_json(part_json, {"status": "complete", "worker": worker_index, "workers": workers, "rows": len(rows), "checkpoint_sha256": hashes, "gradient_npz_sha256": sha256_path(part_npz)})
    print(json.dumps({"status": "complete", "worker": worker_index, "rows": len(rows)}, sort_keys=True))


def _merge(output_dir: Path, workers: int) -> None:
    csv_path = output_dir / "gradient_utility.csv"
    json_path = output_dir / "gradient_utility_summary.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("refusing to overwrite completed V0.4 gradient utility outputs")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    for worker in range(workers):
        part_dir = output_dir / "_gradient_parts"
        metadata = json.loads((part_dir / f"worker_{worker}.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "complete" or metadata.get("gradient_npz_sha256") != sha256_path(part_dir / f"worker_{worker}.npz"):
            raise RuntimeError(f"invalid gradient worker {worker}")
        rows.extend(read_csv(part_dir / f"worker_{worker}.csv"))
        checkpoint_hashes.update(metadata["checkpoint_sha256"])
        with np.load(part_dir / f"worker_{worker}.npz") as gradients:
            for key in gradients.files:
                matrix = gradients[key].astype(np.float64)
                singular = np.linalg.svd(matrix, compute_uv=False)
                energy = np.square(singular)
                probability = energy / max(1.0e-12, float(energy.sum()))
                effective_rank = float(np.exp(-np.sum(probability[probability > 0] * np.log(probability[probability > 0]))))
                summaries.append(
                    {
                        "group_loss": key,
                        "rng_count": int(matrix.shape[0]),
                        "gradient_variance": float(np.mean(np.sum((matrix - matrix.mean(axis=0)) ** 2, axis=1))),
                        "effective_rank": effective_rank,
                        "top_singular_value_energy": float(probability[0]),
                    }
                )
    if len(rows) != 12 * 8:
        raise RuntimeError(f"expected 96 gradient rows, found {len(rows)}")
    write_csv(csv_path, rows)
    summary = {
        "protocol_id": "lcrseg_v0_4_failure_audit",
        "status": "complete",
        "parameter_scope": "projection_head+final_decoder_block_dec1 fixed before analysis",
        "optimizer_steps": 0,
        "hidden_gt_usage": "post_hoc_boundary_grouping_only",
        "checkpoint_sha256": checkpoint_hashes,
        "gradient_groups": summaries,
        "all_finite": all(row["all_gradients_finite"] == "True" for row in rows),
        "old_model_gradient_detected": any(row["old_model_gradient_detected"] == "True" for row in rows),
    }
    write_json(json_path, summary)
    print(json.dumps({"status": "complete", "rows": len(rows), "gradient_groups": len(summaries)}, sort_keys=True))


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
        raise ValueError("V0.4 gradient utility is partitioned over exactly three workers")
    if args.merge:
        if args.worker_index is not None:
            raise ValueError("--merge and --worker-index are mutually exclusive")
        _merge(args.output_dir.resolve(), args.workers)
    else:
        if args.worker_index is None or not 0 <= args.worker_index < args.workers:
            raise ValueError("worker-index must be in [0, workers)")
        _run_worker(args.root.resolve(), args.output_dir.resolve(), args.worker_index, args.workers, torch.device(args.device))


if __name__ == "__main__":
    main()
