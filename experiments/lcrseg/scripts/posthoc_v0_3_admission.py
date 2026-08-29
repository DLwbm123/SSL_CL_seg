#!/usr/bin/env python3
"""Run frozen-checkpoint hidden-GT admission diagnostics for V0.3 R1."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.analysis.diagnostics import _images_and_labels, diagnostic_records
from lcrseg.analysis.v0_1_routing import _interior_grid
from lcrseg.common import sha256_path, write_csv, write_json
from lcrseg.engine.checkpoint import load_checkpoint
from lcrseg.methods.components.learnability import compute_learnability
from lcrseg.methods.components.pseudo_label import build_pseudo_labels
from lcrseg.methods.lcrseg_v0_2a import LCRSegV02AMethod
from lcrseg.methods.lcrseg_v0_3 import LCRSegV03Method
from lcrseg.models import UNet2D


RUN_NAMES = {
    0: "fundus_seed0_lcrseg_v0_2a_r1_progressive_uniform_full200e",
    1: "fundus_seed1_lcrseg_v0_3_r1_progressive_uniform_full200e",
    2: "fundus_seed2_lcrseg_v0_3_r1_progressive_uniform_full200e",
}


def _load(checkpoint: Path, device: torch.device) -> tuple[LCRSegV02AMethod, dict[str, Any]]:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    config = dict(payload["config_resolved"])
    model_config = dict(config["model"])
    model = UNet2D(
        int(model_config["in_channels"]),
        int(model_config["num_classes"]),
        base_channels=int(model_config.get("base_channels", 16)),
        relation_dim=int(model_config.get("relation_dim", 128)),
    ).to(device)
    method_name = payload["method_name"]
    if method_name == "lcrseg_v0_2a":
        method: LCRSegV02AMethod = LCRSegV02AMethod(model, config=dict(config["method"])).to(device)
    elif method_name == "lcrseg_v0_3":
        method = LCRSegV03Method(model, config=dict(config["method"])).to(device)
    else:
        raise ValueError(f"unsupported R1 checkpoint method {method_name}: {checkpoint}")
    method.model.load_state_dict(payload["current_model_state"], strict=True)
    method.load_method_state_dict(payload)
    method.site_id = str(payload["site_id"])
    method.site_index = int(payload["site_index"])
    statistics = payload.get("method_statistics") or {}
    method.total_steps = int(statistics.get("active_site_total_steps") or payload["site_step"])
    method.model.eval()
    if method.old_model is not None:
        method.old_model.eval()
    return method, payload


def _empty_counts() -> dict[str, int]:
    return {
        "candidate_count": 0,
        "candidate_correct": 0,
        "admitted_count": 0,
        "admitted_correct": 0,
        "deferred_count": 0,
        "deferred_correct": 0,
    }


def _add_counts(target: dict[str, int], correct: np.ndarray, candidate: np.ndarray, admitted: np.ndarray) -> None:
    accepted = candidate & admitted
    deferred = candidate & ~admitted
    target["candidate_count"] += int(candidate.sum())
    target["candidate_correct"] += int((candidate & correct).sum())
    target["admitted_count"] += int(accepted.sum())
    target["admitted_correct"] += int((accepted & correct).sum())
    target["deferred_count"] += int(deferred.sum())
    target["deferred_correct"] += int((deferred & correct).sum())


def _row(
    *,
    seed: int,
    site: str,
    site_index: int | str,
    class_id: int,
    region: str,
    target_fraction: float | str,
    counts: dict[str, int],
    checkpoint: str,
    checkpoint_sha256: str,
    gate_scope: str,
) -> dict[str, Any]:
    candidate_count = counts["candidate_count"]
    admitted_count = counts["admitted_count"]
    deferred_count = counts["deferred_count"]
    candidate_accuracy = counts["candidate_correct"] / candidate_count if candidate_count else ""
    admitted_accuracy = counts["admitted_correct"] / admitted_count if admitted_count else ""
    deferred_accuracy = counts["deferred_correct"] / deferred_count if deferred_count else ""
    return {
        "seed": seed,
        "variant": "R1",
        "site": site,
        "site_index": site_index,
        "epoch": 199 if site != "ALL" else "",
        "class_id": class_id,
        "region": region,
        "candidate_count": candidate_count,
        "candidate_accuracy": candidate_accuracy,
        "admitted_count": admitted_count,
        "admitted_accuracy": admitted_accuracy,
        "deferred_count": deferred_count,
        "deferred_accuracy": deferred_accuracy,
        "accuracy_gap_admitted_minus_candidate": (
            admitted_accuracy - candidate_accuracy
            if isinstance(admitted_accuracy, float) and isinstance(candidate_accuracy, float)
            else ""
        ),
        "target_coverage": target_fraction,
        "realized_coverage": admitted_count / candidate_count if candidate_count else "",
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "gate_scope": gate_scope,
        "hidden_gt_scope": "independent_post_hoc_frozen_checkpoint_only",
    }


@torch.no_grad()
def _analyze_checkpoint(
    *, root: Path, seed: int, checkpoint: Path, device: torch.device
) -> tuple[list[dict[str, Any]], dict[int, dict[str, int]]]:
    method, payload = _load(checkpoint, device)
    site = str(payload["site_id"])
    site_index = int(payload["site_index"])
    by_class_region = defaultdict(_empty_counts)
    target_fraction: float | None = None
    for record in diagnostic_records(root, seed=seed, dataset="fundus", site=site):
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
            valid_mask = torch.ones((1, 1, image.shape[-2], image.shape[-1]), dtype=torch.bool, device=device)
            admission = method._compute_admission(
                pseudo,
                learnability,
                valid_mask,
                site_step=max(0, int(payload["site_step"]) - 1),
            )
            if target_fraction is None:
                target_fraction = float(admission.target_fraction)
            elif abs(target_fraction - float(admission.target_fraction)) > 1.0e-12:
                raise AssertionError("post-hoc target fraction changed within one frozen checkpoint")
            grid_shape = tuple(relation.probabilities.shape[-2:])
            grid_label = F.interpolate(
                torch.from_numpy(label).to(device)[None, None].float(), size=grid_shape, mode="nearest"
            )[0, 0].long()
            predicted = pseudo.labels[0].cpu().numpy().reshape(-1)
            valid = pseudo.valid[0, 0].cpu().numpy().astype(bool).reshape(-1)
            correct = pseudo.labels[0].eq(grid_label).cpu().numpy().astype(bool).reshape(-1)
            admitted = admission.mask[0, 0].cpu().numpy().astype(bool).reshape(-1)
            interior = _interior_grid(label, grid_shape).reshape(-1)
            for class_id in (1, 2):
                candidate = valid & (predicted == class_id)
                _add_counts(by_class_region[(class_id, "all")], correct, candidate, admitted)
                _add_counts(by_class_region[(class_id, "boundary")], correct, candidate & ~interior, admitted)
                _add_counts(by_class_region[(class_id, "interior")], correct, candidate & interior, admitted)
    if target_fraction is None:
        raise RuntimeError(f"no diagnostic cases found for {checkpoint}")
    digest = sha256_path(checkpoint)
    rows = [
        _row(
            seed=seed,
            site=site,
            site_index=site_index,
            class_id=class_id,
            region=region,
            target_fraction=target_fraction,
            counts=counts,
            checkpoint=str(checkpoint),
            checkpoint_sha256=digest,
            gate_scope="site_final_checkpoint",
        )
        for (class_id, region), counts in sorted(by_class_region.items())
    ]
    aggregate = {class_id: dict(by_class_region[(class_id, "all")]) for class_id in (1, 2)}
    return rows, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL"))
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/analysis/v0_3")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "fundus_admission_analysis.csv"
    json_path = output_dir / "fundus_admission_analysis.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("refusing to overwrite frozen V0.3 admission analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    seed_totals = defaultdict(_empty_counts)
    for seed, run_name in RUN_NAMES.items():
        run_dir = (args.run_root / run_name).resolve()
        summary = json.loads((run_dir / "run_summary.json").read_text())
        if summary.get("status") != "complete" or int(summary.get("completed_global_steps", -1)) != 13400:
            raise RuntimeError(f"post-hoc requires frozen complete R1 run: {run_dir}")
        if int(summary.get("seed", -1)) != seed or summary.get("variant_id") != "R1":
            raise RuntimeError(f"R1 seed identity mismatch: {run_dir}")
        checkpoints = sorted(run_dir.glob("checkpoint_final_site*_*.pt"))
        if len(checkpoints) != 3:
            raise RuntimeError(f"expected three frozen site checkpoints: {run_dir}")
        for checkpoint in checkpoints:
            checkpoint_rows, totals = _analyze_checkpoint(
                root=args.root.resolve(), seed=seed, checkpoint=checkpoint, device=device
            )
            rows.extend(checkpoint_rows)
            for class_id, counts in totals.items():
                for key, value in counts.items():
                    seed_totals[(seed, class_id)][key] += value
    aggregate_rows = [
        _row(
            seed=seed,
            site="ALL",
            site_index="",
            class_id=class_id,
            region="all",
            target_fraction="",
            counts=counts,
            checkpoint="three_site_final_checkpoints",
            checkpoint_sha256="",
            gate_scope="seed_foreground_class",
        )
        for (seed, class_id), counts in sorted(seed_totals.items())
    ]
    rows.extend(aggregate_rows)
    write_csv(csv_path, rows)
    summary = {
        "status": "complete",
        "hidden_gt_scope": "independent_post_hoc_frozen_checkpoint_only",
        "training_imports_this_module": False,
        "device": str(device),
        "seeds": [0, 1, 2],
        "site_rows": len(rows) - len(aggregate_rows),
        "aggregate_rows": len(aggregate_rows),
        "aggregate": aggregate_rows,
    }
    write_json(json_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
