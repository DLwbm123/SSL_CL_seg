#!/usr/bin/env python3
"""Audit historical relation fidelity and current-site safety for one seed."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.analysis.tarc_v0_1 import TRANSITIONS, labeled_loader, relation_probabilities  # noqa: E402
from lcrseg.analysis.v0_4 import load_frozen_method  # noqa: E402
from lcrseg.common import write_csv, write_json  # noqa: E402


def _margin(probability: torch.Tensor) -> torch.Tensor:
    top = probability.topk(k=2, dim=1).values
    return top[:, 0] - top[:, 1]


def _empty() -> dict[str, float]:
    return defaultdict(float)


@torch.no_grad()
def _previous_fidelity(
    *,
    old_model: torch.nn.Module,
    current_model: torch.nn.Module,
    loader: Any,
    old_anchors: torch.Tensor,
    global_anchors: torch.Tensor,
    class_anchors: torch.Tensor,
    device: torch.device,
) -> list[dict[str, Any]]:
    stats = {class_id: _empty() for class_id in (-1, 0, 1, 2)}
    old_model.eval()
    current_model.eval()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        old_features = old_model(batch.image).relation_features
        current_features = current_model(batch.image).relation_features
        q_old = relation_probabilities(old_features, old_anchors)
        variants = {
            "static": relation_probabilities(current_features, old_anchors),
            "global": relation_probabilities(current_features, global_anchors),
            "class": relation_probabilities(current_features, class_anchors),
        }
        target = F.interpolate(batch.label[:, None].float(), size=q_old.shape[-2:], mode="nearest")[:, 0].long()
        old_pred = q_old.argmax(dim=1)
        old_margin = _margin(q_old)
        for class_id in (-1, 0, 1, 2):
            mask = torch.ones_like(target, dtype=torch.bool) if class_id == -1 else target.eq(class_id)
            count = int(mask.sum())
            if count == 0:
                continue
            item = stats[class_id]
            item["count"] += count
            item["old_accuracy_sum"] += float(old_pred[mask].eq(target[mask]).float().sum())
            for name, probability in variants.items():
                kl = (q_old * (q_old.clamp_min(1.0e-8).log() - probability.clamp_min(1.0e-8).log())).sum(dim=1)
                predicted = probability.argmax(dim=1)
                margin = _margin(probability)
                entropy = -(probability.clamp_min(1.0e-8) * probability.clamp_min(1.0e-8).log()).sum(dim=1)
                item[f"{name}_kl_sum"] += float(kl[mask].sum())
                item[f"{name}_top1_agreement_sum"] += float(predicted[mask].eq(old_pred[mask]).float().sum())
                item[f"{name}_margin_agreement_sum"] += float((1.0 - (margin[mask] - old_margin[mask]).abs()).sum())
                item[f"{name}_accuracy_sum"] += float(predicted[mask].eq(target[mask]).float().sum())
                item[f"{name}_margin_sum"] += float(margin[mask].sum())
                item[f"{name}_entropy_sum"] += float(entropy[mask].sum())
                item["nonfinite"] += int((~torch.isfinite(probability)).sum())
    rows: list[dict[str, Any]] = []
    for class_id, item in stats.items():
        count = int(item["count"])
        if not count:
            continue
        row: dict[str, Any] = {
            "scope": "previous_fidelity",
            "class_id": "ALL" if class_id == -1 else class_id,
            "is_background": class_id == 0,
            "pixel_count": count,
            "old_accuracy": item["old_accuracy_sum"] / count,
            "nonfinite_count": int(item["nonfinite"]),
            "hidden_gt_usage": "post_hoc_visible_previous_val_only",
        }
        for name in ("static", "global", "class"):
            for metric in ("kl", "top1_agreement", "margin_agreement", "accuracy", "margin", "entropy"):
                row[f"{name}_{metric}"] = item[f"{name}_{metric}_sum"] / count
        row["class_kl_reduction_fraction_vs_static"] = (
            row["static_kl"] - row["class_kl"]
        ) / max(1.0e-12, abs(row["static_kl"]))
        row["class_top1_minus_static"] = row["class_top1_agreement"] - row["static_top1_agreement"]
        row["class_margin_agreement_minus_static"] = row["class_margin_agreement"] - row["static_margin_agreement"]
        rows.append(row)
    return rows


@torch.no_grad()
def _current_safety(
    *,
    old_model: torch.nn.Module,
    current_model: torch.nn.Module,
    loader: Any,
    old_anchors: torch.Tensor,
    class_anchors: torch.Tensor,
    device: torch.device,
) -> list[dict[str, Any]]:
    stats = {class_id: _empty() for class_id in (-1, 0, 1, 2)}
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        old_features = old_model(batch.image).relation_features
        current_features = current_model(batch.image).relation_features
        probability = {
            "old": relation_probabilities(old_features, old_anchors),
            "static": relation_probabilities(current_features, old_anchors),
            "class": relation_probabilities(current_features, class_anchors),
        }
        target = F.interpolate(batch.label[:, None].float(), size=next(iter(probability.values())).shape[-2:], mode="nearest")[:, 0].long()
        for class_id in (-1, 0, 1, 2):
            mask = torch.ones_like(target, dtype=torch.bool) if class_id == -1 else target.eq(class_id)
            count = int(mask.sum())
            if not count:
                continue
            item = stats[class_id]
            item["count"] += count
            for name, value in probability.items():
                predicted = value.argmax(dim=1)
                margin = _margin(value)
                entropy = -(value.clamp_min(1.0e-8) * value.clamp_min(1.0e-8).log()).sum(dim=1)
                item[f"{name}_accuracy_sum"] += float(predicted[mask].eq(target[mask]).float().sum())
                item[f"{name}_margin_sum"] += float(margin[mask].sum())
                item[f"{name}_entropy_sum"] += float(entropy[mask].sum())
                item["nonfinite"] += int((~torch.isfinite(value)).sum())
    rows: list[dict[str, Any]] = []
    for class_id, item in stats.items():
        count = int(item["count"])
        if not count:
            continue
        row: dict[str, Any] = {
            "scope": "current_safety",
            "class_id": "ALL" if class_id == -1 else class_id,
            "is_background": class_id == 0,
            "pixel_count": count,
            "nonfinite_count": int(item["nonfinite"]),
            "hidden_gt_usage": "post_hoc_current_train_labeled_and_val_only",
        }
        for name in ("old", "static", "class"):
            for metric in ("accuracy", "margin", "entropy"):
                row[f"{name}_{metric}"] = item[f"{name}_{metric}_sum"] / count
        row["class_accuracy_minus_static"] = row["class_accuracy"] - row["static_accuracy"]
        row["class_margin_minus_static"] = row["class_margin"] - row["static_margin"]
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_dir = args.output_dir.resolve() / f"seed{args.seed}"
    csv_path = seed_dir / "relation_fidelity_audit.csv"
    summary_path = seed_dir / "relation_fidelity_summary.json"
    for path in (csv_path, summary_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite TARC relation audit: {path}")
    rows: list[dict[str, Any]] = []
    device = torch.device(args.device)
    for old_index, current_index in TRANSITIONS:
        bundle = torch.load(seed_dir / f"transport_{old_index}_{current_index}.pt", map_location="cpu")
        old_method, _ = load_frozen_method(Path(bundle["old_checkpoint"]), device)
        current_method, _ = load_frozen_method(Path(bundle["current_checkpoint"]), device)
        anchors = {name: bundle[name].to(device) for name in ("old_anchors", "global_anchors", "class_anchors")}
        previous_loader = labeled_loader(
            args.data_root.resolve(), seed=args.seed, site_id=bundle["old_site_id"], roles=("val",), workers=args.workers
        )
        current_loader = labeled_loader(
            args.data_root.resolve(), seed=args.seed, site_id=bundle["current_site_id"],
            roles=("train_labeled", "val"), workers=args.workers,
        )
        transition = f"{bundle['old_site_id']}->{bundle['current_site_id']}"
        transition_rows = _previous_fidelity(
            old_model=old_method.model, current_model=current_method.model, loader=previous_loader,
            old_anchors=anchors["old_anchors"], global_anchors=anchors["global_anchors"],
            class_anchors=anchors["class_anchors"], device=device,
        )
        transition_rows += _current_safety(
            old_model=old_method.model, current_model=current_method.model, loader=current_loader,
            old_anchors=anchors["old_anchors"], class_anchors=anchors["class_anchors"], device=device,
        )
        for row in transition_rows:
            row.update(seed=args.seed, transition=transition, old_site_id=bundle["old_site_id"], current_site_id=bundle["current_site_id"])
        rows.extend(transition_rows)
        del old_method, current_method
        torch.cuda.empty_cache()
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(csv_path, rows, fieldnames=fieldnames)
    write_json(summary_path, {"protocol_id": "tarcseg_v0_1", "seed": args.seed, "status": "TARC_RELATION_FIDELITY_SEED_AUDIT_COMPLETE", "rows": len(rows), "optimizer_steps": 0})
    print(json.dumps({"status": "complete", "seed": args.seed, "rows": len(rows), "csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
