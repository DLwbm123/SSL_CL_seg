#!/usr/bin/env python3
"""Read-only TARC pairwise-geometry post-mortem for BPRC preregistration."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.analysis.tarc_v0_1 import TRANSITIONS, labeled_loader, relation_probabilities  # noqa: E402
from lcrseg.analysis.v0_4 import load_frozen_method  # noqa: E402
from lcrseg.common import read_csv, write_csv, write_json, write_text  # noqa: E402
from scripts.audit_tarc_relation_fidelity import _margin, _previous_fidelity  # noqa: E402


def _boundary_mask(label: torch.Tensor) -> torch.Tensor:
    one_hot = F.one_hot(label.long(), num_classes=3).permute(0, 3, 1, 2).float()
    local_max = F.max_pool2d(one_hot, kernel_size=3, stride=1, padding=1)
    local_min = -F.max_pool2d(-one_hot, kernel_size=3, stride=1, padding=1)
    return (local_max - local_min).abs().sum(dim=1).gt(0)


def _gram_rows(seed: int, transition: str, anchors: dict[str, torch.Tensor]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    old = F.normalize(anchors["old"][:, 0].float(), dim=1)
    old_gram = old @ old.T
    distortions: dict[str, float] = {}
    for variant, value in anchors.items():
        unit = F.normalize(value[:, 0].float(), dim=1)
        gram = unit @ unit.T
        difference = gram - old_gram
        off_diagonal = ~torch.eye(3, dtype=torch.bool, device=gram.device)
        distortion = float(torch.sqrt(difference[off_diagonal].square().mean()))
        distortions[variant] = distortion
        for class_a, class_b in ((0, 1), (0, 2), (1, 2)):
            rows.append(
                {
                    "seed": seed,
                    "transition": transition,
                    "variant": variant,
                    "class_a": class_a,
                    "class_b": class_b,
                    "pair_cosine": float(gram[class_a, class_b]),
                    "old_pair_cosine": float(old_gram[class_a, class_b]),
                    "pair_cosine_change_vs_old": float(difference[class_a, class_b]),
                    "gram_offdiag_rms_distortion": distortion,
                    "hidden_gt_usage": "none",
                }
            )
    return rows, distortions


@torch.no_grad()
def _margin_rows(
    *,
    seed: int,
    transition: str,
    old_model: torch.nn.Module,
    current_model: torch.nn.Module,
    loader: Any,
    anchors: dict[str, torch.Tensor],
    device: torch.device,
) -> list[dict[str, Any]]:
    variants = ("static", "global", "class")
    accumulators: dict[tuple[str, int, str], dict[str, float]] = {}
    for variant in variants:
        for class_id in (-1, 0, 1, 2):
            for region in ("all", "boundary", "interior"):
                accumulators[(variant, class_id, region)] = {
                    "count": 0.0,
                    "top2_abs_error": 0.0,
                    "all_pair_abs_error": 0.0,
                    "margin_agreement": 0.0,
                }
    old_model.eval()
    current_model.eval()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        old_features = old_model(batch.image).relation_features
        current_features = current_model(batch.image).relation_features
        q_old = relation_probabilities(old_features, anchors["old"])
        q_variants = {name: relation_probabilities(current_features, anchors[name]) for name in variants}
        label = F.interpolate(batch.label[:, None].float(), size=q_old.shape[-2:], mode="nearest")[:, 0].long()
        boundary = _boundary_mask(label)
        old_top2_margin = _margin(q_old)
        old_pair = torch.stack((q_old[:, 0] - q_old[:, 1], q_old[:, 0] - q_old[:, 2], q_old[:, 1] - q_old[:, 2]), dim=1)
        for variant, probability in q_variants.items():
            top2_error = (_margin(probability) - old_top2_margin).abs()
            current_pair = torch.stack((probability[:, 0] - probability[:, 1], probability[:, 0] - probability[:, 2], probability[:, 1] - probability[:, 2]), dim=1)
            pair_error = (current_pair - old_pair).abs().mean(dim=1)
            for class_id in (-1, 0, 1, 2):
                class_mask = torch.ones_like(label, dtype=torch.bool) if class_id == -1 else label.eq(class_id)
                for region, region_mask in (("all", torch.ones_like(boundary)), ("boundary", boundary), ("interior", ~boundary)):
                    mask = class_mask & region_mask
                    count = int(mask.sum())
                    if not count:
                        continue
                    item = accumulators[(variant, class_id, region)]
                    item["count"] += count
                    item["top2_abs_error"] += float(top2_error[mask].sum())
                    item["all_pair_abs_error"] += float(pair_error[mask].sum())
                    item["margin_agreement"] += float((1.0 - top2_error[mask]).sum())
    rows: list[dict[str, Any]] = []
    for (variant, class_id, region), item in accumulators.items():
        count = int(item["count"])
        if not count:
            continue
        rows.append(
            {
                "seed": seed,
                "transition": transition,
                "variant": variant,
                "class_id": "ALL" if class_id == -1 else class_id,
                "region": region,
                "pixel_count": count,
                "top1_top2_margin_abs_error": item["top2_abs_error"] / count,
                "all_pair_margin_abs_error": item["all_pair_abs_error"] / count,
                "margin_agreement": item["margin_agreement"] / count,
                "hidden_gt_usage": "post_hoc_visible_previous_val_only",
            }
        )
    return rows


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return f"""# BPRC V0.1 TARC pairwise-geometry post-mortem

**Status:** `{report['status']}`  
**Optimizer steps:** `0`  
**Hidden-GT training usage:** `none`

The analysis used the frozen TARC R0 checkpoints, anchor views, validation case lists, relation temperature, and exact TARC margin/fidelity functions. It did not change the preregistered BPRC formula or gate.

## Findings

- Class-transport mean off-diagonal Gram distortion: `{summary['mean_class_gram_distortion']:.6f}`.
- Static mean disc-rim margin agreement: `{summary['mean_static_disc_rim_margin_agreement']:.6f}`.
- Class-transport mean disc-rim margin agreement: `{summary['mean_class_disc_rim_margin_agreement']:.6f}`.
- Class-minus-static disc-rim margin agreement: `{summary['mean_disc_rim_margin_delta_class_vs_static']:.6f}`.
- Correlation between class-anchor Gram distortion and disc-rim margin delta: `{summary['gram_distortion_disc_rim_delta_correlation']:.6f}`.

The post-mortem is descriptive only. No transport repair, feature mapping, threshold change, or BPRC formula adjustment was made.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tarc-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    status_dir = args.status_dir.resolve()
    paths = {
        "gram": output_dir / "tarc_anchor_gram_distortion.csv",
        "margin": output_dir / "tarc_pairwise_margin_failure.csv",
        "summary": output_dir / "tarc_failure_summary.json",
        "report_json": status_dir / "BPRC_TARC_POSTMORTEM.json",
        "report_md": status_dir / "BPRC_TARC_POSTMORTEM.md",
    }
    if output_dir.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite BPRC post-mortem artifacts")
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    gram_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    class_distortions: list[float] = []
    disc_deltas: list[float] = []
    for seed in range(3):
        seed_dir = args.tarc_analysis_dir.resolve() / f"seed{seed}"
        canonical = read_csv(seed_dir / "relation_fidelity_audit.csv")
        for old_index, current_index in TRANSITIONS:
            bundle = torch.load(seed_dir / f"transport_{old_index}_{current_index}.pt", map_location="cpu")
            old_method, _ = load_frozen_method(Path(bundle["old_checkpoint"]), device)
            current_method, _ = load_frozen_method(Path(bundle["current_checkpoint"]), device)
            anchors = {
                "old": bundle["old_anchors"].to(device),
                "static": bundle["old_anchors"].to(device),
                "global": bundle["global_anchors"].to(device),
                "class": bundle["class_anchors"].to(device),
            }
            transition = f"{bundle['old_site_id']}->{bundle['current_site_id']}"
            rows, distortion = _gram_rows(seed, transition, anchors)
            gram_rows.extend(rows)
            loader = labeled_loader(args.data_root.resolve(), seed=seed, site_id=bundle["old_site_id"], roles=("val",), workers=args.workers)
            margin_rows.extend(
                _margin_rows(
                    seed=seed, transition=transition, old_model=old_method.model, current_model=current_method.model,
                    loader=loader, anchors=anchors, device=device,
                )
            )
            # Reuse the exact frozen TARC implementation and verify its class-1
            # margin result against the canonical seed artifact.
            exact_rows = _previous_fidelity(
                old_model=old_method.model,
                current_model=current_method.model,
                loader=labeled_loader(args.data_root.resolve(), seed=seed, site_id=bundle["old_site_id"], roles=("val",), workers=args.workers),
                old_anchors=anchors["old"], global_anchors=anchors["global"], class_anchors=anchors["class"], device=device,
            )
            exact_disc = next(row for row in exact_rows if row["class_id"] == 1)
            canonical_disc = next(
                row for row in canonical
                if row["scope"] == "previous_fidelity" and row["class_id"] == "1" and row["transition"] == transition
            )
            reproduced = math.isclose(
                float(exact_disc["class_margin_agreement_minus_static"]),
                float(canonical_disc["class_margin_agreement_minus_static"]),
                abs_tol=1.0e-9,
                rel_tol=0.0,
            )
            reproduction_rows.append(
                {
                    "seed": seed,
                    "transition": transition,
                    "exact_tarc_disc_margin_delta": float(exact_disc["class_margin_agreement_minus_static"]),
                    "canonical_tarc_disc_margin_delta": float(canonical_disc["class_margin_agreement_minus_static"]),
                    "exact_reproduction": reproduced,
                }
            )
            class_distortions.append(distortion["class"])
            disc_deltas.append(float(exact_disc["class_margin_agreement_minus_static"]))
            del old_method, current_method
            torch.cuda.empty_cache()
    static_disc = [row["margin_agreement"] for row in margin_rows if row["variant"] == "static" and row["class_id"] == 1 and row["region"] == "all"]
    class_disc = [row["margin_agreement"] for row in margin_rows if row["variant"] == "class" and row["class_id"] == 1 and row["region"] == "all"]
    correlation = float(np.corrcoef(class_distortions, disc_deltas)[0, 1])
    summary = {
        "mean_class_gram_distortion": float(np.mean(class_distortions)),
        "mean_static_disc_rim_margin_agreement": float(np.mean(static_disc)),
        "mean_class_disc_rim_margin_agreement": float(np.mean(class_disc)),
        "mean_disc_rim_margin_delta_class_vs_static": float(np.mean(np.asarray(class_disc) - np.asarray(static_disc))),
        "gram_distortion_disc_rim_delta_correlation": correlation,
        "exact_tarc_metric_reproduction": all(row["exact_reproduction"] for row in reproduction_rows),
        "pairs": 6,
    }
    report = {
        "protocol_id": "bprcseg_v0_1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "BPRC_TARC_POSTMORTEM_COMPLETE" if summary["exact_tarc_metric_reproduction"] else "HARD_STOP_BPRC_AUDIT_ENGINEERING",
        "optimizer_steps": 0,
        "hidden_gt_training_usage": "none",
        "summary": summary,
        "exact_metric_reproduction_rows": reproduction_rows,
        "interpretation_boundary": "descriptive_only_no_formula_or_gate_change",
    }
    write_csv(paths["gram"], gram_rows)
    write_csv(paths["margin"], margin_rows)
    write_json(paths["summary"], report)
    write_json(paths["report_json"], report)
    write_text(paths["report_md"], _markdown(report))
    print(json.dumps({"status": report["status"], "summary": summary}, indent=2))
    return 0 if report["status"] == "BPRC_TARC_POSTMORTEM_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
