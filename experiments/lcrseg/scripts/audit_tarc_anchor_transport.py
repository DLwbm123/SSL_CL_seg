#!/usr/bin/env python3
"""Audit all-class labeled-evidence anchor transport for one frozen R0 seed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.analysis.tarc_v0_1 import (  # noqa: E402
    TRANSITIONS,
    build_transition_bundle,
    current_frame_oracle,
    labeled_loader,
    tensor_bundle,
)
from lcrseg.common import write_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve() / f"seed{args.seed}"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite TARC seed audit: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    bundles: list[str] = []
    for old_index, current_index in TRANSITIONS:
        bundle, old_model, current_model = build_transition_bundle(
            data_root=args.data_root.resolve(),
            run_root=args.run_root.resolve(),
            seed=args.seed,
            old_site_index=old_index,
            current_site_index=current_index,
            device=device,
            workers=args.workers,
        )
        val_loader = labeled_loader(
            args.data_root.resolve(),
            seed=args.seed,
            site_id=bundle.old_site_id,
            roles=("val",),
            workers=args.workers,
        )
        oracle, oracle_counts = current_frame_oracle(current_model, val_loader, device=device)
        static = F.cosine_similarity(bundle.old_anchors[:, 0], oracle, dim=1)
        global_value = F.cosine_similarity(bundle.global_anchors[:, 0], oracle, dim=1)
        class_value = F.cosine_similarity(bundle.class_anchors[:, 0], oracle, dim=1)
        for class_id in range(3):
            estimate = bundle.transport.class_estimates[class_id]
            rows.append(
                {
                    "seed": args.seed,
                    "transition": f"{bundle.old_site_id}->{bundle.current_site_id}",
                    "old_site_id": bundle.old_site_id,
                    "current_site_id": bundle.current_site_id,
                    "class_id": class_id,
                    "is_background": class_id == 0,
                    "transport_case_count": estimate.case_count,
                    "oracle_case_count": int(oracle_counts[class_id]),
                    "shrinkage": estimate.shrinkage,
                    "variance": estimate.variance,
                    "signal": estimate.signal,
                    "valid_transport": estimate.valid,
                    "global_case_count": bundle.transport.global_case_count,
                    "global_shrinkage": bundle.transport.global_estimate.shrinkage,
                    "static_oracle_cosine": float(static[class_id]),
                    "global_oracle_cosine": float(global_value[class_id]),
                    "class_oracle_cosine": float(class_value[class_id]),
                    "class_minus_static": float(class_value[class_id] - static[class_id]),
                    "class_minus_global": float(class_value[class_id] - global_value[class_id]),
                    "historical_anchor_equal": bundle.historical_anchor_equal,
                    "minimum_relation_pixels": 32,
                    "hidden_gt_usage": "post_hoc_previous_val_only",
                }
            )
        path = output_dir / f"transport_{old_index}_{current_index}.pt"
        if path.exists():
            raise FileExistsError(path)
        torch.save(tensor_bundle(bundle), path)
        bundles.append(str(path))
        del old_model, current_model
        torch.cuda.empty_cache()
    csv_path = output_dir / "anchor_transport_audit.csv"
    summary_path = output_dir / "anchor_transport_summary.json"
    write_csv(csv_path, rows)
    write_json(
        summary_path,
        {
            "protocol_id": "tarcseg_v0_1",
            "seed": args.seed,
            "status": "TARC_ANCHOR_TRANSPORT_SEED_AUDIT_COMPLETE",
            "rows": len(rows),
            "bundles": bundles,
            "optimizer_steps": 0,
            "hidden_gt_usage": "post_hoc_previous_val_only",
        },
    )
    print(json.dumps({"status": "complete", "seed": args.seed, "csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
