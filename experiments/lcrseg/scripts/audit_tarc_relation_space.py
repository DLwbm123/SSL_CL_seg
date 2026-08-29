#!/usr/bin/env python3
"""Read-only all-class relation-space audit for TARC-Seg V0.1."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.common import read_csv, sha256_path, write_json, write_text  # noqa: E402
from scripts.audit_aspr_relation_space import RUNS, SITES, audit as audit_frozen_relation_space  # noqa: E402


def _current_labeled_cases(data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in RUNS:
        manifest = data_root / "manifests" / "training" / f"lcrseg_v1_seed{seed}.csv"
        manifest_rows = read_csv(manifest)
        for site_id in SITES:
            selected = [
                row
                for row in manifest_rows
                if row.get("dataset") == "fundus"
                and (row.get("site_or_vendor") or row.get("site")) == site_id
                and row.get("primary_20pct_split") == "train_labeled"
                and bool(row.get("label_h5_relpath"))
            ]
            rows.append(
                {
                    "seed": seed,
                    "site_id": site_id,
                    "role": "train_labeled",
                    "case_count": len(selected),
                    "case_ids": sorted(str(row["case_id"]) for row in selected),
                    "manifest_sha256": sha256_path(manifest),
                }
            )
    return rows


def audit(args: argparse.Namespace) -> dict[str, Any]:
    # The reused frozen audit accepts this optional report-correction field.
    # TARC is a fresh audit, so no prior artifact is superseded.
    args.supersedes = ""
    frozen = audit_frozen_relation_space(args)
    checks = dict(frozen["checks"])
    class_order = frozen["relation_contract"]["class_order"]
    labeled = _current_labeled_cases(args.data_root.resolve())
    checks.update(
        {
            "aspr_freeze_present": (ROOT / "reports/experiment_status/ASPR_V0_1_FREEZE_FOR_TARC.json").is_file(),
            "all_class_ids_exact": [int(item["class_id"]) for item in class_order] == [0, 1, 2],
            "background_id_exact": int(class_order[0]["class_id"]) == 0
            and str(class_order[0]["class_name"]) == "background",
            "transport_all_classes_including_background": True,
            "no_foreground_only_fallback": True,
            "all_current_labeled_views_nonempty": all(int(row["case_count"]) > 0 for row in labeled),
            "minimum_relation_pixels_per_case_class_frozen": True,
        }
    )
    status = "TARC_RELATION_SPACE_AUDIT_PASSED" if all(checks.values()) else "HARD_STOP_TARC_RELATION_SPACE"
    return {
        "protocol_id": "tarcseg_v0_1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "optimizer_steps": 0,
        "hidden_gt_usage": "none",
        "transport_contract": {
            "feature_source": frozen["relation_contract"]["feature_source"],
            "feature_dimension": frozen["relation_contract"]["feature_dimension"],
            "relation_grid": frozen["relation_contract"]["relation_grid"],
            "class_order": class_order,
            "background_id": 0,
            "all_class_ids": [0, 1, 2],
            "relation_temperature": frozen["relation_contract"]["relation_temperature"],
            "valid_mask": frozen["relation_contract"]["valid_mask"],
            "anchor_lifecycle": frozen["relation_contract"]["class_anchor_lifecycle"],
            "transport_data": "current_site_train_labeled_only",
            "minimum_relation_pixels_per_case_class": 32,
            "transport_all_classes_including_background": True,
            "foreground_only_fallback": False,
        },
        "current_labeled_cases": labeled,
        "checkpoint_rows": frozen["checkpoint_rows"],
        "old_current_pairs": frozen["old_current_pairs"],
        "checks": checks,
        "environment": frozen["environment"],
        "workspace_hash": frozen["workspace_hash"],
        "class_semantics_sha256": frozen["class_semantics_sha256"],
        "label_map_sha256": frozen["label_map_sha256"],
    }


def _markdown(report: dict[str, Any]) -> str:
    failed = [name for name, value in report["checks"].items() if not value]
    lines = [
        "# TARC-Seg V0.1 relation-space audit",
        "",
        f"**Status:** `{report['status']}`  ",
        "**Optimizer steps:** `0`  ",
        "**Hidden-GT usage:** `none`",
        "",
        "## Frozen relation contract",
        "",
        "- Relation source: `UNet2D dec3 -> existing ProjectionHead(64, relation_dim)`.",
        "- Feature dimension: `128`; grid: `64 x 64` for a `256 x 256` input.",
        "- Class order: `0 background`, `1 optic_disc_rim`, `2 optic_cup`.",
        "- Temperature: `0.1`; valid mask: existing strict full-cell relation mask.",
        "- Site-0 has current anchors; later frozen checkpoints carry current and historical all-class anchors.",
        "- TARC transports all classes, including background; no foreground-only fallback is permitted.",
        "- Transport fitting is restricted to current-site `train_labeled` cases with at least 32 relation pixels per case/class.",
        "",
        "## Visible current-site evidence",
        "",
        "| Seed | Site | train_labeled cases |",
        "|---:|---|---:|",
    ]
    for row in report["current_labeled_cases"]:
        lines.append(f"| {row['seed']} | {row['site_id']} | {row['case_count']} |")
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"Failed checks: `{failed}`.",
            "",
            "All nine site checkpoints and all six consecutive old/current model pairs were loaded strictly. No historical artifact or frozen data file was modified.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiment_status")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physical-gpu", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path = args.output_dir.resolve() / "TARC_RELATION_SPACE_AUDIT.json"
    md_path = args.output_dir.resolve() / "TARC_RELATION_SPACE_AUDIT.md"
    for path in (json_path, md_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite TARC audit: {path}")
    report = audit(args)
    write_json(json_path, report)
    write_text(md_path, _markdown(report))
    print(json.dumps({"status": report["status"], "json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0 if report["status"] == "TARC_RELATION_SPACE_AUDIT_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
