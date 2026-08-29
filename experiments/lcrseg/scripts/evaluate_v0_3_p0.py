#!/usr/bin/env python3
"""Audit the frozen V0.3 P0 seed-0 run and evaluate its relation gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import sha256_path, write_json, write_text
from lcrseg.engine.checkpoint import load_checkpoint
from lcrseg.methods.lcrseg_v0_3 import (
    FROZEN_FUNDUS_MANIFEST_HASHES,
    FROZEN_FUNDUS_SPLIT_HASHES,
    FROZEN_V02A_R1_SITE0_SHA256,
)


P0_RUN_NAME = "fundus_seed0_lcrseg_v0_3_p0_progressive_norelation_full200e"
R1_RUN_NAME = "fundus_seed0_lcrseg_v0_2a_r1_progressive_uniform_full200e"
METRICS = ("final_average_dice", "bwt", "incoming_dice", "previous_site_dice")
ZERO_FIELDS = (
    "loss_relation",
    "lambda_relation_effective",
    "relation_denominator",
    "relation_loss_numerator",
    "relation_backward_norm_declared",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"run is not complete: {path}")
    return payload


def _finite_value(value: str) -> bool:
    if value == "":
        return True
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return True

    def recurse(item: Any) -> bool:
        if isinstance(item, bool) or item is None or isinstance(item, str):
            return True
        if isinstance(item, (int, float)):
            return math.isfinite(float(item))
        if isinstance(item, list):
            return all(recurse(child) for child in item)
        if isinstance(item, dict):
            return all(recurse(child) for child in item.values())
        return True

    return recurse(parsed)


def _coverage_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int, int], dict[str, float]] = defaultdict(
        lambda: {"valid": 0.0, "admitted": 0.0, "target_weighted": 0.0}
    )
    for row in rows:
        valid = float(row["valid_count"])
        if valid <= 0:
            continue
        key = (row["site"], int(row["site_index"]), int(row["epoch"]), int(row["predicted_class"]))
        grouped[key]["valid"] += valid
        grouped[key]["admitted"] += float(row["admitted_count"])
        grouped[key]["target_weighted"] += float(row["target_fraction"]) * valid
    errors = []
    for key, values in sorted(grouped.items()):
        target = values["target_weighted"] / values["valid"]
        realized = values["admitted"] / values["valid"]
        errors.append(
            {
                "site": key[0],
                "site_index": key[1],
                "epoch": key[2],
                "predicted_class": key[3],
                "target_fraction": target,
                "realized_fraction": realized,
                "absolute_error": abs(realized - target),
                "valid_count": int(values["valid"]),
            }
        )
    maximum = max((row["absolute_error"] for row in errors), default=math.inf)
    return {
        "aggregation": "site_epoch_predicted_class",
        "nonempty_groups": len(errors),
        "maximum_absolute_error": maximum,
        "threshold": 0.05,
        "passed": bool(errors) and maximum <= 0.05 + 1.0e-12,
        "worst_groups": sorted(errors, key=lambda row: row["absolute_error"], reverse=True)[:10],
    }


def _checkpoint_audit(run_dir: Path) -> dict[str, Any]:
    required = [
        "checkpoint_final_site0_REFUGE.pt",
        "checkpoint_site_0_REFUGE.pt",
        "checkpoint_final_site1_RIM_ONE_r3.pt",
        "checkpoint_site_1_RIM_ONE_r3.pt",
        "checkpoint_final_site2_Drishti_GS.pt",
        "checkpoint_site_2_Drishti_GS.pt",
        "checkpoint_final.pt",
    ]
    records = []
    for name in required:
        path = run_dir / name
        record: dict[str, Any] = {"name": name, "exists": path.is_file() and path.stat().st_size > 0}
        if record["exists"]:
            record["sha256"] = sha256_path(path)
            payload = load_checkpoint(path, map_location="cpu")
            record["readable"] = isinstance(payload, dict)
        else:
            record["readable"] = False
        records.append(record)
    return {
        "records": records,
        "passed": all(record["exists"] and record["readable"] for record in records),
        "site0_parent_hash_exact": all(
            record.get("sha256") == FROZEN_V02A_R1_SITE0_SHA256
            for record in records
            if record["name"] in {"checkpoint_final_site0_REFUGE.pt", "checkpoint_site_0_REFUGE.pt"}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument(
        "--bridge-json",
        type=Path,
        default=PROJECT_ROOT / "reports/experiment_status/V0_3_P0_SITE1_BRIDGE.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports/experiment_status/V0_3_P0_SEED0_COMPLETION.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "reports/experiment_status/V0_3_P0_SEED0_COMPLETION.md",
    )
    args = parser.parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("refusing to overwrite an existing P0 completion report")

    p0_dir = (args.run_root / P0_RUN_NAME).resolve()
    r1_dir = (args.run_root / R1_RUN_NAME).resolve()
    p0 = _summary(p0_dir / "run_summary.json")
    r1 = _summary(r1_dir / "run_summary.json")
    bridge = json.loads(args.bridge_json.read_text())
    train_rows = _read_csv(p0_dir / "train_log.csv")
    coverage_path = p0_dir / "admission_coverage.csv"
    coverage_rows = _read_csv(coverage_path if coverage_path.is_file() else p0_dir / "branch_coverage.csv")

    finite = all(_finite_value(value) for row in train_rows for value in row.values())
    zero_maxima = {
        field: max((abs(float(row.get(field, 0) or 0)) for row in train_rows), default=math.inf)
        for field in ZERO_FIELDS
    }
    safety = {
        "finite_train_log": finite,
        "train_log_rows": len(train_rows),
        "train_log_rows_expected": 5400,
        "optimizer_step_skipped_sum": sum(int(float(row.get("optimizer_step_skipped", 0) or 0)) for row in train_rows),
        "hidden_gt_training_usage_max": max(
            (int(float(row.get("hidden_gt_training_usage", 0) or 0)) for row in train_rows), default=-1
        ),
        "old_model_gradient_detected_max": max(
            (int(float(row.get("old_model_gradient_detected", 0) or 0)) for row in train_rows), default=-1
        ),
        "historical_anchor_changed_max": max(
            (int(float(row.get("historical_anchor_changed", 0) or 0)) for row in train_rows), default=-1
        ),
        "relation_zero_field_maxima": zero_maxima,
        "failure_bundle_absent": not (p0_dir / "failure_bundle.json").exists()
        and not (p0_dir / "failure_bundle").exists(),
    }
    lineage = {
        "protocol_id": p0.get("protocol_id"),
        "variant_id": p0.get("variant_id"),
        "seed": p0.get("seed"),
        "completed_global_steps": p0.get("completed_global_steps"),
        "completed_parent_steps": p0.get("completed_parent_steps"),
        "new_optimizer_steps": p0.get("new_optimizer_steps"),
        "equivalent_full_run_steps": p0.get("equivalent_full_run_steps"),
        "manifest_hash": p0.get("manifest_hash"),
        "split_hash": p0.get("split_hash"),
    }
    lineage_passed = lineage == {
        "protocol_id": "lcrseg_v0_3",
        "variant_id": "P0",
        "seed": 0,
        "completed_global_steps": 13400,
        "completed_parent_steps": 8000,
        "new_optimizer_steps": 5400,
        "equivalent_full_run_steps": 13400,
        "manifest_hash": FROZEN_FUNDUS_MANIFEST_HASHES[0],
        "split_hash": FROZEN_FUNDUS_SPLIT_HASHES[0],
    }
    checkpoints = _checkpoint_audit(p0_dir)
    coverage = _coverage_audit(coverage_rows)
    bridge_passed = (
        bridge.get("status") == "PASSED"
        and bridge.get("parent_checkpoint_sha256") == FROZEN_V02A_R1_SITE0_SHA256
    )
    safety_passed = (
        safety["finite_train_log"]
        and safety["train_log_rows"] == safety["train_log_rows_expected"]
        and safety["optimizer_step_skipped_sum"] == 0
        and safety["hidden_gt_training_usage_max"] == 0
        and safety["old_model_gradient_detected_max"] == 0
        and safety["historical_anchor_changed_max"] == 0
        and safety["failure_bundle_absent"]
        and all(value == 0.0 for value in zero_maxima.values())
    )
    engineering_passed = (
        bridge_passed
        and lineage_passed
        and safety_passed
        and checkpoints["passed"]
        and checkpoints["site0_parent_hash_exact"]
        and coverage["passed"]
    )

    p0_metrics = {metric: float(p0["summary"][metric]) for metric in METRICS}
    r1_metrics = {metric: float(r1["summary"][metric]) for metric in METRICS}
    deltas = {metric: r1_metrics[metric] - p0_metrics[metric] for metric in METRICS}
    thresholds = {
        "final": r1_metrics["final_average_dice"] >= p0_metrics["final_average_dice"] + 0.003,
        "bwt": r1_metrics["bwt"] >= p0_metrics["bwt"] + 0.005,
        "previous": r1_metrics["previous_site_dice"] >= p0_metrics["previous_site_dice"] + 0.005,
        "incoming": r1_metrics["incoming_dice"] >= p0_metrics["incoming_dice"] - 0.010,
    }
    relation_gate_passed = all(thresholds.values())
    status = (
        "P0_SEED0_ENGINEERING_FAILED"
        if not engineering_passed
        else "P0_SEED0_COMPLETE_RELATION_GATE_PASSED"
        if relation_gate_passed
        else "P0_SEED0_COMPLETE_RELATION_GATE_FAILED"
    )
    result = {
        "protocol_id": "lcrseg_v0_3",
        "status": status,
        "engineering_gate_passed": engineering_passed,
        "relation_component_gate_passed": relation_gate_passed,
        "bridge_passed": bridge_passed,
        "lineage": lineage,
        "lineage_passed": lineage_passed,
        "safety_audit": safety,
        "checkpoint_audit": checkpoints,
        "coverage_audit": coverage,
        "p0_metrics": p0_metrics,
        "r1_metrics": r1_metrics,
        "r1_minus_p0": deltas,
        "relation_thresholds": thresholds,
        "run_directories": {"P0": str(p0_dir), "R1": str(r1_dir)},
        "policy": {
            "continue_r0_r1_multiseed_regardless_of_relation_gate": True,
            "p0_seeds_1_2_allowed_by_relation_gate": bool(engineering_passed and relation_gate_passed),
        },
    }
    write_json(args.output_json, result)
    write_text(
        args.output_md,
        "\n".join(
            [
                "# LCR-Seg V0.3 P0 seed-0 completion",
                "",
                f"**Status:** `{status}`",
                "",
                f"- Engineering gate: `{engineering_passed}`",
                f"- Relation component gate: `{relation_gate_passed}`",
                f"- New optimizer steps: `{lineage['new_optimizer_steps']}` (full-equivalent `{lineage['equivalent_full_run_steps']}`)",
                f"- Maximum aggregated coverage error: `{coverage['maximum_absolute_error']}`",
                f"- R1 - P0: `{json.dumps(deltas, sort_keys=True)}`",
                "",
                "R0/R1 seeds 1 and 2 continue regardless of the relation gate. P0 seeds 1 and 2 remain conditional.",
                "",
            ]
        ),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
