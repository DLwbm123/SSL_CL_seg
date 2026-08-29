#!/usr/bin/env python3
"""Compile the preregistered TARC Part-A feasibility gate."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.common import read_csv, sha256_path, write_csv, write_json, write_text  # noqa: E402


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite {key}: {row}")
    return value


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _merge(output_dir: Path, name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed in range(3):
        path = output_dir / f"seed{seed}" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(read_csv(path))
    return rows


def _gate_a(rows: list[dict[str, str]]) -> dict[str, Any]:
    delta = np.asarray([_float(row, "class_minus_static") for row in rows])
    class_seed_positive: dict[str, int] = {}
    class_medians: dict[str, float] = {}
    for class_id in range(3):
        selected = [row for row in rows if int(row["class_id"]) == class_id]
        class_medians[str(class_id)] = float(np.median([_float(row, "class_minus_static") for row in selected]))
        class_seed_positive[str(class_id)] = sum(
            float(np.median([_float(row, "class_minus_static") for row in selected if int(row["seed"]) == seed])) > 0
            for seed in range(3)
        )
    background = [row for row in rows if int(row["class_id"]) == 0]
    metrics = {
        "median_class_minus_static": float(np.median(delta)),
        "fraction_class_gt_static": float(np.mean(delta > 0)),
        "p10_class_minus_static": float(np.quantile(delta, 0.1)),
        "class_median_improvement": class_medians,
        "class_positive_seed_count": class_seed_positive,
        "background_median_improvement": float(np.median([_float(row, "class_minus_static") for row in background])),
        "background_positive_seed_count": class_seed_positive["0"],
        "pairs": len(rows),
    }
    checks = {
        "median_ge_0_020": metrics["median_class_minus_static"] >= 0.020,
        "fraction_gt_static_ge_0_70": metrics["fraction_class_gt_static"] >= 0.70,
        "p10_ge_minus_0_020": metrics["p10_class_minus_static"] >= -0.020,
        "every_class_positive_in_at_least_2_of_3_seeds": all(value >= 2 for value in class_seed_positive.values()),
        "all_transports_valid_with_at_least_two_cases": all(
            _bool(row["valid_transport"]) and int(row["transport_case_count"]) >= 2 for row in rows
        ),
        "historical_anchors_immutable": all(_bool(row["historical_anchor_equal"]) for row in rows),
    }
    background_checks = {
        "background_median_ge_0_010": metrics["background_median_improvement"] >= 0.010,
        "background_positive_in_at_least_2_of_3_seeds": metrics["background_positive_seed_count"] >= 2,
    }
    return {
        "passed": all(checks.values()) and all(background_checks.values()),
        "anchor_checks": checks,
        "background_checks": background_checks,
        "metrics": metrics,
    }


def _gate_b(rows: list[dict[str, str]]) -> dict[str, Any]:
    aggregate = [row for row in rows if row["scope"] == "previous_fidelity" and row["class_id"] == "ALL"]
    by_class = [row for row in rows if row["scope"] == "previous_fidelity" and row["class_id"] != "ALL"]
    class_margin_median = {
        str(class_id): float(np.median([_float(row, "class_margin_agreement_minus_static") for row in by_class if int(row["class_id"]) == class_id]))
        for class_id in range(3)
    }
    reductions = [_float(row, "class_kl_reduction_fraction_vs_static") for row in aggregate]
    metrics = {
        "median_kl_reduction_fraction_vs_static": float(np.median(reductions)),
        "fraction_class_kl_lower_than_static": float(np.mean([_float(row, "class_kl") < _float(row, "static_kl") for row in aggregate])),
        "median_top1_improvement_vs_static": float(np.median([_float(row, "class_top1_minus_static") for row in aggregate])),
        "class_margin_agreement_median_delta_vs_static": class_margin_median,
        "median_class_kl": float(np.median([_float(row, "class_kl") for row in aggregate])),
        "median_global_kl": float(np.median([_float(row, "global_kl") for row in aggregate])),
        "median_class_top1_agreement": float(np.median([_float(row, "class_top1_agreement") for row in aggregate])),
        "median_global_top1_agreement": float(np.median([_float(row, "global_top1_agreement") for row in aggregate])),
        "pairs": len(aggregate),
    }
    checks = {
        "median_kl_reduction_ge_0_10": metrics["median_kl_reduction_fraction_vs_static"] >= 0.10,
        "fraction_kl_lower_ge_0_70": metrics["fraction_class_kl_lower_than_static"] >= 0.70,
        "median_top1_improvement_ge_0_010": metrics["median_top1_improvement_vs_static"] >= 0.010,
        "no_class_median_margin_agreement_drop_gt_0_010": min(class_margin_median.values()) >= -0.010,
        "median_class_kl_le_global": metrics["median_class_kl"] <= metrics["median_global_kl"],
        "median_class_top1_ge_global_minus_0_005": metrics["median_class_top1_agreement"] >= metrics["median_global_top1_agreement"] - 0.005,
        "nonfinite_count_zero": sum(int(row["nonfinite_count"]) for row in aggregate + by_class) == 0,
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}


def _gate_c(rows: list[dict[str, str]]) -> dict[str, Any]:
    aggregate = [row for row in rows if row["scope"] == "current_safety" and row["class_id"] == "ALL"]
    accuracy_delta = [_float(row, "class_accuracy_minus_static") for row in aggregate]
    margin_delta = [_float(row, "class_margin_minus_static") for row in aggregate]
    metrics = {
        "minimum_accuracy_delta_vs_static": min(accuracy_delta),
        "minimum_margin_delta_vs_static": min(margin_delta),
        "median_accuracy_delta_vs_static": float(np.median(accuracy_delta)),
        "median_margin_delta_vs_static": float(np.median(margin_delta)),
        "nonfinite_count": sum(int(row["nonfinite_count"]) for row in rows if row["scope"] == "current_safety"),
        "pairs": len(aggregate),
    }
    checks = {
        "every_pair_accuracy_ge_static_minus_0_005": metrics["minimum_accuracy_delta_vs_static"] >= -0.005,
        "every_pair_margin_ge_static_minus_0_010": metrics["minimum_margin_delta_vs_static"] >= -0.010,
        "nonfinite_count_zero": metrics["nonfinite_count"] == 0,
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}


def _gate_d(rows: list[dict[str, str]]) -> dict[str, Any]:
    key = lambda row: (int(row["seed"]), row["transition"], int(row["update_batch"]))
    t0 = {key(row): _float(row, "previous_val_loss_delta") for row in rows if row["variant"] == "T0"}
    t3 = [row for row in rows if row["variant"] == "T3"]
    previous = {
        variant: [_float(row, "previous_val_loss_delta") for row in rows if row["variant"] == variant]
        for variant in ("T0", "T2", "T3")
    }
    current = {
        variant: [_float(row, "current_val_loss_delta") for row in rows if row["variant"] == variant]
        for variant in ("T0", "T3")
    }
    metrics = {
        "fraction_t3_previous_delta_lower_than_t0": float(np.mean([_float(row, "previous_val_loss_delta") < t0[key(row)] for row in t3])),
        "median_previous_delta_t0": float(np.median(previous["T0"])),
        "median_previous_delta_t2": float(np.median(previous["T2"])),
        "median_previous_delta_t3": float(np.median(previous["T3"])),
        "median_current_delta_t0": float(np.median(current["T0"])),
        "median_current_delta_t3": float(np.median(current["T3"])),
        "update_comparisons": len(t3),
    }
    checks = {
        "t3_reduces_previous_loss_vs_t0_in_at_least_60pct": metrics["fraction_t3_previous_delta_lower_than_t0"] >= 0.60,
        "median_previous_t3_le_t0_minus_1e_4": metrics["median_previous_delta_t3"] <= metrics["median_previous_delta_t0"] - 1.0e-4,
        "median_current_t3_le_t0_plus_2pct_abs_t0": metrics["median_current_delta_t3"] <= metrics["median_current_delta_t0"] + 0.02 * abs(metrics["median_current_delta_t0"]),
        "median_previous_t3_le_t2": metrics["median_previous_delta_t3"] <= metrics["median_previous_delta_t2"],
        "old_model_gradient_zero": all(int(row["old_model_gradient_nonnull"]) == 0 for row in rows),
        "functional_state_never_mutated_checkpoint": all(not _bool(row["checkpoint_or_model_mutated"]) for row in rows),
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}


def _markdown(report: dict[str, Any]) -> str:
    gate = report["gates"]
    a = gate["A_anchor_transport"]["metrics"]
    b = gate["B_relation_fidelity"]["metrics"]
    c = gate["C_current_site_safety"]["metrics"]
    d = gate["D_virtual_step"]["metrics"]
    return f"""# TARC-Seg V0.1 feasibility audit

**Final status:** `{report['status']}`  
**Part B authorized:** `{str(report['part_b_authorized']).lower()}`  
**Optimizer steps:** `0`

## Engineering boundary

- ASPR freeze and TARC relation-space audit passed before this audit.
- Transport used only current-site `train_labeled` cases and all classes `0,1,2`, including background.
- Previous/current validation labels were used only for frozen post-hoc evaluation.
- Functional virtual updates used stateless parameter views; model SHA-256 values were unchanged.
- No hidden GT, unlabeled prototype memory, site modes, optimizer update, checkpoint mutation, or TARC training method was used.

## Gate A — all-class anchor transport: {'PASS' if gate['A_anchor_transport']['passed'] else 'FAIL'}

- median class-minus-static cosine: `{a['median_class_minus_static']:.6f}` (required >= 0.020)
- class better than static: `{a['fraction_class_gt_static']:.2%}` (required >= 70%)
- p10 improvement: `{a['p10_class_minus_static']:.6f}` (required >= -0.020)
- background median improvement: `{a['background_median_improvement']:.6f}` (required >= 0.010)
- positive-seed counts by class: `{a['class_positive_seed_count']}`

## Gate B — historical relation fidelity: {'PASS' if gate['B_relation_fidelity']['passed'] else 'FAIL'}

- median KL reduction: `{b['median_kl_reduction_fraction_vs_static']:.2%}` (required >= 10%)
- class KL lower than static: `{b['fraction_class_kl_lower_than_static']:.2%}` (required >= 70%)
- median top-1 agreement improvement: `{b['median_top1_improvement_vs_static']:.6f}` (required >= 0.010)
- classwise median margin-agreement deltas: `{b['class_margin_agreement_median_delta_vs_static']}`
- failure: class 1 median margin-agreement delta is below `-0.010`.

## Gate C — current-site safety: {'PASS' if gate['C_current_site_safety']['passed'] else 'FAIL'}

- minimum accuracy delta vs static: `{c['minimum_accuracy_delta_vs_static']:.6f}` (required >= -0.005)
- minimum margin delta vs static: `{c['minimum_margin_delta_vs_static']:.6f}` (required >= -0.010)
- non-finite count: `{c['nonfinite_count']}`

## Gate D — functional virtual step: {'PASS' if gate['D_virtual_step']['passed'] else 'FAIL'}

- T3 better than T0 on previous-val delta: `{d['fraction_t3_previous_delta_lower_than_t0']:.2%}` (required >= 60%)
- median previous delta T0/T2/T3: `{d['median_previous_delta_t0']:.8f}` / `{d['median_previous_delta_t2']:.8f}` / `{d['median_previous_delta_t3']:.8f}`
- median current delta T0/T3: `{d['median_current_delta_t0']:.8f}` / `{d['median_current_delta_t3']:.8f}`

## Protocol decision

Gate B is the first failed research gate, so the exact preregistered state is `{report['status']}`. Gate C and Gate D also fail independently. Part B is prohibited: no TARC loss/method/config, equivalence bridge, pilot, or full run may be implemented or launched under V0.1.

## Canonical artifacts

- `reports/analysis/tarcseg_v0_1/anchor_transport_audit.csv`
- `reports/analysis/tarcseg_v0_1/relation_fidelity_audit.csv`
- `reports/analysis/tarcseg_v0_1/virtual_step_audit.csv`
- `reports/experiment_status/TARC_FEASIBILITY_AUDIT.json`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--status-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    status_dir = args.status_dir.resolve()
    canonical = {
        "anchor": analysis_dir / "anchor_transport_audit.csv",
        "relation": analysis_dir / "relation_fidelity_audit.csv",
        "virtual": analysis_dir / "virtual_step_audit.csv",
    }
    report_json = status_dir / "TARC_FEASIBILITY_AUDIT.json"
    report_md = status_dir / "TARC_FEASIBILITY_AUDIT.md"
    for path in (*canonical.values(), report_json, report_md):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite TARC feasibility artifact: {path}")
    anchor_rows = _merge(analysis_dir, "anchor_transport_audit.csv")
    relation_rows = _merge(analysis_dir, "relation_fidelity_audit.csv")
    virtual_rows = _merge(analysis_dir, "virtual_step_audit.csv")
    engineering_checks = {
        "relation_space_passed": json.loads((status_dir / "TARC_RELATION_SPACE_AUDIT.json").read_text())["status"] == "TARC_RELATION_SPACE_AUDIT_PASSED",
        "aspr_freeze_present": (status_dir / "ASPR_V0_1_FREEZE_FOR_TARC.json").is_file(),
        "anchor_rows_exact_18": len(anchor_rows) == 18,
        "relation_rows_exact_48": len(relation_rows) == 48,
        "virtual_rows_exact_768": len(virtual_rows) == 768,
        "virtual_variant_rows_exact": all(sum(row["variant"] == variant for row in virtual_rows) == 192 for variant in ("T0", "T1", "T2", "T3")),
        "all_class_transport_only": {int(row["class_id"]) for row in anchor_rows} == {0, 1, 2},
        "optimizer_steps_zero": all(json.loads((analysis_dir / f"seed{seed}" / "virtual_step_summary.json").read_text())["optimizer_steps"] == 0 for seed in range(3)),
        "all_virtual_models_unchanged": all(
            transition["model_unchanged"] and transition["model_sha256_before"] == transition["model_sha256_after"]
            for seed in range(3)
            for transition in json.loads((analysis_dir / f"seed{seed}" / "virtual_step_summary.json").read_text())["transitions"]
        ),
    }
    if not all(engineering_checks.values()):
        status = "HARD_STOP_TARC_AUDIT_ENGINEERING"
        gates = {}
    else:
        gates = {
            "A_anchor_transport": _gate_a(anchor_rows),
            "B_relation_fidelity": _gate_b(relation_rows),
            "C_current_site_safety": _gate_c(relation_rows),
            "D_virtual_step": _gate_d(virtual_rows),
        }
        if not gates["A_anchor_transport"]["anchor_checks"] or not all(gates["A_anchor_transport"]["anchor_checks"].values()):
            status = "TARC_ANCHOR_TRANSPORT_NOT_SUPPORTED"
        elif not all(gates["A_anchor_transport"]["background_checks"].values()):
            status = "TARC_BACKGROUND_TRANSPORT_NOT_SUPPORTED"
        elif not gates["B_relation_fidelity"]["passed"]:
            status = "TARC_RELATION_FIDELITY_NOT_SUPPORTED"
        elif not gates["C_current_site_safety"]["passed"]:
            status = "TARC_CURRENT_SITE_SAFETY_NOT_SUPPORTED"
        elif not gates["D_virtual_step"]["passed"]:
            status = "TARC_VIRTUAL_STEP_NOT_SUPPORTED"
        else:
            status = "TARC_FEASIBILITY_SUPPORTED"
    write_csv(canonical["anchor"], anchor_rows, fieldnames=list(anchor_rows[0]))
    write_csv(canonical["relation"], relation_rows, fieldnames=list(relation_rows[0]))
    write_csv(canonical["virtual"], virtual_rows, fieldnames=list(virtual_rows[0]))
    report = {
        "protocol_id": "tarcseg_v0_1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "part_b_authorized": status == "TARC_FEASIBILITY_SUPPORTED",
        "optimizer_steps": 0,
        "hidden_gt_training_usage": "none",
        "engineering_checks": engineering_checks,
        "gates": gates,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_path(path)} for name, path in canonical.items()
        },
        "protocol_consequence": (
            "Part B authorized." if status == "TARC_FEASIBILITY_SUPPORTED" else
            "Protocol hard stop before TARC method implementation or training."
        ),
    }
    write_json(report_json, report)
    write_text(report_md, _markdown(report) if gates else f"# TARC feasibility audit\n\n**Status:** `{status}`\n")
    print(json.dumps({"status": status, "part_b_authorized": report["part_b_authorized"], "json": str(report_json), "markdown": str(report_md)}, indent=2))
    return 0 if status == "TARC_FEASIBILITY_SUPPORTED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
