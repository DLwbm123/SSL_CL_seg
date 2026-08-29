#!/usr/bin/env python3
"""Compile the frozen BPRC-X1 exploratory diagnostic gates."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.common import (  # noqa: E402
    canonical_json,
    read_csv,
    sha256_bytes,
    sha256_path,
    write_csv,
    write_json,
    write_text,
)


PROTOCOL_ID = "bprcseg_x1_exploratory"
VARIANTS = ("X0", "X1")
FILENAMES = ("gradient_scale.csv", "virtual_steps.csv", "margin_analysis.csv")
TRANSITIONS = ("REFUGE->RIM_ONE_r3", "RIM_ONE_r3->Drishti_GS")


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite {key}: {row}")
    return value


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _median(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("empty or non-finite gate input")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _quantile(values: list[float], probability: float) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("empty or non-finite gate input")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _key(row: dict[str, str]) -> tuple[int, str, int]:
    return int(row["seed"]), row["transition"], int(row["update_batch"])


def _variant_map(
    rows: list[dict[str, str]], field: str
) -> dict[str, dict[tuple[int, str, int], float]]:
    result = {variant: {} for variant in VARIANTS}
    for row in rows:
        variant = row["variant"]
        if variant not in result:
            raise ValueError(f"unexpected variant: {variant}")
        row_key = _key(row)
        if row_key in result[variant]:
            raise ValueError(f"duplicate {variant} row: {row_key}")
        result[variant][row_key] = _float(row, field)
    if set(result["X0"]) != set(result["X1"]):
        raise ValueError(f"unpaired X0/X1 rows for {field}")
    return result


def _merge(input_root: Path) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    combined = {filename: [] for filename in FILENAMES}
    summaries: list[dict[str, Any]] = []
    for seed in range(3):
        seed_dir = input_root / f"seed{seed}"
        summary_path = seed_dir / "summary.json"
        manifest_path = seed_dir / "fixed_batch_lists.json"
        if not summary_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(summary_path if not summary_path.is_file() else manifest_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary["summary_path"] = str(summary_path)
        summary["manifest_path"] = str(manifest_path)
        summary["manifest_recomputed_sha256"] = sha256_bytes(
            canonical_json(manifest["entries"]).encode("utf-8")
        )
        summary["manifest_recorded_sha256"] = manifest["combined_sha256"]
        summaries.append(summary)
        for filename in FILENAMES:
            path = seed_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            combined[filename].extend(read_csv(path))
    return combined, summaries


def _engineering(
    protocol_path: Path,
    rows: dict[str, list[dict[str, str]]],
    summaries: list[dict[str, Any]],
) -> dict[str, bool]:
    gradient = rows["gradient_scale.csv"]
    virtual = rows["virtual_steps.csv"]
    margin = rows["margin_analysis.csv"]
    expected_keys = {
        (seed, transition, update)
        for seed in range(3)
        for transition in TRANSITIONS
        for update in range(32)
    }
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
    numeric_gradient = (
        "fixed_relation_scale",
        "relation_loss",
        "relation_gradient_norm",
        "relation_gradient_norm_ratio_to_x0",
        "cos_relation_with_sup",
        "cos_relation_with_assim",
        "sup_gradient_norm",
        "assim_gradient_norm",
        "total_gradient_norm",
        "b0_r0_loss_abs_error",
    )
    numeric_virtual = (
        "virtual_step_norm",
        "previous_val_loss_before",
        "previous_val_loss_after",
        "previous_val_loss_delta",
        "previous_val_dice_before",
        "previous_val_dice_after",
        "previous_val_dice_delta",
        "current_val_loss_before",
        "current_val_loss_after",
        "current_val_loss_delta",
        "current_val_dice_before",
        "current_val_dice_after",
        "current_val_dice_delta",
    )
    numeric_margin = (
        "margin_agreement_before",
        "margin_agreement_after",
        "margin_agreement_delta",
        "margin_abs_error_after",
    )
    return {
        "protocol_frozen_before_execution": protocol.get("protocol_id") == PROTOCOL_ID
        and protocol.get("x1_scale") == 1.0 / 3.0
        and protocol.get("pilot_authorization_condition") == "all_diagnostic_gates_pass",
        "three_seed_summaries_complete": len(summaries) == 3
        and [int(summary.get("seed", -1)) for summary in summaries] == [0, 1, 2]
        and all(summary.get("status") == "BPRC_X1_SEED_DIAGNOSTIC_COMPLETE" for summary in summaries),
        "protocol_id_exact": all(row["protocol_id"] == PROTOCOL_ID for row in gradient + virtual + margin),
        "row_counts_exact": len(gradient) == 384 and len(virtual) == 384 and len(margin) == 3456,
        "variant_counts_exact": all(
            sum(row["variant"] == variant for row in gradient) == 192
            and sum(row["variant"] == variant for row in virtual) == 192
            and sum(row["variant"] == variant for row in margin) == 1728
            for variant in VARIANTS
        ),
        "comparison_keys_exact": {_key(row) for row in gradient if row["variant"] == "X0"}
        == expected_keys
        and {_key(row) for row in virtual if row["variant"] == "X0"} == expected_keys,
        "seed_row_counts_exact": all(
            int(summary.get("gradient_rows", -1)) == 128
            and int(summary.get("virtual_rows", -1)) == 128
            and int(summary.get("margin_rows", -1)) == 1152
            for summary in summaries
        ),
        "x1_scale_exact": all(
            _float(row, "fixed_relation_scale") == (1.0 if row["variant"] == "X0" else 1.0 / 3.0)
            for row in gradient
        )
        and all(float(summary.get("x1_fixed_scale", -1.0)) == 1.0 / 3.0 for summary in summaries),
        "exact_bprc_v01_batches_reused": all(
            summary.get("batch_list_sha256")
            == summary.get("reused_bprc_v0_1_batch_list_sha256")
            == summary.get("manifest_recorded_sha256")
            == summary.get("manifest_recomputed_sha256")
            for summary in summaries
        ),
        "models_unchanged": all(
            transition.get("models_unchanged") is True
            and transition.get("current_model_sha256_before") == transition.get("current_model_sha256_after")
            and transition.get("old_model_sha256_before") == transition.get("old_model_sha256_after")
            for summary in summaries
            for transition in summary.get("transitions", [])
        )
        and all(len(summary.get("transitions", [])) == 2 for summary in summaries)
        and all(not _bool(row["checkpoint_or_model_mutated"]) for row in virtual),
        "old_model_gradient_zero": all(int(row["old_model_gradient_nonnull"]) == 0 for row in gradient + virtual),
        "optimizer_steps_zero": all(int(summary.get("optimizer_steps", -1)) == 0 for summary in summaries)
        and all(not _bool(row["optimizer_step_called"]) for row in gradient),
        "hidden_gt_not_used_for_update": all(row["hidden_gt_usage"] == "none" for row in gradient)
        and all(row["hidden_gt_usage"] == "none_visible_update_and_val_only" for row in virtual)
        and all(row["hidden_gt_usage"] == "post_hoc_visible_previous_val_only" for row in margin),
        "exact_margin_function_reused": all(
            row["metric_function"] == "scripts.audit_tarc_relation_fidelity._margin" for row in margin
        ),
        "x0_exact_r0": all(_float(row, "b0_r0_loss_abs_error") <= 1.0e-7 for row in gradient),
        "virtual_step_norm_exact": all(_float(row, "virtual_step_norm") == 1.0e-3 for row in virtual),
        "all_values_finite": all(
            _bool(row["all_finite"]) and all(math.isfinite(float(row[field])) for field in numeric_gradient)
            for row in gradient
        )
        and all(all(math.isfinite(float(row[field])) for field in numeric_virtual) for row in virtual)
        and all(all(math.isfinite(float(row[field])) for field in numeric_margin) for row in margin),
    }


def _gradient_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    ratios = [_float(row, "relation_gradient_norm_ratio_to_x0") for row in rows if row["variant"] == "X1"]
    metrics = {
        "median_x1_to_x0": _median(ratios),
        "p10_x1_to_x0": _quantile(ratios, 0.10),
        "p90_x1_to_x0": _quantile(ratios, 0.90),
        "nonfinite_count": sum(not math.isfinite(value) for value in ratios),
        "comparisons": len(ratios),
    }
    checks = {
        "median_in_0_5_to_2_0": 0.5 <= metrics["median_x1_to_x0"] <= 2.0,
        "p10_ge_0_25": metrics["p10_x1_to_x0"] >= 0.25,
        "p90_le_4_0": metrics["p90_x1_to_x0"] <= 4.0,
        "nonfinite_zero": metrics["nonfinite_count"] == 0,
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _previous_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    values = _variant_map(rows, "previous_val_loss_delta")
    keys = sorted(values["X0"])
    medians = {variant: _median(list(values[variant].values())) for variant in VARIANTS}
    paired = [values["X1"][key] - values["X0"][key] for key in keys]
    metrics = {
        "fraction_x1_lower_than_x0": float(np.mean(np.asarray(paired) < 0.0)),
        "median_previous_delta": medians,
        "median_paired_x1_minus_x0": _median(paired),
        "comparisons": len(keys),
    }
    checks = {
        "paired_better_fraction_ge_0_60": metrics["fraction_x1_lower_than_x0"] >= 0.60,
        "median_x1_le_x0_minus_1e_4": medians["X1"] <= medians["X0"] - 1.0e-4,
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _current_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    loss = _variant_map(rows, "current_val_loss_delta")
    dice = _variant_map(rows, "current_val_dice_delta")
    median_loss = {variant: _median(list(loss[variant].values())) for variant in VARIANTS}
    median_dice = {variant: _median(list(dice[variant].values())) for variant in VARIANTS}
    metrics = {
        "median_current_loss_delta": median_loss,
        "median_current_dice_delta": median_dice,
        "x0_loss_bound": median_loss["X0"] + 0.02 * abs(median_loss["X0"]),
        "x0_dice_bound": median_dice["X0"] - 0.002,
    }
    checks = {
        "x1_loss_within_bound": median_loss["X1"] <= metrics["x0_loss_bound"],
        "x1_dice_within_bound": median_dice["X1"] >= metrics["x0_dice_bound"],
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _margin_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    values = {
        class_id: {
            variant: {
                _key(row): _float(row, "margin_agreement_after")
                for row in rows
                if int(row["class_id"]) == class_id
                and row["region"] == "all"
                and row["variant"] == variant
            }
            for variant in VARIANTS
        }
        for class_id in range(3)
    }
    class_medians: dict[str, float] = {}
    for class_id in range(3):
        if set(values[class_id]["X0"]) != set(values[class_id]["X1"]):
            raise ValueError(f"unpaired class-margin rows: {class_id}")
        class_medians[str(class_id)] = _median([
            values[class_id]["X1"][key] - values[class_id]["X0"][key]
            for key in sorted(values[class_id]["X0"])
        ])
    rim_deltas = [
        values[1]["X1"][key] - values[1]["X0"][key]
        for key in sorted(values[1]["X0"])
    ]
    metrics = {
        "fraction_x1_disc_rim_gt_x0": float(np.mean(np.asarray(rim_deltas) > 0.0)),
        "median_disc_rim_x1_minus_x0": _median(rim_deltas),
        "class_median_x1_minus_x0": class_medians,
        "comparisons": len(rim_deltas),
    }
    checks = {
        "disc_rim_better_fraction_ge_0_60": metrics["fraction_x1_disc_rim_gt_x0"] >= 0.60,
        "median_disc_rim_improvement_ge_0_005": metrics["median_disc_rim_x1_minus_x0"] >= 0.005,
        "no_class_median_below_minus_0_005": min(class_medians.values()) >= -0.005,
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BPRC-X1 exploratory diagnostic",
        "",
        f"**Status:** `{report['status']}`  ",
        f"**Seed-0 pilot authorized:** `{str(report['pilot_authorized']).lower()}`  ",
        "**Candidate:** `X1 = B2 top-2 class-balanced / 3`  ",
        "**Optimizer steps:** `0`",
        "",
        "This is a user-authorized exploratory experiment outside the frozen BPRC V0.1 protocol. BPRC V0.1 artifacts were not changed.",
        "",
        "## Gates",
        "",
        "| Gate | Result | Metrics |",
        "|---|---|---|",
    ]
    labels = {
        "gradient_scale": "Gradient scale",
        "previous_utility": "Previous-site utility",
        "current_safety": "Current-site safety",
        "disc_rim_margin": "Disc-rim margin",
    }
    for key, label in labels.items():
        gate = report["gates"].get(key)
        if gate is None:
            lines.append(f"| {label} | NOT EVALUATED | engineering hard stop |")
        else:
            lines.append(
                f"| {label} | {'PASS' if gate['passed'] else 'FAIL'} | "
                f"`{json.dumps(gate['metrics'], sort_keys=True)}` |"
            )
    lines.extend(["", "## Decision", "", report["consequence"], ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    analysis_dir = args.analysis_dir.resolve()
    status_dir = args.status_dir.resolve()
    protocol_path = args.protocol_json.resolve()
    canonical = {filename: analysis_dir / filename for filename in FILENAMES}
    report_json = status_dir / "BPRC_X1_DIAGNOSTIC_AUDIT.json"
    report_md = status_dir / "BPRC_X1_DIAGNOSTIC_AUDIT.md"
    stop_json = status_dir / "BPRC_X1_EXPLORATORY_STOP.json"
    stop_md = status_dir / "BPRC_X1_EXPLORATORY_STOP.md"
    for path in (*canonical.values(), report_json, report_md, stop_json, stop_md):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite BPRC-X1 artifact: {path}")

    combined, summaries = _merge(input_root)
    engineering = _engineering(protocol_path, combined, summaries)
    if not all(engineering.values()):
        gates: dict[str, Any] = {}
        status = "HARD_STOP_BPRC_X1_ENGINEERING"
    else:
        gates = {
            "gradient_scale": _gradient_gate(combined["gradient_scale.csv"]),
            "previous_utility": _previous_gate(combined["virtual_steps.csv"]),
            "current_safety": _current_gate(combined["virtual_steps.csv"]),
            "disc_rim_margin": _margin_gate(combined["margin_analysis.csv"]),
        }
        if not gates["gradient_scale"]["passed"]:
            status = "BPRC_X1_GRADIENT_SCALE_NOT_SUPPORTED"
        elif not gates["previous_utility"]["passed"]:
            status = "BPRC_X1_PREVIOUS_UTILITY_NOT_SUPPORTED"
        elif not gates["current_safety"]["passed"]:
            status = "BPRC_X1_CURRENT_SAFETY_NOT_SUPPORTED"
        elif not gates["disc_rim_margin"]["passed"]:
            status = "BPRC_X1_DISC_RIM_MARGIN_NOT_SUPPORTED"
        else:
            status = "BPRC_X1_DIAGNOSTIC_SUPPORTED_FOR_SEED0_PILOT"

    analysis_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in combined.items():
        write_csv(canonical[filename], rows, fieldnames=list(rows[0]))
    supported = status == "BPRC_X1_DIAGNOSTIC_SUPPORTED_FOR_SEED0_PILOT"
    consequence = (
        "All frozen diagnostic gates passed; only a seed-0 1000-step pilot is authorized."
        if supported
        else "At least one frozen diagnostic gate failed; stop without method registration or training."
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "epistemic_status": "exploratory_not_originally_preregistered",
        "candidate": "X1=B2_top2_pairwise_class_balanced/3",
        "pilot_authorized": supported,
        "optimizer_steps": 0,
        "engineering_checks": engineering,
        "gates": gates,
        "consequence": consequence,
        "source_protocol": {"path": str(protocol_path), "sha256": sha256_path(protocol_path)},
        "seed_summaries": [
            {
                "seed": summary["seed"],
                "path": summary["summary_path"],
                "sha256": sha256_path(Path(summary["summary_path"])),
                "batch_list_sha256": summary["batch_list_sha256"],
            }
            for summary in summaries
        ],
        "artifacts": {
            filename: {"path": str(path), "sha256": sha256_path(path)}
            for filename, path in canonical.items()
        },
    }
    write_json(report_json, report)
    write_text(report_md, _markdown(report))
    if not supported:
        stop = {
            "protocol_id": PROTOCOL_ID,
            "status": "BPRC_X1_EXPLORATORY_STOP",
            "trigger_status": status,
            "pilot_authorized": False,
            "optimizer_steps": 0,
            "decision_artifact": str(report_json),
            "decision_artifact_sha256": sha256_path(report_json),
            "bprc_v0_1_unchanged": True,
        }
        write_json(stop_json, stop)
        write_text(
            stop_md,
            f"# BPRC-X1 exploratory stop\n\n**Trigger:** `{status}`\n\n"
            "The fixed exploratory diagnostic did not pass every gate. No method, config, optimizer step, or training pilot is authorized. BPRC V0.1 remains unchanged.\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "pilot_authorized": supported,
                "json": str(report_json),
                "markdown": str(report_md),
            },
            indent=2,
        )
    )
    return 0 if supported else 3


if __name__ == "__main__":
    raise SystemExit(main())
