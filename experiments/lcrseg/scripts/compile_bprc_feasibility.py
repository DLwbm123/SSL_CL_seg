#!/usr/bin/env python3
"""Compile the preregistered BPRC-Seg V0.1 Part-A feasibility gate."""
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


FILENAMES = (
    "feasibility_gradient_scale.csv",
    "feasibility_virtual_steps.csv",
    "feasibility_margin_analysis.csv",
)
VARIANTS = ("B0", "B1", "B2", "B3")
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
        raise RuntimeError("gate input is empty or non-finite")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _quantile(values: list[float], probability: float) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("gate input is empty or non-finite")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _key(row: dict[str, str]) -> tuple[int, str, int]:
    return int(row["seed"]), row["transition"], int(row["update_batch"])


def _by_variant(
    rows: list[dict[str, str]], key: str
) -> dict[str, dict[tuple[int, str, int], float]]:
    result = {variant: {} for variant in VARIANTS}
    for row in rows:
        variant = row["variant"]
        if variant not in result:
            raise ValueError(f"unexpected BPRC variant: {variant}")
        row_key = _key(row)
        if row_key in result[variant]:
            raise ValueError(f"duplicate row for {variant}: {row_key}")
        result[variant][row_key] = _float(row, key)
    expected = set(result["B0"])
    if any(set(result[variant]) != expected for variant in VARIANTS):
        raise ValueError(f"unpaired BPRC rows for metric {key}")
    return result


def _merge(input_root: Path) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    combined = {filename: [] for filename in FILENAMES}
    summaries: list[dict[str, Any]] = []
    for seed in range(3):
        seed_dir = input_root / f"seed{seed}"
        summary_path = seed_dir / "feasibility_summary.json"
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
        summary["manifest_recorded_sha256"] = manifest.get("combined_sha256")
        summaries.append(summary)
        for filename in FILENAMES:
            path = seed_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            combined[filename].extend(read_csv(path))
    return combined, summaries


def _engineering(
    input_root: Path,
    status_dir: Path,
    combined: dict[str, list[dict[str, str]]],
    summaries: list[dict[str, Any]],
) -> dict[str, bool]:
    gradient = combined["feasibility_gradient_scale.csv"]
    virtual = combined["feasibility_virtual_steps.csv"]
    margin = combined["feasibility_margin_analysis.csv"]
    gradient_numeric = (
        "relation_loss",
        "relation_gradient_norm",
        "relation_gradient_norm_ratio_to_b0",
        "cos_relation_with_sup",
        "cos_relation_with_assim",
        "sup_gradient_norm",
        "assim_gradient_norm",
        "total_gradient_norm",
        "boundary_relation_loss",
        "interior_relation_loss",
        "probability_sum_error",
        "b0_r0_loss_abs_error",
    )
    virtual_numeric = (
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
    margin_numeric = (
        "margin_agreement_before",
        "margin_agreement_after",
        "margin_agreement_delta",
        "margin_abs_error_after",
    )
    freeze_path = status_dir / "TARC_V0_1_FREEZE_FOR_BPRC.json"
    reuse_path = status_dir / "BPRC_METRIC_REUSE_AUDIT.json"
    postmortem_path = status_dir / "BPRC_TARC_POSTMORTEM.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.is_file() else {}
    reuse = json.loads(reuse_path.read_text(encoding="utf-8")) if reuse_path.is_file() else {}
    postmortem = json.loads(postmortem_path.read_text(encoding="utf-8")) if postmortem_path.is_file() else {}
    expected_keys = {
        (seed, transition, update)
        for seed in range(3)
        for transition in TRANSITIONS
        for update in range(32)
    }
    checks = {
        "tarc_freeze_status_exact": freeze.get("status") == "TARC_V0_1_FROZEN_FOR_BPRC"
        and freeze.get("tarc_status") == "TARC_RELATION_FIDELITY_NOT_SUPPORTED"
        and freeze.get("tarc_training_method_exists") is False,
        "exact_tarc_metric_reuse_audit_passed": reuse.get("status") == "BPRC_METRIC_REUSE_AUDIT_PASSED",
        "descriptive_pairwise_postmortem_complete": postmortem.get("status") == "BPRC_TARC_POSTMORTEM_COMPLETE",
        "three_seed_summaries_complete": len(summaries) == 3
        and [int(summary.get("seed", -1)) for summary in summaries] == [0, 1, 2]
        and all(summary.get("status") == "BPRC_FEASIBILITY_SEED_AUDIT_COMPLETE" for summary in summaries),
        "row_counts_exact": len(gradient) == 768 and len(virtual) == 768 and len(margin) == 6912,
        "variant_rows_exact": all(
            sum(row["variant"] == variant for row in gradient) == 192
            and sum(row["variant"] == variant for row in virtual) == 192
            and sum(row["variant"] == variant for row in margin) == 1728
            for variant in VARIANTS
        ),
        "fixed_comparison_keys_exact": {_key(row) for row in gradient if row["variant"] == "B0"}
        == expected_keys
        and {_key(row) for row in virtual if row["variant"] == "B0"} == expected_keys,
        "seed_summary_row_counts_exact": all(
            int(summary.get("gradient_rows", -1)) == 256
            and int(summary.get("virtual_rows", -1)) == 256
            and int(summary.get("margin_rows", -1)) == 2304
            for summary in summaries
        ),
        "fixed_batch_manifest_sha_valid": all(
            summary.get("batch_list_sha256") == summary.get("manifest_recorded_sha256")
            == summary.get("manifest_recomputed_sha256")
            for summary in summaries
        ),
        "models_and_checkpoints_not_mutated": all(
            transition.get("models_unchanged") is True
            and transition.get("current_model_sha256_before") == transition.get("current_model_sha256_after")
            and transition.get("old_model_sha256_before") == transition.get("old_model_sha256_after")
            for summary in summaries
            for transition in summary.get("transitions", [])
        )
        and all(len(summary.get("transitions", [])) == 2 for summary in summaries)
        and all(not _bool(row["checkpoint_or_model_mutated"]) for row in virtual),
        "old_model_gradient_zero": all(int(row["old_model_gradient_nonnull"]) == 0 for row in virtual),
        "optimizer_steps_zero": all(int(summary.get("optimizer_steps", -1)) == 0 for summary in summaries)
        and all(not _bool(row["optimizer_step_called"]) for row in gradient),
        "hidden_gt_post_hoc_only": all(
            row["hidden_gt_usage"] == "post_hoc_boundary_grouping_only" for row in gradient
        )
        and all(row["hidden_gt_usage"] == "none_visible_update_and_val_only" for row in virtual)
        and all(row["hidden_gt_usage"] == "post_hoc_visible_previous_val_only" for row in margin),
        "exact_tarc_margin_function_reused": all(
            row["metric_function"] == "scripts.audit_tarc_relation_fidelity._margin" for row in margin
        ),
        "b0_exact_r0_loss_equivalence": all(
            _float(row, "b0_r0_loss_abs_error") <= 1.0e-7 for row in gradient if row["variant"] == "B0"
        ),
        "virtual_step_norm_exact": all(_float(row, "virtual_step_norm") == 1.0e-3 for row in virtual),
        "all_gradient_scales_finite": all(
            _bool(row["all_finite"]) and all(math.isfinite(float(row[key])) for key in gradient_numeric)
            for row in gradient
        ),
        "all_virtual_metrics_finite": all(
            all(math.isfinite(float(row[key])) for key in virtual_numeric) for row in virtual
        ),
        "all_margin_metrics_finite": all(
            all(math.isfinite(float(row[key])) for key in margin_numeric) for row in margin
        ),
    }
    return checks


def _gradient_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    ratios = [
        _float(row, "relation_gradient_norm_ratio_to_b0")
        for row in rows
        if row["variant"] == "B3"
    ]
    metrics = {
        "median_b3_to_b0_gradient_norm_ratio": _median(ratios),
        "p10_b3_to_b0_gradient_norm_ratio": _quantile(ratios, 0.10),
        "p90_b3_to_b0_gradient_norm_ratio": _quantile(ratios, 0.90),
        "nonfinite_count": sum(not math.isfinite(float(row["relation_gradient_norm_ratio_to_b0"])) for row in rows),
        "comparisons": len(ratios),
    }
    checks = {
        "median_ratio_in_0_5_to_2_0": 0.5 <= metrics["median_b3_to_b0_gradient_norm_ratio"] <= 2.0,
        "p10_ratio_ge_0_25": metrics["p10_b3_to_b0_gradient_norm_ratio"] >= 0.25,
        "p90_ratio_le_4_0": metrics["p90_b3_to_b0_gradient_norm_ratio"] <= 4.0,
        "nonfinite_count_zero": metrics["nonfinite_count"] == 0,
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _previous_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    paired = _by_variant(rows, "previous_val_loss_delta")
    keys = sorted(paired["B0"])
    medians = {variant: _median(list(paired[variant].values())) for variant in VARIANTS}
    metrics = {
        "fraction_b3_delta_lower_than_b0": float(np.mean([paired["B3"][key] < paired["B0"][key] for key in keys])),
        "median_previous_loss_delta": medians,
        "comparisons": len(keys),
    }
    checks = {
        "b3_better_than_b0_in_at_least_60pct": metrics["fraction_b3_delta_lower_than_b0"] >= 0.60,
        "median_b3_le_b0_minus_1e_4": medians["B3"] <= medians["B0"] - 1.0e-4,
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _current_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    loss = _by_variant(rows, "current_val_loss_delta")
    dice = _by_variant(rows, "current_val_dice_delta")
    median_loss = {variant: _median(list(loss[variant].values())) for variant in VARIANTS}
    median_dice = {variant: _median(list(dice[variant].values())) for variant in VARIANTS}
    metrics = {
        "median_current_loss_delta": median_loss,
        "median_current_dice_delta": median_dice,
        "b0_loss_2pct_safety_bound": median_loss["B0"] + 0.02 * abs(median_loss["B0"]),
        "b0_dice_minus_0_002_bound": median_dice["B0"] - 0.002,
    }
    checks = {
        "median_b3_loss_within_b0_2pct_abs_bound": median_loss["B3"] <= metrics["b0_loss_2pct_safety_bound"],
        "median_b3_dice_ge_b0_minus_0_002": median_dice["B3"] >= metrics["b0_dice_minus_0_002_bound"],
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _margin_pairs(
    rows: list[dict[str, str]], *, region: str = "all"
) -> dict[int, dict[str, dict[tuple[int, str, int], float]]]:
    result = {
        class_id: {variant: {} for variant in VARIANTS}
        for class_id in range(3)
    }
    for row in rows:
        if row["region"] != region:
            continue
        class_id = int(row["class_id"])
        variant = row["variant"]
        row_key = _key(row)
        if row_key in result[class_id][variant]:
            raise ValueError(f"duplicate margin row for class={class_id}, variant={variant}, key={row_key}")
        result[class_id][variant][row_key] = _float(row, "margin_agreement_after")
    for class_id in range(3):
        expected = set(result[class_id]["B0"])
        if any(set(result[class_id][variant]) != expected for variant in VARIANTS):
            raise ValueError(f"unpaired margin rows for class {class_id}")
    return result


def _disc_rim_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    pairs = _margin_pairs(rows)
    rim = pairs[1]
    keys = sorted(rim["B0"])
    rim_deltas = [rim["B3"][key] - rim["B0"][key] for key in keys]
    class_medians = {
        str(class_id): _median([
            pairs[class_id]["B3"][key] - pairs[class_id]["B0"][key]
            for key in sorted(pairs[class_id]["B0"])
        ])
        for class_id in range(3)
    }
    metrics = {
        "fraction_b3_disc_rim_margin_gt_b0": float(np.mean(np.asarray(rim_deltas) > 0.0)),
        "median_disc_rim_b3_minus_b0": _median(rim_deltas),
        "class_median_b3_minus_b0": class_medians,
        "comparisons": len(keys),
    }
    checks = {
        "b3_improves_disc_rim_in_at_least_60pct": metrics["fraction_b3_disc_rim_margin_gt_b0"] >= 0.60,
        "median_disc_rim_b3_minus_b0_ge_0_005": metrics["median_disc_rim_b3_minus_b0"] >= 0.005,
        "no_class_median_below_minus_0_005": min(class_medians.values()) >= -0.005,
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _beyond_class_balance_gate(
    virtual_rows: list[dict[str, str]], margin_rows: list[dict[str, str]]
) -> dict[str, Any]:
    previous = _by_variant(virtual_rows, "previous_val_loss_delta")
    margins = _margin_pairs(margin_rows)[1]
    median_previous = {variant: _median(list(previous[variant].values())) for variant in VARIANTS}
    median_rim = {variant: _median(list(margins[variant].values())) for variant in VARIANTS}
    metrics = {
        "median_previous_loss_delta": median_previous,
        "median_disc_rim_margin_agreement_after": median_rim,
        "disc_rim_b3_minus_b1": median_rim["B3"] - median_rim["B1"],
    }
    checks = {
        "median_previous_b3_le_b1_minus_5e_5": median_previous["B3"] <= median_previous["B1"] - 5.0e-5,
        "median_disc_rim_b3_ge_b1_plus_0_003": median_rim["B3"] >= median_rim["B1"] + 0.003,
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _all_competitors_gate(
    virtual_rows: list[dict[str, str]], margin_rows: list[dict[str, str]]
) -> dict[str, Any]:
    previous = _by_variant(virtual_rows, "previous_val_loss_delta")
    margins = _margin_pairs(margin_rows)
    median_previous = {variant: _median(list(previous[variant].values())) for variant in VARIANTS}
    class_deltas = {
        str(class_id): _median([
            margins[class_id]["B3"][key] - margins[class_id]["B2"][key]
            for key in sorted(margins[class_id]["B2"])
        ])
        for class_id in range(3)
    }
    metrics = {
        "median_previous_loss_delta": median_previous,
        "class_median_margin_b3_minus_b2": class_deltas,
    }
    checks = {
        "median_previous_b3_le_b2_plus_5e_5": median_previous["B3"] <= median_previous["B2"] + 5.0e-5,
        "no_class_margin_b3_below_b2_by_more_than_0_003": min(class_deltas.values()) >= -0.003,
    }
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BPRC-Seg V0.1 feasibility audit",
        "",
        f"**Final status:** `{report['status']}`  ",
        f"**Part B authorized:** `{str(report['part_b_authorized']).lower()}`  ",
        "**Optimizer steps:** `0`  ",
        "**Hidden GT training usage:** `none`",
        "",
        "## Engineering boundary",
        "",
        "- TARC remained frozen at `TARC_RELATION_FIDELITY_NOT_SUPPORTED`.",
        "- The audit reused the exact frozen TARC metric functions and R0 objective path.",
        "- B0/B1/B2/B3 were evaluated on fixed 32-update/16-previous-val/16-current-val batches for each of two transitions and three seeds.",
        "- Functional updates were stateless with norm `1e-3`; optimizer steps, checkpoint mutation, old-model gradients, and hidden-GT training usage were zero.",
        "- Boundary/interior and class diagnostic labels were used post-hoc only.",
        "",
        "## Gate results",
        "",
        "| Gate | Result | Metrics |",
        "|---|---|---|",
    ]
    labels = {
        "gradient_scale": "Gradient scale",
        "previous_utility": "B3 vs B0 previous-site utility",
        "current_safety": "Current-site safety",
        "disc_rim_margin": "Disc-rim margin",
        "pairwise_beyond_class_balance": "B3 beyond B1 class balance",
        "all_competitors": "B3 all competitors beyond B2 top-2",
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
    lines.extend(
        [
            "",
            "## Protocol decision",
            "",
            report["protocol_consequence"],
            "",
            "## Canonical artifacts",
            "",
            "- `reports/analysis/bprcseg_v0_1/feasibility_gradient_scale.csv`",
            "- `reports/analysis/bprcseg_v0_1/feasibility_virtual_steps.csv`",
            "- `reports/analysis/bprcseg_v0_1/feasibility_margin_analysis.csv`",
            "- `reports/experiment_status/BPRC_FEASIBILITY_AUDIT.json`",
            "",
        ]
    )
    return "\n".join(lines)


def _stop_markdown(report: dict[str, Any]) -> str:
    return f"""# Stop new relation methods

**BPRC status:** `{report['status']}`

BPRC-Seg V0.1 did not pass every preregistered feasibility gate. Under the frozen protocol:

- do not register or train a BPRC method;
- do not create BPRC training configs or run an optimizer;
- stop proposing new relation-coordinate or relation-loss variants;
- any future work is limited to source-faithful DC2T/JASCL-PAS reproduction and preregistered strong baselines under a new protocol.

The decision evidence is `reports/experiment_status/BPRC_FEASIBILITY_AUDIT.json`.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--status-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    analysis_dir = args.analysis_dir.resolve()
    status_dir = args.status_dir.resolve()
    canonical = {filename: analysis_dir / filename for filename in FILENAMES}
    report_json = status_dir / "BPRC_FEASIBILITY_AUDIT.json"
    report_md = status_dir / "BPRC_FEASIBILITY_AUDIT.md"
    stop_json = status_dir / "STOP_NEW_RELATION_METHODS.json"
    stop_md = status_dir / "STOP_NEW_RELATION_METHODS.md"
    for path in (*canonical.values(), report_json, report_md, stop_json, stop_md):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite BPRC feasibility artifact: {path}")
    combined, summaries = _merge(input_root)
    engineering_checks = _engineering(input_root, status_dir, combined, summaries)
    if not all(engineering_checks.values()):
        gates: dict[str, Any] = {}
        status = "HARD_STOP_BPRC_AUDIT_ENGINEERING"
    else:
        gates = {
            "gradient_scale": _gradient_gate(combined["feasibility_gradient_scale.csv"]),
            "previous_utility": _previous_gate(combined["feasibility_virtual_steps.csv"]),
            "current_safety": _current_gate(combined["feasibility_virtual_steps.csv"]),
            "disc_rim_margin": _disc_rim_gate(combined["feasibility_margin_analysis.csv"]),
            "pairwise_beyond_class_balance": _beyond_class_balance_gate(
                combined["feasibility_virtual_steps.csv"], combined["feasibility_margin_analysis.csv"]
            ),
            "all_competitors": _all_competitors_gate(
                combined["feasibility_virtual_steps.csv"], combined["feasibility_margin_analysis.csv"]
            ),
        }
        if not gates["gradient_scale"]["passed"]:
            status = "BPRC_GRADIENT_SCALE_NOT_SUPPORTED"
        elif not gates["previous_utility"]["passed"]:
            status = "BPRC_PREVIOUS_UTILITY_NOT_SUPPORTED"
        elif not gates["current_safety"]["passed"]:
            status = "BPRC_CURRENT_SAFETY_NOT_SUPPORTED"
        elif not gates["disc_rim_margin"]["passed"]:
            status = "BPRC_DISC_RIM_MARGIN_NOT_SUPPORTED"
        elif not gates["pairwise_beyond_class_balance"]["passed"]:
            status = "BPRC_PAIRWISE_NOT_BEYOND_CLASS_BALANCE"
        elif not gates["all_competitors"]["passed"]:
            status = "BPRC_ALL_COMPETITORS_NOT_SUPPORTED"
        else:
            status = "BPRC_FEASIBILITY_SUPPORTED"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in combined.items():
        write_csv(canonical[filename], rows, fieldnames=list(rows[0]))
    supported = status == "BPRC_FEASIBILITY_SUPPORTED"
    consequence = (
        "All feasibility gates passed. Part B method implementation is authorized; this audit itself performed no training."
        if supported
        else "Protocol hard stop before BPRC method/config implementation or training. STOP_NEW_RELATION_METHODS is binding."
    )
    report = {
        "protocol_id": "bprcseg_v0_1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "part_b_authorized": supported,
        "optimizer_steps": 0,
        "hidden_gt_training_usage": "none",
        "engineering_checks": engineering_checks,
        "gates": gates,
        "protocol_consequence": consequence,
        "seed_summaries": [
            {
                "seed": summary["seed"],
                "summary_path": summary["summary_path"],
                "summary_sha256": sha256_path(Path(summary["summary_path"])),
                "fixed_batch_manifest_path": summary["manifest_path"],
                "fixed_batch_manifest_sha256": sha256_path(Path(summary["manifest_path"])),
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
            "protocol_id": "bprcseg_v0_1",
            "status": "STOP_NEW_RELATION_METHODS",
            "trigger_status": status,
            "part_b_authorized": False,
            "optimizer_steps": 0,
            "allowed_future_scope": [
                "source-faithful DC2T reproduction under a new protocol",
                "source-faithful JASCL-PAS reproduction under a new protocol",
                "preregistered strong baselines under a new protocol",
            ],
            "forbidden_scope": [
                "BPRC method registration or training",
                "new relation-coordinate variants",
                "new relation-loss variants",
            ],
            "decision_artifact": str(report_json),
            "decision_artifact_sha256": sha256_path(report_json),
        }
        write_json(stop_json, stop)
        write_text(stop_md, _stop_markdown(report))
    print(
        json.dumps(
            {
                "status": status,
                "part_b_authorized": supported,
                "json": str(report_json),
                "markdown": str(report_md),
                "stop_artifacts_created": not supported,
            },
            indent=2,
        )
    )
    return 0 if supported else 3


if __name__ == "__main__":
    raise SystemExit(main())
