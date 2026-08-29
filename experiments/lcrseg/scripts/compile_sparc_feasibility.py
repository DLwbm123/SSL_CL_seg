#!/usr/bin/env python3
"""Compile all preregistered SPARC-Seg V0.1 feasibility gates."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.common import sha256_path, write_csv, write_json, write_text  # noqa: E402


SEEDS = (0, 1, 2)
FOREGROUND_IDS = (1, 2)
EXPECTED_PAIRS = 6
EXPECTED_GRADIENT_ROWS = 3 * 2 * 32
EXPECTED_VIRTUAL_ROWS = 3 * 2 * 32 * 6


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.median(finite)) if finite else float("nan")


def _aggregate(seed_dirs: list[Path], name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed_dir in seed_dirs:
        rows.extend(_read_csv(seed_dir / name))
    return rows


def _prototype_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    lookup = {
        (int(row["seed"]), row["transition"], row["selector"], int(row["class_id"])): row
        for row in rows
    }
    per_class: dict[str, Any] = {}
    overall = True
    for class_id in FOREGROUND_IDS:
        improvements: list[float] = []
        coverages: list[float] = []
        pair_records: list[dict[str, Any]] = []
        pair_keys = sorted({(key[0], key[1]) for key in lookup if key[3] == class_id})
        for seed, transition in pair_keys:
            baseline = lookup.get((seed, transition, "r0_candidate", class_id))
            current = lookup.get((seed, transition, "current_pas", class_id))
            if baseline is None or current is None:
                continue
            improvement = _float(current, "precision") - _float(baseline, "precision")
            coverage = _float(current, "coverage")
            improvements.append(improvement)
            coverages.append(coverage)
            pair_records.append(
                {"seed": seed, "transition": transition, "precision_improvement": improvement, "coverage": coverage}
            )
        passed = (
            len(improvements) == EXPECTED_PAIRS
            and all(math.isfinite(value) for value in improvements + coverages)
            and float(np.mean(improvements)) >= 0.020
            and sum(value > 0 for value in improvements) >= 4
            and sum(value >= 0.05 for value in coverages) >= 4
        )
        per_class[str(class_id)] = {
            "pair_count": len(improvements),
            "mean_precision_improvement": float(np.mean(improvements)) if improvements else float("nan"),
            "positive_pairs": sum(value > 0 for value in improvements),
            "coverage_ge_0_05_pairs": sum(value >= 0.05 for value in coverages),
            "pairs": pair_records,
            "passed": passed,
        }
        overall &= passed
    return {"passed": overall, "per_class": per_class}


def _partition_gates(rows: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    lookup = {
        (int(row["seed"]), row["transition"], row["partition"], int(row["class_id"])): row
        for row in rows
    }
    separation: dict[str, Any] = {}
    spatial: dict[str, Any] = {}
    separation_pass = True
    spatial_pass = True
    for class_id in FOREGROUND_IDS:
        pair_keys = sorted({(key[0], key[1]) for key in lookup if key[3] == class_id})
        differences: list[float] = []
        stable_coverages: list[float] = []
        plastic_coverages: list[float] = []
        stable_boundary_correct = stable_boundary_true = 0
        stable_interior_correct = stable_interior_true = 0
        plastic_boundary_correct = 0
        for seed, transition in pair_keys:
            stable = lookup.get((seed, transition, "stable", class_id))
            plastic = lookup.get((seed, transition, "plastic", class_id))
            if stable is None or plastic is None:
                continue
            differences.append(_float(stable, "precision") - _float(plastic, "precision"))
            stable_coverages.append(_float(stable, "coverage"))
            plastic_coverages.append(_float(plastic, "coverage"))
            stable_boundary_correct += int(float(stable["boundary_correct"]))
            stable_boundary_true += int(float(stable["boundary_true"]))
            stable_interior_correct += int(float(stable["interior_correct"]))
            stable_interior_true += int(float(stable["interior_true"]))
            plastic_boundary_correct += int(float(plastic["boundary_correct"]))
        sep_passed = (
            len(differences) == EXPECTED_PAIRS
            and all(math.isfinite(value) for value in differences + stable_coverages + plastic_coverages)
            and _median(differences) >= 0.030
            and sum(value > 0 for value in differences) >= 4
            and sum(value >= 0.02 for value in stable_coverages) >= 4
            and sum(value >= 0.02 for value in plastic_coverages) >= 4
        )
        separation[str(class_id)] = {
            "pair_count": len(differences),
            "median_stable_minus_plastic_precision": _median(differences),
            "stable_gt_plastic_pairs": sum(value > 0 for value in differences),
            "stable_coverage_ge_0_02_pairs": sum(value >= 0.02 for value in stable_coverages),
            "plastic_coverage_ge_0_02_pairs": sum(value >= 0.02 for value in plastic_coverages),
            "passed": sep_passed,
        }
        separation_pass &= sep_passed
        boundary_coverage = stable_boundary_correct / stable_boundary_true if stable_boundary_true else float("nan")
        interior_coverage = stable_interior_correct / stable_interior_true if stable_interior_true else float("nan")
        ratio = boundary_coverage / interior_coverage if interior_coverage > 0 else float("nan")
        spatial_passed = math.isfinite(ratio) and ratio >= 0.20 and plastic_boundary_correct > 0
        spatial[str(class_id)] = {
            "stable_boundary_coverage": boundary_coverage,
            "stable_interior_coverage": interior_coverage,
            "boundary_to_interior_ratio": ratio,
            "plastic_boundary_correct_pixels": plastic_boundary_correct,
            "passed": spatial_passed,
        }
        spatial_pass &= spatial_passed
    return {"passed": separation_pass, "per_class": separation}, {"passed": spatial_pass, "per_class": spatial}


def _feature_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    lookup = {
        (
            int(row["seed"]),
            row["transition"],
            row["layer"],
            row["partition"],
            int(row["class_id"]),
            row["region"],
        ): row
        for row in rows
    }
    differences: list[float] = []
    distance_order: list[bool] = []
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        transitions = sorted({key[1] for key in lookup if key[0] == seed})
        for transition in transitions:
            for layer in ("dec3", "dec1"):
                for class_id in FOREGROUND_IDS:
                    stable = lookup.get((seed, transition, layer, "stable", class_id, "all"))
                    plastic = lookup.get((seed, transition, layer, "plastic", class_id, "all"))
                    if stable is None or plastic is None:
                        continue
                    stable_cos = _float(stable, "old_current_cosine")
                    plastic_cos = _float(plastic, "old_current_cosine")
                    stable_l2 = _float(stable, "old_current_l2")
                    plastic_l2 = _float(plastic, "old_current_l2")
                    if not all(math.isfinite(value) for value in (stable_cos, plastic_cos, stable_l2, plastic_l2)):
                        continue
                    difference = stable_cos - plastic_cos
                    ordered = stable_l2 <= plastic_l2
                    differences.append(difference)
                    distance_order.append(ordered)
                    records.append(
                        {
                            "seed": seed,
                            "transition": transition,
                            "layer": layer,
                            "class_id": class_id,
                            "stable_minus_plastic_cosine": difference,
                            "stable_l2_le_plastic": ordered,
                        }
                    )
    expected = 3 * 2 * 2 * 2
    fraction = sum(distance_order) / len(distance_order) if distance_order else 0.0
    passed = len(differences) == expected and _median(differences) >= 0.030 and fraction >= 0.70
    return {
        "passed": passed,
        "comparison_count": len(differences),
        "expected_comparisons": expected,
        "median_stable_minus_plastic_cosine": _median(differences),
        "stable_distance_le_plastic_fraction": fraction,
        "comparisons": records,
    }


def _gradient_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    ratios = [_float(row, "sfm_to_relation_ratio") for row in rows if _bool(row.get("finite", ""))]
    all_finite = len(ratios) == len(rows) and all(math.isfinite(value) for value in ratios)
    median = _median(ratios)
    p10 = float(np.quantile(ratios, 0.10)) if ratios else float("nan")
    p90 = float(np.quantile(ratios, 0.90)) if ratios else float("nan")
    localization = all(_bool(row.get("localization_pass", "")) for row in rows)
    old_grad_zero = all(int(float(row.get("old_model_gradient_nonnull", "0"))) == 0 for row in rows)
    passed = (
        len(rows) == EXPECTED_GRADIENT_ROWS
        and all_finite
        and 0.10 <= median <= 1.50
        and p90 <= 3.0
        and p10 >= 0.02
        and localization
        and old_grad_zero
    )
    return {
        "passed": passed,
        "row_count": len(rows),
        "expected_rows": EXPECTED_GRADIENT_ROWS,
        "median_ratio": median,
        "p10_ratio": p10,
        "p90_ratio": p90,
        "nonfinite_count": len(rows) - len(ratios),
        "localization_all_pass": localization,
        "old_model_gradient_all_zero": old_grad_zero,
    }


def _virtual_gates(rows: list[dict[str, str]]) -> dict[str, Any]:
    pivot: dict[tuple[int, str, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (int(row["seed"]), row["transition"], int(row["update_batch"]))
        pivot.setdefault(key, {})[row["variant"]] = row
    complete = len(rows) == EXPECTED_VIRTUAL_ROWS and all(set(values) == {"S0", "S1", "S2", "S3", "S4", "S5"} for values in pivot.values())
    variants: dict[str, dict[str, float]] = {}
    for variant in ("S0", "S1", "S2", "S3", "S4", "S5"):
        selected = [values[variant] for values in pivot.values() if variant in values]
        variants[variant] = {
            "previous_loss_delta_median": _median([_float(row, "previous_val_loss_delta") for row in selected]),
            "current_loss_delta_median": _median([_float(row, "current_val_loss_delta") for row in selected]),
            "previous_dice_delta_median": _median([_float(row, "previous_val_dice_delta") for row in selected]),
            "current_dice_delta_median": _median([_float(row, "current_val_dice_delta") for row in selected]),
        }
    paired_better = [
        _float(values["S3"], "previous_val_loss_delta") < _float(values["S1"], "previous_val_loss_delta")
        for values in pivot.values()
        if "S3" in values and "S1" in values
    ]
    better_fraction = sum(paired_better) / len(paired_better) if paired_better else 0.0
    s1, s2, s3, s4, s5 = (variants[name] for name in ("S1", "S2", "S3", "S4", "S5"))
    previous = complete and better_fraction >= 0.60 and s3["previous_loss_delta_median"] <= s1["previous_loss_delta_median"] - 1.0e-4
    current = (
        complete
        and s3["current_loss_delta_median"]
        <= s1["current_loss_delta_median"] + 0.02 * abs(s1["current_loss_delta_median"])
        and s3["current_dice_delta_median"] >= s1["current_dice_delta_median"] - 0.002
    )
    targeting = (
        complete
        and s3["previous_loss_delta_median"] <= s4["previous_loss_delta_median"]
        and s3["current_loss_delta_median"]
        <= s4["current_loss_delta_median"] + 0.02 * abs(s4["current_loss_delta_median"])
        and s3["current_loss_delta_median"] <= s5["current_loss_delta_median"]
        and s3["previous_loss_delta_median"] <= s5["previous_loss_delta_median"] + 1.0e-4
        and s3["previous_loss_delta_median"] <= s1["previous_loss_delta_median"]
        and s3["current_loss_delta_median"] <= s2["current_loss_delta_median"]
    )
    return {
        "complete": complete,
        "row_count": len(rows),
        "expected_rows": EXPECTED_VIRTUAL_ROWS,
        "paired_s3_better_than_s1_previous_fraction": better_fraction,
        "variant_medians": variants,
        "previous_gate_passed": previous,
        "current_safety_gate_passed": current,
        "targeting_and_complementarity_gate_passed": targeting,
    }


def _status(engineering: bool, gates: dict[str, Any]) -> str:
    if not engineering:
        return "HARD_STOP_SPARC_AUDIT_ENGINEERING"
    if not gates["A1_current_pas"]["passed"]:
        return "SPARC_PAS_NOT_SUPPORTED"
    if not gates["A2_partition"]["passed"]:
        return "SPARC_PARTITION_NOT_SUPPORTED"
    if not gates["A3_spatial"]["passed"]:
        return "SPARC_SPATIAL_COVERAGE_NOT_SUPPORTED"
    if not gates["B_feature_separation"]["passed"]:
        return "SPARC_FEATURE_SEPARATION_NOT_SUPPORTED"
    if not gates["B_gradient_scale"]["passed"]:
        return "SPARC_GRADIENT_SCALE_NOT_SUPPORTED"
    if not gates["C_virtual"]["previous_gate_passed"]:
        return "SPARC_VIRTUAL_PREVIOUS_NOT_SUPPORTED"
    if not gates["C_virtual"]["current_safety_gate_passed"]:
        return "SPARC_CURRENT_SAFETY_NOT_SUPPORTED"
    if not gates["C_virtual"]["targeting_and_complementarity_gate_passed"]:
        return "SPARC_TARGETING_NOT_SUPPORTED"
    return "SPARC_FEASIBILITY_SUPPORTED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "experiment_status")
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    report_dir = args.report_dir.resolve()
    outputs = [
        analysis_dir / "prototype_validation_quality.csv",
        analysis_dir / "partition_quality.csv",
        analysis_dir / "feature_separation.csv",
        analysis_dir / "gradient_scale.csv",
        analysis_dir / "virtual_steps.csv",
        report_dir / "SPARC_FEASIBILITY_AUDIT.json",
        report_dir / "SPARC_FEASIBILITY_AUDIT.md",
    ]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite SPARC compiled artifact: {path}")
    seed_dirs = [analysis_dir / f"seed{seed}" for seed in SEEDS]
    required = [
        seed_dir / name
        for seed_dir in seed_dirs
        for name in (
            "visible_summary.json",
            "posthoc_summary.json",
            "prototype_validation_quality.csv",
            "partition_quality.csv",
            "feature_separation.csv",
            "gradient_scale.csv",
            "virtual_steps.csv",
            "batch_manifest.json",
        )
    ]
    missing = [str(path) for path in required if not path.is_file()]
    source_ok = (report_dir / "SPARC_SOURCE_AUDIT.json").is_file() and json.loads(
        (report_dir / "SPARC_SOURCE_AUDIT.json").read_text()
    ).get("status") == "SPARC_SOURCE_AUDIT_PASSED"
    path_ok = (report_dir / "SPARC_MODEL_PATH_AUDIT.json").is_file() and json.loads(
        (report_dir / "SPARC_MODEL_PATH_AUDIT.json").read_text()
    ).get("status") == "SPARC_MODEL_PATH_AUDIT_PASSED"
    freeze_ok = (report_dir / "SPARC_PREVIOUS_METHOD_FREEZE.json").is_file()
    engineering = not missing and source_ok and path_ok and freeze_ok
    if missing:
        raise FileNotFoundError("missing seed artifacts: " + ", ".join(missing))
    prototype_rows = _aggregate(seed_dirs, "prototype_validation_quality.csv")
    partition_rows = _aggregate(seed_dirs, "partition_quality.csv")
    feature_rows = _aggregate(seed_dirs, "feature_separation.csv")
    gradient_rows = _aggregate(seed_dirs, "gradient_scale.csv")
    virtual_rows = _aggregate(seed_dirs, "virtual_steps.csv")
    write_csv(outputs[0], prototype_rows)
    write_csv(outputs[1], partition_rows)
    write_csv(outputs[2], feature_rows)
    write_csv(outputs[3], gradient_rows)
    write_csv(outputs[4], virtual_rows)
    a1 = _prototype_gate(prototype_rows)
    a2, a3 = _partition_gates(partition_rows)
    feature = _feature_gate(feature_rows)
    gradient = _gradient_gate(gradient_rows)
    virtual = _virtual_gates(virtual_rows)
    gates = {
        "A1_current_pas": a1,
        "A2_partition": a2,
        "A3_spatial": a3,
        "B_feature_separation": feature,
        "B_gradient_scale": gradient,
        "C_virtual": virtual,
    }
    status = _status(engineering, gates)
    visible = [json.loads((seed_dir / "visible_summary.json").read_text()) for seed_dir in seed_dirs]
    posthoc = [json.loads((seed_dir / "posthoc_summary.json").read_text()) for seed_dir in seed_dirs]
    evidence = {str(path.relative_to(ROOT)): sha256_path(path) for path in outputs[:5]}
    payload = {
        "protocol_id": "sparcseg_v0_1",
        "status": status,
        "optimizer_steps": 0,
        "training_method_registered": False,
        "training_configs_created": False,
        "hidden_gt_training_usage": "none",
        "hidden_gt_analysis_usage": "independent_post_hoc_only",
        "engineering_checks": {
            "passed": engineering,
            "missing_artifacts": missing,
            "previous_method_freeze": freeze_ok,
            "source_audit_passed": source_ok,
            "model_path_audit_passed": path_ok,
        },
        "gates": gates,
        "seed_visible_summaries": visible,
        "seed_posthoc_summaries": posthoc,
        "evidence_sha256": evidence,
        "protocol_consequence": (
            "Part B is authorized only because all preregistered feasibility gates passed."
            if status == "SPARC_FEASIBILITY_SUPPORTED"
            else "Protocol hard stop before SPARC method/config implementation or training."
        ),
    }
    write_json(outputs[5], payload)
    lines = [
        "# SPARC-Seg V0.1 feasibility audit",
        "",
        f"**Final status:** `{status}`  ",
        "**Optimizer steps:** `0`  ",
        "**Hidden-GT training usage:** `none`  ",
        "**Hidden-GT analysis usage:** `independent post-hoc only`",
        "",
        "## Gate summary",
        "",
        "| Gate | Passed | Key evidence |",
        "|---|---:|---|",
        f"| A1 current PAS | `{a1['passed']}` | Per-class details in JSON and `prototype_validation_quality.csv` |",
        f"| A2 stable/plastic | `{a2['passed']}` | Per-class details in JSON and `partition_quality.csv` |",
        f"| A3 spatial coverage | `{a3['passed']}` | 3-pixel processed boundary band, per-class ratios in JSON |",
        f"| B feature separation | `{feature['passed']}` | median cosine gap `{feature['median_stable_minus_plastic_cosine']}`; distance-order fraction `{feature['stable_distance_le_plastic_fraction']}` |",
        f"| B gradient scale/localization | `{gradient['passed']}` | median `{gradient['median_ratio']}`; p10 `{gradient['p10_ratio']}`; p90 `{gradient['p90_ratio']}` |",
        f"| C1 previous utility | `{virtual['previous_gate_passed']}` | paired-better fraction `{virtual['paired_s3_better_than_s1_previous_fraction']}` |",
        f"| C2 current safety | `{virtual['current_safety_gate_passed']}` | variant medians in JSON |",
        f"| C3/C4 targeting and complementarity | `{virtual['targeting_and_complementarity_gate_passed']}` | S3 versus S1/S2/S4/S5 medians in JSON |",
        "",
        "## Protocol boundary",
        "",
        "- Prototypes used all and only current-site `train_labeled` cases, with per-case normalization, equal case weighting, final normalization, and the frozen 32-cell minimum.",
        "- The visible audit used 32 fixed clean current-unlabeled update batches plus 16 previous/current validation batches for each of six seed-transition pairs.",
        "- Hidden labels were resolved only by separate post-hoc process invocations after visible artifacts were frozen; they were used only for mask/feature quality metrics.",
        "- Uniform relation KD, R0 pseudo targets, frozen checkpoints, data, manifests, and splits were not modified.",
        "- No optimizer step, SPARC method registration, or SPARC training configuration occurred.",
        "",
        "## Consequence",
        "",
        payload["protocol_consequence"],
        "",
    ]
    write_text(outputs[6], "\n".join(lines))
    print(json.dumps({"status": status, "json": str(outputs[5]), "markdown": str(outputs[6])}, indent=2))
    return 0 if status == "SPARC_FEASIBILITY_SUPPORTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
