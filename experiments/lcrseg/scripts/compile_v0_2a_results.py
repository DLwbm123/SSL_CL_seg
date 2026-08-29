#!/usr/bin/env python3
"""Compile preregistered V0.2a Fundus tables and evaluate the final gate."""
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

from lcrseg.common import write_csv, write_json, write_text


RUN_NAMES = {
    "R0": "fundus_seed0_lcrseg_uniform_relation_kd_full200e",
    "R1": "fundus_seed0_lcrseg_v0_2a_r1_progressive_uniform_full200e",
    "R2": "fundus_seed0_lcrseg_v0_2a_r2_legacy_teacherreject_full200e",
    "R3": "fundus_seed0_lcrseg_v0_2a_r3_progressive_teacherreject_full200e",
    "U0": "fundus_seed0_lcrseg_v0_2_r0_uniform_full200e",
}
METRICS = ("final_average_dice", "bwt", "incoming_dice", "previous_site_dice")


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return list(csv.DictReader(path.open()))


def _finite_training(run_dir: Path) -> tuple[bool, int, int, float, float]:
    rows = _csv(run_dir / "train_log.csv")
    required = ("loss_total", "loss_sup", "loss_assim", "loss_relation")
    finite = all(all(math.isfinite(float(row[key])) for key in required) for row in rows)
    hidden = max((int(float(row.get("hidden_gt_training_usage", 0))) for row in rows), default=0)
    skipped = sum(int(float(row.get("optimizer_step_skipped", 0))) for row in rows)
    rejection = max((float(row.get("teacher_rejected_fraction", 0)) for row in rows), default=0.0)
    ess_ratios = [
        float(row["relation_effective_sample_size"]) / float(row["relation_valid_count"])
        for row in rows
        if row.get("relation_effective_sample_size") not in (None, "")
        and float(row.get("relation_valid_count", 0)) > 0
    ]
    return finite, hidden, skipped, rejection, min(ess_ratios, default=1.0)


def _summary(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "run_summary.json").read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete formal run: {run_dir}")
    return payload


def _main_and_factorial(run_dirs: dict[str, Path], output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, float]] = {}
    for variant, run_dir in run_dirs.items():
        payload = _summary(run_dir)
        values = {key: float(payload["summary"][key]) for key in METRICS}
        summaries[variant] = values
        main_rows.append(
            {
                "variant": variant,
                "formal_status": "auxiliary_not_formal_r0" if variant == "U0" else "formal",
                "run_dir": str(run_dir),
                "completed_global_steps": int(payload["completed_global_steps"]),
                **values,
            }
        )
    factorial_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        r0, r1, r2, r3 = (summaries[key][metric] for key in ("R0", "R1", "R2", "R3"))
        factorial_rows.append(
            {
                "metric": metric,
                "R0_A0_C0": r0,
                "R1_A1_C0": r1,
                "R2_A0_C1": r2,
                "R3_A1_C1": r3,
                "assimilation_main_effect": ((r1 + r3) - (r0 + r2)) / 2.0,
                "consolidation_main_effect": ((r2 + r3) - (r0 + r1)) / 2.0,
                "interaction": r3 - r2 - r1 + r0,
            }
        )
    write_csv(output_dir / "main_results.csv", main_rows)
    write_csv(output_dir / "factorial_effects.csv", factorial_rows)
    return main_rows, factorial_rows


def _matrices_and_classwise(run_dirs: dict[str, Path], output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    matrix_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    forgetting_rows: list[dict[str, Any]] = []
    for variant, run_dir in run_dirs.items():
        rows = _csv(run_dir / "site_matrix_long.csv")
        for row in rows:
            matrix_rows.append({"variant": variant, "run_dir": str(run_dir), **row})
            for class_id in (1, 2):
                class_rows.append(
                    {
                        "variant": variant,
                        "trained_site": row["trained_site"],
                        "trained_site_index": row["trained_site_index"],
                        "evaluation_site": row["evaluation_site"],
                        "class_id": class_id,
                        "dice": row[f"dice_class_{class_id}"],
                        "asd": row[f"asd_class_{class_id}"],
                        "hd95": row[f"hd95_class_{class_id}"],
                    }
                )
        final_index = max(int(row["trained_site_index"]) for row in rows)
        final_rows = {row["evaluation_site"]: row for row in rows if int(row["trained_site_index"]) == final_index}
        diagonal = {row["trained_site"]: row for row in rows if row["trained_site"] == row["evaluation_site"]}
        for site, start in diagonal.items():
            if site not in final_rows:
                continue
            forgetting_rows.append(
                {
                    "variant": variant,
                    "site": site,
                    "diagonal_dice": start["mean_foreground_dice"],
                    "final_dice": final_rows[site]["mean_foreground_dice"],
                    "forgetting_final_minus_diagonal": float(final_rows[site]["mean_foreground_dice"])
                    - float(start["mean_foreground_dice"]),
                }
            )
    write_csv(output_dir / "site_matrices.csv", matrix_rows)
    write_csv(output_dir / "classwise_results.csv", class_rows)
    write_csv(output_dir / "per_site_forgetting.csv", forgetting_rows)
    return matrix_rows, class_rows, forgetting_rows


def _admission(run_dirs: dict[str, Path], output_dir: Path) -> list[dict[str, Any]]:
    posthoc_path = output_dir / "posthoc_classwise_metrics.csv"
    posthoc = {}
    if posthoc_path.is_file():
        posthoc = {
            (row["variant"], row["site"], int(row["class_id"])): row
            for row in _csv(posthoc_path)
        }
    grouped: dict[tuple[str, str, int, int, int], dict[str, float]] = defaultdict(
        lambda: {"valid": 0.0, "admitted": 0.0, "deferred": 0.0, "target_weighted": 0.0}
    )
    for variant in ("R1", "R3"):
        for row in _csv(run_dirs[variant] / "branch_coverage.csv"):
            key = (variant, row["site"], int(row["site_index"]), int(row["epoch"]), int(row["predicted_class"]))
            valid = int(row["valid_count"])
            grouped[key]["valid"] += valid
            grouped[key]["admitted"] += int(row["admitted_count"])
            grouped[key]["deferred"] += int(row["deferred_count"])
            grouped[key]["target_weighted"] += float(row["target_fraction"]) * valid
    rows: list[dict[str, Any]] = []
    for (variant, site, site_index, epoch, class_id), values in sorted(grouped.items()):
        valid = values["valid"]
        diagnostic = posthoc.get((variant, site, class_id), {})
        rows.append(
            {
                "variant": variant,
                "site": site,
                "site_index": site_index,
                "epoch": epoch,
                "class_id": class_id,
                "target_coverage": values["target_weighted"] / valid if valid else "",
                "realized_coverage": values["admitted"] / valid if valid else "",
                "valid_pseudo_count": int(valid),
                "admitted_pixel_count": int(values["admitted"]),
                "deferred_pixel_count": int(values["deferred"]),
                "posthoc_final_pseudo_label_accuracy": diagnostic.get("pseudo_label_accuracy", ""),
                "posthoc_final_boundary_interior_admitted_ratio": diagnostic.get("boundary_interior_admitted_ratio", ""),
                "hidden_gt_scope": "post_hoc_final_checkpoint_only" if diagnostic else "not_yet_available",
            }
        )
    write_csv(output_dir / "admission_coverage.csv", rows)
    return rows


def _teacher_and_ess(run_dirs: dict[str, Path], output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    teacher_rows: list[dict[str, Any]] = []
    teacher_json: list[dict[str, Any]] = []
    ess_grouped: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"ratio": [], "rejection": []}
    )
    for variant in ("R2", "R3"):
        run_dir = run_dirs[variant]
        for path in sorted(run_dir.glob("site_*/teacher_validity_calibrator.csv")):
            site = path.parent.name.removeprefix("site_")
            teacher_rows.extend({"variant": variant, "site": site, **row} for row in _csv(path))
            state_path = path.with_suffix(".json")
            teacher_json.append({"variant": variant, "site": site, "state": json.loads(state_path.read_text())})
        for row in _csv(run_dir / "train_log.csv"):
            key = (variant, row["site_id"], int(row["epoch"]))
            valid = float(row.get("relation_valid_count", 0))
            if valid > 0:
                ess_grouped[key]["ratio"].append(float(row["relation_effective_sample_size"]) / valid)
                ess_grouped[key]["rejection"].append(float(row["teacher_rejected_fraction"]))
    ess_rows = [
        {
            "variant": variant,
            "site": site,
            "epoch": epoch,
            "steps": len(values["ratio"]),
            "relation_ess_ratio_mean": sum(values["ratio"]) / len(values["ratio"]),
            "relation_ess_ratio_min": min(values["ratio"]),
            "rejected_fraction_mean": sum(values["rejection"]) / len(values["rejection"]),
            "rejected_fraction_max": max(values["rejection"]),
        }
        for (variant, site, epoch), values in sorted(ess_grouped.items())
        if values["ratio"]
    ]
    write_csv(output_dir / "teacher_validity_calibrator_tables.csv", teacher_rows)
    posthoc_path = output_dir / "teacher_validity_posthoc.csv"
    posthoc_rows = _csv(posthoc_path) if posthoc_path.is_file() else []
    write_csv(output_dir / "teacher_validity_calibration.csv", posthoc_rows if posthoc_rows else teacher_rows)
    posthoc_summary_path = output_dir / "posthoc_teacher_metrics.json"
    write_json(
        output_dir / "teacher_validity_calibration.json",
        {
            "calibrators": teacher_json,
            "posthoc_summary": json.loads(posthoc_summary_path.read_text()) if posthoc_summary_path.is_file() else {},
        },
    )
    write_csv(output_dir / "relation_effective_sample_size.csv", ess_rows)
    return teacher_rows, ess_rows


def _gradients(run_dirs: dict[str, Path], output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in ("R1", "R2", "R3"):
        path = run_dirs[variant] / "analysis/gradient_cosine.csv"
        if path.is_file():
            rows.extend({"variant": variant, **row} for row in _csv(path))
    write_csv(output_dir / "gradient_diagnostics.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/analysis/v0_2a")
    parser.add_argument("--completion-json", type=Path, default=PROJECT_ROOT / "reports/experiment_status/V0_2A_FUNDUS_COMPLETION.json")
    parser.add_argument("--completion-md", type=Path, default=PROJECT_ROOT / "reports/experiment_status/V0_2A_FUNDUS_COMPLETION.md")
    args = parser.parse_args()
    run_dirs = {variant: (args.run_root / name).resolve() for variant, name in RUN_NAMES.items()}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    main_rows, factorial_rows = _main_and_factorial(run_dirs, output_dir)
    matrix_rows, class_rows, forgetting_rows = _matrices_and_classwise(run_dirs, output_dir)
    admission_rows = _admission(run_dirs, output_dir)
    teacher_rows, ess_rows = _teacher_and_ess(run_dirs, output_dir)
    gradient_rows = _gradients(run_dirs, output_dir)
    bridge = json.loads((PROJECT_ROOT / "reports/experiment_status/V0_2A_R0_BRIDGE_REPORT.json").read_text())
    pilot = json.loads((PROJECT_ROOT / "reports/experiment_status/V0_2A_PILOT_REPORT.json").read_text())
    audits = {}
    for variant in ("R1", "R2", "R3"):
        finite, hidden, skipped, rejection, ess_min = _finite_training(run_dirs[variant])
        audits[variant] = {
            "finite": finite,
            "hidden_gt_training_usage_max": hidden,
            "optimizer_steps_skipped": skipped,
            "maximum_rejection_fraction": rejection,
            "minimum_relation_ess_ratio": ess_min,
        }
    engineering_passed = bool(bridge.get("passed")) and bool(pilot.get("passed")) and all(
        audit["finite"]
        and audit["hidden_gt_training_usage_max"] == 0
        and audit["optimizer_steps_skipped"] == 0
        and audit["maximum_rejection_fraction"] <= 0.20 + 1.0e-12
        and audit["minimum_relation_ess_ratio"] >= 0.80 - 1.0e-12
        for audit in audits.values()
    )
    values = {row["variant"]: row for row in main_rows}
    r0, r3 = values["R0"], values["R3"]
    numeric_thresholds = {
        "final": r3["final_average_dice"] >= r0["final_average_dice"] + 0.003,
        "bwt": r3["bwt"] >= r0["bwt"] + 0.005,
        "incoming": r3["incoming_dice"] >= r0["incoming_dice"] - 0.010,
        "previous": r3["previous_site_dice"] >= r0["previous_site_dice"],
    }
    pareto = not any(
        other["variant"] != "R3"
        and all(float(other[key]) >= float(r3[key]) for key in METRICS)
        and any(float(other[key]) > float(r3[key]) for key in METRICS)
        for other in main_rows
        if other["variant"] != "U0"
    )
    posthoc_path = output_dir / "posthoc_teacher_metrics.json"
    posthoc = json.loads(posthoc_path.read_text()) if posthoc_path.is_file() else {}
    retained_better = posthoc.get("R3_retained_old_correctness_gt_rejected")
    foreground_sacrifice_passed = posthoc.get("R3_no_foreground_class_drop_over_0_01")
    research_evaluated = retained_better is not None and foreground_sacrifice_passed is not None
    research_passed = (
        research_evaluated
        and all(numeric_thresholds.values())
        and pareto
        and bool(retained_better)
        and bool(foreground_sacrifice_passed)
        and audits["R3"]["minimum_relation_ess_ratio"] >= 0.80
    )
    status = (
        "FUNDUS_V0_2A_GATE_PASSED"
        if engineering_passed and research_passed
        else "POSTHOC_REQUIRED"
        if engineering_passed and not research_evaluated
        else "FUNDUS_V0_2A_GATE_FAILED"
    )
    completion = {
        "protocol_id": "lcrseg_v0_2a",
        "status": status,
        "engineering_gate_passed": engineering_passed,
        "research_gate_evaluated": research_evaluated,
        "research_gate_passed": research_passed,
        "numeric_thresholds": numeric_thresholds,
        "pareto_non_dominated": pareto,
        "retained_better_than_rejected": retained_better,
        "no_foreground_class_drop_over_0_01": foreground_sacrifice_passed,
        "audits": audits,
        "run_directories": {key: str(value) for key, value in run_dirs.items()},
        "artifact_counts": {
            "main_results": len(main_rows),
            "factorial_effects": len(factorial_rows),
            "site_matrix_rows": len(matrix_rows),
            "classwise_rows": len(class_rows),
            "forgetting_rows": len(forgetting_rows),
            "admission_rows": len(admission_rows),
            "teacher_calibration_rows": len(teacher_rows),
            "ess_rows": len(ess_rows),
            "gradient_rows": len(gradient_rows),
        },
        "prostate_allowed": bool(engineering_passed and research_passed),
    }
    write_json(args.completion_json, completion)
    write_text(
        args.completion_md,
        "\n".join(
            [
                "# LCR-Seg V0.2a Fundus completion",
                "",
                f"**Status:** `{status}`",
                "",
                f"- Engineering gate: `{engineering_passed}`",
                f"- Research gate evaluated: `{research_evaluated}`",
                f"- Research gate passed: `{research_passed}`",
                f"- R3 Pareto non-dominated: `{pareto}`",
                f"- Prostate pilot allowed: `{completion['prostate_allowed']}`",
                "",
                "No Prostate run is permitted unless status is `FUNDUS_V0_2A_GATE_PASSED`.",
                "",
            ]
        ),
    )
    print(json.dumps(completion, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
