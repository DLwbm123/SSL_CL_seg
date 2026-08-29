#!/usr/bin/env python3
"""Read-only unified trajectory audit of the completed SR-GAS V0.1a pilots."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


RUN_NAMES = {
    "A1": "fundus_seed0_srgas_a1_pilot1000",
    "A2": "fundus_seed0_srgas_a2_pilot1000",
    "A3": "fundus_seed0_srgas_a3_pilot1000",
    "A4": "fundus_seed0_srgas_a4_pilot1000",
    "A5": "fundus_seed0_srgas_a5_pilot1000",
}
SITE_COLUMNS = {
    "REFUGE": "refuge_mean_foreground_dice",
    "RIM_ONE_r3": "rim_one_mean_foreground_dice",
}
DIAGNOSTIC_COLUMNS = (
    "perturbation_l2_ratio",
    "sensitivity_p10",
    "sensitivity_p50",
    "sensitivity_p90",
    "noise_scale_p10",
    "noise_scale_p50",
    "noise_scale_p90",
    "sensitivity_cosine",
    "a5_a4_noise_scale_l1",
    "classifier_angular_drift",
    "gradient_cosine_ssl_stable",
)


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float | None:
    if value in (None, "", "unavailable"):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _display(value: float | None) -> float | str:
    return "unavailable" if value is None else value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty audit table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalized_auc(points: list[tuple[int, float]]) -> float:
    if len(points) < 2:
        raise ValueError("trajectory AUC requires at least two evaluations")
    area = sum(
        (right_step - left_step) * (left_value + right_value) * 0.5
        for (left_step, left_value), (right_step, right_value) in zip(points, points[1:])
    )
    return area / float(points[-1][0] - points[0][0])


def audit(run_root: Path, output_dir: Path) -> dict[str, Any]:
    trajectories: dict[str, list[dict[str, str]]] = {}
    train_rows: dict[str, dict[int, dict[str, str]]] = {}
    for variant, name in RUN_NAMES.items():
        run = run_root / name
        trajectories[variant] = _rows(run / "pilot_trajectory.csv")
        train_rows[variant] = {int(row["site_step"]): row for row in _rows(run / "train_log.csv")}
        steps = [int(row["site_step"]) for row in trajectories[variant]]
        if steps != list(range(50, 1001, 50)):
            raise RuntimeError(f"{variant} trajectory is not the frozen 50-step grid: {steps}")

    baseline = {
        (int(row["site_step"]), site): float(row[column])
        for row in trajectories["A1"]
        for site, column in SITE_COLUMNS.items()
    }
    trajectory_output: list[dict[str, Any]] = []
    diagnostic_output: list[dict[str, Any]] = []
    recovery_output: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"status": "SRGAS_V0_1A_TRAJECTORY_AUDITED", "variants": {}}

    for variant, rows in trajectories.items():
        summary["variants"][variant] = {}
        for site, column in SITE_COLUMNS.items():
            points = [(int(row["site_step"]), float(row[column])) for row in rows]
            deltas = [(step, value - baseline[(step, site)]) for step, value in points]
            worst_step, worst_delta = min(deltas, key=lambda item: item[1])
            auc = _normalized_auc(points)
            base_auc = _normalized_auc([(step, baseline[(step, site)]) for step, _ in points])
            recovery_candidates = [step for step, delta in deltas if step >= worst_step and delta >= -0.015]
            recovery_time = recovery_candidates[0] - worst_step if recovery_candidates else None
            summary["variants"][variant][site] = {
                "worst_step": worst_step,
                "worst_drop": max(0.0, -worst_delta),
                "normalized_auc": auc,
                "normalized_auc_delta_vs_a1": auc - base_auc,
                "recovery_time_to_original_gate": _display(recovery_time),
            }
            recovery_output.append({"variant": variant, "site": site, **summary["variants"][variant][site]})
            for step, value in points:
                trajectory_output.append(
                    {
                        "variant": variant,
                        "site": site,
                        "eval_step": step,
                        "dice": value,
                        "delta_vs_a1": value - baseline[(step, site)],
                        "worst_drop": max(0.0, -worst_delta),
                        "normalized_auc": auc,
                        "recovery_time_to_original_gate": _display(recovery_time),
                    }
                )
        for trajectory_row in rows:
            step = int(trajectory_row["site_step"])
            source = train_rows[variant].get(step, {})
            diagnostic_output.append(
                {
                    "variant": variant,
                    "eval_step": step,
                    **{column: _display(_float(source.get(column))) for column in DIAGNOSTIC_COLUMNS},
                }
            )

    _write_csv(output_dir / "v01a_trajectory_metrics.csv", trajectory_output)
    _write_csv(output_dir / "v01a_noise_sensitivity_trajectory.csv", diagnostic_output)
    _write_csv(output_dir / "v01a_recovery_analysis.csv", recovery_output)
    summary["unavailable_policy"] = "Metrics absent from frozen V0.1a logs are recorded as unavailable; no run was repeated."
    (output_dir / "v01a_trajectory_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/jiangsuiyang/SSL_CL/runs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports/analysis/srgas_v0_2",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.run_root, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
