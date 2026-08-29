#!/usr/bin/env python3
"""Evaluate the preregistered Fundus LCR-Seg V0.2 R0-R3 research gate."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import write_json


VARIANTS = {
    "r0": ("fundus_seed0_lcrseg_v0_2_r0_uniform_full200e", (False, False, False)),
    "r1": ("fundus_seed0_lcrseg_v0_2_r1_learnability_full200e", (True, False, False)),
    "r2": ("fundus_seed0_lcrseg_v0_2_r2_compatibility_full200e", (False, True, True)),
    "r3": ("fundus_seed0_lcrseg_v0_2_r3_asymmetric_full200e", (True, True, True)),
}
METRICS = ("final_average_dice", "bwt", "incoming_dice", "previous_site_dice")


def _condition(passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **evidence}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _list(value: Any) -> list[float] | None:
    if isinstance(value, list):
        candidate = value
    elif isinstance(value, str) and value.strip():
        try:
            candidate = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    else:
        return None
    if not isinstance(candidate, (list, tuple)):
        return None
    result: list[float] = []
    for item in candidate:
        number = _float(item)
        if number is None:
            return None
        result.append(number)
    return result


def _run_evidence(run_dir: Path, variant: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    run_dir = Path(run_dir).resolve()
    expected_name, expected_flags = VARIANTS[variant]
    try:
        summary = _read_json(run_dir / "run_summary.json")
        config = _read_json(run_dir / "config.yaml")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return {"run_dir": str(run_dir)}, [str(exc)]
    metrics = {name: _float((summary.get("summary") or {}).get(name)) for name in METRICS}
    if run_dir.name != expected_name:
        errors.append(f"run directory name mismatch: expected {expected_name}, got {run_dir.name}")
    if summary.get("status") != "complete":
        errors.append("run_summary status is not complete")
    if summary.get("method") != "lcrseg_v0_2" or summary.get("dataset") != "fundus" or int(summary.get("seed", -1)) != 0:
        errors.append("run_summary does not declare the preregistered V0.2 Fundus seed-0 scope")
    if int(summary.get("completed_global_steps", -1)) != 13400:
        errors.append(f"completed_global_steps is {summary.get('completed_global_steps')}, expected 13400")
    method = dict(config.get("method") or {})
    actual_flags = tuple(bool(method.get(key)) for key in ("progressive_admission", "compatibility_calibration", "compatibility_rejection"))
    if method.get("name") != "lcrseg_v0_2" or str(method.get("version")) != "0.2" or actual_flags != expected_flags:
        errors.append(f"method identity or routing flags differ from {variant}: {actual_flags}")
    data = dict(config.get("data") or {})
    training = dict(config.get("training") or {})
    if data.get("evaluation_role") != "val" or data.get("site_order") != ["REFUGE", "RIM_ONE_r3", "Drishti_GS"]:
        errors.append("data scope/order is not the preregistered Fundus validation protocol")
    if int(training.get("epochs_per_site", -1)) != 200:
        errors.append("epochs_per_site is not 200")
    if not bool(data.get("require_readonly")):
        errors.append("frozen inputs were not required read-only")
    if any(value is None for value in metrics.values()):
        errors.append("one or more primary metrics are missing or non-finite")
    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "manifest_hash": summary.get("manifest_hash"),
        "split_hash": summary.get("split_hash"),
        "completed_global_steps": summary.get("completed_global_steps"),
    }, errors


def _admission_gate(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "train_log.csv"
    if not path.is_file():
        return _condition(False, reason="train_log.csv missing")
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    parse_errors: list[str] = []
    for row in _read_csv(path):
        progress = _float(row.get("site_progress"))
        candidates = _list(row.get("assim_candidate_counts_by_class"))
        fractions = _list(row.get("assim_selected_fraction_by_class"))
        if progress is None or candidates is None or fractions is None or len(candidates) != len(fractions):
            continue
        site = str(row.get("site_id") or "")
        for class_id, (candidate, fraction) in enumerate(zip(candidates, fractions, strict=True)):
            # A one- or two-pixel class necessarily rounds to 100%; the
            # preregistered persistent-coverage check is meaningful only once
            # a class has at least ten candidate relation-grid pixels.
            if candidate >= 10:
                grouped[(site, class_id)].append((progress, fraction))
    evidence: dict[str, Any] = {}
    passed = bool(grouped)
    for key, values in sorted(grouped.items()):
        early = [fraction for progress, fraction in values if progress <= 0.10]
        late = [fraction for progress, fraction in values if progress >= 0.90]
        fractions = [fraction for _, fraction in values]
        if not early or not late:
            parse_errors.append(f"{key} lacks a usable early or late admission sample")
            passed = False
            continue
        early_mean = float(sum(early) / len(early))
        late_mean = float(sum(late) / len(late))
        max_fraction = float(max(fractions))
        endpoint_ok = abs(early_mean - 0.4) <= 0.15 and abs(late_mean - 0.8) <= 0.15 and late_mean + 0.02 >= early_mean
        coverage_ok = max_fraction <= 0.90 + 1.0e-9
        passed = passed and endpoint_ok and coverage_ok
        evidence[f"{key[0]}:class{key[1]}"] = {
            "samples": len(values),
            "early_mean": early_mean,
            "late_mean": late_mean,
            "max_fraction": max_fraction,
            "endpoint_ok": endpoint_ok,
            "coverage_ok": coverage_ok,
        }
    return _condition(passed and not parse_errors, per_site_class=evidence, parse_errors=parse_errors)


def _calibration_gate(analysis_dir: Path, expected_sites: list[str], num_classes: int) -> dict[str, Any]:
    path = Path(analysis_dir) / "calibration_tables.csv"
    if not path.is_file():
        return _condition(False, reason="calibration_tables.csv missing")
    rows = _read_csv(path)
    by_mapping: dict[tuple[str, str, str, str], list[tuple[int, float]]] = defaultdict(list)
    fallbacks: set[tuple[str, str, str]] = set()
    for row in rows:
        site = str(row.get("site_id") or "")
        epoch = str(row.get("epoch") or "")
        scope = str(row.get("scope") or "")
        class_id = str(row.get("class_id") or "")
        if scope == "class_fallback_global":
            fallbacks.add((site, epoch, class_id))
            continue
        probability = _float(row.get("pava_probability"))
        index = _float(row.get("bin"))
        if probability is not None and index is not None:
            by_mapping[(site, epoch, scope, class_id)].append((int(index), probability))
    evidence: dict[str, Any] = {}
    passed = True
    for site in expected_sites[1:]:
        epochs = sorted({epoch for observed_site, epoch, scope, class_id in by_mapping if observed_site == site and scope == "global" and class_id == ""})
        if not epochs:
            evidence[site] = {"reason": "no global calibrator mapping"}
            passed = False
            continue
        site_evidence: dict[str, Any] = {}
        for epoch in epochs:
            global_values = [value for _, value in sorted(by_mapping[(site, epoch, "global", "")])]
            global_monotonic = all(right + 1.0e-12 >= left for left, right in zip(global_values, global_values[1:]))
            classes: dict[str, Any] = {}
            for class_id in range(num_classes):
                key = (site, epoch, "class", str(class_id))
                values = [value for _, value in sorted(by_mapping.get(key, []))]
                fallback = (site, epoch, str(class_id)) in fallbacks
                class_monotonic = all(right + 1.0e-12 >= left for left, right in zip(values, values[1:])) if values else fallback and global_monotonic
                classes[str(class_id)] = {"mapping_bins": len(values), "fallback_to_global": fallback, "monotonic": class_monotonic}
                passed = passed and class_monotonic
            site_evidence[epoch] = {"global_bins": len(global_values), "global_monotonic": global_monotonic, "classes": classes}
            passed = passed and global_monotonic
        evidence[site] = site_evidence
    return _condition(passed, mappings=evidence)


def _rejection_cap_gate(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "train_log.csv"
    if not path.is_file():
        return _condition(False, reason="train_log.csv missing")
    maximum = 0.0
    observed = 0
    for row in _read_csv(path):
        fractions = _list(row.get("compat_rejected_fraction_by_class"))
        candidates = _list(row.get("relation_valid_counts_by_class"))
        if fractions is None or candidates is None or len(fractions) != len(candidates):
            continue
        for fraction, candidate in zip(fractions, candidates, strict=True):
            if candidate > 0:
                maximum = max(maximum, fraction)
                observed += 1
    return _condition(observed > 0 and maximum <= 0.2 + 1.0e-9, observed_class_step_rows=observed, maximum_fraction=maximum)


def _ess_gate(r0_analysis: Path, r3_analysis: Path) -> dict[str, Any]:
    def global_consolidation(path: Path) -> dict[str, float]:
        rows = _read_csv(path / "effective_sample_size.csv")
        result: dict[str, float] = {}
        for row in rows:
            if row.get("route") == "consolidation" and row.get("scope") == "global":
                value = _float(row.get("effective_pixel_count"))
                if value is not None:
                    result[str(row.get("site_id") or "")] = value
        return result
    try:
        r0 = global_consolidation(Path(r0_analysis))
        r3 = global_consolidation(Path(r3_analysis))
    except FileNotFoundError as exc:
        return _condition(False, reason=str(exc))
    common = sorted(set(r0).intersection(r3))
    ratios = {site: r3[site] / r0[site] for site in common if r0[site] > 0.0}
    return _condition(bool(ratios) and all(value >= 0.70 for value in ratios.values()), ratios=ratios, required_minimum=0.70)


def _hidden_gt_gate(test_gate: Path, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not Path(test_gate).is_file():
        return _condition(False, reason="V0.2 test gate evidence is missing")
    try:
        evidence = _read_json(Path(test_gate))
    except (ValueError, json.JSONDecodeError) as exc:
        return _condition(False, reason=str(exc))
    configurations_clean = all("diagnostic" not in json.dumps(run, sort_keys=True).lower() for run in runs.values())
    passed = bool(evidence.get("all_tests_passed")) and bool(evidence.get("hidden_gt_leakage_tests_passed")) and configurations_clean
    return _condition(passed, test_gate=str(test_gate), configurations_clean=configurations_clean, test_evidence=evidence)


def _performance_conditions(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    r0, r1, r2, r3 = (metrics[key] for key in ("r0", "r1", "r2", "r3"))
    return {
        "r3_primary_thresholds": _condition(
            r3["final_average_dice"] >= 0.6551 and r3["bwt"] > -0.1185 and r3["incoming_dice"] >= 0.7241 and r3["previous_site_dice"] >= 0.6709,
            final=r3["final_average_dice"],
            bwt=r3["bwt"],
            incoming=r3["incoming_dice"],
            previous=r3["previous_site_dice"],
        ),
        "r1_not_worse_than_r0_on_both_final_and_incoming": _condition(
            not (r1["final_average_dice"] < r0["final_average_dice"] and r1["incoming_dice"] < r0["incoming_dice"]),
            r0=r0,
            r1=r1,
        ),
        "r2_bwt_or_previous_improves_with_incoming_tolerance": _condition(
            (r2["bwt"] > r0["bwt"] or r2["previous_site_dice"] > r0["previous_site_dice"]) and r2["incoming_dice"] >= r0["incoming_dice"] - 0.005,
            r0=r0,
            r2=r2,
        ),
        "r3_improves_two_of_final_bwt_previous_over_r0": _condition(
            sum(
                r3[name] > r0[name]
                for name in ("final_average_dice", "bwt", "previous_site_dice")
            ) >= 2,
            r0=r0,
            r3=r3,
            improved_count=sum(r3[name] > r0[name] for name in ("final_average_dice", "bwt", "previous_site_dice")),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in VARIANTS:
        parser.add_argument(f"--{key}", type=Path, required=True, help=f"completed {key.upper()} run directory")
    parser.add_argument("--analysis-r0", type=Path, default=PROJECT_ROOT / "reports" / "analysis" / "v0_2_r0")
    parser.add_argument("--analysis-r1", type=Path, default=PROJECT_ROOT / "reports" / "analysis" / "v0_2_r1")
    parser.add_argument("--analysis-r2", type=Path, default=PROJECT_ROOT / "reports" / "analysis" / "v0_2_r2")
    parser.add_argument("--analysis-r3", type=Path, default=PROJECT_ROOT / "reports" / "analysis" / "v0_2_r3")
    parser.add_argument("--test-gate", type=Path, default=PROJECT_ROOT / "reports" / "experiment_status" / "V0_2_TEST_GATE.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "experiment_status" / "LCRSEG_V0_2_GATE.json")
    args = parser.parse_args()

    run_evidence: dict[str, dict[str, Any]] = {}
    preflight_errors: dict[str, list[str]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        evidence, errors = _run_evidence(getattr(args, variant), variant)
        run_evidence[variant] = evidence
        if errors:
            preflight_errors[variant] = errors
            continue
        metrics[variant] = evidence["metrics"]
    conditions: dict[str, dict[str, Any]] = {}
    if not preflight_errors:
        conditions.update(_performance_conditions(metrics))
        r3_config = _read_json(Path(args.r3) / "config.yaml")
        expected_sites = list(r3_config["data"]["site_order"])
        num_classes = int(r3_config["model"]["num_classes"])
        conditions["classwise_progressive_admission"] = _admission_gate(args.r3)
        conditions["calibrated_classwise_correctness_non_decreasing"] = _calibration_gate(args.analysis_r3, expected_sites, num_classes)
        conditions["per_class_rejection_cap"] = _rejection_cap_gate(args.r3)
        conditions["consolidation_ess_at_least_70pct_of_r0"] = _ess_gate(args.analysis_r0, args.analysis_r3)
        conditions["no_hidden_gt_leakage"] = _hidden_gt_gate(args.test_gate, run_evidence)
    passed = not preflight_errors and bool(conditions) and all(item["passed"] for item in conditions.values())
    result = {
        "protocol": "LCR-Seg V0.2 Fundus R0-R3 preregistered gate",
        "run_evidence": run_evidence,
        "preflight_errors": preflight_errors,
        "conditions": conditions,
        "status": "passed" if passed else "FUNDUS_V0_2_RESEARCH_GATE_NOT_MET",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
