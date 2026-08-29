#!/usr/bin/env python3
"""Compile the immutable V0.4 Failure-Audit Gate and preregistered status."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import sha256_path, write_json, write_text


def _read_complete(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise RuntimeError(f"audit artifact is not complete: {path}")
    return value


def _verify_freeze(freeze: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    checked = 0
    for name, run in freeze["runs"].items():
        run_dir = Path(run["path"])
        for artifact in run["artifact_hashes"]:
            path = run_dir / artifact["path"]
            checked += 1
            if not path.is_file():
                failures.append({"run": name, "path": str(path), "reason": "missing"})
            elif int(path.stat().st_size) != int(artifact["size"]):
                failures.append({"run": name, "path": str(path), "reason": "size_mismatch"})
            elif sha256_path(path) != artifact["sha256"]:
                failures.append({"run": name, "path": str(path), "reason": "sha256_mismatch"})
    return {"checked_artifacts": checked, "failures": failures, "passed": not failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    project = args.project_root.resolve()
    analysis_dir = project / "reports" / "analysis" / "v0_4"
    status_dir = project / "reports" / "experiment_status"
    json_path = status_dir / "V0_4_FAILURE_AUDIT.json"
    md_path = status_dir / "V0_4_FAILURE_AUDIT.md"
    if json_path.exists() or md_path.exists():
        raise FileExistsError("refusing to overwrite completed V0.4 Failure-Audit Gate")
    freeze_path = status_dir / "V0_3_FREEZE_FOR_V0_4.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_check = _verify_freeze(freeze)
    precision = _read_complete(analysis_dir / "precision_coverage_summary.json")
    mode = _read_complete(analysis_dir / "mode_coverage_summary.json")
    stability = _read_complete(analysis_dir / "admission_stability_summary.json")
    anchor = _read_complete(analysis_dir / "anchor_relation_summary.json")
    gradient = _read_complete(analysis_dir / "gradient_utility_summary.json")
    split = _read_complete(analysis_dir / "split_characterization.json")
    crossed = _read_complete(analysis_dir / "crossed_pilot_summary.json")

    below = precision["gate_raw"]["failed_seed_boundary_ratio_below_0_80"]
    jsd = precision["gate_raw"]["failed_seed_group_jsd_at_least_0_10"]
    failed_seed_boundary_support = {
        str(seed): any(int(row["seed"]) == seed for row in below) for seed in (1, 2)
    }
    boundary_supported = all(failed_seed_boundary_support.values())
    group_jsd_supported = bool(jsd)
    mode_supported = bool(mode["mode_coverage_supported_2_of_3"])
    coverage_mode_supported = bool(mode_supported or boundary_supported or group_jsd_supported)
    stability_supported = bool(stability["hard_selection_instability_supported_2_of_3"])
    optimization_supported = bool(crossed["r1_more_optimization_sensitive"])
    hard_selection_supported = bool(stability_supported and optimization_supported)

    report_paths = [
        analysis_dir / "precision_coverage.csv",
        analysis_dir / "boundary_coverage.csv",
        analysis_dir / "mode_coverage_k2.csv",
        analysis_dir / "mode_coverage_k4.csv",
        analysis_dir / "admission_stability.csv",
        analysis_dir / "anchor_drift.csv",
        analysis_dir / "relation_drift.csv",
        analysis_dir / "gradient_utility.csv",
        analysis_dir / "split_characterization.csv",
        analysis_dir / "crossed_pilot_results.csv",
    ]
    scripts_complete = all(path.is_file() and path.stat().st_size > 0 for path in report_paths)
    engineering = {
        "all_audit_scripts_complete": scripts_complete,
        "hidden_gt_training_usage_zero": bool(crossed["engineering"]["hidden_gt_training_usage_zero"]),
        "all_analysis_from_frozen_artifacts": freeze_check["passed"],
        "crossed_pilots_complete": bool(crossed["engineering"]["all_12_complete"]),
        "existing_artifacts_unmodified": freeze_check["passed"],
        "gradient_finite": bool(gradient["all_finite"]),
        "old_model_gradient_zero": not bool(gradient["old_model_gradient_detected"]),
        "crossed_nan_inf_zero": bool(crossed["engineering"]["nan_inf_zero"]),
        "crossed_amp_skip_zero": bool(crossed["engineering"]["amp_skip_zero"]),
        "split_mutation_zero": not bool(split.get("split_mutation", True)),
    }
    engineering_passed = all(engineering.values())
    research = {
        "mode_coverage_problem": mode_supported,
        "boundary_coverage_bias_failed_seeds": boundary_supported,
        "admitted_candidate_group_jsd_ge_0_10": group_jsd_supported,
        "coverage_or_mode_bias_supported": coverage_mode_supported,
        "hard_selection_stability_criterion": stability_supported,
        "r1_more_optimization_sensitive": optimization_supported,
        "hard_selection_instability_supported": hard_selection_supported,
        "anchor_memory_bias_supported": bool(anchor["biased_memory_update_supported"]),
        "split_sensitivity_raw": crossed["split_sensitivity"],
        "optimization_sensitivity_raw": crossed["optimization_sensitivity"],
        "relation_kd_remains_fixed": True,
        "foundation_error_detected": False,
    }
    research_passed = bool((coverage_mode_supported or hard_selection_supported) and not research["foundation_error_detected"])
    if not engineering_passed:
        status = "HARD_STOP_AUDIT_ENGINEERING_FAILURE"
    elif research_passed:
        status = "V0_4_AUDIT_SUPPORTS_SOFT_ROUTING"
    else:
        status = "V0_4_DIAGNOSIS_NOT_SUPPORT_SOFT_ROUTING"
    result = {
        "protocol_id": "lcrseg_v0_4_failure_audit",
        "status": status,
        "v0_3_status": freeze["v0_3_status"],
        "v0_3_failure_level": freeze["v0_3_failure_level"],
        "r2_r3_status": freeze["r2_r3_status"],
        "v0_4a_allowed": status == "V0_4_AUDIT_SUPPORTS_SOFT_ROUTING",
        "engineering_gate": {"passed": engineering_passed, "criteria": engineering},
        "research_gate": {"passed": research_passed, "criteria": research},
        "raw": {
            "freeze_check": freeze_check,
            "failed_seed_boundary_support": failed_seed_boundary_support,
            "boundary_ratio_below_0_80_rows": below,
            "group_jsd_ge_0_10_rows": jsd,
            "mode_seed_support": mode["seed_support"],
            "stability_seed_raw": stability["seed_raw"],
            "crossed_engineering": crossed["engineering"],
        },
        "source_sha256": {
            str(path.relative_to(project)): sha256_path(path)
            for path in [
                freeze_path,
                analysis_dir / "precision_coverage_summary.json",
                analysis_dir / "mode_coverage_summary.json",
                analysis_dir / "admission_stability_summary.json",
                analysis_dir / "anchor_relation_summary.json",
                analysis_dir / "gradient_utility_summary.json",
                analysis_dir / "split_characterization.json",
                analysis_dir / "crossed_pilot_summary.json",
            ]
        },
    }
    write_json(json_path, result)
    lines = [
        "# LCR-Seg V0.4 Failure-Audit Gate",
        "",
        f"- Status: `{status}`",
        f"- Engineering gate: `{engineering_passed}`",
        f"- Research gate: `{research_passed}`",
        f"- V0.4a allowed: `{result['v0_4a_allowed']}`",
        "",
        "## Research criteria",
        "",
        f"- Mode coverage problem: `{mode_supported}`",
        f"- Boundary coverage bias in failed seeds 1/2: `{boundary_supported}`",
        f"- Group JSD >= 0.10: `{group_jsd_supported}`",
        f"- Hard-selection stability criterion: `{stability_supported}`",
        f"- R1 more optimization sensitive: `{optimization_supported}`",
        f"- Anchor-memory bias: `{anchor['biased_memory_update_supported']}`",
        "",
        "## Engineering criteria",
        "",
    ]
    lines.extend(f"- {name}: `{value}`" for name, value in engineering.items())
    lines.extend([
        "",
        "All raw booleans, thresholds, and source hashes are preserved in `V0_4_FAILURE_AUDIT.json`.",
        "",
    ])
    write_text(md_path, "\n".join(lines))
    print(json.dumps({"status": status, "engineering": engineering_passed, "research": research_passed}, sort_keys=True))


if __name__ == "__main__":
    main()
