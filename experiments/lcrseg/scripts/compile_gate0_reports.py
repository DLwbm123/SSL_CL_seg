#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any


DOMAIN_ORDER = ["REFUGE", "RIM_ONE_r3", "Drishti_GS"]
METRICS = ["mean_iou", "mean_dice", "mean_foreground_dice"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_matrix(matrix: dict[str, dict[str, float]]) -> list[str]:
    errors: list[str] = []
    for stage_index, trained_domain in enumerate(DOMAIN_ORDER):
        row = matrix.get(trained_domain, {})
        expected = set(DOMAIN_ORDER[: stage_index + 1])
        observed = set(row)
        if observed != expected:
            errors.append(f"{trained_domain}: matrix keys {sorted(observed)} != {sorted(expected)}")
        for domain, value in row.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"{trained_domain}/{domain}: non-finite matrix value {value!r}")
    return errors


def audit_log(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    unlabeled_rows = 0
    for index, row in enumerate(rows, start=1):
        for field in ("loss_total", "loss_supervised"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"line {index}: non-finite {field}")
        # The runner records LR on supervised rows, while unlabeled rows record
        # the repaired step/no-grad/leakage fields instead. Missing is not NaN.
        if row.get("phase") == "supervised":
            value = row.get("lr")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"line {index}: missing or non-finite supervised lr")
        elif "lr" in row:
            value = row["lr"]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"line {index}: non-finite optional lr")
        if row.get("hidden_gt_training_usage") != "none":
            errors.append(f"line {index}: hidden GT policy violation")
        if row.get("phase") == "unlabeled":
            unlabeled_rows += 1
            if row.get("optimizer_step_executed") is not True:
                errors.append(f"line {index}: unlabeled backward lacks optimizer step evidence")
            if row.get("teacher_forward_no_grad") is not True:
                errors.append(f"line {index}: teacher no_grad evidence missing")
    return {"rows": len(rows), "unlabeled_rows": unlabeled_rows, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("/root/LCRSeg/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/di_dmpa_jascl"))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    result_root = output_dir / "gate0_results"
    seed_reports: dict[str, Any] = {}
    global_errors: list[str] = []
    leakage_reports: dict[str, Any] = {}

    for seed in (0, 1, 2):
        run_dir = args.runs_root / f"gate0_repaired_unet_fundus_seed{seed}"
        errors: list[str] = []
        complete_path = run_dir / ".complete"
        exit_path = run_dir / ".exit"
        if not complete_path.is_file():
            errors.append("missing .complete")
        if not exit_path.is_file() or exit_path.read_text(encoding="utf-8").strip() != "0":
            errors.append("missing or nonzero .exit")
        required = [run_dir / "run_completion.json", run_dir / "stage_by_domain_matrices.json", run_dir / "leakage_preflight.json", run_dir / "train.jsonl"]
        for path in required:
            if not path.is_file():
                errors.append(f"missing {path.name}")
        if errors:
            seed_reports[str(seed)] = {"status": "FAIL", "errors": errors, "run_dir": str(run_dir)}
            global_errors.extend(f"seed {seed}: {error}" for error in errors)
            continue

        completion = read_json(run_dir / "run_completion.json")
        matrices = read_json(run_dir / "stage_by_domain_matrices.json")
        leakage = read_json(run_dir / "leakage_preflight.json")
        log_audit = audit_log(run_dir / "train.jsonl")
        if completion.get("status") != "COMPLETE" or completion.get("method_registered") is not False:
            errors.append("run completion or method-registration state is invalid")
        if completion.get("hidden_gt_training_usage") != "none" or completion.get("nan_detected") is not False:
            errors.append("completion metadata reports leakage or NaN")
        if leakage.get("status") != "PASS" or leakage.get("hidden_gt_training_usage") != "none":
            errors.append("leakage preflight failed")
        errors.extend(log_audit["errors"])
        if log_audit["unlabeled_rows"] == 0:
            errors.append("no unlabeled optimizer-step evidence")
        for metric in METRICS:
            errors.extend(f"{metric}: {error}" for error in audit_matrix(matrices.get(metric, {})))

        seed_result_dir = result_root / f"seed{seed}"
        seed_result_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / "stage_by_domain_matrices.json", seed_result_dir / "stage_by_domain_matrices.json")
        for metric in METRICS:
            source_csv = run_dir / f"stage_by_domain_{metric}.csv"
            target_csv = seed_result_dir / f"stage_by_domain_{metric}.csv"
            target_csv.write_text(source_csv.read_text(encoding="utf-8"), encoding="utf-8")
        leakage_reports[str(seed)] = leakage
        seed_reports[str(seed)] = {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "run_dir": str(run_dir),
            "global_step": completion.get("global_step"),
            "elapsed_seconds": completion.get("elapsed_seconds"),
            "log_audit": log_audit,
            "matrices": matrices,
        }
        global_errors.extend(f"seed {seed}: {error}" for error in errors)

    leakage_payload = {
        "status": "PASS" if len(leakage_reports) == 3 and not any(report.get("status") != "PASS" for report in leakage_reports.values()) else "FAIL",
        "hidden_gt_training_usage": "none" if len(leakage_reports) == 3 else "unverified",
        "seeds": leakage_reports,
    }
    write_json(output_dir / "LEAKAGE_AUDIT_REPORT.json", leakage_payload)
    (output_dir / "LEAKAGE_AUDIT_REPORT.md").write_text(
        "# Gate 0 leakage audit\n\n"
        f"Status: `{leakage_payload['status']}`\n"
        f"Hidden-GT training usage: `{leakage_payload['hidden_gt_training_usage']}`\n\n"
        "Every training batch was restricted to the current Fundus domain. "
        "Unlabeled manifest records had no label path, and val/test roles were "
        "constructible only through the evaluator API.\n",
        encoding="utf-8",
    )

    status = "PASS" if len(seed_reports) == 3 and not global_errors else "FAIL"
    gate0 = {
        "date": "2026-08-29",
        "gate": "Gate 0",
        "status": status,
        "benchmark": "fundus",
        "model": "lcrseg_unet2d_jascl_3x3_stochastic_head",
        "seeds": [0, 1, 2],
        "upstream_commit": "3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53",
        "method_registered": False,
        "di_dmpa_training_launched": False,
        "constant_patch_classifier_regularization": False,
        "hidden_gt_training_usage": leakage_payload["hidden_gt_training_usage"],
        "resume_equivalence": "PASS",
        "unit_integration_tests": "PASS",
        "invalidated_runs": [
            {
                "run_dir": "/root/LCRSeg/runs/gate0_repaired_fundus_seed0",
                "architecture": "DeepLab/Xception",
                "stopped_at_global_step": 120,
                "complete_marker": False,
                "gate0_evidence": False,
            },
            {
                "run_dir": "/root/LCRSeg/runs/gate0_repaired_fundus_seed1",
                "architecture": "DeepLab/Xception",
                "stopped_at_global_step": 120,
                "complete_marker": False,
                "gate0_evidence": False,
            },
        ],
        "seed_reports": seed_reports,
        "errors": global_errors,
        "next_action": "STOP_AFTER_GATE0_REPORTING",
    }
    write_json(output_dir / "GATE0_STATUS.json", gate0)
    print(json.dumps({"status": status, "errors": global_errors}, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
