#!/usr/bin/env python3
"""Fail-closed compiler: reports are inputs, never fabricated PASS constants."""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from di_dmpa_jascl.config import load_yaml, resolved_config_hash, sha256_file
from di_dmpa_jascl.metrics import write_json
from di_dmpa_jascl.provenance import git_revision

OBJECTIVE = "probability_mse_on_joint_pas_validity"
METRICS = ("mean_iou", "mean_dice", "mean_foreground_dice")
REPORTS = ("UNIT_INTEGRATION_TEST_REPORT.json", "RESUME_EQUIVALENCE_REPORT.json",
           "PAS_GRADIENT_AUDIT.json", "LEAKAGE_AUDIT_REPORT.json", "EVAL_STOCHASTICITY_AUDIT.json")
STATE_GROUPS = ("student", "ema_teacher", "optimizer", "scheduler", "gas_state", "prototypes",
                "rng_state", "sampler_state", "stage_state", "evaluation_matrices", "best_metric",
                "deterministic_evaluation_output")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def audit_matrix(matrix, domains):
    errors = []
    if set(matrix) != set(domains):
        errors.append("matrix stages differ from frozen domains")
    for i, stage in enumerate(domains):
        row = matrix.get(stage, {})
        if set(row) != set(domains[:i+1]):
            errors.append(f"{stage}: incomplete lower-triangular matrix")
        if any(not finite(value) for value in row.values()):
            errors.append(f"{stage}: non-finite matrix")
    return errors


def audit_log(path, *, config_hash=None, git_commit=None, domains=None):
    errors, rows, unlabeled, coverage, gradients = [], [], 0, {}, {}
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            errors.append(f"line {index}: malformed JSON")
            continue
        rows.append(row)
        prefix = f"line {index}: "
        for key in ("loss_total", "loss_supervised"):
            if not finite(row.get(key)):
                errors.append(prefix + f"non-finite {key}")
        for key, expected in (("config_hash", config_hash), ("git_commit", git_commit)):
            if expected is not None and row.get(key) != expected:
                errors.append(prefix + f"{key} mismatch")
        if row.get("hidden_gt_training_usage") != "none":
            errors.append(prefix + "hidden GT violation")
        if row.get("objective_name") != OBJECTIVE:
            errors.append(prefix + "objective_name mismatch (legacy hard-label MSE forbidden)")
        if row.get("phase") == "supervised":
            if not finite(row.get("lr")):
                errors.append(prefix + "missing or non-finite supervised lr")
        elif row.get("phase") == "unlabeled":
            unlabeled += 1
            domain = row.get("domain")
            count = row.get("pas_joint_valid_pixels")
            grad = row.get("student_unsupervised_gradient_norm")
            if not finite(count) or count < 0:
                errors.append(prefix + "invalid PAS count")
                count = 0
            coverage[domain] = coverage.get(domain, 0) + count
            gradients[domain] = max(gradients.get(domain, 0), grad if finite(grad) else 0)
            for field in ("loss_consistency", "pas_joint_coverage", "student_unsupervised_gradient_norm",
                          "student_total_gradient_norm"):
                if not finite(row.get(field)):
                    errors.append(prefix + f"missing/non-finite {field}")
            if row.get("consistency_requires_grad") is not True:
                errors.append(prefix + "detached consistency")
            for field in ("teacher_forward_no_grad", "optimizer_step_executed", "stochastic_classifier_train_mode"):
                if row.get(field) is not True:
                    errors.append(prefix + f"{field} must be true")
            if row.get("teacher_nonnull_gradient_count") != 0 or row.get("prototype_requires_grad") is not False:
                errors.append(prefix + "teacher/prototype gradient leak")
        else:
            errors.append(prefix + "unknown phase")
    if not unlabeled:
        errors.append("no unlabeled objective evidence")
    for domain in domains or coverage:
        if coverage.get(domain, 0) <= 0:
            errors.append(f"{domain}: zero PAS coverage")
        if gradients.get(domain, 0) <= 1e-8:
            errors.append(f"{domain}: zero unlabeled gradient")
    if [row.get("global_step") for row in rows] != list(range(1, len(rows)+1)):
        errors.append("non-contiguous global-step trajectory")
    return {"rows": len(rows), "unlabeled_rows": unlabeled, "valid_pixels_by_domain": coverage,
            "max_gradient_by_domain": gradients, "errors": errors}


def validate_report(name, report, commit, hashes, domains):
    errors = []
    if report.get("status") != "PASS":
        errors.append(f"{name}: status is not PASS")
    if report.get("git_commit") != commit or report.get("config_hashes") != hashes:
        errors.append(f"{name}: missing/mismatched exact source/config provenance")
    if name == "UNIT_INTEGRATION_TEST_REPORT.json":
        if report.get("exit_code") != 0 or report.get("failed") != 0 or report.get("skipped") != 0 or report.get("passed", 0) < 14:
            errors.append("unit/integration report lacks a complete real passing test run")
        if not report.get("test_cases") or not report.get("transcript_sha256"):
            errors.append("unit/integration report lacks test cases/transcript")
    elif name == "RESUME_EQUIVALENCE_REPORT.json":
        cases = report.get("trajectories", {})
        for case in ("mid_supervised", "mid_unlabeled", "before_stage_transition", "after_stage_transition"):
            result = cases.get(case, {})
            groups = result.get("groups", {})
            if result.get("status") != "PASS" or not result.get("reference") or not result.get("candidate"):
                errors.append(f"resume trajectory missing/failed: {case}")
            for group in STATE_GROUPS:
                value = groups.get(group, {})
                if value.get("within_tolerance") is not True or not finite(value.get("max_abs_difference")):
                    errors.append(f"resume missing/failed state group {case}/{group}")
        if report.get("atol") != 1e-6 or report.get("rtol") != 1e-6:
            errors.append("resume tolerance drift")
    elif name == "PAS_GRADIENT_AUDIT.json":
        records = report.get("domains", {})
        if set(records) != set(domains) or report.get("data_kind") != "real_fundus_unlabeled":
            errors.append("real three-domain gradient audit is missing")
        for domain in domains:
            row = records.get(domain, {})
            if row.get("joint_valid_pixels", 0) <= 0 or row.get("student_unsupervised_gradient_norm", 0) <= 1e-8:
                errors.append(f"{domain}: zero PAS coverage/unlabeled gradient")
            if row.get("consistency_requires_grad") is not True or row.get("teacher_nonnull_gradient_count") != 0:
                errors.append(f"{domain}: detached consistency or teacher gradient")
            if row.get("prototype_requires_grad") is not False or row.get("hidden_gt_training_usage") != "none":
                errors.append(f"{domain}: prototype/hidden GT leak")
            if row.get("stochastic_classifier_train_mode") is not True or not row.get("exact_checkpoint"):
                errors.append(f"{domain}: missing classifier/checkpoint audit evidence")
            if row.get("config_hash") != hashes["B0"] or row.get("git_commit") != commit:
                errors.append(f"{domain}: provenance mismatch")
            for field in ("joint_coverage", "consistency_loss", "student_unsupervised_gradient_norm", "student_total_gradient_norm"):
                if not finite(row.get(field)):
                    errors.append(f"{domain}: missing/non-finite {field}")
    elif name == "LEAKAGE_AUDIT_REPORT.json":
        if report.get("hidden_gt_training_usage") != "none" or set(report.get("seeds", {})) != {"0", "1", "2"}:
            errors.append("missing real three-seed leakage evidence")
        for seed, audit in report.get("seeds", {}).items():
            if audit.get("status") != "PASS" or set(audit.get("domains", {})) != set(domains):
                errors.append(f"leakage seed {seed} failed")
            for domain, row in audit.get("domains", {}).items():
                if row.get("unlabeled_records_with_label_path") != 0 or row.get("observed_training_domains") != [domain]:
                    errors.append(f"leakage seed {seed}/{domain} violates current-domain-only")
    elif name == "EVAL_STOCHASTICITY_AUDIT.json":
        if len(report.get("stochastic_single_draw", {}).get("values", [])) != 20:
            errors.append("20-draw stochastic evaluation missing")
        if report.get("deterministic_repeat_count") != 2 or report.get("deterministic_repeat_max_absolute_difference") != 0:
            errors.append("posterior-mean evaluation is not exactly repeatable")
        if report.get("deterministic_rng_unchanged") is not True or not report.get("checkpoint"):
            errors.append("deterministic evaluation RNG/checkpoint evidence missing")
        if report.get("formal_evaluation_policy") != "posterior_mean_preselected":
            errors.append("formal evaluation policy changed")
    return errors


def config_contract():
    configs = {key: load_yaml(ROOT / f"configs/gate0_repaired_v2/{name}.yaml") for key, name in
               (("C0", "fundus_lambda_u0"), ("B0", "fundus_pas_probmse"))}
    a, b = copy.deepcopy(configs["C0"]), copy.deepcopy(configs["B0"])
    assert a["training"].pop("lambda_u") == 0.0
    assert b["training"].pop("lambda_u") == 0.5
    if a != b:
        raise RuntimeError("C0/B0 differ beyond lambda_u")
    return configs, {key: resolved_config_hash(config) for key, config in configs.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("/root/LCRSeg/runs"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/di_dmpa_jascl")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--source-commit", help="audited code commit; later documentation-only commits are permitted")
    args = parser.parse_args()
    configs, hashes = config_contract()
    domains, commit = configs["B0"]["data"]["domain_order"], args.source_commit or git_revision(ROOT)
    subprocess.run(["git", "-C", str(ROOT), "diff", "--exit-code", commit, "--",
                    "di_dmpa_jascl", "scripts", "tests/gate0", "configs/gate0_repaired_v2"], check=True)
    errors, evidence, reports, results = [], {}, {}, {}
    for name in REPORTS:
        path = args.output_dir / name
        try:
            report = json.loads(path.read_text())
            failures = validate_report(name, report, commit, hashes, domains)
            if name == "UNIT_INTEGRATION_TEST_REPORT.json":
                for artifact, hash_field in (("pytest_output.txt", "transcript_sha256"), ("pytest.xml", "junit_sha256")):
                    if sha256_file(args.output_dir / artifact) != report.get(hash_field):
                        failures.append(f"unit report artifact hash mismatch: {artifact}")
            if name in ("PAS_GRADIENT_AUDIT.json", "EVAL_STOCHASTICITY_AUDIT.json"):
                entries = report.get("domains", {}).values() if name.startswith("PAS") else [report]
                for entry in entries:
                    checkpoint = entry.get("exact_checkpoint", entry.get("checkpoint", ""))
                    if not checkpoint or sha256_file(checkpoint) != entry.get("checkpoint_sha256"):
                        failures.append(f"{name}: checkpoint hash evidence missing/mismatched")
            if name == "RESUME_EQUIVALENCE_REPORT.json":
                for trajectory in report.get("trajectories", {}).values():
                    for key in ("reference", "candidate"):
                        if sha256_file(trajectory[key]) != trajectory.get(key + "_sha256"):
                            failures.append(f"resume {key} artifact hash mismatch")
            reports[name] = report
            evidence[name] = {"sha256": sha256_file(path), "errors": failures}
            errors.extend(failures)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{name}: missing/invalid report: {exc}")
    preflight_pass = not errors
    if not args.preflight:
        for key, prefix in (("C0", "lambda0"), ("B0", "pas_probmse")):
            results[key] = {}
            for seed in args.seeds:
                run = args.runs_root / f"gate0_v2_{prefix}_fundus_seed{seed}"
                run_errors = []
                try:
                    if not (run / ".complete").is_file() or (run / ".exit").read_text().strip() != "0":
                        raise ValueError("missing completion/exit-0 marker")
                    completion = json.loads((run / "run_completion.json").read_text())
                    metadata = json.loads((run / "run_metadata.json").read_text())
                    matrices = json.loads((run / "stage_by_domain_matrices.json").read_text())
                    resolved = load_yaml(run / "resolved_config.yaml")
                    if resolved_config_hash(resolved) != hashes[key]:
                        run_errors.append("resolved config hash mismatch")
                    for payload in (completion, metadata):
                        if payload.get("config_hash") != hashes[key] or payload.get("git_commit") != commit:
                            run_errors.append("source/config provenance mismatch")
                        if payload.get("objective_name") != OBJECTIVE or payload.get("evaluation_classifier") != "posterior_mean":
                            run_errors.append("objective/evaluation semantic mismatch")
                        if payload.get("method_registered") is not False or payload.get("hidden_gt_training_usage") != "none":
                            run_errors.append("method/leakage boundary failed")
                    if completion.get("status") != "COMPLETE" or completion.get("nan_detected") is not False or completion.get("di_dmpa_training_launched") is not False:
                        run_errors.append("invalid completion status")
                    log = audit_log(run / "train.jsonl", config_hash=hashes[key], git_commit=commit, domains=domains)
                    run_errors.extend(log["errors"])
                    if completion.get("global_step") != log["rows"]:
                        run_errors.append("completion/log step count mismatch")
                    for metric in METRICS:
                        run_errors.extend(audit_matrix(matrices.get(metric, {}), domains))
                    if not (run / "last.pt").is_file():
                        run_errors.append("missing final checkpoint")
                    for i, domain in enumerate(domains):
                        stage = run / f"stage_{i}_{domain}"
                        for required in ("best.pt", "stage_completion.json", "validation_pas_precision.json", "test_metrics.json"):
                            if not (stage / required).is_file():
                                run_errors.append(f"missing {domain}/{required}")
                    result_dir = args.output_dir / "gate0_results_v2" / key / f"seed{seed}"
                    result_dir.mkdir(parents=True, exist_ok=True)
                    for path in run.glob("stage_by_domain_*"):
                        shutil.copyfile(path, result_dir / path.name)
                    summaries = {}
                    for metric, matrix in matrices.items():
                        final = matrix[domains[-1]]
                        forgetting = {d: max(matrix[stage][d] for stage in domains[i:] if d in matrix[stage])-final[d]
                                      for i, d in enumerate(domains[:-1])}
                        summaries[metric] = {"current_domain": final[domains[-1]],
                            "historical_domains": {d: final[d] for d in domains[:-1]},
                            "final_domain_mean": statistics.mean(final.values()), "forgetting": forgetting}
                    results[key][str(seed)] = {"errors": run_errors, "summary": summaries, "log_audit": log,
                                              "run_dir": str(run), "config_hash": hashes[key], "git_commit": commit}
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    run_errors.append(str(exc))
                    results[key][str(seed)] = {"errors": run_errors, "run_dir": str(run)}
                errors.extend(f"{key}/seed{seed}: {error}" for error in run_errors)
    full = not args.preflight and sorted(args.seeds) == [0, 1, 2]
    overall = "PASS" if full and not errors else "BLOCKED_PENDING_PAS_REPAIR_AND_RERUN"
    if reports.get("PAS_GRADIENT_AUDIT.json", {}).get("status") == "BLOCKED_ZERO_PAS_COVERAGE":
        overall = "BLOCKED_ZERO_PAS_COVERAGE"
    status = {"status": overall, "overall_status": overall,
        "engineering_plumbing_status": reports.get(REPORTS[0], {}).get("status", "MISSING"),
        "ssl_objective_status": reports.get("PAS_GRADIENT_AUDIT.json", {}).get("status", "MISSING"),
        "evaluation_semantics_status": reports.get("EVAL_STOCHASTICITY_AUDIT.json", {}).get("status", "MISSING"),
        "resume_status": reports.get("RESUME_EQUIVALENCE_REPORT.json", {}).get("status", "MISSING"),
        "leakage_status": reports.get("LEAKAGE_AUDIT_REPORT.json", {}).get("status", "MISSING"),
        "method_off_switch_parity_status": "NOT_APPLICABLE_METHOD_NOT_IMPLEMENTED",
        "method_registered": False, "di_dmpa_training_launched": False,
        "git_commit": commit, "config_hashes": hashes, "evidence": evidence, "runs": results,
        "preflight_pass": preflight_pass, "errors": errors, "seeds_checked": args.seeds,
        "next_action": "STOP_FOR_INDEPENDENT_REVIEW" if full or errors else "SEED0_PAIR_GATES_ONLY"}
    write_json(args.output_dir / "GATE0_STATUS.json", status)
    if full and not errors:
        paired = {metric: [results["B0"][str(seed)]["summary"][metric]["final_domain_mean"] -
                           results["C0"][str(seed)]["summary"][metric]["final_domain_mean"] for seed in (0,1,2)]
                  for metric in METRICS}
        write_json(args.output_dir / "C0_VS_B0_PAIRED_COMPARISON.json",
                   {"git_commit": commit, "config_hashes": hashes, "delta_B0_minus_C0": paired,
                    "performance_superiority_required_for_gate": False})
    print(json.dumps(status, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
