#!/usr/bin/env python3
"""PPC-SHOR V0.6B executor-accounting recovery and development adjudication."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

import ppc_shor_v0_6a as a
from di_dmpa_gate1.binding import safe_asset
from shor_jascl_v0_3.core import shor_routes, top1_lowest


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DOCS = ROOT / "docs/ppc_shor_v0_6b"
PREREG = DOCS / "PPC_SHOR_V0_6B_PREREGISTRATION.json"
REMOTE = "https://github.com/DLwbm123/SSL_CL_seg.git"
POLICIES = tuple("C%d" % value for value in range(9))
EXPECTED_SELECTED = (
    "k010_tau095", "k010_tau095", "k010_tau095", "k100_tau090", "k100_tau090")
EXPECTED_RATIOS = (
    0.07792207792207792, 0.08108108108108109, 0.08641975308641975,
    0.07692307692307693, 0.08108108108108109)
PUBLIC_FILES = {
    "PPC_SHOR_V0_6B_PREREGISTRATION.json",
    "PPC_SHOR_V0_6B_PREREGISTRATION.md",
    "PPC_SHOR_V0_6B_RECOVERY_QUALIFICATION.json",
    "PPC_SHOR_V0_6B_DESIGN_PREFLIGHT.json",
    "PPC_SHOR_V0_6B_SELECTED_CANDIDATES.json",
    "PPC_SHOR_V0_6B_STATUS.json",
    "PPC_SHOR_V0_6B_FINAL_REPORT.md",
    "PPC_SHOR_V0_6B_METRICS.csv",
    "PPC_SHOR_V0_6B_ROUTING.csv",
    "PPC_SHOR_V0_6B_CALIBRATION.csv",
    "PPC_SHOR_V0_6B_STABILITY.csv",
    "PPC_SHOR_V0_6B_FAILURES_AND_WARNINGS.md",
    "PPC_SHOR_V0_6B_EXACT_COMMANDS.md",
    "PPC_SHOR_V0_6B_MANIFEST.json",
    "PPC_SHOR_V0_6B_PUBLICATION_RECEIPT.json",
}


class ProtocolViolation(a.ProtocolViolation):
    pass


class RecoveryEquivalence(ProtocolViolation):
    status = "BLOCKED_RECOVERY_EQUIVALENCE"


def require(condition, message, error=ProtocolViolation):
    if not condition:
        raise error(message)


sha256_file = a.sha256_file
canonical_hash = a.canonical_hash
array_hash = a.array_hash
read_json = a.read_json
write_json_new = a.write_json_new
write_text_new = a.write_text_new
write_csv_new = a.write_csv_new
save_npz_new = a.save_npz_new


def load_protocol():
    protocol = read_json(PREREG)
    require(protocol["registration_id"] == "PPC_SHOR_V0_6B_EXECUTOR_ACCOUNTING_RECOVERY"
            and protocol["immutable_history"]["PPC_SHOR_V0_6A"]["scientific_status"]
            == "BLOCKED_PROTOCOL_OR_LEAKAGE", "wrong V0.6B registration")
    return protocol


def base_protocol(protocol):
    base = read_json(a.PREREG)
    frozen = protocol["method_frozen"]
    require(base["population"]["calibration"]["rows"] == protocol["population"]["calibration"]["rows"]
            and base["population"]["segmentation_value"]["rows"]
            == protocol["population"]["segmentation_value"]["rows"]
            and base["candidates"]["kappa"] == frozen["kappa"]
            and base["candidates"]["tau"] == frozen["tau"]
            and base["candidates"]["rho"] == frozen["rho"]
            and base["bootstrap"]["master_seed"] == frozen["bayesian_master_seed"],
            "V0.6A/V0.6B scientific binding changed")
    return base


def effective_probability_levels(model):
    probability = np.asarray(model["probability"], dtype=np.float64)
    require(probability.ndim == 1 and len(probability) > 0, "invalid PAV probability levels")
    bits = probability.view(np.uint64)
    return int(1 + np.count_nonzero(bits[1:] != bits[:-1]))


_fit_calibrators_v0_6a = a.fit_calibrators


def fit_calibrators(rows, weights):
    """Keep PAV fits unchanged; replace only the effective parameter accounting."""
    state = _fit_calibrators_v0_6a(rows, weights)
    ratios = []
    for model in state["pooled"].values():
        model["effective_probability_levels"] = effective_probability_levels(model)
        ratios.append(model["effective_probability_levels"] / model["unique_patients"])
    for model in state["local"].values():
        if model["fallback"]:
            continue
        model["effective_probability_levels"] = effective_probability_levels(model)
        ratios.append(model["effective_probability_levels"] / model["unique_patients"])
    state["parameter_ratios"] = ratios
    state["max_parameter_ratio"] = max(ratios, default=math.inf)
    return state


# Reused V0.6A calibration and sensitivity functions resolve this module global at runtime.
a.fit_calibrators = fit_calibrators


def prediction_accounting(rows, predictions):
    predictions = np.asarray(predictions, dtype=np.float64)
    require(predictions.ndim == 3 and predictions.shape[1:] == (len(rows), 2),
            "calibrator prediction shape changed")
    validity = np.sum(np.isfinite(predictions), axis=0, dtype=np.int64)
    top = top1_lowest(np.stack([row["alpha"] for row in rows]))
    eligibility = np.zeros(len(rows), dtype=np.int64)
    historical = top < 2
    eligibility[historical] = validity[np.flatnonzero(historical), top[historical]]
    return validity, eligibility


def run_calibration_fold(output, fold, calibration, value, folds, protocol):
    report = a.run_calibration_fold(output, fold, calibration, value, folds, protocol)
    for row in report["candidates"]:
        kappa = math.inf if row["kappa"] == "inf" else float(row["kappa"])
        predictions = np.stack([a.calibrated_probabilities(state, report["outer"], kappa)
                                for state in report["states"]])
        validity, eligibility = prediction_accounting(report["outer"], predictions)
        row["minimum_outer_finite_predictions"] = int(validity.min())
        row["prediction_validity_count_min"] = int(validity.min())
        row["prediction_validity_count_max"] = int(validity.max())
        row["route_eligibility_count_min"] = int(eligibility.min())
        row["route_eligibility_count_max"] = int(eligibility.max())
    return report


def design_preflight(output, calibration, value, folds, protocol):
    reports = [run_calibration_fold(output, fold, calibration, value, folds, protocol)
               for fold in range(5)]
    identifiers = [a.candidate_id(kappa, tau) for kappa in a.KAPPAS for tau in a.TAUS]
    global_routes = {}
    for identifier in identifiers:
        route = np.full(len(value), 2, dtype=np.int64)
        for report in reports:
            route[np.flatnonzero(folds == report["fold"])] = report["arrays"][identifier + "_C6"]
        global_routes[identifier] = route
    hashes = {identifier: array_hash(route) for identifier, route in global_routes.items()}
    duplicates = a.mark_duplicates(hashes)
    frequencies = {identifier: float(np.mean(route < 2))
                   for identifier, route in global_routes.items()}
    gates = {
        "inner_unique_patients_ge_80": all(report["inner_unique_patients"] >= 80 for report in reports),
        "pooled_active_support_ge_30": all(report["pooled_active_support"][str(h)] >= 30
                                            for report in reports for h in (0, 1)),
        "bayesian_fits_ge_195": all(report["bayesian_feasible"][str(h)] >= 195
                                     for report in reports for h in (0, 1)),
        "outer_finite_predictions_ge_190": all(row["minimum_outer_finite_predictions"] >= 190
                                                 for report in reports for row in report["candidates"]),
        "candidate_route_frequency_15_to_85_percent": any(0.15 <= value_ <= 0.85
                                                            for value_ in frequencies.values()),
        "at_least_two_candidate_routes": len(set(hashes.values())) >= 2,
        "not_all_candidates_C0": any(np.any(route < 2) for route in global_routes.values()),
        "no_global_feasibility_early_return": True,
        "parameters_affect_or_duplicate_marked": all(
            duplicates[identifier] is not None or list(hashes.values()).count(hashes[identifier]) == 1
            for identifier in identifiers),
        "calibration_parameter_ratio_le_0_10": all(
            report["max_calibration_parameter_ratio"] <= 0.10 for report in reports),
        "pre_GT_seals_verified": all(report["preseal"]["status"]
                                      == "PASS_PRE_GT_CALIBRATION_ARTIFACTS_SEALED"
                                      for report in reports),
    }
    public_reports = [{key: report[key] for key in (
        "fold", "inner_rows", "inner_unique_patients", "outer_rows", "outer_unique_patients",
        "pooled_active_support", "bayesian_feasible", "max_calibration_parameter_ratio",
        "candidates", "preseal")} for report in reports]
    preflight = {
        "schema_version": 1,
        "registration_id": protocol["registration_id"],
        "status": ("PASS_PPC_SHOR_V0_6B_DESIGN_PREFLIGHT" if all(gates.values())
                   else "BLOCKED_DESIGN_DEGENERATE_BEFORE_GT"),
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "population": a.support_counts(calibration, value, folds),
        "candidate_global_route_frequency": frequencies,
        "candidate_global_route_sha256": hashes,
        "duplicate_of": duplicates,
        "unique_candidate_route_arrays": len(set(hashes.values())),
        "folds": public_reports,
        "accounting": {
            "prediction_validity_denominator": "per case and historical expert",
            "route_eligibility_denominator": "ridge-top1 historical expert only",
            "effective_probability_levels": "contiguous bitwise-identical float64 runs"
        },
        "outer_GT_reads": 0,
        "outer_domain_reads": 0,
        "v0_4_formal_03_reads": 0,
    }
    write_json_new(output / "public/PPC_SHOR_V0_6B_DESIGN_PREFLIGHT.json", preflight)
    return preflight, reports, global_routes


def copy_registration(output):
    public = Path(output) / "public"
    public.mkdir(parents=True, exist_ok=True)
    for name in ("PPC_SHOR_V0_6B_PREREGISTRATION.json", "PPC_SHOR_V0_6B_PREREGISTRATION.md"):
        target = public / name
        with target.open("xb") as handle:
            handle.write((DOCS / name).read_bytes())
            handle.flush()
            os.fsync(handle.fileno())


def file_manifest(root, excluded=()):
    root = Path(root)
    excluded = {Path(value).as_posix() for value in excluded}
    files = [path for path in root.rglob("*") if path.is_file()
             and "public" not in path.relative_to(root).parts
             and path.relative_to(root).as_posix() not in excluded]
    entries = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                "sha256": sha256_file(path)} for path in sorted(files)]
    return {"files": len(entries), "bytes": sum(row["bytes"] for row in entries),
            "content_sha256": canonical_hash(entries), "entries": entries}


def verify_manifest(root, manifest):
    root = Path(root)
    for row in manifest["entries"]:
        path = root / row["path"]
        require(path.is_file() and path.stat().st_size == row["bytes"]
                and sha256_file(path) == row["sha256"], "manifest artifact changed: " + row["path"])
    require(canonical_hash(manifest["entries"]) == manifest["content_sha256"],
            "manifest aggregate changed")


def verify_v0_6a_private(protocol):
    root = Path(protocol["v0_6a_recovery_binding"]["private_root_from_execution_receipt"])
    receipt_path = root / protocol["v0_6a_recovery_binding"]["execution_receipt"]
    receipt = read_json(receipt_path)
    require(receipt["code_commit"] == protocol["v0_6a_recovery_binding"]["source_commit"],
            "V0.6A receipt source changed", RecoveryEquivalence)
    manifest = file_manifest(root)
    require(manifest["files"] == protocol["v0_6a_recovery_binding"]["private_files"],
            "V0.6A private file count changed", RecoveryEquivalence)
    for fold in range(5):
        seal_path = root / "candidate_preseals" / ("fold%d.json" % fold)
        seal = read_json(seal_path)
        require(seal["status"] == "PASS_PRE_GT_CALIBRATION_ARTIFACTS_SEALED"
                and sha256_file(root / "calibration_models" / ("fold%d.json" % fold)) == seal["model_sha256"]
                and sha256_file(root / "bootstrap_weights" / ("fold%d.npz" % fold)) == seal["bootstrap_sha256"]
                and sha256_file(root / "candidate_routes" / ("fold%d.npz" % fold)) == seal["route_sha256"],
                "V0.6A preseal failed", RecoveryEquivalence)
    return root, manifest


def probability_fields(value, prefix=""):
    output = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "probability":
                output[prefix + "/probability"] = np.asarray(item, dtype=np.float64)
            else:
                output.update(probability_fields(item, prefix + "/" + str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            output.update(probability_fields(item, prefix + "/" + str(index)))
    return output


def write_role_caches(output, reports, value, folds):
    for report in reports:
        rows = [{"seed": row["seed"], "patient_id": row["patient_id"],
                 "domain_index": row["domain_index"], "alpha": row["alpha"].tolist(),
                 "score": a.score_matrix([row])[0].tolist()} for row in report["inner"]]
        write_json_new(output / "inner_calibration" / ("fold%d.json" % report["fold"]), rows)
    outer = [{"case_id": row["case_id"], "patient_id": row["patient_id"], "seed": row["seed"],
              "alpha": row["alpha"].tolist(), "image_h5_relpath": row["image_h5_relpath"],
              "image_sha256": row["image_sha256"], "row_index": row["row_index"],
              "fold": int(folds[index])} for index, row in enumerate(value)]
    validate_outer_blind_rows(outer)
    write_json_new(output / "outer_blind_cache.json", outer)


def validate_outer_blind_rows(rows):
    forbidden = {"domain", "domain_index", "site_or_vendor", "label", "label_h5_relpath", "label_sha256"}
    require(bool(rows) and all(not forbidden.intersection(row) for row in rows),
            "outer blind cache contains evaluator-only fields")
    return True


def sanitized_test_report(path, source_sha):
    report = read_json(path)
    require(report["status"] == "PASS" and report["source_sha256"] == source_sha
            and report["failures"] == report["errors"] == report["skips"] == 0,
            "qualification tests failed")
    require(sha256_file(report["junit_path"]) == report["junit_sha256"]
            and sha256_file(report["pytest_output_path"]) == report["pytest_output_sha256"],
            "qualification test evidence changed")
    return {key: report[key] for key in ("status", "tests", "failures", "errors", "skips",
                                          "source_sha256", "junit_sha256", "pytest_output_sha256")}


def qualification_equivalence(protocol, output, reports, preflight, global_routes, v0_root):
    old_preflight = read_json(a.DOCS / "PPC_SHOR_V0_6A_DESIGN_PREFLIGHT.json")
    route_arrays_verified = 0
    probability_arrays_verified = 0
    weight_arrays_verified = 0
    for fold, report in enumerate(reports):
        old_model = read_json(v0_root / "calibration_models" / ("fold%d.json" % fold))
        new_model = read_json(output / "calibration_models" / ("fold%d.json" % fold))
        old_probability = probability_fields(old_model)
        new_probability = probability_fields(new_model)
        require(old_probability.keys() == new_probability.keys(), "PAV model set changed", RecoveryEquivalence)
        for key in old_probability:
            require(np.array_equal(old_probability[key].view(np.uint64),
                                   new_probability[key].view(np.uint64)),
                    "PAV fitted probability changed", RecoveryEquivalence)
            probability_arrays_verified += 1
        with np.load(v0_root / "bootstrap_weights" / ("fold%d.npz" % fold), allow_pickle=False) as old, \
                np.load(output / "bootstrap_weights" / ("fold%d.npz" % fold), allow_pickle=False) as new:
            require(old.files == new.files, "Bayesian weight fields changed", RecoveryEquivalence)
            for key in old.files:
                require(np.array_equal(old[key], new[key]), "Bayesian weights changed", RecoveryEquivalence)
                weight_arrays_verified += 1
        with np.load(v0_root / "candidate_routes" / ("fold%d.npz" % fold), allow_pickle=False) as old, \
                np.load(output / "candidate_routes" / ("fold%d.npz" % fold), allow_pickle=False) as new:
            for identifier in protocol["method_frozen"]["candidate_ids"]:
                key = identifier + "_C6"
                require(np.array_equal(old[key], new[key]), "candidate route changed", RecoveryEquivalence)
                route_arrays_verified += 1
        old_rows = {row["candidate_id"]: row for row in old_preflight["folds"][fold]["candidates"]}
        for row in report["candidates"]:
            old_row = old_rows[row["candidate_id"]]
            for key in set(old_row) - {"minimum_outer_finite_predictions"}:
                require(row[key] == old_row[key], "calibration raw value changed: " + key,
                        RecoveryEquivalence)
            require(row["route_sha256"] == old_row["route_sha256"],
                    "route SHA changed", RecoveryEquivalence)
    pass_counts = [sum(row["calibration_pass"] for row in report["candidates"]) for report in reports]
    feasible = [[report["bayesian_feasible"][str(h)] for h in (0, 1)] for report in reports]
    ratios = [report["max_calibration_parameter_ratio"] for report in reports]
    selected = [row["candidate_id"] for row in a.select_candidates(reports)]
    require(pass_counts == protocol["recovery_qualification"]["expected_calibration_pass_counts"]
            and feasible == protocol["recovery_qualification"]["expected_bayesian_feasible_per_fold_expert"]
            and ratios == list(EXPECTED_RATIOS)
            and preflight["unique_candidate_route_arrays"] == 10
            and preflight["all_gates_pass"]
            and selected == list(EXPECTED_SELECTED), "registered recovery values changed", RecoveryEquivalence)
    require(preflight["candidate_global_route_sha256"]
            == old_preflight["candidate_global_route_sha256"],
            "global candidate route SHA changed", RecoveryEquivalence)
    for report in reports:
        require(all(row["prediction_validity_count_min"] == 200
                    and row["prediction_validity_count_max"] == 200 for row in report["candidates"]),
                "prediction-validity denominator changed", RecoveryEquivalence)
    stitched = np.full(198, 2, dtype=np.int64)
    for report, identifier in zip(reports, EXPECTED_SELECTED):
        indices = np.flatnonzero(np.asarray([row["fold"] for row in read_json(output / "outer_blind_cache.json")])
                                 == report["fold"])
        stitched[indices] = report["arrays"][identifier + "_C6"]
    binding = protocol["v0_6a_recovery_binding"]["stitched_C6"]
    require(int(np.sum(stitched < 2)) == binding["route_count"]
            and float(np.mean(stitched < 2)) == binding["route_frequency"]
            and array_hash(stitched) == binding["route_sha256"],
            "stitched registered C6 changed", RecoveryEquivalence)
    return {
        "bayesian_weight_arrays_verified": weight_arrays_verified,
        "PAV_probability_arrays_bitwise_verified": probability_arrays_verified,
        "candidate_fold_route_arrays_bitwise_verified": route_arrays_verified,
        "route_SHA_verified": route_arrays_verified,
        "calibration_pass_counts": pass_counts,
        "bayesian_feasible_per_fold_expert": feasible,
        "prediction_validity_per_case_expert": 200,
        "corrected_parameter_ratio_fold_maxima": ratios,
        "unique_candidate_route_arrays": preflight["unique_candidate_route_arrays"],
        "corrected_design_preflight": "11/11",
        "selected_candidates": selected,
        "stitched_C6": binding,
    }


def qualify(args):
    protocol = load_protocol()
    base = base_protocol(protocol)
    require(str(args.output.resolve()).startswith("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/"),
            "qualification output is not on canonical NAS")
    require(not args.output.exists(), "qualification output already exists")
    source_sha = sha256_file(__file__)
    tests = sanitized_test_report(args.test_report, source_sha)
    registration_commit = "0b3f490b6436625217152ddbf388ffa78db43198"
    remote = subprocess.check_output(
        ["git", "ls-remote", REMOTE, "refs/heads/" + protocol["branch"]], text=True).split()
    require(remote and remote[0] == registration_commit,
            "registration commit is not remotely verified")
    v0_root, v0_manifest = verify_v0_6a_private(protocol)
    args.output.mkdir(parents=True)
    copy_registration(args.output)
    calibration, value, _, _ = a.load_population(base)
    folds = a.fold_assignments(value)
    execution_protocol = dict(base)
    execution_protocol["registration_id"] = protocol["registration_id"]
    preflight, reports, global_routes = design_preflight(
        args.output, calibration, value, folds, execution_protocol)
    write_role_caches(args.output, reports, value, folds)
    equivalence = qualification_equivalence(protocol, args.output, reports, preflight,
                                            global_routes, v0_root)
    selected_payload = {
        "schema_version": 1,
        "registration_id": protocol["registration_id"],
        "status": "PASS_FIXED_SELECTION_RECOVERED",
        "selected_candidates": [{"fold": fold, "candidate_id": identifier}
                                for fold, identifier in enumerate(EXPECTED_SELECTED)],
        "stitched_C6": protocol["v0_6a_recovery_binding"]["stitched_C6"],
        "selection_reopened_in_formal": False
    }
    write_json_new(args.output / "public/PPC_SHOR_V0_6B_SELECTED_CANDIDATES.json", selected_payload)
    private = file_manifest(args.output, excluded=("QUALIFICATION_PRIVATE_MANIFEST.json",))
    write_json_new(args.output / "QUALIFICATION_PRIVATE_MANIFEST.json", private)
    receipt = {
        "schema_version": 1,
        "registration_id": protocol["registration_id"],
        "status": "PASS_PPC_SHOR_V0_6B_RECOVERY_QUALIFICATION",
        "source_sha256": source_sha,
        "registration_commit": registration_commit,
        "v0_6a_private": {"files": v0_manifest["files"], "bytes": v0_manifest["bytes"],
                           "content_sha256": v0_manifest["content_sha256"], "preseals_verified": 5},
        "equivalence": equivalence,
        "tests": tests,
        "role_isolation": {
            "outer_blind_cache_fields": ["case_id", "patient_id", "seed", "alpha",
                                          "image_h5_relpath", "image_sha256", "row_index", "fold"],
            "outer_domain_materialized_in_controller_before_seal": 0,
            "outer_domain_used_for_fit": 0,
            "outer_domain_used_for_selection": 0,
            "outer_domain_evaluator_reads": 0
        },
        "counters": {"outer_GT_reads": 0, "expert_forward_batches": 0,
                     "segmentation_optimizer_updates": 0, "router_updates": 0,
                     "v0_4_formal_03_reads": 0},
        "private_qualification_inventory": {key: private[key] for key in
                                             ("files", "bytes", "content_sha256")},
        "formal_GT_access_reservation_created": False,
    }
    write_json_new(args.output / "public/PPC_SHOR_V0_6B_RECOVERY_QUALIFICATION.json", receipt)
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}, sort_keys=True))
    return receipt


def source_gate(protocol, code_commit, qualification):
    branch = subprocess.check_output(
        ["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()
    require(branch == protocol["branch"] and head == code_commit and not dirty,
            "wrong or dirty qualified-freeze source")
    require(subprocess.check_output(
        ["git", "-C", str(REPO), "rev-list", "--count",
         protocol["base_commit"] + ".." + code_commit], text=True).strip() == "2",
        "qualified freeze is not the second registered commit")
    require(subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", code_commit + "^"], text=True).strip()
        == qualification["registration_commit"], "qualified freeze parent changed")
    expected = {
        "experiments/lcrseg/docs/ppc_shor_v0_6b/PPC_SHOR_V0_6B_PREREGISTRATION.json",
        "experiments/lcrseg/docs/ppc_shor_v0_6b/PPC_SHOR_V0_6B_PREREGISTRATION.md",
        "experiments/lcrseg/docs/ppc_shor_v0_6b/PPC_SHOR_V0_6B_RECOVERY_QUALIFICATION.json",
        "experiments/lcrseg/docs/ppc_shor_v0_6b/PPC_SHOR_V0_6B_DESIGN_PREFLIGHT.json",
        "experiments/lcrseg/docs/ppc_shor_v0_6b/PPC_SHOR_V0_6B_SELECTED_CANDIDATES.json",
        "experiments/lcrseg/ppc_shor_v0_6b.py",
        "experiments/lcrseg/tests/ppc_shor_v0_6b/test_protocol.py",
    }
    changed = set(subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "--name-only", protocol["base_commit"], code_commit],
        text=True).splitlines())
    require(changed == expected, "qualified-freeze file set changed")
    immutable = (
        "experiments/lcrseg/docs/shor_jascl_v0_3",
        "experiments/lcrseg/docs/shor_jascl_v0_3_1",
        "experiments/lcrseg/docs/shor_v0_4_fixed_policy_test",
        "experiments/lcrseg/docs/rc_shor_v0_5",
        "experiments/lcrseg/docs/rc_shor_v0_5_erratum",
        "experiments/lcrseg/docs/ppc_shor_v0_6a",
        "experiments/lcrseg/rc_shor_v0_5.py",
        "experiments/lcrseg/ppc_shor_v0_6a.py",
        "experiments/lcrseg/tests/rc_shor_v0_5",
        "experiments/lcrseg/tests/ppc_shor_v0_6a",
    )
    for path in immutable:
        require(subprocess.run(
            ["git", "-C", str(REPO), "diff", "--quiet", protocol["base_commit"],
             code_commit, "--", path]).returncode == 0, "immutable predecessor changed")
    remote = subprocess.check_output(
        ["git", "ls-remote", REMOTE, "refs/heads/" + protocol["branch"]], text=True).split()
    require(remote and remote[0] == code_commit, "qualified freeze is not remotely verified")
    require(sha256_file(__file__) == qualification["source_sha256"],
            "qualified source SHA changed")
    return {"branch": branch, "freeze_commit": code_commit, "remote_sha": remote[0]}


def load_qualification(protocol, root):
    root = Path(root)
    receipt = read_json(root / "public/PPC_SHOR_V0_6B_RECOVERY_QUALIFICATION.json")
    require(receipt["status"] == "PASS_PPC_SHOR_V0_6B_RECOVERY_QUALIFICATION"
            and receipt["equivalence"]["corrected_design_preflight"] == "11/11"
            and receipt["equivalence"]["selected_candidates"] == list(EXPECTED_SELECTED),
            "qualification did not pass")
    manifest = read_json(root / "QUALIFICATION_PRIVATE_MANIFEST.json")
    verify_manifest(root, manifest)
    for name in ("PPC_SHOR_V0_6B_RECOVERY_QUALIFICATION.json",
                 "PPC_SHOR_V0_6B_DESIGN_PREFLIGHT.json",
                 "PPC_SHOR_V0_6B_SELECTED_CANDIDATES.json"):
        require(sha256_file(root / "public" / name) == sha256_file(DOCS / name),
                "published qualification differs from private receipt")
    return receipt, manifest


def load_reports(root):
    root = Path(root)
    preflight = read_json(root / "public/PPC_SHOR_V0_6B_DESIGN_PREFLIGHT.json")
    blind = read_json(root / "outer_blind_cache.json")
    validate_outer_blind_rows(blind)
    for row in blind:
        row["alpha"] = np.asarray(row["alpha"], dtype=np.float64)
    reports = []
    for fold in range(5):
        inner = read_json(root / "inner_calibration" / ("fold%d.json" % fold))
        for row in inner:
            row["alpha"] = np.asarray(row["alpha"], dtype=np.float64)
        model = read_json(root / "calibration_models" / ("fold%d.json" % fold))
        with np.load(root / "bootstrap_weights" / ("fold%d.npz" % fold), allow_pickle=False) as source:
            weights = np.asarray(source["weights"])
        with np.load(root / "candidate_routes" / ("fold%d.npz" % fold), allow_pickle=False) as source:
            arrays = {key: np.asarray(source[key]) for key in source.files}
        outer = [row for row in blind if row["fold"] == fold]
        public = preflight["folds"][fold]
        reports.append({**public, "inner": inner, "outer": outer, "base_state": model["base"],
                        "states": model["bayesian"], "weights": weights, "arrays": arrays})
    return preflight, reports, blind


def copy_qualification(output, qualification_root):
    output = Path(output)
    qualification_root = Path(qualification_root)
    copy_registration(output)
    for directory in ("calibration_models", "bootstrap_weights", "candidate_routes",
                      "candidate_preseals", "inner_calibration"):
        shutil.copytree(qualification_root / directory, output / directory)
    shutil.copy2(qualification_root / "outer_blind_cache.json", output / "outer_blind_cache.json")
    for name in ("PPC_SHOR_V0_6B_RECOVERY_QUALIFICATION.json",
                 "PPC_SHOR_V0_6B_DESIGN_PREFLIGHT.json",
                 "PPC_SHOR_V0_6B_SELECTED_CANDIDATES.json"):
        shutil.copy2(qualification_root / "public" / name, output / "public" / name)


class NoUpdateGuard(contextlib.AbstractContextManager):
    def __init__(self):
        self.calls = {"backward": 0, "optimizer_construction": 0, "optimizer_step": 0}
        self.originals = []

    def _block(self, name):
        def blocked(*_args, **_kwargs):
            self.calls[name] += 1
            raise ProtocolViolation("runtime update guard blocked " + name)
        return blocked

    def __enter__(self):
        torch = a.v4.torch
        self.originals.append((torch.Tensor, "backward", torch.Tensor.backward))
        torch.Tensor.backward = self._block("backward")
        self.originals.append((torch.optim.Optimizer, "__init__", torch.optim.Optimizer.__init__))
        torch.optim.Optimizer.__init__ = self._block("optimizer_construction")
        classes = list(torch.optim.Optimizer.__subclasses__())
        while classes:
            cls = classes.pop()
            classes.extend(cls.__subclasses__())
            if "step" in cls.__dict__:
                self.originals.append((cls, "step", cls.step))
                cls.step = self._block("optimizer_step")
        return self

    def __exit__(self, *_exc):
        for owner, name, value in reversed(self.originals):
            setattr(owner, name, value)
        return False


def validate_probability_cache(path, rows):
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    require(value.shape == (rows, 3, 384, 384) and value.dtype == np.float32,
            "expert probability cache schema changed")
    for start in range(0, rows, 4):
        block = np.asarray(value[start:start + 4])
        require(np.isfinite(block).all() and np.all(block >= 0) and np.all(block <= 1)
                and np.allclose(block.sum(1), 1.0, atol=1e-5, rtol=0),
                "invalid expert probability cache")
    return sha256_file(path)


def verify_materialization_inputs(value, protocol):
    image_hashes = {}
    for row in value:
        path = safe_asset(protocol["inputs"]["data_root"], row["image_h5_relpath"])
        a.verify_file(path, row["image_sha256"])
        image_hashes["%d:%s" % (row["seed"], row["case_id"])] = row["image_sha256"]
    checkpoints = a.baseline_checkpoints(protocol)
    checkpoint_hashes = {}
    for key, row in checkpoints.items():
        a.verify_file(row["path"], row["sha256"])
        checkpoint_hashes["%d:%d" % key] = row["sha256"]
    deterministic = a.v4.configure_determinism()
    require(deterministic == {"deterministic_algorithms": True, "cudnn_deterministic": True,
                              "cudnn_benchmark": False, "matmul_tf32": False,
                              "cudnn_tf32": False}, "deterministic runtime changed")
    return image_hashes, checkpoint_hashes, deterministic


def stability_diagnostics(routes):
    routes = np.asarray(routes)
    require(routes.ndim == 2 and routes.shape[0] > 0, "invalid constituent routes")
    agreement = np.asarray([max(Counter(routes[:, index].tolist()).values()) / len(routes)
                            for index in range(routes.shape[1])], dtype=np.float64)
    return {
        "constituent_route_semantics": "constituent_bayesian_routes",
        "modal_disagreement": float(np.mean(1.0 - agreement)),
        "any_flip_case_fraction": float(np.mean([len(set(routes[:, index].tolist())) > 1
                                                  for index in range(routes.shape[1])])),
        "median_case_consensus": float(np.median(agreement)),
        "route_agreement": float(np.mean(agreement)),
    }


def final_ensemble_route(rows, predictions, tau, rho=0.80):
    require(np.asarray(predictions).shape[0] == 200, "final ensemble requires 200 predictions")
    return a.route_policy(rows, predictions, tau, rho=rho, minimum_predictions=190)


def create_reservation(path, payload):
    return write_json_new(path, payload)


class EvaluatorAccess:
    def __init__(self):
        self.sealed_folds = set()
        self.outer_domain_reads = 0
        self.outer_GT_reads = 0
        self.verified_manifests = set()

    def seal(self, fold):
        self.sealed_folds.add(int(fold))

    def require(self, fold):
        require(int(fold) in self.sealed_folds, "outer truth requested before verified seal")


def reveal_rows(protocol, expected, fold, access):
    access.require(fold)
    wanted = {(row["seed"], row["case_id"]): index for index, row in enumerate(expected)}
    found = {}
    for spec in protocol["inputs"]["seed_manifests"]:
        seed = spec["seed"]
        path = Path(protocol["inputs"]["data_root"]) / "manifests/training" / (
            "lcrseg_v1_seed%d.csv" % seed)
        if seed not in access.verified_manifests:
            a.verify_file(path, spec["sha256"])
            access.verified_manifests.add(seed)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (seed, row["case_id"])
                if key not in wanted:
                    continue
                require(row["dataset"] == "fundus" and row["primary_20pct_split"] == "train_labeled"
                        and row["site_or_vendor"] in a.DOMAINS and row["label_h5_relpath"]
                        and row["label_sha256"], "outer evaluator row changed")
                found[wanted[key]] = row
                access.outer_domain_reads += 1
    require(set(found) == set(range(len(expected))), "outer evaluator population incomplete")
    return [found[index] for index in range(len(expected))]


def read_label(row, protocol, fold, access):
    access.require(fold)
    path = safe_asset(protocol["inputs"]["data_root"], row["label_h5_relpath"])
    a.verify_file(path, row["label_sha256"])
    with h5py.File(path, "r") as handle:
        label = np.asarray(handle["label"][...], dtype=np.int64)
    require(label.shape == (384, 384) and set(np.unique(label)).issubset({0, 1, 2}),
            "outer label geometry/value changed")
    access.outer_GT_reads += 1
    return label


def public_manifest(output, scientific_status):
    public = Path(output) / "public"
    observed = {path.name for path in public.iterdir() if path.is_file()}
    require(observed == PUBLIC_FILES - {"PPC_SHOR_V0_6B_MANIFEST.json"},
            "public report set incomplete")
    entries = [{"path": name, "bytes": (public / name).stat().st_size,
                "sha256": sha256_file(public / name)} for name in sorted(observed)]
    payload = {"schema_version": 1, "registration_id": "PPC_SHOR_V0_6B_EXECUTOR_ACCOUNTING_RECOVERY",
               "scientific_status": scientific_status, "files": len(entries),
               "bytes": sum(row["bytes"] for row in entries),
               "content_sha256": canonical_hash(entries), "entries": entries,
               "private_artifacts_published": False}
    write_json_new(public / "PPC_SHOR_V0_6B_MANIFEST.json", payload)
    return payload


def exact_commands():
    return """# PPC-SHOR V0.6B exact commands

Qualification and formal execution use the registered source with the project NAS wrapper:

```bash
bash experiments/lcrseg/scripts/with_nas_storage.sh python experiments/lcrseg/ppc_shor_v0_6b.py --mode qualify --output QUALIFICATION_ROOT --test-report TEST_REPORT
bash experiments/lcrseg/scripts/with_nas_storage.sh python experiments/lcrseg/ppc_shor_v0_6b.py --mode formal --output FORMAL_ROOT --qualification-root QUALIFICATION_ROOT --code-commit QUALIFIED_FREEZE_COMMIT --device cuda:1
```

The qualified freeze remote SHA is verified by the formal source gate before materialization.
V0.4 `formal_03` is not an input.
"""


def execute_formal(args, protocol, base, qualification, publication, reports, value):
    output = args.output
    metadata = {"registration_id": protocol["registration_id"], "code_commit": args.code_commit,
                "source": publication, "segmentation_optimizer_steps": 0,
                "segmentation_parameter_updates": 0, "router_optimizer_steps": 0,
                "router_parameter_updates": 0}
    image_hashes, checkpoint_hashes, deterministic = verify_materialization_inputs(value, base)
    selected = []
    for report, identifier in zip(reports, EXPECTED_SELECTED):
        matches = [row for row in report["candidates"] if row["candidate_id"] == identifier]
        require(len(matches) == 1 and matches[0]["calibration_pass"], "fixed candidate unavailable")
        selected.append(matches[0])
    sensitivity = [a.sensitivity_analysis(output, report, chosen, base)
                   for report, chosen in zip(reports, selected)]
    with NoUpdateGuard() as update_guard:
        expert_paths, forward_batches = a.materialize_experts(
            output, value, base, args.device, metadata)
    require(update_guard.calls == {"backward": 0, "optimizer_construction": 0, "optimizer_step": 0},
            "runtime update guard observed a forbidden call")
    expert_hashes = {"seed%d_expert%d" % key: validate_probability_cache(path, 66)
                     for key, path in expert_paths.items()}
    expert_arrays, local = a.expert_views(value, expert_paths)
    thresholds = a.frozen_thresholds(base, Path(base["inputs"]["oof_root"]))
    access = EvaluatorAccess()
    folds = np.asarray([row["fold"] for row in value], dtype=np.int64)
    global_routes = {policy: np.full(len(value), -1, dtype=np.int64) for policy in POLICIES[:7]}
    constituent_routes = np.full((200, len(value)), 2, dtype=np.int64)
    prediction_validity = np.zeros((len(value), 2), dtype=np.int64)
    route_eligibility = np.zeros(len(value), dtype=np.int64)
    fold_prediction_paths = {}
    seal_receipts = []
    for report, chosen, sensitive in zip(reports, selected, sensitivity):
        fold = report["fold"]
        indices = np.flatnonzero(folds == fold)
        rows = report["outer"]
        identifier = chosen["candidate_id"]
        alpha = np.stack([row["alpha"] for row in rows])
        top = top1_lowest(alpha)
        routes = {
            "C0": np.full(len(rows), 2, dtype=np.int64),
            "C1": top,
            "C2": np.full(len(rows), -1, dtype=np.int64),
            "C3": np.full(len(rows), 2, dtype=np.int64),
            "C4": report["arrays"][identifier + "_C4"],
            "C5": report["arrays"][identifier + "_C5"],
            "C6": report["arrays"][identifier + "_C6"],
        }
        c3 = np.empty(len(rows), dtype=np.int64)
        for seed in range(3):
            mask = np.asarray([row["seed"] == seed for row in rows])
            if mask.any():
                c3[mask] = shor_routes(alpha[mask], stage=2, thresholds=thresholds[seed])
        routes["C3"] = c3
        for policy in POLICIES[:7]:
            global_routes[policy][indices] = routes[policy]
        constituent_routes[:, indices] = report["arrays"][identifier + "_C6_realizations"]
        kappa = math.inf if chosen["kappa"] == "inf" else float(chosen["kappa"])
        probabilities = np.stack([a.calibrated_probabilities(state, rows, kappa)
                                  for state in report["states"]])
        validity, eligibility = prediction_accounting(rows, probabilities)
        prediction_validity[indices] = validity
        route_eligibility[indices] = eligibility
        observed_route, _, _, _ = final_ensemble_route(rows, probabilities, chosen["tau"], rho=a.RHO)
        require(np.array_equal(observed_route, routes["C6"]), "selected C6 route changed")
        prediction_paths = a.materialize_fold_predictions(
            output, fold, rows, routes, expert_arrays, local)
        fold_prediction_paths[fold] = prediction_paths
        case_order_path = write_json_new(
            output / "case_order" / ("fold%d.json" % fold),
            [{"row_index": row["row_index"], "seed": row["seed"], "case_id": row["case_id"]}
             for row in rows])
        sealed_files = {
            (Path("calibration_models") / ("fold%d.json" % fold)).as_posix():
                report["preseal"]["model_sha256"],
            (Path("bootstrap_weights") / ("fold%d.npz" % fold)).as_posix():
                report["preseal"]["bootstrap_sha256"],
            (Path("candidate_routes") / ("fold%d.npz" % fold)).as_posix():
                report["preseal"]["route_sha256"],
            sensitive["relative_path"]: sensitive["artifact_sha256"],
            case_order_path.relative_to(output).as_posix(): sha256_file(case_order_path),
        }
        sealed_files.update({path.relative_to(output).as_posix(): sha256_file(path)
                             for path in prediction_paths.values()})
        sealed_files.update({path.relative_to(output).as_posix(): expert_hashes["seed%d_expert%d" % key]
                             for key, path in expert_paths.items()})
        seal = {
            "status": "PASS_OUTER_CANDIDATES_SEALED_BEFORE_GT",
            "fold": fold,
            "outer_GT_reads": 0,
            "outer_domain_reads": 0,
            "selected_candidate": chosen,
            "case_order_sha256": sha256_file(case_order_path),
            "prediction_validity_count_min": int(validity.min()),
            "prediction_validity_count_max": int(validity.max()),
            "route_eligibility_count_min": int(eligibility.min()),
            "route_eligibility_count_max": int(eligibility.max()),
            "sealed_files": sealed_files,
        }
        seal_path = write_json_new(output / "candidate_seals" / ("fold%d.json" % fold), seal)
        a.verify_candidate_seal(seal_path, output)
        access.seal(fold)
        seal_receipts.append({"fold": fold, "seal_sha256": sha256_file(seal_path),
                              "reverified": True, "files": len(sealed_files)})
    require(np.all(prediction_validity == 200), "formal prediction-validity count changed")
    require(int(np.sum(global_routes["C6"] < 2)) == 155
            and array_hash(global_routes["C6"])
            == protocol["v0_6a_recovery_binding"]["stitched_C6"]["route_sha256"],
            "formal fixed C6 binding changed")
    for fold in range(5):
        a.verify_candidate_seal(output / "candidate_seals" / ("fold%d.json" % fold), output)
    reservation_path = args.qualification_root.parent / "FORMAL_GT_ACCESS_RESERVATION.json"
    reservation = {
        "status": "FORMAL_GT_ACCESS_RESERVED",
        "registration_id": protocol["registration_id"],
        "qualified_freeze_commit": args.code_commit,
        "qualification_sha256": sha256_file(
            args.qualification_root / "public/PPC_SHOR_V0_6B_RECOVERY_QUALIFICATION.json"),
        "formal_output": str(output),
        "candidate_seals": seal_receipts,
        "outer_domain_reads_before_reservation": 0,
        "outer_GT_reads_before_reservation": 0,
    }
    create_reservation(reservation_path, reservation)
    reservation_sha = sha256_file(reservation_path)

    case_rows = []
    expert_fg = np.empty((len(value), 3))
    expert_class = np.empty((len(value), 3, 2))
    domains = np.empty(len(value), dtype=np.int64)
    for report in reports:
        fold = report["fold"]
        indices = np.flatnonzero(folds == fold)
        rows = report["outer"]
        a.verify_candidate_seal(output / "candidate_seals" / ("fold%d.json" % fold), output)
        revealed = reveal_rows(base, rows, fold, access)
        predictions = {policy: np.load(path, mmap_mode="r", allow_pickle=False)
                       for policy, path in fold_prediction_paths[fold].items()}
        for local_index, (index, blind, full) in enumerate(zip(indices, rows, revealed)):
            label = read_label(full, base, fold, access)
            domain = a.DOMAINS.index(full["site_or_vendor"])
            domains[index] = domain
            local_position = local[(blind["seed"], blind["case_id"])]
            expert_metrics = []
            for expert in range(3):
                hard = np.argmax(expert_arrays[(blind["seed"], expert)][local_position], axis=0)
                metric = a.v4.case_metrics(hard, label)
                expert_metrics.append(metric)
                expert_fg[index, expert] = metric["foreground_dice"]
                expert_class[index, expert] = (metric["rim_dice"], metric["cup_dice"])
            for policy in POLICIES[:7]:
                metric = a.v4.case_metrics(predictions[policy][local_index], label)
                case_rows.append({"row_index": int(index), "fold": fold, "seed": blind["seed"],
                                  "case_id": blind["case_id"], "patient_id": blind["patient_id"],
                                  "domain_index": domain, "domain": a.DOMAINS[domain], "policy": policy,
                                  "route": int(global_routes[policy][index]), **metric})
            c7, c8 = domain, int(np.argmax(expert_fg[index]))
            for policy, expert in (("C7", c7), ("C8", c8)):
                case_rows.append({"row_index": int(index), "fold": fold, "seed": blind["seed"],
                                  "case_id": blind["case_id"], "patient_id": blind["patient_id"],
                                  "domain_index": domain, "domain": a.DOMAINS[domain], "policy": policy,
                                  "route": expert, **expert_metrics[expert]})
                global_routes.setdefault(policy, np.full(len(value), -1, dtype=np.int64))[index] = expert
    with (output / "case_metrics.jsonl").open("x", encoding="utf-8") as handle:
        for row in case_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    utility_fg = expert_fg[:, :2] - expert_fg[:, 2, None]
    utility_class = expert_class[:, :2] - expert_class[:, 2, None, :]
    seeds = np.asarray([row["seed"] for row in value])
    patients = np.asarray([row["patient_id"] for row in value])
    summaries = {policy: a.route_metrics(global_routes[policy], utility_fg, utility_class,
                                          seeds, domains, patients)
                 for policy in ("C3", "C4", "C5", "C6")}
    constituent_metrics = [a.route_metrics(route, utility_fg, utility_class, seeds, domains, patients)
                           for route in constituent_routes]
    diagnostics = stability_diagnostics(constituent_routes)
    stability = {
        "constituent_bayesian_routes": {
            "replicates": 200,
            "historical_gain_p10": float(np.quantile(
                [row["historical_gain"] for row in constituent_metrics], 0.10)),
            "shared_gain_p10": float(np.quantile(
                [row["shared_gain"] for row in constituent_metrics], 0.10)),
            "current_domain_drop_p90": float(np.quantile(
                [row["current_domain_drop"] for row in constituent_metrics], 0.90)),
            "maximum_seed_domain_drop_p90": float(np.quantile(
                [row["maximum_seed_domain_drop"] for row in constituent_metrics], 0.90)),
            "not_full_policy_realizations": True,
        },
        "final_ensemble_C6": {
            **diagnostics,
            "prediction_validity_count_min": int(prediction_validity.min()),
            "prediction_validity_count_median": float(np.median(prediction_validity)),
            "prediction_validity_count_max": int(prediction_validity.max()),
            "route_eligibility_count_min": int(route_eligibility.min()),
            "route_eligibility_count_median": float(np.median(route_eligibility)),
            "route_eligibility_count_max": int(route_eligibility.max()),
            "predictions": 200,
            "rho": a.RHO,
        },
        "per_fold_expert_feasible": [
            {"fold": report["fold"], "historical_expert": h,
             "numerator": report["bayesian_feasible"][str(h)], "denominator": 200}
            for report in reports for h in (0, 1)],
    }
    metrics = a.aggregate_case_metrics(case_rows)
    overall = {row["policy"]: row for row in metrics if row["level"] == "overall"}
    c6, c3 = summaries["C6"], summaries["C3"]
    domain_oracle_gap = overall["C7"]["foreground_dice"] - overall["C6"]["foreground_dice"]
    nondegenerate = (0.15 <= c6["route_frequency"] <= 0.85 and c6["route_count"] >= 30
                     and all(np.any(global_routes["C6"][folds == fold] < 2) for fold in range(5)))
    value_gate = (nondegenerate and c6["three_domain_gain"] >= 0.15
                  and c6["historical_gain"] >= 0.25 and c6["REFUGE_gain"] > 0
                  and c6["RIM_ONE_r3_gain"] > 0 and c6["positive_seed_count"] == 3
                  and domain_oracle_gap <= 0.06)
    noninferiority = (c6["three_domain_gain"] - c3["three_domain_gain"] >= -0.010
                      and c6["historical_gain"] - c3["historical_gain"] >= -0.020)
    constituent = stability["constituent_bayesian_routes"]
    ensemble = stability["final_ensemble_C6"]
    safety_gate = (c6["current_domain_drop"] <= 0.010
                   and c6["maximum_current_class_drop"] <= 0.015
                   and c6["maximum_seed_domain_drop"] <= 0.020
                   and constituent["current_domain_drop_p90"] <= 0.015
                   and constituent["maximum_seed_domain_drop_p90"] <= 0.025)
    stability_gate = (all(row["numerator"] >= 195 for row in stability["per_fold_expert_feasible"])
                      and constituent["historical_gain_p10"] >= 0.20
                      and constituent["shared_gain_p10"] >= 0.10
                      and ensemble["modal_disagreement"] <= 0.15
                      and ensemble["median_case_consensus"] >= 0.80)
    if not safety_gate:
        status = "FAIL_PPC_SHOR_CURRENT_SAFETY"
    elif not value_gate or not noninferiority:
        status = "FAIL_PPC_SHOR_VALUE"
    elif not stability_gate:
        status = "FAIL_PPC_SHOR_STABILITY"
    else:
        status = "PASS_PPC_SHOR_V0_6B_DEVELOPMENT_FEASIBILITY"
    receipt = {"status": "COMMAND_COMPLETED", "scientific_status": status,
               "code_commit": args.code_commit, "formal_attempt": 1,
               "reservation_sha256": reservation_sha, "main_merged": False,
               "external_test_launched": False}
    write_json_new(output / "EXECUTION_RECEIPT.json", receipt)
    private = file_manifest(output, excluded=("PPC_SHOR_V0_6B_PRIVATE_MANIFEST.json",))
    write_json_new(output / "PPC_SHOR_V0_6B_PRIVATE_MANIFEST.json", private)
    gates = {
        "isolation": True,
        "nondegeneracy": nondegenerate,
        "value": value_gate,
        "C3_noninferiority": noninferiority,
        "current_safety": safety_gate,
        "stability": stability_gate,
    }
    gate_values = {
        "C6_route_frequency": {"value": c6["route_frequency"], "threshold": [0.15, 0.85]},
        "C6_historical_route_count": {"value": c6["route_count"], "threshold_min": 30},
        "three_domain_gain": {"value": c6["three_domain_gain"], "threshold_min": 0.15},
        "historical_gain": {"value": c6["historical_gain"], "threshold_min": 0.25},
        "REFUGE_gain": {"value": c6["REFUGE_gain"], "threshold": ">0"},
        "RIM_ONE_r3_gain": {"value": c6["RIM_ONE_r3_gain"], "threshold": ">0"},
        "positive_seed_count": {"value": c6["positive_seed_count"], "required": 3},
        "domain_oracle_gap": {"value": domain_oracle_gap, "threshold_max": 0.06},
        "C6_minus_C3_overall_gain": {"value": c6["three_domain_gain"] - c3["three_domain_gain"],
                                      "threshold_min": -0.010},
        "C6_minus_C3_historical_gain": {"value": c6["historical_gain"] - c3["historical_gain"],
                                         "threshold_min": -0.020},
        "current_domain_drop": {"value": c6["current_domain_drop"], "threshold_max": 0.010},
        "maximum_current_class_drop": {"value": c6["maximum_current_class_drop"],
                                        "threshold_max": 0.015},
        "maximum_seed_domain_drop": {"value": c6["maximum_seed_domain_drop"],
                                      "threshold_max": 0.020},
        "current_domain_drop_p90": {"value": constituent["current_domain_drop_p90"],
                                     "threshold_max": 0.015},
        "maximum_seed_domain_drop_p90": {"value": constituent["maximum_seed_domain_drop_p90"],
                                          "threshold_max": 0.025},
        "historical_gain_p10": {"value": constituent["historical_gain_p10"],
                                 "threshold_min": 0.20},
        "shared_gain_p10": {"value": constituent["shared_gain_p10"], "threshold_min": 0.10},
        "modal_disagreement": {"value": ensemble["modal_disagreement"], "threshold_max": 0.15},
        "median_case_consensus": {"value": ensemble["median_case_consensus"],
                                  "threshold_min": 0.80},
        "any_flip_case_fraction": {"value": ensemble["any_flip_case_fraction"],
                                    "diagnostic_only": True},
    }
    controls = {policy: {key: value for key, value in row.items() if key in
                ("foreground_dice", "rim_dice", "cup_dice", "mean_iou")}
                for policy, row in overall.items()}
    status_payload = {
        "schema_version": 1,
        "registration_id": protocol["registration_id"],
        "scientific_status": status,
        "formal_attempt": 1,
        "V0_6A_status_changed": False,
        "recovery_qualification": qualification,
        "development_population": read_json(
            output / "public/PPC_SHOR_V0_6B_DESIGN_PREFLIGHT.json")["population"],
        "controls": controls,
        "routing": {key: {k: v for k, v in row.items() if k not in ("delta", "class_delta")}
                    for key, row in summaries.items()},
        "selected_candidates": [{"fold": fold, **row} for fold, row in enumerate(selected)],
        "stability": stability,
        "sensitivity": sensitivity,
        "domain_oracle_gap": domain_oracle_gap,
        "C6_minus_C3": {"overall_gain": c6["three_domain_gain"] - c3["three_domain_gain"],
                         "historical_gain": c6["historical_gain"] - c3["historical_gain"]},
        "gates": gates,
        "gate_values": gate_values,
        "isolation": {
            "outer_domain_materialized_in_controller_before_seal": 0,
            "outer_domain_used_for_fit": 0,
            "outer_domain_used_for_selection": 0,
            "outer_domain_evaluator_reads": access.outer_domain_reads,
            "outer_GT_reads": access.outer_GT_reads,
            "outer_GT_reads_before_verified_seal": 0,
            "outer_domain_reads_before_verified_seal": 0,
            "v0_4_formal_03_reads": 0,
            "test_domain_inputs_to_C6": 0,
            "segmentation_training_steps": 0,
            "segmentation_optimizer_steps": 0,
            "segmentation_parameter_updates": 0,
            "router_optimizer_steps": 0,
            "router_parameter_updates": 0,
            "segmentation_expert_forward_batches": forward_batches,
            "runtime_update_guard": update_guard.calls,
            "old_artifact_mutations": 0,
        },
        "integrity": {"verified_image_files": len(image_hashes),
                      "verified_checkpoint_files": len(checkpoint_hashes),
                      "expert_probability_cache_sha256": expert_hashes,
                      "deterministic_runtime": deterministic,
                      "candidate_seal_reverification": seal_receipts,
                      "reservation_sha256": reservation_sha},
        "tests": qualification["tests"],
        "private_artifact_inventory": {key: private[key] for key in
                                       ("files", "bytes", "content_sha256")},
        "publication": publication,
        "main_merged": False,
        "external_test_launched": False,
        "report_commit": None,
        "report_commit_resolution": "third commit containing these exact report bytes",
    }
    public = output / "public"
    write_json_new(public / "PPC_SHOR_V0_6B_STATUS.json", status_payload)
    write_csv_new(public / "PPC_SHOR_V0_6B_METRICS.csv", metrics)
    calibration_rows = [dict(fold=report["fold"], selected=row["candidate_id"] == chosen["candidate_id"],
                             **row) for report, chosen in zip(reports, selected)
                        for row in report["candidates"]]
    write_csv_new(public / "PPC_SHOR_V0_6B_CALIBRATION.csv", calibration_rows)
    routing_rows = [{"level": "overall", "policy": policy,
                     **{key: value for key, value in summaries[policy].items()
                        if key not in ("delta", "class_delta")}}
                    for policy in ("C3", "C4", "C5", "C6")]
    routing_rows += [{"level": "outer_fold_selection", "policy": "C6", "fold": fold,
                      "candidate_id": row["candidate_id"], "kappa": row["kappa"],
                      "tau": row["tau"], "route_frequency": row["outer_route_frequency"],
                      "prediction_validity_denominator": row["prediction_validity_count_min"],
                      "route_eligibility_denominator": row["route_eligibility_count_min"]}
                     for fold, row in enumerate(selected)]
    write_csv_new(public / "PPC_SHOR_V0_6B_ROUTING.csv", routing_rows)
    stability_rows = [{"scope": "overall", "metric": key, "value": value["value"],
                       "threshold": value.get("threshold", value.get("threshold_min",
                                     value.get("threshold_max", "diagnostic")))}
                      for key, value in gate_values.items() if key in (
                          "historical_gain_p10", "shared_gain_p10", "current_domain_drop_p90",
                          "maximum_seed_domain_drop_p90", "modal_disagreement",
                          "median_case_consensus", "any_flip_case_fraction")]
    stability_rows += [{"scope": "fold%d_expert%d" % (row["fold"], row["historical_expert"]),
                        "metric": "bayesian_feasible", "value": row["numerator"], "threshold": 195}
                       for row in stability["per_fold_expert_feasible"]]
    write_csv_new(public / "PPC_SHOR_V0_6B_STABILITY.csv", stability_rows)
    report = """# PPC-SHOR V0.6B final report

## Outcome

The single registered development outer-OOF adjudication completed with **%s**. V0.6A remains
`BLOCKED_PROTOCOL_OR_LEAKAGE`; no prior status or artifact was changed. This is development
evidence, not external confirmation.

Recovery equivalence passed before the qualified freeze: V0.6A Bayesian weights, PAV fitted
probabilities, all 60 candidate-fold routes and hashes were reproduced exactly. The two V0.6A
false-negative gates were repaired by separating prediction validity from route eligibility and
by counting contiguous bitwise-identical PAV probability levels. Constituent Bayesian routes are
used only for gain/drop quantiles; ensemble modal disagreement is the hard stability statistic,
while any-flip fraction is diagnostic.

The formal GT reservation was created only after five full candidate seals, expert probability
caches, predictions, routes, calibration models, bootstrap weights, sensitivity artifacts and
case order were durably written and reverified. C6 routed %d/198 cases (%.6f). Three-domain gain
was %.6f, historical gain %.6f, current-domain drop %.6f and domain-oracle gap %.6f.

Stability: historical-gain p10 %.6f, shared-gain p10 %.6f, current-drop p90 %.6f,
maximum seed-domain-drop p90 %.6f, modal disagreement %.6f, any-flip fraction %.6f and median
case consensus %.6f. Prediction validity was 200/200 for every case/expert.

Outer evaluator domain/GT reads were %d/%d and occurred after verified seals. Segmentation and
router optimizer/update counts were zero. V0.4 `formal_03` reads, main merges and external tests
were zero.
""" % (status, c6["route_count"], c6["route_frequency"], c6["three_domain_gain"],
       c6["historical_gain"], c6["current_domain_drop"], domain_oracle_gap,
       constituent["historical_gain_p10"], constituent["shared_gain_p10"],
       constituent["current_domain_drop_p90"], constituent["maximum_seed_domain_drop_p90"],
       ensemble["modal_disagreement"], ensemble["any_flip_case_fraction"],
       ensemble["median_case_consensus"], access.outer_domain_reads, access.outer_GT_reads)
    write_text_new(public / "PPC_SHOR_V0_6B_FINAL_REPORT.md", report)
    failed = [name for name, passed in gates.items() if not passed]
    write_text_new(public / "PPC_SHOR_V0_6B_FAILURES_AND_WARNINGS.md",
                   "# Failures and warnings\n\n- Scientific status: `%s`.\n- Failed gates: %s.\n"
                   "- All Fundus rows are development-consumed; no external claim is permitted.\n"
                   % (status, ", ".join(failed) if failed else "none"))
    write_text_new(public / "PPC_SHOR_V0_6B_EXACT_COMMANDS.md", exact_commands())
    publication_receipt = {"schema_version": 1, "branch": protocol["branch"],
                           "registration_commit": qualification["registration_commit"],
                           "qualified_freeze_commit": args.code_commit,
                           "qualified_freeze_remote_sha": publication["remote_sha"],
                           "report_commit": None,
                           "report_commit_resolution": "the third commit containing this file",
                           "repository": "https://github.com/DLwbm123/SSL_CL_seg",
                           "main_merged": False, "external_test_launched": False}
    write_json_new(public / "PPC_SHOR_V0_6B_PUBLICATION_RECEIPT.json", publication_receipt)
    public_manifest(output, status)
    return status_payload


def formal(args):
    protocol = load_protocol()
    base = base_protocol(protocol)
    require(args.code_commit, "formal mode requires --code-commit")
    require(args.qualification_root and args.qualification_root.is_dir(),
            "formal mode requires qualification root")
    require(str(args.output.resolve()).startswith("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/"),
            "formal output is not on canonical NAS")
    require(not args.output.exists(), "formal output already exists")
    qualification, _ = load_qualification(protocol, args.qualification_root)
    publication = source_gate(protocol, args.code_commit, qualification)
    reservation_path = args.qualification_root.parent / "FORMAL_GT_ACCESS_RESERVATION.json"
    require(not reservation_path.exists(), "REFUSED_AFTER_FORMAL_GT_ACCESS_RESERVATION")
    preflight, reports, value = load_reports(args.qualification_root)
    require(preflight["all_gates_pass"] and preflight["status"]
            == "PASS_PPC_SHOR_V0_6B_DESIGN_PREFLIGHT", "qualified design preflight changed")
    args.output.mkdir(parents=True)
    copy_qualification(args.output, args.qualification_root)
    result = execute_formal(args, protocol, base, qualification, publication, reports, value)
    print(json.dumps({"status": result["scientific_status"], "output": str(args.output)}, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("qualify", "formal"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qualification-root", type=Path)
    parser.add_argument("--test-report", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.mode == "qualify":
        require(args.test_report and args.test_report.is_file(),
                "qualification mode requires test report")
        qualify(args)
    else:
        formal(args)


if __name__ == "__main__":
    main()
