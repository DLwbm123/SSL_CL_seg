#!/usr/bin/env python3
"""One-shot PPC-SHOR V0.6A development-feasibility execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

import rc_shor_v0_5 as rc
import shor_v0_4_test as v4
from di_dmpa_gate1.binding import safe_asset
from shor_jascl_v0_3.core import shor_routes, top1_lowest


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DOCS = ROOT / "docs/ppc_shor_v0_6a"
PREREG = DOCS / "PPC_SHOR_V0_6A_PREREGISTRATION.json"
REMOTE = "https://github.com/DLwbm123/SSL_CL_seg.git"
DOMAINS = ("REFUGE", "RIM_ONE_r3", "Drishti_GS")
POLICIES = tuple("C%d" % value for value in range(9))
KAPPAS = (10.0, 30.0, 100.0, math.inf)
TAUS = (0.90, 0.95, 0.98)
RHO = 0.80


class ProtocolViolation(RuntimeError):
    status = "BLOCKED_PROTOCOL_OR_LEAKAGE"


class RequiredInputMissing(ProtocolViolation):
    status = "BLOCKED_REQUIRED_FROZEN_INPUT_MISSING"


class DesignDegenerate(ProtocolViolation):
    status = "BLOCKED_DESIGN_DEGENERATE_BEFORE_GT"


class CalibrationFailure(ProtocolViolation):
    status = "FAIL_PPC_SHOR_CALIBRATION"


def require(condition, message, error=ProtocolViolation):
    if not condition:
        raise error(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        handle.flush(); os.fsync(handle.fileno())
    return path


def write_text_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value); handle.flush(); os.fsync(handle.fileno())
    return path


def write_csv_new(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            fields.extend(key for key in row if key not in fields)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
    return path


def save_npz_new(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
    return path


def load_protocol():
    protocol = read_json(PREREG)
    require(protocol["registration_id"] == "PPC_SHOR_V0_6A_DEVELOPMENT_FEASIBILITY"
            and protocol["immutable_history"]["SHOR_V0_3_1"]["H5"] is False,
            "wrong PPC-SHOR registration")
    return protocol


def source_gate(protocol, code_commit):
    branch = subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()
    require(branch == protocol["branch"] and head == code_commit and not dirty,
            "wrong or dirty execution source")
    require(subprocess.check_output(["git", "-C", str(REPO), "rev-list", "--count",
                                     protocol["base_commit"] + ".." + code_commit], text=True).strip() == "1",
            "execution source is not the unique freeze commit")
    changed = set(subprocess.check_output(["git", "-C", str(REPO), "diff", "--name-only",
                                           protocol["base_commit"], code_commit], text=True).splitlines())
    expected = {
        "experiments/lcrseg/docs/ppc_shor_v0_6a/PPC_SHOR_V0_6A_PREREGISTRATION.json",
        "experiments/lcrseg/docs/ppc_shor_v0_6a/PPC_SHOR_V0_6A_PREREGISTRATION.md",
        "experiments/lcrseg/ppc_shor_v0_6a.py",
        "experiments/lcrseg/tests/ppc_shor_v0_6a/test_protocol.py",
    }
    require(changed == expected, "freeze commit file set changed")
    for immutable in ("experiments/lcrseg/docs/shor_jascl_v0_3",
                      "experiments/lcrseg/docs/shor_jascl_v0_3_1",
                      "experiments/lcrseg/docs/shor_v0_4_fixed_policy_test",
                      "experiments/lcrseg/docs/rc_shor_v0_5",
                      "experiments/lcrseg/rc_shor_v0_5.py",
                      "experiments/lcrseg/tests/rc_shor_v0_5"):
        require(subprocess.run(["git", "-C", str(REPO), "diff", "--quiet",
                                protocol["base_commit"], code_commit, "--", immutable]).returncode == 0,
                "immutable predecessor changed")
    remote = subprocess.check_output(["git", "ls-remote", REMOTE,
                                      "refs/heads/" + protocol["branch"]], text=True).split()
    require(remote and remote[0] == code_commit, "freeze commit is not published")
    return {"branch": branch, "freeze_commit": code_commit, "remote_sha": remote[0]}


def verify_file(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing input: " + str(path), RequiredInputMissing)
    require(sha256_file(path) == expected, "input SHA mismatch: " + str(path), RequiredInputMissing)
    return path


def verify_oof_bundle(protocol):
    spec = protocol["inputs"]["oof_bundle"]
    root = Path(protocol["inputs"]["oof_root"])
    manifest_path = verify_file(root / spec["manifest"], spec["manifest_sha256"])
    manifest = read_json(manifest_path)
    require(manifest["files"] == spec["files"] and manifest["bytes"] == spec["bytes"]
            and manifest["content_sha256"] == spec["content_sha256"]
            and canonical_hash(manifest["entries"]) == spec["content_sha256"],
            "OOF bundle manifest identity changed", RequiredInputMissing)
    entries = {row["path"]: row for row in manifest["entries"]}
    for item in protocol["inputs"]["stage2_oof"]:
        require(item["path"] in entries and all(entries[item["path"]][key] == item[key]
                for key in ("bytes", "sha256")), "OOF entry binding changed", RequiredInputMissing)
        verify_file(root / item["path"], item["sha256"])
    threshold = protocol["inputs"]["frozen_shor_threshold_manifest"]
    verify_file(root / threshold["path"], threshold["sha256"])
    return root


def blind_manifests(protocol):
    root = Path(protocol["inputs"]["data_root"])
    output = {}
    required = ("dataset", "primary_20pct_split", "case_id", "patient_id",
                "image_h5_relpath", "image_sha256")
    for spec in protocol["inputs"]["seed_manifests"]:
        seed = spec["seed"]
        path = verify_file(root / "manifests/training" / ("lcrseg_v1_seed%d.csv" % seed), spec["sha256"])
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle); header = next(reader)
            require(all(field in header for field in required), "blind manifest schema changed")
            index = {field: header.index(field) for field in required}
            rows = {}
            for values in reader:
                if values[index["dataset"]] != "fundus":
                    continue
                row = {field: values[index[field]] for field in required[1:]}
                rows[row["case_id"]] = row
        require(len(rows) == 660, "Fundus manifest population changed")
        output[seed] = rows
    return output


def load_population(protocol):
    oof_root = verify_oof_bundle(protocol)
    manifests = blind_manifests(protocol)
    calibration = []
    for item in protocol["inputs"]["stage2_oof"]:
        seed = item["seed"]
        with np.load(oof_root / item["path"], allow_pickle=False) as source:
            case_ids = source["case_ids"].astype(str)
            domains = np.asarray(source["domain_indices"], dtype=np.int64)
            alpha = np.asarray(source["alpha"], dtype=np.float64)
        require(len(case_ids) == len(set(case_ids)) == 330 and alpha.shape == (330, 3)
                and np.isfinite(alpha).all() and np.allclose(alpha.sum(1), 1.0, atol=1e-12, rtol=1e-12),
                "stage2 OOF schema changed", RequiredInputMissing)
        for local, (case_id, domain) in enumerate(zip(case_ids, domains)):
            require(case_id in manifests[seed] and domain in (0, 1, 2), "OOF/manifest alignment changed")
            row = manifests[seed][case_id]
            calibration.append({"seed": seed, "case_id": case_id, "patient_id": row["patient_id"],
                                "domain_index": int(domain), "alpha": alpha[local],
                                "role": row["primary_20pct_split"]})
    value = []
    lookup = {(row["seed"], row["case_id"]): row for row in calibration}
    for seed, rows in manifests.items():
        for case_id, row in rows.items():
            if row["primary_20pct_split"] != "train_labeled":
                continue
            require((seed, case_id) in lookup, "train_labeled value row is outside frozen OOF")
            source = lookup[(seed, case_id)]
            value.append({"seed": seed, "case_id": case_id, "patient_id": row["patient_id"],
                          "alpha": source["alpha"], "image_h5_relpath": row["image_h5_relpath"],
                          "image_sha256": row["image_sha256"]})
    value.sort(key=lambda row: (row["seed"], row["case_id"]))
    for index, row in enumerate(value):
        row["row_index"] = index
    require(len(calibration) == 990 and len({row["patient_id"] for row in calibration}) == 575
            and len(value) == 198 and len({row["patient_id"] for row in value}) == 177,
            "registered PPC population changed", RequiredInputMissing)
    return calibration, value, manifests, oof_root


def fold_assignments(value):
    patients = sorted({row["patient_id"] for row in value}, key=lambda item: (
        hashlib.sha256(("ppc-shor-v0.6a-fold\0" + item).encode()).hexdigest(), item))
    mapping = {patient: index % 5 for index, patient in enumerate(patients)}
    folds = np.asarray([mapping[row["patient_id"]] for row in value], dtype=np.int64)
    require([int(np.sum(folds == fold)) for fold in range(5)] == [41, 42, 37, 38, 40],
            "outer fold counts changed")
    return folds


def score_matrix(rows):
    alpha = np.stack([row["alpha"] for row in rows])
    return np.stack([np.log(alpha[:, h] + 1e-12) - np.log(alpha[:, 2] + 1e-12)
                     for h in (0, 1)], axis=1)


def patient_row_weights(rows, patient_weights=None):
    counts = Counter(row["patient_id"] for row in rows)
    patient_weights = ({patient: 1.0 for patient in counts} if patient_weights is None else patient_weights)
    weights = np.asarray([patient_weights[row["patient_id"]] / counts[row["patient_id"]] for row in rows])
    for patient, count in counts.items():
        selected = np.asarray([row["patient_id"] == patient for row in rows])
        require(abs(weights[selected].sum() - patient_weights[patient]) < 1e-10,
                "patient total row weight changed")
    return weights


def bootstrap_weights(rows, master_seed, replicate, ordinary=False):
    patient_domain = {}
    for row in rows:
        patient = row["patient_id"]
        require(patient not in patient_domain or patient_domain[patient] == row["domain_index"],
                "patient crosses dataset domains")
        patient_domain[patient] = row["domain_index"]
    rng = np.random.Generator(np.random.PCG64(int(master_seed) + int(replicate)))
    result = {}
    for domain in range(3):
        patients = sorted(patient for patient, observed in patient_domain.items() if observed == domain)
        require(patients, "empty bootstrap domain")
        if ordinary:
            counts = Counter(patients[int(index)] for index in rng.integers(0, len(patients), len(patients)))
            result.update({patient: float(counts[patient]) for patient in patients})
        else:
            draw = rng.dirichlet(np.ones(len(patients))) * len(patients)
            result.update({patient: float(weight) for patient, weight in zip(patients, draw)})
            require(all(result[patient] > 0 for patient in patients), "Bayesian patient weight is not positive")
    return patient_row_weights(rows, result)


def pav_fit(x, y, weights):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    active = (weights > 0) & np.isfinite(x) & np.isfinite(y)
    require(active.any(), "empty isotonic support")
    order = np.lexsort((np.arange(len(x))[active], x[active]))
    sx, sy, sw = x[active][order], y[active][order], weights[active][order]
    blocks = []
    for value in np.unique(sx):
        selected = sx == value; weight = float(sw[selected].sum())
        blocks.append([float(value), weight, float(np.sum(sw[selected] * sy[selected]))])
        while len(blocks) >= 2 and blocks[-2][2] / blocks[-2][1] > blocks[-1][2] / blocks[-1][1]:
            right = blocks.pop(); left = blocks.pop()
            blocks.append([right[0], left[1] + right[1], left[2] + right[2]])
    return {"upper": [row[0] for row in blocks], "probability": [row[2] / row[1] for row in blocks],
            "fit_weight": [row[1] for row in blocks], "blocks": len(blocks)}


def pav_predict(model, x):
    upper = np.asarray(model["upper"]); probability = np.asarray(model["probability"])
    index = np.searchsorted(upper, np.asarray(x), side="left")
    return probability[np.minimum(index, len(probability) - 1)]


def effective_patients(rows, mask, weights):
    totals = {}
    for row, selected, weight in zip(rows, mask, weights):
        if selected and weight > 0:
            totals[row["patient_id"]] = totals.get(row["patient_id"], 0.0) + float(weight)
    value = np.asarray(list(totals.values()))
    return 0.0 if not len(value) else float(value.sum() ** 2 / np.sum(value ** 2))


def fit_calibrators(rows, weights):
    score = score_matrix(rows); top = top1_lowest(np.stack([row["alpha"] for row in rows]))
    labels = np.asarray([row["domain_index"] for row in rows]); seeds = np.asarray([row["seed"] for row in rows])
    patients = np.asarray([row["patient_id"] for row in rows])
    state = {"pooled": {}, "local": {}, "feasible": {}, "parameter_ratios": []}
    for historical in (0, 1):
        active = top == historical
        support = len(set(patients[active & (weights > 0)].tolist()))
        target = (labels == historical).astype(float)
        feasible = support >= 30 and len(set(target[active & (weights > 0)].tolist())) == 2
        state["feasible"][str(historical)] = bool(feasible)
        if not feasible:
            continue
        pooled = pav_fit(score[active, historical], target[active], weights[active])
        pooled.update(unique_patients=support)
        state["pooled"][str(historical)] = pooled
        state["parameter_ratios"].append(pooled["blocks"] / support)
        for seed in range(3):
            local_mask = active & (seeds == seed)
            local_support = len(set(patients[local_mask & (weights > 0)].tolist()))
            local_feasible = local_support >= 10 and len(set(target[local_mask & (weights > 0)].tolist())) == 2
            key = "%d:%d" % (seed, historical)
            if local_feasible:
                local = pav_fit(score[local_mask, historical], target[local_mask], weights[local_mask])
                local.update(unique_patients=local_support,
                             n_eff=effective_patients(rows, local_mask, weights), fallback=False)
                state["local"][key] = local
                state["parameter_ratios"].append(local["blocks"] / local_support)
            else:
                state["local"][key] = {"unique_patients": local_support, "n_eff": 0.0, "fallback": True}
    state["max_parameter_ratio"] = max(state["parameter_ratios"], default=math.inf)
    return state


def calibrated_probabilities(state, rows, kappa, pooled_only=False):
    score = score_matrix(rows); seeds = np.asarray([row["seed"] for row in rows])
    output = np.full((len(rows), 2), np.nan)
    for historical in (0, 1):
        if not state["feasible"].get(str(historical), False):
            continue
        pooled = pav_predict(state["pooled"][str(historical)], score[:, historical])
        output[:, historical] = pooled
        if pooled_only or math.isinf(kappa):
            continue
        for seed in range(3):
            selected = seeds == seed
            if not selected.any():
                continue
            local = state["local"]["%d:%d" % (seed, historical)]
            if local["fallback"]:
                continue
            weight = local["n_eff"] / (local["n_eff"] + kappa)
            output[selected, historical] = (weight * pav_predict(local, score[selected, historical])
                                             + (1.0 - weight) * pooled[selected])
    return output


def route_policy(rows, predictions, tau, rho=RHO, minimum_predictions=190):
    """The shared final/full-realization route; denominator is finite predictions."""
    predictions = np.asarray(predictions, dtype=np.float64)
    top = top1_lowest(np.stack([row["alpha"] for row in rows]))
    route = np.full(len(rows), 2, dtype=np.int64)
    median = np.full(len(rows), np.nan); consensus = np.zeros(len(rows)); denominator = np.zeros(len(rows), int)
    for index, historical in enumerate(top):
        if historical not in (0, 1):
            continue
        valid = np.isfinite(predictions[:, index, historical])
        denominator[index] = int(valid.sum())
        if denominator[index] < minimum_predictions:
            continue
        values = predictions[valid, index, historical]
        median[index] = float(np.median(values)); consensus[index] = float(np.mean(values >= tau))
        if median[index] >= tau and consensus[index] >= rho:
            route[index] = int(historical)
    return route, median, consensus, denominator


def calibration_metrics(route, rows, weights):
    route = np.asarray(route); weights = np.asarray(weights)
    domains = np.asarray([row["domain_index"] for row in rows])
    top = top1_lowest(np.stack([row["alpha"] for row in rows]))
    selected = route < 2; correct = selected & (route == domains)
    precision_den = float(weights[selected].sum()); precision_num = float(weights[correct].sum())
    current = domains == 2
    beneficial = (domains < 2) & (top == domains)
    recall_den = float(weights[beneficial].sum()); recall_num = float(weights[correct & beneficial].sum())
    accepted_patients = len({row["patient_id"] for row, take in zip(rows, correct & (domains < 2)) if take})
    return {"route_precision": None if precision_den == 0 else precision_num / precision_den,
            "route_precision_numerator": precision_num, "route_precision_denominator": precision_den,
            "current_false_override": float(weights[selected & current].sum() / weights[current].sum()),
            "historical_recall": None if recall_den == 0 else recall_num / recall_den,
            "historical_recall_numerator": recall_num, "historical_recall_denominator": recall_den,
            "accepted_historical_patients": accepted_patients}


def candidate_id(kappa, tau):
    return "k%s_tau%03d" % ("inf" if math.isinf(kappa) else "%03d" % int(kappa), round(tau * 100))


def aligned_global(rows, count):
    ordered = sorted(rows, key=lambda row: row["row_index"])
    require([row["row_index"] for row in ordered] == list(range(count)), "global row_index alignment failed")
    return ordered


def mark_duplicates(hashes):
    members = {}
    for identifier, digest in hashes.items():
        members.setdefault(digest, []).append(identifier)
    return {identifier: min(members[digest]) if len(members[digest]) > 1 else None
            for identifier, digest in hashes.items()}


def quantile_or_none(values, q):
    return None if not values or any(value is None for value in values) else float(np.quantile(values, q))


def route_agreement(routes):
    routes = np.asarray(routes)
    if routes.size == 0:
        return 0.0
    return float(np.mean([max(Counter(routes[:, index].tolist()).values()) / len(routes)
                          for index in range(routes.shape[1])]))


def candidate_sort_key(row):
    kappa_rank = math.inf if row["kappa"] == "inf" else float(row["kappa"])
    return (-row["historical_recall_p10"], row["current_false_override_p90"],
            -row["route_agreement"], -kappa_rank, -row["tau"], row["candidate_id"])


def run_calibration_fold(output, fold, calibration, value, value_folds, protocol):
    held_patients = {row["patient_id"] for row, assigned in zip(value, value_folds) if assigned == fold}
    inner = [row for row in calibration if row["patient_id"] not in held_patients]
    outer = [row for row, assigned in zip(value, value_folds) if assigned == fold]
    require(not held_patients & {row["patient_id"] for row in inner}, "outer patient leaked into calibration")
    base_weights = patient_row_weights(inner)
    base_state = fit_calibrators(inner, base_weights)
    weights = np.stack([bootstrap_weights(inner, protocol["bootstrap"]["master_seed"] + fold * 1000,
                                          replicate) for replicate in range(200)])
    states = [fit_calibrators(inner, weights[replicate]) for replicate in range(200)]
    feasible = {str(h): np.asarray([state["feasible"].get(str(h), False) for state in states], dtype=bool)
                for h in (0, 1)}
    model_path = write_json_new(output / "calibration_models" / ("fold%d.json" % fold),
                                {"fold": fold, "base": base_state, "bayesian": states})
    weight_path = save_npz_new(output / "bootstrap_weights" / ("fold%d.npz" % fold), weights=weights,
                               feasible_h0=feasible["0"], feasible_h1=feasible["1"])
    candidates, arrays = [], {}
    for kappa in KAPPAS:
        partial_inner = np.stack([calibrated_probabilities(state, inner, kappa) for state in states])
        partial_outer = np.stack([calibrated_probabilities(state, outer, kappa) for state in states])
        pooled_outer = np.stack([calibrated_probabilities(state, outer, math.inf, pooled_only=True)
                                 for state in states])
        base_outer = calibrated_probabilities(base_state, outer, kappa)[None, ...]
        for tau in TAUS:
            identifier = candidate_id(kappa, tau)
            replicate_routes, metrics = [], []
            for replicate in range(200):
                route = route_policy(inner, partial_inner[replicate:replicate + 1], tau,
                                     minimum_predictions=1)[0]
                replicate_routes.append(route)
                metrics.append(calibration_metrics(route, inner, base_weights))
            replicate_routes = np.stack(replicate_routes)
            precision_p10 = quantile_or_none([row["route_precision"] for row in metrics], 0.10)
            current_p90 = quantile_or_none([row["current_false_override"] for row in metrics], 0.90)
            recall_p10 = quantile_or_none([row["historical_recall"] for row in metrics], 0.10)
            accepted_median = float(np.median([row["accepted_historical_patients"] for row in metrics]))
            selected_route, median, consensus, denominator = route_policy(outer, partial_outer, tau)
            c4 = route_policy(outer, pooled_outer, tau)[0]
            c5 = route_policy(outer, base_outer, tau, rho=0.0, minimum_predictions=1)[0]
            arrays[identifier + "_C4"] = c4
            arrays[identifier + "_C5"] = c5
            arrays[identifier + "_C6"] = selected_route
            arrays[identifier + "_C6_realizations"] = np.stack([
                route_policy(outer, partial_outer[replicate:replicate + 1], tau,
                             minimum_predictions=1)[0] for replicate in range(200)])
            calibration_pass = (precision_p10 is not None and precision_p10 >= 0.98
                                and current_p90 is not None and current_p90 <= 0.02
                                and recall_p10 is not None and recall_p10 >= 0.30
                                and accepted_median >= 30
                                and all(int(feasible[str(h)].sum()) >= 195 for h in (0, 1)))
            candidates.append({
                "candidate_id": identifier, "kappa": "inf" if math.isinf(kappa) else int(kappa),
                "tau": tau, "rho": RHO, "calibration_pass": bool(calibration_pass),
                "route_precision_p10": precision_p10,
                "current_false_override_p90": current_p90,
                "historical_recall_p10": recall_p10,
                "median_accepted_historical_patients": accepted_median,
                "route_agreement": route_agreement(replicate_routes),
                "outer_route_count": int(np.sum(selected_route < 2)),
                "outer_route_frequency": float(np.mean(selected_route < 2)),
                "minimum_outer_finite_predictions": int(denominator.min()),
                "route_sha256": array_hash(selected_route),
                "median_probability_sha256": array_hash(median),
                "consensus_sha256": array_hash(consensus),
            })
    route_path = save_npz_new(output / "candidate_routes" / ("fold%d.npz" % fold), **arrays)
    seal = {
        "status": "PASS_PRE_GT_CALIBRATION_ARTIFACTS_SEALED",
        "fold": fold,
        "outer_GT_reads": 0,
        "outer_domain_reads": 0,
        "model_sha256": sha256_file(model_path),
        "bootstrap_sha256": sha256_file(weight_path),
        "route_sha256": sha256_file(route_path),
        "candidate_count": len(candidates),
    }
    seal_path = write_json_new(output / "candidate_preseals" / ("fold%d.json" % fold), seal)
    observed = read_json(seal_path)
    require(observed == seal and sha256_file(model_path) == seal["model_sha256"]
            and sha256_file(weight_path) == seal["bootstrap_sha256"]
            and sha256_file(route_path) == seal["route_sha256"], "pre-GT calibration seal failed")
    pooled_support = {str(h): base_state["pooled"].get(str(h), {}).get("unique_patients", 0) for h in (0, 1)}
    ratios = [state["max_parameter_ratio"] for state in [base_state] + states]
    return {
        "fold": fold, "inner_rows": len(inner),
        "inner_unique_patients": len({row["patient_id"] for row in inner}),
        "outer_rows": len(outer), "outer_unique_patients": len(held_patients),
        "pooled_active_support": pooled_support,
        "bayesian_feasible": {str(h): int(feasible[str(h)].sum()) for h in (0, 1)},
        "max_calibration_parameter_ratio": max(ratios),
        "candidates": candidates, "arrays": arrays,
        "preseal": {**seal, "seal_sha256": sha256_file(seal_path)},
        "inner": inner, "outer": outer, "base_state": base_state, "states": states,
        "weights": weights,
    }


def support_counts(calibration, value, folds):
    per_seed_domain = []
    for seed in range(3):
        for domain in range(3):
            rows = [row for row in calibration if row["seed"] == seed and row["domain_index"] == domain]
            per_seed_domain.append({"seed": seed, "domain_index": domain, "rows": len(rows),
                                    "unique_patients": len({row["patient_id"] for row in rows})})
    overlap = [[len({row["patient_id"] for row in calibration if row["seed"] == left}
                    & {row["patient_id"] for row in calibration if row["seed"] == right})
                for right in range(3)] for left in range(3)]
    value_overlap = [[len({row["patient_id"] for row in value if row["seed"] == left}
                          & {row["patient_id"] for row in value if row["seed"] == right})
                      for right in range(3)] for left in range(3)]
    return {
        "calibration_rows": len(calibration),
        "calibration_unique_patients": len({row["patient_id"] for row in calibration}),
        "segmentation_value_rows": len(value),
        "segmentation_value_unique_patients": len({row["patient_id"] for row in value}),
        "calibration_seed_domain": per_seed_domain,
        "calibration_cross_seed_patient_overlap": overlap,
        "segmentation_cross_seed_patient_overlap": value_overlap,
        "outer_folds": [{"fold": fold, "rows": int(np.sum(folds == fold)),
                         "unique_patients": len({row["patient_id"] for row, assigned in zip(value, folds)
                                                 if assigned == fold})} for fold in range(5)],
    }


def design_preflight(output, calibration, value, folds, protocol):
    reports = [run_calibration_fold(output, fold, calibration, value, folds, protocol) for fold in range(5)]
    identifiers = [candidate_id(kappa, tau) for kappa in KAPPAS for tau in TAUS]
    global_routes = {}
    for identifier in identifiers:
        route = np.full(len(value), 2, dtype=np.int64)
        for report in reports:
            indices = np.flatnonzero(folds == report["fold"])
            route[indices] = report["arrays"][identifier + "_C6"]
        global_routes[identifier] = route
    hashes = {identifier: array_hash(route) for identifier, route in global_routes.items()}
    duplicates = mark_duplicates(hashes)
    frequencies = {identifier: float(np.mean(route < 2)) for identifier, route in global_routes.items()}
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
        "parameters_affect_or_duplicate_marked": all(duplicates[identifier] is not None
                                                       or list(hashes.values()).count(hashes[identifier]) == 1
                                                       for identifier in identifiers),
        "calibration_parameter_ratio_le_0_10": all(report["max_calibration_parameter_ratio"] <= 0.10
                                                    for report in reports),
        "pre_GT_seals_verified": all(report["preseal"]["status"]
                                      == "PASS_PRE_GT_CALIBRATION_ARTIFACTS_SEALED" for report in reports),
    }
    public_reports = []
    for report in reports:
        public_reports.append({key: report[key] for key in ("fold", "inner_rows", "inner_unique_patients",
                              "outer_rows", "outer_unique_patients", "pooled_active_support",
                              "bayesian_feasible", "max_calibration_parameter_ratio", "candidates", "preseal")})
    preflight = {
        "schema_version": 1,
        "registration_id": protocol["registration_id"],
        "status": ("PASS_PPC_SHOR_V0_6A_DESIGN_PREFLIGHT" if all(gates.values())
                   else "BLOCKED_DESIGN_DEGENERATE_BEFORE_GT"),
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "population": support_counts(calibration, value, folds),
        "candidate_global_route_frequency": frequencies,
        "candidate_global_route_sha256": hashes,
        "duplicate_of": duplicates,
        "unique_candidate_route_arrays": len(set(hashes.values())),
        "folds": public_reports,
        "outer_GT_reads": 0,
        "outer_domain_reads": 0,
        "v0_4_formal_03_reads": 0,
    }
    write_json_new(output / "public/PPC_SHOR_V0_6A_DESIGN_PREFLIGHT.json", preflight)
    return preflight, reports, global_routes


def select_candidates(reports):
    selected = []
    for report in reports:
        feasible = [row for row in report["candidates"] if row["calibration_pass"]]
        if not feasible:
            raise CalibrationFailure("no candidate passed in fold %d" % report["fold"])
        selected.append(min(feasible, key=candidate_sort_key))
    return selected


def sensitivity_analysis(output, report, selected, protocol):
    inner, outer = report["inner"], report["outer"]
    kappa = math.inf if selected["kappa"] == "inf" else float(selected["kappa"])
    tau = selected["tau"]
    ordinary_weights = np.stack([bootstrap_weights(
        inner, protocol["bootstrap"]["sensitivities"]["ordinary_clustered_bootstrap"]["master_seed"]
        + report["fold"] * 1000, replicate, ordinary=True) for replicate in range(200)])
    ordinary_routes, ordinary_feasible = [], {"0": 0, "1": 0}
    for weights in ordinary_weights:
        state = fit_calibrators(inner, weights)
        for historical in (0, 1):
            ordinary_feasible[str(historical)] += int(state["feasible"].get(str(historical), False))
        prediction = calibrated_probabilities(state, outer, kappa)[None, ...]
        ordinary_routes.append(route_policy(outer, prediction, tau, minimum_predictions=1)[0])
    patients = sorted({row["patient_id"] for row in inner})
    base = patient_row_weights(inner)
    jackknife_routes, jackknife_feasible = [], {"0": 0, "1": 0}
    for patient in patients:
        weights = base.copy()
        weights[np.asarray([row["patient_id"] == patient for row in inner])] = 0.0
        state = fit_calibrators(inner, weights)
        for historical in (0, 1):
            jackknife_feasible[str(historical)] += int(state["feasible"].get(str(historical), False))
        prediction = calibrated_probabilities(state, outer, kappa)[None, ...]
        jackknife_routes.append(route_policy(outer, prediction, tau, minimum_predictions=1)[0])
    path = save_npz_new(output / "sensitivity" / ("fold%d.npz" % report["fold"]),
                        ordinary_weights=ordinary_weights,
                        ordinary_routes=np.stack(ordinary_routes),
                        jackknife_routes=np.stack(jackknife_routes))
    return {"ordinary_replicates": 200, "ordinary_feasible": ordinary_feasible,
            "ordinary_route_agreement": route_agreement(ordinary_routes),
            "jackknife_replicates": len(patients), "jackknife_feasible": jackknife_feasible,
            "jackknife_route_agreement": route_agreement(jackknife_routes),
            "artifact_sha256": sha256_file(path), "relative_path": path.relative_to(output).as_posix()}


def frozen_thresholds(protocol, oof_root):
    path = oof_root / protocol["inputs"]["frozen_shor_threshold_manifest"]["path"]
    manifest = read_json(path); output = {}
    for seed in range(3):
        rows = [row for row in manifest["formal"] if row["seed"] == seed and row["stage_index"] == 2]
        require(len(rows) == 2 and all(row["feasible"] for row in rows), "frozen C3 threshold missing")
        output[seed] = {row["historical_domain"]: float(row["threshold"]) for row in rows}
    return output


def baseline_checkpoints(protocol):
    spec = protocol["inputs"]["baseline_snapshot_manifest"]
    path = verify_file(REPO / spec["path"], spec["sha256"])
    rows = read_json(path)["checkpoints"]
    checkpoints = {(row["seed"], row["stage_index"]): row for row in rows}
    require(set(checkpoints) == {(seed, stage) for seed in range(3) for stage in range(3)},
            "nine frozen checkpoints missing", RequiredInputMissing)
    return checkpoints


def materialize_experts(output, value, protocol, device, metadata):
    checkpoints = baseline_checkpoints(protocol)
    paths, batches = {}, 0
    for seed in range(3):
        rows = [row for row in value if row["seed"] == seed]
        require(len(rows) == 66, "seed value population changed")
        for row in rows:
            path = safe_asset(protocol["inputs"]["data_root"], row["image_h5_relpath"])
            require(path.is_file(), "missing value image", RequiredInputMissing)
        for expert in range(3):
            paths[(seed, expert)] = v4.predict_expert(output, seed, expert, rows,
                checkpoints[(seed, expert)], protocol["inputs"]["data_root"], device, metadata)
        batches += math.ceil(len(rows) / v4.BATCH_SIZE) * 3
    return paths, batches


def expert_views(value, paths):
    by_seed = {seed: [row for row in value if row["seed"] == seed] for seed in range(3)}
    local = {(seed, row["case_id"]): index for seed, rows in by_seed.items() for index, row in enumerate(rows)}
    arrays = {(seed, expert): np.load(paths[(seed, expert)], mmap_mode="r", allow_pickle=False)
              for seed in range(3) for expert in range(3)}
    return arrays, local


def materialize_fold_predictions(output, fold, rows, routes, arrays, local):
    paths = {}
    for policy in tuple("C%d" % value for value in range(7)):
        path = output / "candidate_predictions" / ("fold%d_%s.npy" % (fold, policy))
        path.parent.mkdir(parents=True, exist_ok=True)
        target = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                           shape=(len(rows), 384, 384))
        for index, row in enumerate(rows):
            seed, position = row["seed"], local[(row["seed"], row["case_id"])]
            if policy == "C2":
                alpha = row["alpha"]
                probability = sum(float(alpha[expert]) * arrays[(seed, expert)][position]
                                  for expert in range(3))
                target[index] = np.argmax(probability, axis=0).astype(np.uint8)
            else:
                expert = int(routes[policy][index])
                target[index] = np.argmax(arrays[(seed, expert)][position], axis=0).astype(np.uint8)
        target.flush(); del target; paths[policy] = path
    return paths


class EvaluatorAccess:
    def __init__(self):
        self.sealed_folds = set(); self.outer_domain_reads = 0; self.outer_GT_reads = 0

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
        path = Path(protocol["inputs"]["data_root"]) / "manifests/training" / ("lcrseg_v1_seed%d.csv" % seed)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (seed, row["case_id"])
                if key not in wanted:
                    continue
                require(row["dataset"] == "fundus" and row["primary_20pct_split"] == "train_labeled"
                        and row["site_or_vendor"] in DOMAINS and row["label_h5_relpath"],
                        "outer evaluator row changed", RequiredInputMissing)
                found[wanted[key]] = row; access.outer_domain_reads += 1
    require(set(found) == set(range(len(expected))), "outer evaluator population incomplete")
    return [found[index] for index in range(len(expected))]


def read_label(row, protocol, fold, access):
    access.require(fold)
    path = safe_asset(protocol["inputs"]["data_root"], row["label_h5_relpath"])
    require(path.is_file(), "missing outer label", RequiredInputMissing)
    with h5py.File(path, "r") as handle:
        label = np.asarray(handle["label"][...], dtype=np.int64)
    require(label.shape == (384, 384), "outer label geometry changed", RequiredInputMissing)
    access.outer_GT_reads += 1
    return label


def route_metrics(routes, utility_fg, utility_class, seeds, domains, patients):
    routes = np.asarray(routes); delta = np.zeros(len(routes)); class_delta = np.zeros((len(routes), 2))
    for historical in (0, 1):
        selected = routes == historical
        delta[selected] = utility_fg[selected, historical]
        class_delta[selected] = utility_class[selected, historical]
    def balanced(value, selected_domains=(0, 1, 2)):
        groups = [float(np.mean(value[(seeds == seed) & (domains == domain)]))
                  for seed in range(3) for domain in selected_domains
                  if np.any((seeds == seed) & (domains == domain))]
        require(len(groups) == 3 * len(selected_domains), "missing metric group")
        return float(np.mean(groups))
    selected = routes < 2; precision_den = int(selected.sum())
    precision_num = int(np.sum(delta[selected] > 0))
    beneficial = (domains < 2) & (np.max(utility_fg, axis=1) > 0)
    current_gain = balanced(delta, (2,))
    seed_domain = [float(np.mean(delta[(seeds == seed) & (domains == domain)]))
                   for seed in range(3) for domain in range(3)]
    return {
        "three_domain_gain": balanced(delta), "shared_gain": balanced(delta),
        "historical_gain": balanced(delta, (0, 1)),
        "REFUGE_gain": balanced(delta, (0,)), "RIM_ONE_r3_gain": balanced(delta, (1,)),
        "current_domain_drop": max(0.0, -current_gain),
        "maximum_current_class_drop": max([0.0] + [-balanced(class_delta[:, c], (2,)) for c in range(2)]),
        "maximum_seed_domain_drop": max([0.0] + [-value for value in seed_domain]),
        "positive_seed_count": int(sum(np.mean(seed_domain[seed * 3:(seed + 1) * 3]) > 0 for seed in range(3))),
        "route_frequency": float(np.mean(selected)), "route_count": precision_den,
        "route_precision": None if precision_den == 0 else precision_num / precision_den,
        "route_precision_numerator": precision_num, "route_precision_denominator": precision_den,
        "historical_recall": (float(np.mean(selected[beneficial])) if beneficial.any() else None),
        "historical_recall_numerator": int(np.sum(selected & beneficial)),
        "historical_recall_denominator": int(beneficial.sum()),
        "delta": delta, "class_delta": class_delta,
    }


def aggregate_case_metrics(case_rows):
    rows = []
    dimensions = [("overall", lambda row: "all"), ("seed", lambda row: str(row["seed"])),
                  ("domain", lambda row: row["domain"]),
                  ("seed_domain", lambda row: "%d:%s" % (row["seed"], row["domain"]))]
    for level, key in dimensions:
        for value in sorted({key(row) for row in case_rows}):
            for policy in POLICIES:
                selected = [row for row in case_rows if row["policy"] == policy and key(row) == value]
                if not selected:
                    continue
                def mean(field):
                    groups = {}
                    for row in selected:
                        group = ((row["seed"], row["domain_index"]) if level == "overall" else
                                 (row["domain_index"],) if level == "seed" else
                                 (row["seed"],) if level == "domain" else (0,))
                        groups.setdefault(group, []).append(row[field])
                    return float(np.mean([np.mean(values) for values in groups.values()]))
                rows.append({"level": level, "key": value, "policy": policy, "cases": len(selected),
                             "foreground_dice": mean("foreground_dice"), "rim_dice": mean("rim_dice"),
                             "cup_dice": mean("cup_dice"), "mean_iou": mean("mean_iou")})
    return rows


def verify_candidate_seal(path, output):
    seal = read_json(path)
    require(seal["status"] == "PASS_OUTER_CANDIDATES_SEALED_BEFORE_GT"
            and seal["outer_GT_reads"] == seal["outer_domain_reads"] == 0,
            "candidate seal status changed")
    for relative, expected in seal["sealed_files"].items():
        require(sha256_file(output / relative) == expected, "post-seal artifact changed: " + relative)
    return seal


def private_inventory(output):
    files = [path for path in Path(output).rglob("*") if path.is_file()
             and "public" not in path.relative_to(output).parts]
    return {"files": len(files), "bytes": sum(path.stat().st_size for path in files)}


def copy_registration(output):
    public = Path(output) / "public"; public.mkdir(parents=True, exist_ok=True)
    for name in ("PPC_SHOR_V0_6A_PREREGISTRATION.json", "PPC_SHOR_V0_6A_PREREGISTRATION.md"):
        target = public / name
        with target.open("xb") as handle:
            handle.write((DOCS / name).read_bytes()); handle.flush(); os.fsync(handle.fileno())


def public_manifest(output, scientific_status):
    public = Path(output) / "public"
    expected = {
        "PPC_SHOR_V0_6A_PREREGISTRATION.json", "PPC_SHOR_V0_6A_PREREGISTRATION.md",
        "PPC_SHOR_V0_6A_DESIGN_PREFLIGHT.json", "PPC_SHOR_V0_6A_STATUS.json",
        "PPC_SHOR_V0_6A_FINAL_REPORT.md", "PPC_SHOR_V0_6A_METRICS.csv",
        "PPC_SHOR_V0_6A_ROUTING.csv", "PPC_SHOR_V0_6A_CALIBRATION.csv",
        "PPC_SHOR_V0_6A_FAILURES_AND_WARNINGS.md", "PPC_SHOR_V0_6A_EXACT_COMMANDS.md",
    }
    observed = {path.name for path in public.iterdir() if path.is_file()}
    require(observed == expected, "public output set incomplete")
    entries = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
               for path in sorted(public.iterdir())]
    manifest = {"schema_version": 1, "status": "PASS_PUBLIC_REPORT_BUNDLE_COMPLETE",
                "scientific_status": scientific_status, "entries": entries,
                "files": len(entries), "bytes": sum(row["bytes"] for row in entries),
                "content_sha256": canonical_hash(entries), "private_artifacts_published": False}
    write_json_new(public / "PPC_SHOR_V0_6A_MANIFEST.json", manifest)


def exact_commands(_output):
    return """# PPC-SHOR V0.6A exact commands

Private paths are represented by bound variables in the public record.

```bash
git switch codex/ppc-shor-v0-6a-development
python -m pytest -q experiments/lcrseg/tests/ppc_shor_v0_6a
CODE_COMMIT=$(git rev-parse HEAD)
bash experiments/lcrseg/scripts/with_nas_storage.sh python experiments/lcrseg/ppc_shor_v0_6a.py \\
  --output <NAS_CREATE_ONLY_FORMAL_01> --code-commit "$CODE_COMMIT" \\
  --test-report <NAS_TEST_REPORT> --device cuda:1
```

V0.4 `formal_03` was not an input.
"""


def publish_early_stop(output, protocol, preflight, status, reason, tests, publication):
    public = Path(output) / "public"
    calibration = [dict(fold=fold["fold"], **row) for fold in preflight["folds"]
                   for row in fold["candidates"]]
    routing = [{"candidate_id": key, "route_frequency": value,
                "route_sha256": preflight["candidate_global_route_sha256"][key],
                "duplicate_of": preflight["duplicate_of"][key]}
               for key, value in preflight["candidate_global_route_frequency"].items()]
    write_csv_new(public / "PPC_SHOR_V0_6A_CALIBRATION.csv", calibration)
    write_csv_new(public / "PPC_SHOR_V0_6A_ROUTING.csv", routing)
    write_csv_new(public / "PPC_SHOR_V0_6A_METRICS.csv", [],
                  ["level", "key", "policy", "cases", "foreground_dice", "rim_dice", "cup_dice", "mean_iou"])
    inventory = private_inventory(output)
    payload = {
        "schema_version": 1, "registration_id": protocol["registration_id"],
        "scientific_status": status, "reason": reason, "formal_attempt": 1,
        "design_preflight": preflight["status"], "outer_GT_reads": 0, "outer_domain_reads": 0,
        "v0_4_formal_03_reads": 0, "segmentation_training_steps": 0,
        "segmentation_optimizer_steps": 0, "segmentation_parameter_updates": 0,
        "router_optimizer_steps": 0, "router_parameter_updates": 0,
        "C0_C8": None, "candidate_seal_reverification": "not reached after required stop",
        "tests": tests, "publication": publication, "private_artifact_inventory": inventory,
        "main_merged": False, "external_test_launched": False,
        "report_commit": None, "report_commit_resolution": "second commit adding exact report bytes",
    }
    write_json_new(public / "PPC_SHOR_V0_6A_STATUS.json", payload)
    write_text_new(public / "PPC_SHOR_V0_6A_FINAL_REPORT.md",
                   "# PPC-SHOR V0.6A final report\n\n**%s**\n\n%s\n\n"
                   "The required stop occurred before all segmentation GT/domain access. No C0-C8 value "
                   "evaluation, model update, main merge, or external test occurred.\n" % (status, reason))
    write_text_new(public / "PPC_SHOR_V0_6A_FAILURES_AND_WARNINGS.md",
                   "# Failures and warnings\n\n- `%s`: %s\n- All Fundus data remain development-consumed.\n"
                   % (status, reason))
    write_text_new(public / "PPC_SHOR_V0_6A_EXACT_COMMANDS.md", exact_commands(output))
    public_manifest(output, status)
    return payload


def execute_formal(output, protocol, calibration, value, folds, oof_root, reports, selected,
                   device, metadata, tests, publication):
    sensitivity = [sensitivity_analysis(output, report, chosen, protocol)
                   for report, chosen in zip(reports, selected)]
    expert_paths, forward_batches = materialize_experts(output, value, protocol, device, metadata)
    expert_arrays, local = expert_views(value, expert_paths)
    thresholds = frozen_thresholds(protocol, oof_root)
    access = EvaluatorAccess()
    global_routes = {policy: np.full(len(value), -1, dtype=np.int64) for policy in POLICIES[:7]}
    global_realizations = np.full((200, len(value)), 2, dtype=np.int64)
    final_consensus = np.zeros(len(value)); final_denominator = np.zeros(len(value), dtype=int)
    fold_prediction_paths, seal_receipts = {}, []
    for report, chosen, sensitive in zip(reports, selected, sensitivity):
        fold = report["fold"]; indices = np.flatnonzero(folds == fold); rows = report["outer"]
        identifier = chosen["candidate_id"]
        alpha = np.stack([row["alpha"] for row in rows]); top = top1_lowest(alpha)
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
        global_realizations[:, indices] = report["arrays"][identifier + "_C6_realizations"]
        kappa = math.inf if chosen["kappa"] == "inf" else float(chosen["kappa"])
        probability = np.stack([calibrated_probabilities(state, rows, kappa) for state in report["states"]])
        observed_route, _, consensus, denominator = route_policy(rows, probability, chosen["tau"])
        require(np.array_equal(observed_route, routes["C6"]), "selected C6 route changed")
        final_consensus[indices] = consensus; final_denominator[indices] = denominator
        prediction_paths = materialize_fold_predictions(output, fold, rows, routes, expert_arrays, local)
        fold_prediction_paths[fold] = prediction_paths
        sealed_files = {
            (Path("calibration_models") / ("fold%d.json" % fold)).as_posix(): report["preseal"]["model_sha256"],
            (Path("bootstrap_weights") / ("fold%d.npz" % fold)).as_posix(): report["preseal"]["bootstrap_sha256"],
            (Path("candidate_routes") / ("fold%d.npz" % fold)).as_posix(): report["preseal"]["route_sha256"],
            sensitive["relative_path"]: sensitive["artifact_sha256"],
        }
        sealed_files.update({path.relative_to(output).as_posix(): sha256_file(path)
                             for path in prediction_paths.values()})
        seal = {"status": "PASS_OUTER_CANDIDATES_SEALED_BEFORE_GT", "fold": fold,
                "outer_GT_reads": 0, "outer_domain_reads": 0, "selected_candidate": chosen,
                "case_order_sha256": canonical_hash([(row["seed"], row["case_id"]) for row in rows]),
                "sealed_files": sealed_files}
        seal_path = write_json_new(output / "candidate_seals" / ("fold%d.json" % fold), seal)
        verify_candidate_seal(seal_path, output); access.seal(fold)
        seal_receipts.append({"fold": fold, "seal_sha256": sha256_file(seal_path),
                              "reverified": True, "files": len(sealed_files)})

    require(all(np.all(global_routes[policy] >= (-1 if policy == "C2" else 0)) for policy in POLICIES[:7]),
            "global route arrays incomplete")
    case_rows = []; expert_fg = np.empty((len(value), 3)); expert_class = np.empty((len(value), 3, 2))
    domains = np.empty(len(value), dtype=np.int64)
    for report in reports:
        fold = report["fold"]; indices = np.flatnonzero(folds == fold); rows = report["outer"]
        for path in [output / "candidate_seals" / ("fold%d.json" % fold)]:
            verify_candidate_seal(path, output)
        revealed = reveal_rows(protocol, rows, fold, access)
        predictions = {policy: np.load(path, mmap_mode="r", allow_pickle=False)
                       for policy, path in fold_prediction_paths[fold].items()}
        for local_index, (index, blind, full) in enumerate(zip(indices, rows, revealed)):
            label = read_label(full, protocol, fold, access)
            domain = DOMAINS.index(full["site_or_vendor"]); domains[index] = domain
            expected_domain = next(row["domain_index"] for row in calibration
                                   if row["seed"] == blind["seed"] and row["case_id"] == blind["case_id"])
            require(domain == expected_domain, "frozen OOF domain label changed")
            local_position = local[(blind["seed"], blind["case_id"])]
            expert_metrics = []
            for expert in range(3):
                hard = np.argmax(expert_arrays[(blind["seed"], expert)][local_position], axis=0)
                metric = v4.case_metrics(hard, label); expert_metrics.append(metric)
                expert_fg[index, expert] = metric["foreground_dice"]
                expert_class[index, expert] = (metric["rim_dice"], metric["cup_dice"])
            for policy in POLICIES[:7]:
                metric = v4.case_metrics(predictions[policy][local_index], label)
                case_rows.append({"row_index": int(index), "fold": fold, "seed": blind["seed"],
                                  "case_id": blind["case_id"], "patient_id": blind["patient_id"],
                                  "domain_index": domain, "domain": DOMAINS[domain], "policy": policy,
                                  "route": int(global_routes[policy][index]), **metric})
            c7, c8 = domain, int(np.argmax(expert_fg[index]))
            for policy, expert in (("C7", c7), ("C8", c8)):
                metric = expert_metrics[expert]
                case_rows.append({"row_index": int(index), "fold": fold, "seed": blind["seed"],
                                  "case_id": blind["case_id"], "patient_id": blind["patient_id"],
                                  "domain_index": domain, "domain": DOMAINS[domain], "policy": policy,
                                  "route": expert, **metric})
                global_routes.setdefault(policy, np.full(len(value), -1, dtype=np.int64))[index] = expert

    with (output / "case_metrics.jsonl").open("x", encoding="utf-8") as handle:
        for row in case_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    utility_fg = expert_fg[:, :2] - expert_fg[:, 2, None]
    utility_class = expert_class[:, :2] - expert_class[:, 2, None, :]
    seeds = np.asarray([row["seed"] for row in value]); patients = np.asarray([row["patient_id"] for row in value])
    summaries = {policy: route_metrics(global_routes[policy], utility_fg, utility_class,
                                        seeds, domains, patients) for policy in ("C3", "C4", "C5", "C6")}
    stability_rows = [route_metrics(route, utility_fg, utility_class, seeds, domains, patients)
                      for route in global_realizations]
    stability = {
        "historical_gain_p10": float(np.quantile([row["historical_gain"] for row in stability_rows], 0.10)),
        "shared_gain_p10": float(np.quantile([row["shared_gain"] for row in stability_rows], 0.10)),
        "current_domain_drop_p90": float(np.quantile([row["current_domain_drop"] for row in stability_rows], 0.90)),
        "maximum_seed_domain_drop_p90": float(np.quantile(
            [row["maximum_seed_domain_drop"] for row in stability_rows], 0.90)),
        "route_disagreement": float(np.mean([len(set(global_realizations[:, index].tolist())) > 1
                                             for index in range(len(value))])),
        "median_per_case_route_consensus": float(np.median([
            max(Counter(global_realizations[:, index].tolist()).values()) / 200.0
            for index in range(len(value))])),
        "finite_prediction_denominator_min": int(final_denominator.min()),
        "finite_prediction_denominator_median": float(np.median(final_denominator)),
        "finite_prediction_denominator_max": int(final_denominator.max()),
        "per_fold_expert_feasible": [{"fold": report["fold"], "historical_expert": h,
                                      "numerator": report["bayesian_feasible"][str(h)], "denominator": 200}
                                     for report in reports for h in (0, 1)],
        "full_policy_realizations_use_route_policy": True,
        "per_unit_global_intersection_conflated": False,
    }
    metrics = aggregate_case_metrics(case_rows)
    overall = {row["policy"]: row for row in metrics if row["level"] == "overall"}
    c6, c3 = summaries["C6"], summaries["C3"]
    domain_oracle_gap = overall["C7"]["foreground_dice"] - overall["C6"]["foreground_dice"]
    nondegenerate = (0.15 <= c6["route_frequency"] <= 0.85 and c6["route_count"] >= 30
                     and all(np.any(global_routes["C6"][folds == fold] < 2) for fold in range(5)))
    value_gate = (nondegenerate and c6["three_domain_gain"] >= 0.15 and c6["historical_gain"] >= 0.25
                  and c6["REFUGE_gain"] > 0 and c6["RIM_ONE_r3_gain"] > 0
                  and c6["positive_seed_count"] == 3 and domain_oracle_gap <= 0.06)
    noninferiority = ((c6["three_domain_gain"] - c3["three_domain_gain"]) >= -0.010
                      and (c6["historical_gain"] - c3["historical_gain"]) >= -0.020)
    safety_gate = (c6["current_domain_drop"] <= 0.010 and c6["maximum_current_class_drop"] <= 0.015
                   and c6["maximum_seed_domain_drop"] <= 0.020
                   and stability["current_domain_drop_p90"] <= 0.015
                   and stability["maximum_seed_domain_drop_p90"] <= 0.025)
    stability_gate = (all(row["numerator"] >= 195 for row in stability["per_fold_expert_feasible"])
                      and stability["historical_gain_p10"] >= 0.20 and stability["shared_gain_p10"] >= 0.10
                      and stability["route_disagreement"] <= 0.15
                      and stability["median_per_case_route_consensus"] >= 0.80)
    if not safety_gate:
        status = "FAIL_PPC_SHOR_CURRENT_SAFETY"
    elif not value_gate or not noninferiority:
        status = "FAIL_PPC_SHOR_VALUE"
    elif not stability_gate:
        status = "FAIL_PPC_SHOR_STABILITY"
    else:
        status = "PASS_PPC_SHOR_DEVELOPMENT_FEASIBILITY"
    public = output / "public"
    write_csv_new(public / "PPC_SHOR_V0_6A_METRICS.csv", metrics)
    calibration_rows = [dict(fold=report["fold"], selected=row["candidate_id"] == chosen["candidate_id"], **row)
                        for report, chosen in zip(reports, selected) for row in report["candidates"]]
    write_csv_new(public / "PPC_SHOR_V0_6A_CALIBRATION.csv", calibration_rows)
    routing_rows = [{"level": "overall", "policy": policy,
                     **{key: value for key, value in summaries[policy].items() if key not in ("delta", "class_delta")}}
                    for policy in ("C3", "C4", "C5", "C6")]
    routing_rows += [{"level": "outer_fold_selection", "policy": "C6", "fold": fold,
                      "candidate_id": row["candidate_id"], "kappa": row["kappa"], "tau": row["tau"],
                      "route_frequency": row["outer_route_frequency"],
                      "finite_denominator": row["minimum_outer_finite_predictions"]}
                     for fold, row in enumerate(selected)]
    write_csv_new(public / "PPC_SHOR_V0_6A_ROUTING.csv", routing_rows)
    controls = {policy: {key: value for key, value in row.items() if key in
                ("foreground_dice", "rim_dice", "cup_dice", "mean_iou")}
                for policy, row in overall.items()}
    inventory = private_inventory(output)
    status_payload = {
        "schema_version": 1, "registration_id": protocol["registration_id"],
        "scientific_status": status, "formal_attempt": 1,
        "development_population": support_counts(calibration, value, folds),
        "controls": controls, "routing": {key: {k: v for k, v in row.items()
                                                if k not in ("delta", "class_delta")}
                                          for key, row in summaries.items()},
        "selected_candidates": selected, "stability": stability,
        "sensitivity": sensitivity, "domain_oracle_gap": domain_oracle_gap,
        "C6_minus_C3": {"overall_gain": c6["three_domain_gain"] - c3["three_domain_gain"],
                         "historical_gain": c6["historical_gain"] - c3["historical_gain"]},
        "gates": {"isolation": True, "nondegeneracy": nondegenerate, "value": value_gate,
                  "C3_noninferiority": noninferiority, "current_safety": safety_gate,
                  "stability": stability_gate},
        "isolation": {"v0_4_formal_03_reads": 0, "outer_GT_reads_before_verified_seal": 0,
                      "outer_domain_reads_before_verified_seal": 0, "outer_GT_reads": access.outer_GT_reads,
                      "outer_domain_reads": access.outer_domain_reads, "test_domain_inputs_to_C6": 0,
                      "segmentation_training_steps": 0, "segmentation_optimizer_steps": 0,
                      "segmentation_parameter_updates": 0, "router_optimizer_steps": 0,
                      "router_parameter_updates": 0, "segmentation_expert_forward_batches": forward_batches,
                      "old_artifact_mutations": 0},
        "candidate_seal_reverification": seal_receipts, "tests": tests, "publication": publication,
        "private_artifact_inventory": inventory, "main_merged": False,
        "external_test_launched": False, "report_commit": None,
        "report_commit_resolution": "second commit adding exact report bytes",
    }
    write_json_new(public / "PPC_SHOR_V0_6A_STATUS.json", status_payload)
    report = """# PPC-SHOR V0.6A final report

## Outcome

The sole development outer-OOF execution completed with **%s**. This is not held-out or
external confirmation; all Fundus data are development-consumed.

The population was 990 calibration rows / 575 patients and 198 own-seed `train_labeled`
segmentation rows / 177 patients. C6 routed %.6f of rows (%d/198). Its three-domain gain was
%.6f, historical gain %.6f, current-domain drop %.6f, and domain-oracle gap %.6f.

Stability: shared-gain p10 %.6f, historical-gain p10 %.6f, current-drop p90 %.6f,
maximum seed-domain-drop p90 %.6f, route disagreement %.6f, and median per-case route
consensus %.6f. Finite prediction denominators were %d to %d. Full-policy realizations used
the same `route_policy` function as final C6.

All candidate/model/bootstrap/prediction seals were reverified before %d outer GT/domain
reads. Segmentation and router optimizer/update counts were zero. V0.4 `formal_03` reads,
main merges, and external tests were zero.
""" % (status, c6["route_frequency"], c6["route_count"], c6["three_domain_gain"],
       c6["historical_gain"], c6["current_domain_drop"], domain_oracle_gap,
       stability["shared_gain_p10"], stability["historical_gain_p10"],
       stability["current_domain_drop_p90"], stability["maximum_seed_domain_drop_p90"],
       stability["route_disagreement"], stability["median_per_case_route_consensus"],
       stability["finite_prediction_denominator_min"], stability["finite_prediction_denominator_max"],
       access.outer_GT_reads)
    write_text_new(public / "PPC_SHOR_V0_6A_FINAL_REPORT.md", report)
    failed = [key for key, value in status_payload["gates"].items() if not value]
    write_text_new(public / "PPC_SHOR_V0_6A_FAILURES_AND_WARNINGS.md",
                   "# Failures and warnings\n\n- Scientific status: `%s`.\n- Failed gates: %s.\n"
                   "- All Fundus data are development-consumed; no external claim is permitted.\n"
                   % (status, ", ".join(failed) if failed else "none"))
    write_text_new(public / "PPC_SHOR_V0_6A_EXACT_COMMANDS.md", exact_commands(output))
    public_manifest(output, status)
    return status_payload


def load_test_report(path, code_commit):
    report = read_json(path)
    require(report["status"] == "PASS" and report["code_commit"] == code_commit
            and report["failures"] == report["errors"] == report["skips"] == 0,
            "freeze test report failed")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--test-report", required=True, type=Path)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    protocol = load_protocol()
    require(str(args.output.resolve()).startswith("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/"),
            "formal output is not on canonical NAS")
    require(not args.output.exists(), "REFUSED_AFTER_STATUS_EXISTS")
    publication = source_gate(protocol, args.code_commit)
    tests = load_test_report(args.test_report, args.code_commit)
    args.output.mkdir(parents=True)
    copy_registration(args.output)
    metadata = {"registration_id": protocol["registration_id"], "code_commit": args.code_commit,
                "source": publication, "segmentation_optimizer_steps": 0,
                "segmentation_parameter_updates": 0, "router_optimizer_steps": 0,
                "router_parameter_updates": 0}
    calibration, value, _, oof_root = load_population(protocol)
    folds = fold_assignments(value)
    preflight, reports, _ = design_preflight(args.output, calibration, value, folds, protocol)
    if not preflight["all_gates_pass"]:
        failed = [key for key, passed in preflight["gates"].items() if not passed]
        result = publish_early_stop(args.output, protocol, preflight,
                                    "BLOCKED_DESIGN_DEGENERATE_BEFORE_GT",
                                    "Failed GT-before-design gates: " + ", ".join(failed), tests, publication)
    else:
        try:
            selected = select_candidates(reports)
        except CalibrationFailure as error:
            result = publish_early_stop(args.output, protocol, preflight, error.status, str(error),
                                        tests, publication)
        else:
            result = execute_formal(args.output, protocol, calibration, value, folds, oof_root,
                                    reports, selected, args.device, metadata, tests, publication)
    write_json_new(args.output / "EXECUTION_RECEIPT.json",
                   {"status": "COMMAND_COMPLETED", "scientific_status": result["scientific_status"],
                    "code_commit": args.code_commit, "formal_attempt": 1,
                    "main_merged": False, "external_test_launched": False})
    print(json.dumps({"status": result["scientific_status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
