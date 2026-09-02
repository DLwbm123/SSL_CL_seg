#!/usr/bin/env python3
"""One-shot grouped outer-OOF evaluation of preregistered RC-SHOR V0.5."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

import h5py
import numpy as np
import torch
from scipy import ndimage

from di_dmpa_gate1.binding import safe_asset
from di_dmpa_gate1c_v2.binding import no_updates
from shor_jascl_v0_3.core import historical_score, shor_routes, top1_lowest
import shor_v0_4_test as v4


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DOCS = ROOT / "docs/rc_shor_v0_5"
PREREG = DOCS / "RC_SHOR_V0_5_PREREGISTRATION.json"
AUDIT = DOCS / "RC_SHOR_V0_5_SOURCE_AUDIT.json"
REMOTE = "https://github.com/DLwbm123/SSL_CL_seg.git"
DOMAINS = ("REFUGE", "RIM_ONE_r3", "Drishti_GS")
POLICIES = tuple("C%d" % index for index in range(9))
PUBLIC_FILES = (
    "RC_SHOR_V0_5_STATUS.json",
    "RC_SHOR_V0_5_FINAL_REPORT.md",
    "RC_SHOR_V0_5_METRICS.csv",
    "RC_SHOR_V0_5_ROUTING.csv",
    "RC_SHOR_V0_5_FAILURES_AND_WARNINGS.md",
    "RC_SHOR_V0_5_EXACT_COMMANDS.md",
)
BATCH_SIZE = 8


class ProtocolViolation(RuntimeError):
    status = "BLOCKED_PROTOCOL_OR_LEAKAGE"


class RequiredInputMissing(RuntimeError):
    status = "BLOCKED_REQUIRED_FROZEN_INPUT_MISSING"


def require(condition, message, error=ProtocolViolation):
    if not condition:
        raise error(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    return v4.sha256_file(path)


def canonical_hash(value):
    return v4.canonical_hash(value)


def array_hash(value):
    return v4.array_hash(value)


def write_json_new(path, value):
    return v4.write_json_new(path, value)


def write_csv_new(path, rows):
    return v4.write_csv_new(path, rows)


def write_text_new(path, value):
    return v4.write_text_new(path, value)


def load_protocol():
    protocol = read_json(PREREG)
    audit = read_json(AUDIT)
    require(protocol["registration_id"] == "RC_SHOR_V0_5_RISK_CONTROLLED_STABILITY", "wrong preregistration")
    require(protocol["base_commit"] == "6e42a04c4ea0547aeb89d430f96b551294cc3aaf"
            and protocol["branch"] == "codex/shor-v0-5-rc-stability", "wrong source boundary")
    require(protocol["immutable_history"]["SHOR_V0_3_1"]["H5"] is False
            and protocol["immutable_history"]["SHOR_V0_4"]["scientific_status"]
            == "PASS_FIXED_POLICY_TEST_EFFECTIVENESS", "history changed")
    require(audit["status"] == "PASS_SOURCE_DATA_AND_LEAKAGE_AUDIT"
            and audit["leakage_decision"]["decision"] == "CONTINUE_TO_PREREGISTRATION", "audit did not pass")
    return protocol, audit


def source_gate(protocol, code_commit, test_report):
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()
    require(head == code_commit and branch == protocol["branch"] and not dirty, "wrong or dirty execution source")
    count = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-list", "--count", protocol["base_commit"] + ".." + code_commit],
        text=True).strip()
    require(count == "1", "formal execution requires exactly one freeze commit")
    expected = {
        "experiments/lcrseg/docs/rc_shor_v0_5/RC_SHOR_V0_5_SOURCE_AUDIT.json",
        "experiments/lcrseg/docs/rc_shor_v0_5/RC_SHOR_V0_5_PREREGISTRATION.json",
        "experiments/lcrseg/docs/rc_shor_v0_5/RC_SHOR_V0_5_PREREGISTRATION.md",
        "experiments/lcrseg/rc_shor_v0_5.py",
        "experiments/lcrseg/tests/rc_shor_v0_5/test_protocol.py",
    }
    changed = set(subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "--name-only", protocol["base_commit"], code_commit],
        text=True).splitlines())
    require(changed == expected, "unregistered freeze-commit file set")
    for immutable in ("experiments/lcrseg/docs/shor_jascl_v0_3",
                      "experiments/lcrseg/docs/shor_jascl_v0_3_1",
                      "experiments/lcrseg/docs/shor_v0_4_fixed_policy_test"):
        quiet = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", protocol["base_commit"],
                                code_commit, "--", immutable]).returncode
        require(quiet == 0, "immutable historical directory changed")
    remote = subprocess.check_output(
        ["git", "ls-remote", REMOTE, "refs/heads/" + protocol["branch"]], text=True).split()
    require(remote and remote[0] == code_commit, "freeze commit is not published")
    report = read_json(test_report)
    require(report["status"] == "PASS" and report["code_commit"] == code_commit
            and report["failures"] == report["errors"] == report["skips"] == 0, "test gate failed")
    require(sha256_file(report["junit_path"]) == report["junit_sha256"]
            and sha256_file(report["pytest_output_path"]) == report["pytest_output_sha256"], "test evidence changed")
    return {"branch": branch, "code_commit": code_commit, "remote_sha": remote[0],
            "test_report_sha256": sha256_file(test_report)}


def require_file(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing frozen input: %s" % path, RequiredInputMissing)
    observed = sha256_file(path)
    require(observed == expected, "frozen input SHA mismatch: %s" % path, RequiredInputMissing)
    return observed


def blind_legal_rows(protocol):
    """Select legal rows without indexing domain or label fields."""
    data_root = Path(protocol["frozen_inputs"]["data_root"])
    required = ("dataset", "primary_20pct_split", "case_id", "patient_id", "image_h5_relpath", "image_sha256")
    parsed = []
    for asset in protocol["frozen_inputs"]["seed_assets"]:
        seed = asset["seed"]
        manifest = data_root / "manifests/training" / ("lcrseg_v1_seed%d.csv" % seed)
        require_file(manifest, asset["manifest_sha256"])
        split = data_root / "splits" / ("fundus_seed%d.json" % seed)
        require_file(split, asset["split_sha256"])
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            require(all(name in header for name in required), "manifest blind fields missing")
            index = {name: header.index(name) for name in required}
            rows = []
            for values in reader:
                if values[index["dataset"]] == "fundus":
                    rows.append({name: values[index[name]] for name in required})
        parsed.append(rows)
    reserved_cases = {row["case_id"] for rows in parsed for row in rows
                      if row["primary_20pct_split"] in ("val", "test")}
    reserved_patients = {row["patient_id"] for rows in parsed for row in rows
                         if row["primary_20pct_split"] in ("val", "test")}
    legal = []
    for seed, rows in enumerate(parsed):
        for row in rows:
            if (row["primary_20pct_split"] == "train_labeled" and row["case_id"] not in reserved_cases
                    and row["patient_id"] not in reserved_patients):
                legal.append({"seed": seed, "case_id": row["case_id"], "patient_id": row["patient_id"],
                              "image_h5_relpath": row["image_h5_relpath"], "image_sha256": row["image_sha256"]})
    legal.sort(key=lambda row: (row["seed"], row["case_id"]))
    require(len(legal) == 49 and len({row["case_id"] for row in legal}) == 37
            and len({row["patient_id"] for row in legal}) == 37, "legal utility population changed")
    for row in legal:
        require_file(safe_asset(data_root, row["image_h5_relpath"]), row["image_sha256"])
    return legal


def fold_assignments(rows):
    patients = sorted({row["patient_id"] for row in rows},
                      key=lambda value: (hashlib.sha256(("rc-shor-v0.5-fold\0" + value).encode()).hexdigest(), value))
    mapping = {patient: index % 5 for index, patient in enumerate(patients)}
    folds = np.asarray([mapping[row["patient_id"]] for row in rows], dtype=np.int64)
    require(set(folds.tolist()) == set(range(5))
            and all(len({rows[i]["patient_id"] for i in np.flatnonzero(folds == fold)}) >= 7
                    for fold in range(5)), "outer fold assignment changed")
    return folds


class FoldAccess:
    def __init__(self, fold, train_ids, eval_ids):
        self.fold = int(fold)
        self.train_ids = set(train_ids)
        self.eval_ids = set(eval_ids)
        require(not self.train_ids & self.eval_ids, "outer train/eval overlap")
        self.sealed = False
        self.outer_GT_reads = 0
        self.outer_domain_reads = 0
        self.training_GT_reads = 0
        self.training_domain_reads = 0

    def permit(self, case_id, evaluator):
        allowed = self.eval_ids if evaluator else self.train_ids
        require(case_id in allowed, "case outside declared outer role")
        if evaluator:
            require(self.sealed, "outer evaluator requested before candidate seal")

    def mark_sealed(self, seal_path):
        seal = read_json(seal_path)
        require(seal["outer_GT_reads"] == seal["outer_domain_reads"] == 0
                and seal["status"] == "PASS_OUTER_CANDIDATES_SEALED_BEFORE_GT", "contaminated candidate seal")
        self.sealed = True


def full_rows(protocol, blind_rows, indices, access, evaluator):
    data_root = Path(protocol["frozen_inputs"]["data_root"])
    wanted = {(blind_rows[index]["seed"], blind_rows[index]["case_id"]): index for index in indices}
    found = {}
    for asset in protocol["frozen_inputs"]["seed_assets"]:
        seed = asset["seed"]
        manifest = data_root / "manifests/training" / ("lcrseg_v1_seed%d.csv" % seed)
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (seed, row["case_id"])
                if key not in wanted:
                    continue
                access.permit(row["case_id"], evaluator)
                require(row["dataset"] == "fundus" and row["primary_20pct_split"] == "train_labeled"
                        and row["site_or_vendor"] in DOMAINS and row["label_h5_relpath"]
                        and row["label_sha256"], "outer row semantics changed")
                found[wanted[key]] = row
                if evaluator:
                    access.outer_domain_reads += 1
                else:
                    access.training_domain_reads += 1
    require(set(found) == set(indices), "outer rows incomplete")
    return [found[index] for index in indices]


def read_label(protocol, row, access, evaluator):
    access.permit(row["case_id"], evaluator)
    path = safe_asset(protocol["frozen_inputs"]["data_root"], row["label_h5_relpath"])
    require_file(path, row["label_sha256"])
    with h5py.File(path, "r") as handle:
        label = np.asarray(handle["label"][...], dtype=np.int64)
    require(label.shape == (384, 384), "label geometry changed", RequiredInputMissing)
    if evaluator:
        access.outer_GT_reads += 1
    else:
        access.training_GT_reads += 1
    return label


def foreground_boundary(mask):
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[1:, :] != mask[:-1, :]
    return boundary


def probability_statistics(probability):
    hard = np.argmax(probability, axis=0).astype(np.uint8)
    fg = hard > 0
    entropy = -np.sum(probability * np.log(probability + 1e-12), axis=0)
    components = ndimage.label(fg, structure=np.asarray([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8))[1]
    boundary = foreground_boundary(fg)
    stats = [float(entropy.mean()), float((probability[1] + probability[2]).mean()),
             float((hard == 1).mean()), float((hard == 2).mean()), float(components), float(boundary.mean())]
    return np.asarray(stats, dtype=np.float64), hard, boundary


def feature_matrix(descriptors, alpha, expert_arrays):
    descriptors = np.asarray(descriptors, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    count = len(descriptors)
    require(descriptors.shape == (count, 102) and alpha.shape == (count, 3), "base feature schema changed")
    features = np.empty((count, 2, 141), dtype=np.float64)
    hard = np.empty((3, count, 384, 384), dtype=np.uint8)
    stats = np.empty((count, 18), dtype=np.float64)
    boundaries = np.empty((3, count, 384, 384), dtype=bool)
    for index in range(count):
        for expert in range(3):
            value, hard[expert, index], boundaries[expert, index] = probability_statistics(expert_arrays[expert][index])
            stats[index, expert * 6:(expert + 1) * 6] = value
    pair = np.empty((count, 15), dtype=np.float64)
    for index in range(count):
        cursor = 0
        for left, right in ((0,1),(0,2),(1,2)):
            p, q = expert_arrays[left][index], expert_arrays[right][index]
            m = 0.5 * (p + q)
            js = 0.5 * np.sum(p * np.log((p + 1e-12) / (m + 1e-12)), axis=0)
            js += 0.5 * np.sum(q * np.log((q + 1e-12) / (m + 1e-12)), axis=0)
            values = (float(js.mean()), float((hard[left, index] != hard[right, index]).mean()),
                      abs(float((hard[left, index] == 1).mean()) - float((hard[right, index] == 1).mean())),
                      abs(float((hard[left, index] == 2).mean()) - float((hard[right, index] == 2).mean())),
                      float((boundaries[left, index] != boundaries[right, index]).mean()))
            pair[index, cursor:cursor + 5] = values
            cursor += 5
    order = np.sort(alpha, axis=1)
    common = np.concatenate([descriptors, alpha, (order[:, -1] - order[:, -2])[:, None],
                             (-np.sum(alpha * np.log(alpha + 1e-12), axis=1))[:, None], stats, pair], axis=1)
    require(common.shape == (count, 140), "common RC feature schema changed")
    for historical in (0, 1):
        contrast = (np.log(alpha[:, historical] + 1e-12) - np.log(alpha[:, 2] + 1e-12))[:, None]
        features[:, historical] = np.concatenate([common[:, :105], contrast, common[:, 105:]], axis=1)
    require(features.shape == (count, 2, 141) and np.isfinite(features).all(), "nonfinite RC feature")
    return features, hard


def grouped_folds(groups, salt):
    unique = sorted(set(groups), key=lambda value: (hashlib.sha256((salt + "\0" + value).encode()).hexdigest(), value))
    require(len(unique) >= 5, "fewer than five groups")
    mapping = {group: index % 5 for index, group in enumerate(unique)}
    return np.asarray([mapping[group] for group in groups], dtype=np.int64)


def fit_ridge(x, y, weights, regularization):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    active = weights > 0
    require(x.ndim == 2 and y.shape == weights.shape == (len(x),) and active.sum() >= 2
            and np.isfinite(x[active]).all() and np.isfinite(y[active]).all(), "invalid ridge active support")
    w = weights[active]
    mean = np.average(x[active], axis=0, weights=w)
    variance = np.average((x[active] - mean) ** 2, axis=0, weights=w)
    std = np.sqrt(np.maximum(variance, 0.0)); scale = np.where(std > 0, std, 1.0)
    z = (x[active] - mean) / scale
    intercept = float(np.average(y[active], weights=w))
    root_w = np.sqrt(w)
    zw = z * root_w[:, None]
    target = (y[active] - intercept) * root_w
    dual = np.linalg.solve(zw @ zw.T + float(regularization) * np.eye(len(zw)), target)
    beta = zw.T @ dual
    require(np.isfinite(beta).all(), "nonfinite ridge model")
    return {"mean": mean, "scale": scale, "beta": beta, "intercept": intercept,
            "lambda": float(regularization), "support_rows": int(active.sum())}


def predict_ridge(model, x, return_z=False):
    z = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]
    value = model["intercept"] + z @ model["beta"]
    require(np.isfinite(value).all() and np.isfinite(z).all(), "nonfinite utility prediction")
    return (value, z) if return_z else value


def oof_predict(x, y, weights, groups, regularization, salt):
    weights = np.asarray(weights, dtype=np.float64)
    folds = grouped_folds(groups, salt)
    output = np.full(len(y), np.nan, dtype=np.float64)
    for fold in range(5):
        train = (folds != fold) & (weights > 0)
        held = (folds == fold) & (weights > 0)
        if not held.any() or train.sum() < 2:
            continue
        output[held] = predict_ridge(fit_ridge(x, y, weights * train, regularization), x[held])
    require(np.isfinite(output[weights > 0]).all(), "incomplete active OOF utility prediction")
    require(np.isnan(output[weights == 0]).all(), "inactive NaN sentinel changed")
    return output


def select_lambda(x, y, groups, grid, salt):
    weights = np.ones(len(y), dtype=np.float64)
    rows = []
    for value in grid:
        prediction = oof_predict(x, y, weights, groups, value, salt)
        mse = float(np.mean((prediction - y) ** 2))
        rows.append({"lambda": float(value), "oof_mse": mse})
    selected = min(rows, key=lambda row: (row["oof_mse"], -row["lambda"]))
    return selected["lambda"], rows


def higher_quantile(value, q):
    return float(np.quantile(np.asarray(value, dtype=np.float64), q, method="higher"))


def conformal_q(prediction, truth, weights, seeds, domains):
    active = np.asarray(weights) > 0
    residual = np.asarray(prediction) - np.asarray(truth)
    values = []
    for seed in range(3):
        for domain in range(3):
            selected = active & (np.asarray(seeds) == seed) & (np.asarray(domains) == domain)
            require(selected.any(), "empty required conformal seed/domain group")
            values.append(higher_quantile(residual[selected], 0.90))
    require(np.isfinite(values).all(), "nonfinite conformal quantile")
    return max(values), values


def bootstrap_multiplicity(groups, domains, master_seed, replicate):
    groups = np.asarray(groups); domains = np.asarray(domains, dtype=np.int64)
    group_domain = {}
    for group, domain in zip(groups, domains):
        require(group not in group_domain or group_domain[group] == domain, "patient crosses dataset domains")
        group_domain[group] = int(domain)
    rng = np.random.Generator(np.random.PCG64(int(master_seed) + int(replicate)))
    multiplicity = {group: 0 for group in group_domain}
    for domain in range(3):
        pool = sorted(group for group, value in group_domain.items() if value == domain)
        require(pool, "empty bootstrap dataset domain")
        for index in rng.integers(0, len(pool), size=len(pool)):
            multiplicity[pool[int(index)]] += 1
    row_weights = np.asarray([multiplicity[group] for group in groups], dtype=np.float64)
    return row_weights, len([value for value in multiplicity.values() if value > 0])


def save_npz_new(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush(); os.fsync(handle.fileno())
    return path


def fit_base_models(features, utilities, groups, seeds, domains, grid, salt):
    count = len(groups)
    models, selected, oof_lcb, calibration = [], [], np.empty((count, 2), dtype=np.float64), []
    for historical in (0, 1):
        x, y = features[:, historical], utilities[:, historical]
        regularization, cv = select_lambda(x, y, groups, grid, salt + "-h%d" % historical)
        oof = oof_predict(x, y, np.ones(count), groups, regularization, salt + "-h%d" % historical)
        q, group_q = conformal_q(oof, y, np.ones(count), seeds, domains)
        model = fit_ridge(x, y, np.ones(count), regularization)
        models.append(model); selected.append(regularization); oof_lcb[:, historical] = oof - q
        calibration.append({"historical_expert": historical, "selected_lambda": regularization,
                            "q": q, "group_q": group_q, "lambda_cv": cv})
    return models, selected, oof_lcb, calibration


def fit_bootstrap_ensemble(features, utilities, groups, seeds, domains, eval_features, selected_lambdas,
                           master_seed, salt):
    require(np.isfinite(features).all() and np.isfinite(utilities).all()
            and np.isfinite(eval_features).all(), "nonfinite active model input")
    replicates, train_count, eval_count = 100, len(groups), len(eval_features)
    train_lcb = np.full((replicates, train_count, 2), np.nan, dtype=np.float64)
    eval_lcb = np.full((replicates, eval_count, 2), np.nan, dtype=np.float64)
    multiplicities = np.zeros((replicates, train_count), dtype=np.float64)
    support = np.zeros(replicates, dtype=np.int64)
    feasible = np.zeros(replicates, dtype=bool)
    beta = np.full((replicates, 2, features.shape[2]), np.nan, dtype=np.float64)
    means = np.full_like(beta, np.nan); scales = np.full_like(beta, np.nan)
    intercept = np.full((replicates, 2), np.nan, dtype=np.float64)
    q_values = np.full((replicates, 2), np.nan, dtype=np.float64)
    for replicate in range(replicates):
        weights, support[replicate] = bootstrap_multiplicity(groups, domains, master_seed, replicate)
        multiplicities[replicate] = weights
        if support[replicate] < 15:
            continue
        try:
            local_train, local_eval, states = [], [], []
            for historical in (0, 1):
                x, y = features[:, historical], utilities[:, historical]
                oof = oof_predict(x, y, weights, groups, selected_lambdas[historical],
                                   salt + "-b%d-h%d" % (replicate, historical))
                q, _ = conformal_q(oof, y, weights, seeds, domains)
                state = fit_ridge(x, y, weights, selected_lambdas[historical])
                local_train.append(oof - q)
                local_eval.append(predict_ridge(state, eval_features[:, historical]) - q)
                states.append((state, q))
            train_lcb[replicate] = np.stack(local_train, axis=1)
            eval_lcb[replicate] = np.stack(local_eval, axis=1)
            for historical, (state, q) in enumerate(states):
                beta[replicate, historical] = state["beta"]
                means[replicate, historical] = state["mean"]
                scales[replicate, historical] = state["scale"]
                intercept[replicate, historical] = state["intercept"]
                q_values[replicate, historical] = q
            feasible[replicate] = True
        except ProtocolViolation:
            # A bootstrap with an empty empirical group is an infeasible registered draw.
            continue
    require(np.isnan(train_lcb[~(multiplicities > 0).repeat(2).reshape(replicates, train_count, 2)]).all(),
            "inactive bootstrap sentinel was filled")
    return {"train_lcb": train_lcb, "eval_lcb": eval_lcb, "multiplicity": multiplicities,
            "support": support, "feasible": feasible, "beta": beta, "mean": means, "scale": scales,
            "intercept": intercept, "q": q_values}


def route_from_ensemble(lcb, feasible, support, epsilon, rho, ood=None):
    lcb = np.asarray(lcb, dtype=np.float64); feasible = np.asarray(feasible, dtype=bool)
    count = lcb.shape[1]
    route = np.full(count, 2, dtype=np.int64)
    median = np.full((count, 2), np.nan, dtype=np.float64)
    consensus = np.zeros((count, 2), dtype=np.float64)
    ood = np.zeros((count, 2), dtype=bool) if ood is None else np.asarray(ood, dtype=bool)
    if feasible.sum() < 90:
        return route, median, consensus
    for index in range(count):
        active = feasible & np.isfinite(lcb[:, index]).all(axis=1)
        if not active.any():
            continue
        median[index] = np.median(lcb[active, index], axis=0)
        winners = np.argmax(lcb[active, index], axis=1)
        for historical in (0, 1):
            consensus[index, historical] = np.sum(
                (winners == historical) & (lcb[active, index, historical] > epsilon)) / 100.0
    chosen = np.argmax(median, axis=1)
    for index, historical in enumerate(chosen):
        if (np.isfinite(median[index]).all() and median[index, historical] > epsilon
                and consensus[index, historical] >= rho
                and np.sum(feasible & (np.asarray(support) >= 15)) >= 90 and not ood[index, historical]):
            route[index] = int(historical)
    return route, median, consensus


def route_from_base(models, features, q_values, epsilon):
    lcb = np.empty((len(features), 2), dtype=np.float64)
    ood = np.zeros((len(features), 2), dtype=bool)
    for historical in (0, 1):
        prediction, z = predict_ridge(models[historical], features[:, historical], return_z=True)
        lcb[:, historical] = prediction - q_values[historical]
        ood[:, historical] = np.max(np.abs(z), axis=1) > 8.0
    chosen = np.argmax(lcb, axis=1)
    route = np.full(len(features), 2, dtype=np.int64)
    for index, historical in enumerate(chosen):
        if lcb[index, historical] > epsilon and not ood[index, historical]:
            route[index] = int(historical)
    return route, lcb, ood


def routes_for_bootstrap_draws(lcb, feasible, epsilon):
    output = np.full((100, lcb.shape[1]), 2, dtype=np.int64)
    for replicate in range(100):
        if not feasible[replicate]:
            continue
        active = np.isfinite(lcb[replicate]).all(axis=1)
        chosen = np.argmax(lcb[replicate, active], axis=1)
        best = lcb[replicate, np.flatnonzero(active), chosen]
        accepted = np.flatnonzero(active)[best > epsilon]
        output[replicate, accepted] = chosen[best > epsilon]
    return output


def metric_arrays(indices, blind_rows, protocol, access, evaluator, hard_masks):
    records = full_rows(protocol, blind_rows, indices, access, evaluator)
    count = len(indices)
    domains = np.empty(count, dtype=np.int64)
    expert_fg = np.empty((count, 3), dtype=np.float64)
    expert_class = np.empty((count, 3, 2), dtype=np.float64)
    expert_iou = np.empty((count, 3), dtype=np.float64)
    labels = []
    for local, (index, row) in enumerate(zip(indices, records)):
        domains[local] = DOMAINS.index(row["site_or_vendor"])
        label = read_label(protocol, row, access, evaluator); labels.append(label)
        for expert in range(3):
            metrics = v4.case_metrics(hard_masks[expert, index], label)
            expert_fg[local, expert] = metrics["foreground_dice"]
            expert_class[local, expert] = (metrics["rim_dice"], metrics["cup_dice"])
            expert_iou[local, expert] = metrics["mean_iou"]
    utility_fg = expert_fg[:, :2] - expert_fg[:, 2, None]
    utility_class = expert_class[:, :2] - expert_class[:, 2, None, :]
    return {"domains": domains, "expert_fg": expert_fg, "expert_class": expert_class,
            "expert_iou": expert_iou, "utility_fg": utility_fg, "utility_class": utility_class,
            "labels": labels}


def balanced_group_mean(value, seeds, domains, selected_domains=(0,1,2)):
    groups = []
    for seed in range(3):
        for domain in selected_domains:
            selected = (seeds == seed) & (domains == domain)
            if selected.any():
                groups.append(float(np.mean(value[selected])))
    require(groups, "empty balanced metric")
    return float(np.mean(groups))


def route_summary(routes, utility_fg, utility_class, seeds, domains):
    routes = np.asarray(routes, dtype=np.int64)
    delta = np.zeros(len(routes), dtype=np.float64)
    class_delta = np.zeros((len(routes), 2), dtype=np.float64)
    for historical in (0, 1):
        selected = routes == historical
        delta[selected] = utility_fg[selected, historical]
        class_delta[selected] = utility_class[selected, historical]
    group_gain = {}
    for seed in range(3):
        for domain in range(3):
            selected = (seeds == seed) & (domains == domain)
            if selected.any():
                group_gain[(seed, domain)] = float(delta[selected].mean())
    shared = balanced_group_mean(delta, seeds, domains)
    historical = balanced_group_mean(delta, seeds, domains, (0,1))
    current_gain = balanced_group_mean(delta, seeds, domains, (2,))
    current_class = [balanced_group_mean(class_delta[:, cls], seeds, domains, (2,)) for cls in range(2)]
    beneficial = np.max(utility_fg, axis=1) > 0
    selected_historical = routes < 2
    selected_utility = np.zeros(len(routes), dtype=np.float64)
    for historical_expert in (0, 1):
        mask = routes == historical_expert
        selected_utility[mask] = utility_fg[mask, historical_expert]
    return {
        "shared_gain": shared,
        "three_domain_gain": shared,
        "historical_gain": historical,
        "REFUGE_gain": balanced_group_mean(delta, seeds, domains, (0,)),
        "RIM_ONE_r3_gain": balanced_group_mean(delta, seeds, domains, (1,)),
        "current_domain_drop": max(0.0, -current_gain),
        "maximum_current_class_drop": max([0.0] + [-value for value in current_class]),
        "maximum_seed_domain_drop": max([0.0] + [-value for value in group_gain.values()]),
        "positive_seed_count": int(sum(np.mean([value for (s, _), value in group_gain.items() if s == seed]) > 0
                                       for seed in range(3))),
        "route_precision": float(np.mean(selected_utility[selected_historical] > 0)) if selected_historical.any() else 1.0,
        "historical_route_recall": float(np.mean(selected_historical[beneficial])) if beneficial.any() else 1.0,
        "current_false_override": float(np.mean(selected_historical[domains == 2])),
        "historical_missed_override": float(np.mean(~selected_historical[(domains < 2) & beneficial]))
            if ((domains < 2) & beneficial).any() else 0.0,
        "route_frequency": float(np.mean(selected_historical)),
        "delta": delta,
        "class_delta": class_delta,
    }


def stability_summary(draw_routes, utility_fg, utility_class, seeds, domains, feasible):
    summaries = [route_summary(routes, utility_fg, utility_class, seeds, domains) for routes in draw_routes]
    value = lambda key: np.asarray([row[key] for row in summaries], dtype=np.float64)
    return {"shared_gain_p10": float(np.quantile(value("shared_gain"), 0.10)),
            "historical_gain_p10": float(np.quantile(value("historical_gain"), 0.10)),
            "current_domain_drop_p90": float(np.quantile(value("current_domain_drop"), 0.90)),
            "maximum_seed_domain_drop_p90": float(np.quantile(value("maximum_seed_domain_drop"), 0.90)),
            "feasible_replicates": int(np.asarray(feasible).sum()),
            "feasible_fraction": float(np.asarray(feasible).mean()),
            "route_disagreement": float(np.mean(np.apply_along_axis(lambda x: len(set(x.tolist())) > 1,
                                                                     0, draw_routes)))}


def candidate_rows(ensemble, base_models, calibration, features, utilities, utility_class, groups, seeds, domains,
                   grid):
    q_values = [row["q"] for row in calibration]
    _, _, ood = route_from_base(base_models, features, q_values, 0.0)
    rows = []
    for rho in (0.70, 0.80, 0.90):
        for epsilon in (0.0, 0.005, 0.010):
            route, _, _ = route_from_ensemble(ensemble["train_lcb"], ensemble["feasible"], ensemble["support"],
                                               epsilon, rho, ood)
            draws = routes_for_bootstrap_draws(ensemble["train_lcb"], ensemble["feasible"], epsilon)
            point = route_summary(route, utilities, utility_class, seeds, domains)
            stability = stability_summary(draws, utilities, utility_class, seeds, domains, ensemble["feasible"])
            safety = (point["current_domain_drop"] <= 0.010
                      and point["maximum_current_class_drop"] <= 0.015
                      and point["maximum_seed_domain_drop"] <= 0.020)
            stable = (stability["current_domain_drop_p90"] <= 0.015
                      and stability["maximum_seed_domain_drop_p90"] <= 0.025
                      and stability["shared_gain_p10"] >= 0.08
                      and stability["historical_gain_p10"] >= 0.12
                      and stability["feasible_replicates"] >= 90)
            rows.append({"candidate_id": "rho%03d_eps%03d" % (round(rho * 100), round(epsilon * 1000)),
                         "rho": rho, "epsilon": epsilon, "all_inner_stability_and_safety_gates": bool(safety and stable),
                         **{key: stability[key] for key in ("historical_gain_p10", "current_domain_drop_p90",
                                                            "maximum_seed_domain_drop_p90", "shared_gain_p10",
                                                            "feasible_replicates")},
                         "point_historical_gain": point["historical_gain"], "point_shared_gain": point["shared_gain"]})
    selected = min(rows, key=lambda row: (-int(row["all_inner_stability_and_safety_gates"]),
                                          -row["historical_gain_p10"], row["current_domain_drop_p90"],
                                          row["maximum_seed_domain_drop_p90"], -row["rho"], -row["epsilon"],
                                          -min(grid), row["candidate_id"]))
    return selected, rows


def c3_routes(alpha, seeds, thresholds):
    output = np.empty(len(alpha), dtype=np.int64)
    for seed in range(3):
        selected = seeds == seed
        if selected.any():
            output[selected] = shor_routes(alpha[selected], stage=2, thresholds=thresholds[seed])
    return output


def c4_routes(alpha, seeds, bootstrap_thresholds, rho):
    top = top1_lowest(alpha); votes = np.zeros((len(alpha), 2), dtype=np.float64)
    for replicate in range(100):
        frozen_replicate = replicate % 5
        for seed in range(3):
            for historical in (0, 1):
                row = bootstrap_thresholds[(seed, historical, frozen_replicate)]
                selected = (seeds == seed) & (top == historical)
                if row["feasible"]:
                    votes[selected, historical] += (historical_score(alpha[selected], 2, historical)
                                                    >= float(row["threshold"]))
    votes /= 100.0
    output = np.full(len(alpha), 2, dtype=np.int64)
    for index, historical in enumerate(top):
        if historical < 2 and votes[index, historical] >= rho:
            output[index] = int(historical)
    return output, votes


def load_frozen(protocol):
    baseline_path = REPO / protocol["frozen_inputs"]["baseline_snapshot_manifest"]["path"]
    require_file(baseline_path, protocol["frozen_inputs"]["baseline_snapshot_manifest"]["sha256"])
    baseline = read_json(baseline_path)
    checkpoints = {(row["seed"], row["stage_index"]): row for row in baseline["checkpoints"]}
    require(set(checkpoints) == {(seed, stage) for seed in range(3) for stage in range(3)},
            "nine frozen checkpoints unavailable", RequiredInputMissing)
    try:
        routers, thresholds, threshold_hashes, threshold_path, router_path = v4.load_frozen_policy(protocol)
    except (v4.RequiredInputMissing, v4.ProtocolViolation) as error:
        raise RequiredInputMissing(str(error)) from error
    threshold_manifest = read_json(threshold_path)
    bootstrap_thresholds = {}
    for row in threshold_manifest["bootstrap"]:
        if row["stage_index"] == 2:
            bootstrap_thresholds[(row["seed"], row["historical_domain"], row["replicate"])] = row
    require(set(bootstrap_thresholds) == {(seed, historical, replicate) for seed in range(3)
                                          for historical in (0,1) for replicate in range(5)},
            "frozen SHOR bootstrap thresholds incomplete", RequiredInputMissing)
    return checkpoints, routers, thresholds, threshold_hashes, bootstrap_thresholds, baseline_path, threshold_path, router_path


def materialize_blind_inputs(args, protocol, rows, checkpoints, routers, metadata):
    count = len(rows)
    descriptors = np.empty((count, 102), dtype=np.float64)
    alpha = np.empty((count, 3), dtype=np.float64)
    hard = np.empty((3, count, 384, 384), dtype=np.uint8)
    features = np.empty((count, 2, 141), dtype=np.float64)
    probabilities = [np.empty((count, 3, 384, 384), dtype=np.float32) for _ in range(3)]
    hashes = {"descriptors": {}, "probabilities": {}, "features": None}
    batches = 0
    with no_updates():
        for seed in range(3):
            indices = np.asarray([index for index, row in enumerate(rows) if row["seed"] == seed], dtype=np.int64)
            selected = [rows[index] for index in indices]
            descriptor, descriptor_path = v4.extract_test_descriptors(
                args.output, seed, selected, checkpoints[(seed, 0)], protocol["frozen_inputs"]["data_root"],
                args.device, metadata)
            descriptors[indices] = descriptor
            alpha[indices] = v4.router_probabilities(descriptor, routers[seed])
            paths = []
            for expert in range(3):
                path = v4.predict_expert(args.output, seed, expert, selected, checkpoints[(seed, expert)],
                                         protocol["frozen_inputs"]["data_root"], args.device, metadata)
                paths.append(path)
                probabilities[expert][indices] = np.load(path, mmap_mode="r", allow_pickle=False)
                hashes["probabilities"]["seed%d_expert%d" % (seed, expert)] = sha256_file(path)
            local_features, local_hard = feature_matrix(descriptor, alpha[indices],
                                                        [np.load(path, mmap_mode="r", allow_pickle=False)
                                                         for path in paths])
            features[indices] = local_features; hard[:, indices] = local_hard
            hashes["descriptors"][str(seed)] = sha256_file(descriptor_path)
            batches += int(math.ceil(len(selected) / BATCH_SIZE)) * 4
    cache = save_npz_new(args.output / "feature_cache/RC_SHOR_FEATURES.npz", descriptors=descriptors,
                         alpha=alpha, features=features, hard_masks=hard)
    hashes["features"] = sha256_file(cache)
    write_json_new(args.output / "PRIVATE_CASE_ORDER.json", {"rows": rows,
                   "case_order_sha256": canonical_hash([(row["seed"], row["case_id"]) for row in rows])})
    return descriptors, alpha, features, hard, probabilities, hashes, batches


def materialize_policy_predictions(path, eval_indices, routes, alpha, hard, probabilities):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                       shape=(len(eval_indices), 384, 384))
    for local, index in enumerate(eval_indices):
        if routes is None:
            fused = sum(float(alpha[index, expert]) * probabilities[expert][index] for expert in range(3))
            output[local] = np.argmax(fused, axis=0).astype(np.uint8)
        else:
            output[local] = hard[int(routes[local]), index]
    output.flush(); del output
    return path


def execute_outer_folds(args, protocol, rows, folds, alpha, features, hard, probabilities, thresholds,
                        bootstrap_thresholds):
    count = len(rows); seeds_all = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    case_rows, candidate_rows_all = [], []
    global_utility = np.empty((count, 2), dtype=np.float64)
    global_utility_class = np.empty((count, 2, 2), dtype=np.float64)
    global_expert_fg = np.empty((count, 3), dtype=np.float64)
    global_draw_routes = np.full((100, count), 2, dtype=np.int64)
    global_c3_routes = np.full(count, 2, dtype=np.int64)
    global_feasible = np.ones(100, dtype=bool)
    selected_candidates, seals = [], []
    access_totals = {"training_GT_reads": 0, "training_domain_reads": 0,
                     "outer_GT_reads": 0, "outer_domain_reads": 0}
    for fold in range(5):
        train_indices = np.flatnonzero(folds != fold); eval_indices = np.flatnonzero(folds == fold)
        access = FoldAccess(fold, [rows[i]["case_id"] for i in train_indices],
                            [rows[i]["case_id"] for i in eval_indices])
        train = metric_arrays(train_indices, rows, protocol, access, False, hard)
        train_features = features[train_indices]
        train_groups = np.asarray([rows[i]["patient_id"] for i in train_indices])
        train_seeds = seeds_all[train_indices]
        grid = protocol["model"]["lambda_grid"]
        base_models, selected_lambdas, train_oof_lcb, calibration = fit_base_models(
            train_features, train["utility_fg"], train_groups, train_seeds, train["domains"], grid,
            "rc-shor-fold%d" % fold)
        ensemble = fit_bootstrap_ensemble(
            train_features, train["utility_fg"], train_groups, train_seeds, train["domains"],
            features[eval_indices], selected_lambdas,
            protocol["router_bootstrap"]["master_seed"] + fold * 1000, "rc-shor-fold%d" % fold)
        selected, candidates = candidate_rows(
            ensemble, base_models, calibration, train_features, train["utility_fg"], train["utility_class"],
            train_groups, train_seeds, train["domains"], grid)
        for candidate in candidates:
            candidate_rows_all.append({"fold": fold, **candidate})
        selected_candidates.append({"fold": fold, "selected_lambdas": selected_lambdas, **selected})
        q_values = [row["q"] for row in calibration]
        c5, base_lcb, ood = route_from_base(base_models, features[eval_indices], q_values, selected["epsilon"])
        c6, median_lcb, consensus = route_from_ensemble(
            ensemble["eval_lcb"], ensemble["feasible"], ensemble["support"], selected["epsilon"],
            selected["rho"], ood)
        c3 = c3_routes(alpha[eval_indices], seeds_all[eval_indices], thresholds)
        c4, c4_votes = c4_routes(alpha[eval_indices], seeds_all[eval_indices], bootstrap_thresholds,
                                 selected["rho"])
        routes = {
            "C0": np.full(len(eval_indices), 2, dtype=np.int64),
            "C1": top1_lowest(alpha[eval_indices]),
            "C2": None,
            "C3": c3,
            "C4": c4,
            "C5": c5,
            "C6": c6,
        }
        prediction_paths = {}
        for policy in POLICIES[:7]:
            prediction_paths[policy] = materialize_policy_predictions(
                args.output / ("candidate_predictions/fold%d_%s.npy" % (fold, policy)), eval_indices,
                routes[policy], alpha, hard, probabilities)
        model_path = save_npz_new(
            args.output / ("model_states/fold%d.npz" % fold),
            base_beta=np.stack([model["beta"] for model in base_models]),
            base_mean=np.stack([model["mean"] for model in base_models]),
            base_scale=np.stack([model["scale"] for model in base_models]),
            base_intercept=np.asarray([model["intercept"] for model in base_models]),
            base_q=np.asarray(q_values), selected_lambdas=np.asarray(selected_lambdas),
            bootstrap_beta=ensemble["beta"], bootstrap_mean=ensemble["mean"],
            bootstrap_scale=ensemble["scale"], bootstrap_intercept=ensemble["intercept"],
            bootstrap_q=ensemble["q"])
        bootstrap_path = save_npz_new(
            args.output / ("bootstrap_draws/fold%d.npz" % fold), multiplicity=ensemble["multiplicity"],
            support=ensemble["support"], feasible=ensemble["feasible"],
            train_oof_lcb=ensemble["train_lcb"], eval_lcb=ensemble["eval_lcb"])
        conformal_path = args.output / ("conformal_states/fold%d.json" % fold)
        write_json_new(conformal_path, {"fold": fold, "calibration": calibration,
                                       "selected_candidate": selected})
        route_path = save_npz_new(args.output / ("routes/fold%d.npz" % fold),
                                  **{key: value for key, value in routes.items() if value is not None},
                                  c4_votes=c4_votes, base_lcb=base_lcb, median_lcb=median_lcb,
                                  consensus=consensus, ood=ood)
        seal = {
            "status": "PASS_OUTER_CANDIDATES_SEALED_BEFORE_GT",
            "fold": fold,
            "case_order_sha256": canonical_hash([(rows[i]["seed"], rows[i]["case_id"]) for i in eval_indices]),
            "model_state_sha256": sha256_file(model_path),
            "conformal_state_sha256": sha256_file(conformal_path),
            "bootstrap_draw_sha256": sha256_file(bootstrap_path),
            "feature_sha256": array_hash(features[eval_indices]),
            "route_sha256": sha256_file(route_path),
            "prediction_sha256": {policy: sha256_file(path) for policy, path in prediction_paths.items()},
            "selected_candidate": selected,
            "outer_GT_reads": access.outer_GT_reads,
            "outer_domain_reads": access.outer_domain_reads,
            "segmentation_optimizer_steps": 0,
            "segmentation_parameter_updates": 0,
            "test_domain_inputs_to_C6": 0,
        }
        seal_path = args.output / ("candidate_seals/fold%d.json" % fold)
        write_json_new(seal_path, seal); access.mark_sealed(seal_path); seals.append(seal_path)
        evaluated = metric_arrays(eval_indices, rows, protocol, access, True, hard)
        global_utility[eval_indices] = evaluated["utility_fg"]
        global_utility_class[eval_indices] = evaluated["utility_class"]
        global_expert_fg[eval_indices] = evaluated["expert_fg"]
        draw_routes = routes_for_bootstrap_draws(ensemble["eval_lcb"], ensemble["feasible"], selected["epsilon"])
        global_draw_routes[:, eval_indices] = draw_routes
        global_feasible &= ensemble["feasible"]
        global_c3_routes[eval_indices] = c3
        c7 = evaluated["domains"].copy()
        c8 = np.argmax(evaluated["expert_fg"], axis=1)
        oracle_paths = {}
        for policy, oracle_route in (("C7", c7), ("C8", c8)):
            oracle_paths[policy] = materialize_policy_predictions(
                args.output / ("evaluator_oracle_predictions/fold%d_%s.npy" % (fold, policy)),
                eval_indices, oracle_route, alpha, hard, probabilities)
        all_routes = {**routes, "C7": c7, "C8": c8}
        predictions = {policy: np.load(path, mmap_mode="r", allow_pickle=False)
                       for policy, path in {**prediction_paths, **oracle_paths}.items()}
        for local, index in enumerate(eval_indices):
            label = evaluated["labels"][local]
            for policy in POLICIES:
                metrics = v4.case_metrics(predictions[policy][local], label)
                route = -1 if all_routes[policy] is None else int(all_routes[policy][local])
                item = {"fold": fold, "row_index": int(index), "case_id": rows[index]["case_id"],
                        "patient_id": rows[index]["patient_id"], "seed": int(rows[index]["seed"]),
                        "domain_index": int(evaluated["domains"][local]), "domain": DOMAINS[evaluated["domains"][local]],
                        "policy": policy, "route": route, **metrics}
                if policy == "C6":
                    h = int(np.argmax(median_lcb[local])) if np.isfinite(median_lcb[local]).all() else 0
                    item.update(consensus=float(consensus[local, h]), utility_LCB=float(median_lcb[local, h])
                                if np.isfinite(median_lcb[local, h]) else -1e9,
                                support=int(np.median(ensemble["support"])))
                case_rows.append(item)
        for key in access_totals:
            access_totals[key] += getattr(access, key)
    require(np.isfinite(global_utility).all() and np.isfinite(global_utility_class).all(),
            "incomplete outer utility evaluation")
    with (args.output / "case_metrics.jsonl").open("x", encoding="utf-8") as handle:
        for row in case_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    save_npz_new(args.output / "utility_targets/outer_oof.npz", utility_fg=global_utility,
                 utility_class=global_utility_class, expert_fg=global_expert_fg)
    write_csv_new(args.output / "calibration_curves.csv", candidate_rows_all)
    write_json_new(args.output / "cross_fit_assignments.json", {"folds": folds.tolist(),
                   "case_order_sha256": canonical_hash([(row["seed"], row["case_id"]) for row in rows])})
    return {"case_rows": case_rows, "utility_fg": global_utility, "utility_class": global_utility_class,
            "expert_fg": global_expert_fg, "draw_routes": global_draw_routes,
            "c3_routes": global_c3_routes, "global_feasible": global_feasible,
            "selected_candidates": selected_candidates, "seals": seals, "access": access_totals}


def aggregate_metrics(case_rows):
    output = []
    dimensions = [("overall", lambda row: "all"), ("seed", lambda row: str(row["seed"])),
                  ("domain", lambda row: row["domain"]),
                  ("seed_domain", lambda row: "%d:%s" % (row["seed"], row["domain"]))]
    for level, key_fn in dimensions:
        keys = sorted({key_fn(row) for row in case_rows})
        for key in keys:
            for policy in POLICIES:
                selected = [row for row in case_rows if row["policy"] == policy and key_fn(row) == key]
                if not selected:
                    continue
                def mean(field):
                    if level == "seed_domain":
                        return float(np.mean([row[field] for row in selected]))
                    groups = {}
                    for row in selected:
                        group = ((row["domain_index"],) if level == "seed" else
                                 (row["seed"],) if level == "domain" else
                                 (row["seed"], row["domain_index"]))
                        groups.setdefault(group, []).append(row[field])
                    return float(np.mean([np.mean(values) for values in groups.values()]))
                output.append({"row_type": "metric", "level": level, "key": key, "policy": policy,
                               "cases": len(selected),
                               "foreground_dice": mean("foreground_dice"), "rim_dice": mean("rim_dice"),
                               "cup_dice": mean("cup_dice"), "mean_iou": mean("mean_iou")})
    for cls, field in (("rim", "rim_dice"), ("cup", "cup_dice")):
        for policy in POLICIES:
            selected = [row[field] for row in case_rows if row["policy"] == policy]
            output.append({"row_type": "metric", "level": "foreground_class", "key": cls,
                           "policy": policy, "cases": len(selected), field: float(np.mean(selected))})
    c6 = [row for row in case_rows if row["policy"] == "C6"]
    for field, level in (("consensus", "consensus_decile"), ("utility_LCB", "utility_LCB_decile"),
                         ("support", "support_size_bin")):
        values = np.asarray([row[field] for row in c6], dtype=np.float64)
        edges = np.quantile(values, np.linspace(0, 1, 11))
        bins = np.searchsorted(edges[1:-1], values, side="right")
        for value in sorted(set(bins.tolist())):
            selected = [row for row, bin_ in zip(c6, bins) if bin_ == value]
            output.append({"row_type": "metric", "level": level, "key": int(value), "policy": "C6",
                           "cases": len(selected), "foreground_dice": float(np.mean([r["foreground_dice"] for r in selected]))})
    return output


def routing_rows(result, seeds, domains):
    rows = []
    for policy in ("C3", "C4", "C5", "C6"):
        route = np.asarray([row["route"] for row in result["case_rows"] if row["policy"] == policy], dtype=np.int64)
        summary = route_summary(route, result["utility_fg"], result["utility_class"], seeds, domains)
        rows.append({"level": "overall", "policy": policy, **{key: value for key, value in summary.items()
                     if key not in ("delta", "class_delta")}})
    rows.extend({"level": "outer_fold_selection", "policy": "C6", **row}
                for row in result["selected_candidates"])
    return rows


def evaluation_bootstrap(delta, seeds, domains, patients, master_seed, clustered):
    rng = np.random.Generator(np.random.PCG64(master_seed)); draws = []
    for _ in range(2000):
        weights = np.zeros(len(delta), dtype=np.float64)
        if clustered:
            for domain in range(3):
                pool = sorted(set(patients[domains == domain]))
                sampled = rng.integers(0, len(pool), size=len(pool))
                counts = {patient: 0 for patient in pool}
                for index in sampled: counts[pool[int(index)]] += 1
                for index in np.flatnonzero(domains == domain): weights[index] = counts[patients[index]]
        else:
            for seed in range(3):
                for domain in range(3):
                    index = np.flatnonzero((seeds == seed) & (domains == domain))
                    sampled = rng.integers(0, len(index), size=len(index))
                    for value in sampled: weights[index[int(value)]] += 1
        group = []
        for seed in range(3):
            for domain in range(3):
                selected = (seeds == seed) & (domains == domain) & (weights > 0)
                if not selected.any(): break
                group.append(float(np.average(delta[selected], weights=weights[selected])))
            else: continue
            break
        if len(group) == 9: draws.append(float(np.mean(group)))
    require(draws, "no feasible evaluation bootstrap draw")
    return {"draws": 2000, "feasible_draws": len(draws),
            "estimate": balanced_group_mean(delta, seeds, domains),
            "lower95": float(np.quantile(draws, 0.025)), "upper95": float(np.quantile(draws, 0.975)),
            "draw_sha256": array_hash(np.asarray(draws, dtype=np.float64))}


def decide_and_report(args, protocol, audit, publication, metadata, rows, result, test_report):
    seeds = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    patients = np.asarray([row["patient_id"] for row in rows])
    c6_rows = sorted([row for row in result["case_rows"] if row["policy"] == "C6"], key=lambda row: row["row_index"])
    domains = np.asarray([row["domain_index"] for row in c6_rows], dtype=np.int64)
    c6_routes_ = np.asarray([row["route"] for row in c6_rows], dtype=np.int64)
    c6 = route_summary(c6_routes_, result["utility_fg"], result["utility_class"], seeds, domains)
    c3 = route_summary(result["c3_routes"], result["utility_fg"], result["utility_class"], seeds, domains)
    stability = stability_summary(result["draw_routes"], result["utility_fg"], result["utility_class"],
                                  seeds, domains, result["global_feasible"])
    c3_stability = stability_summary(np.tile(result["c3_routes"], (100,1)), result["utility_fg"],
                                     result["utility_class"], seeds, domains, np.ones(100, dtype=bool))
    overall = {row["policy"]: row for row in aggregate_metrics(result["case_rows"])
               if row["level"] == "overall"}
    domain_oracle_gap = overall["C7"]["foreground_dice"] - overall["C6"]["foreground_dice"]
    regret_c6 = overall["C8"]["foreground_dice"] - overall["C6"]["foreground_dice"]
    regret_c3 = overall["C8"]["foreground_dice"] - overall["C3"]["foreground_dice"]
    regret_reduction = ((regret_c3 - regret_c6) / regret_c3) if regret_c3 > 0 else float(regret_c6 <= 0)
    isolation_gate = (audit["history"]["v0_4"]["formal_03_content_reads"] == 0
                      and all(read_json(path)["outer_GT_reads"] == read_json(path)["outer_domain_reads"] == 0
                              for path in result["seals"])
                      and result["access"]["outer_GT_reads"] == result["access"]["outer_domain_reads"] == 49)
    value_gate = (c6["three_domain_gain"] >= 0.10 and c6["historical_gain"] >= 0.15
                  and c6["REFUGE_gain"] > 0 and c6["RIM_ONE_r3_gain"] > 0
                  and c6["positive_seed_count"] == 3 and domain_oracle_gap <= 0.06)
    safety_gate = (c6["current_domain_drop"] <= 0.010 and c6["maximum_current_class_drop"] <= 0.015
                   and c6["maximum_seed_domain_drop"] <= 0.020)
    stability_gate = (stability["current_domain_drop_p90"] <= 0.015
                      and stability["maximum_seed_domain_drop_p90"] <= 0.025
                      and stability["shared_gain_p10"] >= 0.08
                      and stability["historical_gain_p10"] >= 0.12
                      and stability["feasible_replicates"] >= 90)
    incremental = {
        "C6_minus_C3_historical_gain": c6["historical_gain"] - c3["historical_gain"],
        "C6_minus_C3_overall_gain": c6["shared_gain"] - c3["shared_gain"],
        "per_case_oracle_regret_reduction_fraction": regret_reduction,
        "C6_minus_C3_current_drop_p90": stability["current_domain_drop_p90"] - c3_stability["current_domain_drop_p90"],
    }
    incremental_gate = (incremental["C6_minus_C3_historical_gain"] >= -0.005
                        and incremental["C6_minus_C3_overall_gain"] >= -0.005
                        and regret_reduction >= 0.20
                        and incremental["C6_minus_C3_current_drop_p90"] <= 0.002)
    if not isolation_gate:
        status = "BLOCKED_PROTOCOL_OR_LEAKAGE"
    elif not safety_gate:
        status = "FAIL_RC_SHOR_CURRENT_SAFETY"
    elif not value_gate:
        status = "FAIL_RC_SHOR_VALUE"
    elif not stability_gate or not incremental_gate:
        status = "FAIL_RC_SHOR_STABILITY"
    else:
        status = "PASS_RC_SHOR_STABILITY_RECOVERY"
    public_metrics = aggregate_metrics(result["case_rows"])
    public_routing = routing_rows(result, seeds, domains)
    clustered = evaluation_bootstrap(c6["delta"], seeds, domains, patients,
                                     protocol["evaluation_bootstrap"]["master_seed"], True)
    ordinary = evaluation_bootstrap(c6["delta"], seeds, domains, patients,
                                    protocol["evaluation_bootstrap"]["master_seed"] + 1, False)
    gate_values = {
        "isolation": {"pass": isolation_gate, "v0_4_formal_03_reads": 0,
                      "outer_GT_reads_before_seal": 0, "outer_domain_reads_before_seal": 0,
                      "test_domain_inputs_to_C6": 0, "segmentation_optimizer_steps": 0,
                      "segmentation_parameter_updates": 0, "old_artifact_mutations": 0},
        "value": {"pass": value_gate, "three_domain_gain": [c6["three_domain_gain"], 0.10],
                  "historical_gain": [c6["historical_gain"], 0.15], "REFUGE_gain": [c6["REFUGE_gain"], 0.0],
                  "RIM_ONE_r3_gain": [c6["RIM_ONE_r3_gain"], 0.0],
                  "positive_seed_count": [c6["positive_seed_count"], 3],
                  "domain_oracle_gap": [domain_oracle_gap, 0.06]},
        "current_safety": {"pass": safety_gate, "current_domain_drop": [c6["current_domain_drop"], 0.010],
                           "maximum_current_class_drop": [c6["maximum_current_class_drop"], 0.015],
                           "maximum_seed_domain_drop": [c6["maximum_seed_domain_drop"], 0.020]},
        "stability": {"pass": stability_gate, "current_domain_drop_p90": [stability["current_domain_drop_p90"], 0.015],
                      "maximum_seed_domain_drop_p90": [stability["maximum_seed_domain_drop_p90"], 0.025],
                      "shared_gain_p10": [stability["shared_gain_p10"], 0.08],
                      "historical_gain_p10": [stability["historical_gain_p10"], 0.12],
                      "feasible_replicates": [stability["feasible_replicates"], 90]},
        "incremental": {"pass": incremental_gate, **incremental},
    }
    private_before = [path for path in args.output.rglob("*") if path.is_file()]
    status_value = {
        "schema_version": 1, "registration_id": protocol["registration_id"], "scientific_status": status,
        "completed_at": v4.now(), "execution_code_commit": args.code_commit, "formal_attempt": 1,
        "additional_attempt_authorized": False, "gates": gate_values,
        "C3": {key: value for key, value in c3.items() if key not in ("delta", "class_delta")},
        "C6": {key: value for key, value in c6.items() if key not in ("delta", "class_delta")},
        "C3_C6_direct_comparison": incremental,
        "H5_recovered": bool(stability_gate), "clustered_bootstrap": clustered,
        "ordinary_case_bootstrap_sensitivity": ordinary,
        "controls": {policy: {key: overall[policy][key] for key in ("foreground_dice","rim_dice","cup_dice","mean_iou")}
                     for policy in POLICIES},
        "routing_stability": stability, "selected_candidates": result["selected_candidates"],
        "isolation": {"candidate_seals": [sha256_file(path) for path in result["seals"]],
                      "v0_4_formal_03_reads": 0, **result["access"], "C6_domain_inputs": 0,
                      "frozen_cache_materialization_batches": metadata["frozen_cache_materialization_batches"],
                      "segmentation_expert_forward_batches": metadata["segmentation_expert_forward_batches"],
                      "segmentation_training_steps": 0, "segmentation_optimizer_steps": 0,
                      "segmentation_parameter_updates": 0, "router_closed_form_fits_nonzero": True,
                      "router_optimizer_steps": 0},
        "tests": read_json(test_report), "predecessors_unchanged": True,
        "repeat_final_evaluation": "REFUSED_AFTER_STATUS_EXISTS", "main_merged": False,
        "next_stage_started": False, "private_artifact_inventory_before_public":
            {"files": len(private_before), "bytes": sum(path.stat().st_size for path in private_before)},
        "report_commit": None, "report_commit_resolution": "second Git commit adding exact public report bytes",
    }
    public = args.output / "public"; public.mkdir()
    write_json_new(public / PUBLIC_FILES[0], status_value)
    write_csv_new(public / PUBLIC_FILES[2], public_metrics)
    write_csv_new(public / PUBLIC_FILES[3], public_routing)
    failures = [name for name, gate in gate_values.items() if not gate["pass"]]
    warnings = "# RC-SHOR V0.5 failures and warnings\n\n- Scientific status: `%s`.\n- Failed gates: %s.\n- The legal leakage-free pool contains only 37 unique patients; this limitation was frozen before evaluation.\n- SHOR V0.3.1 and V0.4 remain unchanged. No formal retry is authorized.\n" % (status, ", ".join(failures) or "none")
    write_text_new(public / PUBLIC_FILES[4], warnings)
    commands = "# RC-SHOR V0.5 exact commands\n\n```sh\nPYTHONPATH=experiments/lcrseg python -m pytest --import-mode=importlib experiments/lcrseg/tests/rc_shor_v0_5 experiments/lcrseg/tests/shor_jascl_v0_3 experiments/lcrseg/tests/shor_v0_4_test --junitxml=PYTEST_XML\nbash experiments/lcrseg/scripts/with_nas_storage.sh python experiments/lcrseg/rc_shor_v0_5.py --output NAS_CREATE_ONLY_ROOT --code-commit FREEZE_COMMIT --test-report TEST_REPORT --device cuda:0\n```\n"
    write_text_new(public / PUBLIC_FILES[5], commands)
    report = """# RC-SHOR V0.5 final report

## Outcome

**{status}**

RC-SHOR V0.5 used 49 seed-case observations from 37 leakage-free train-labeled patients in grouped five-fold outer OOF evaluation. SHOR V0.3.1 remains `FAIL_SELECTIVE_OVERRIDE_STABILITY` with H5 false; SHOR V0.4 remains `PASS_FIXED_POLICY_TEST_EFFECTIVENESS`.

## Gates

| Gate | Pass | Raw evidence |
|---|---:|---|
| isolation | {isolation} | V0.4 formal reads 0; all five seals preceded outer GT/domain; model updates 0 |
| value | {value} | overall gain {overall:.6f}; historical {historical:.6f}; REFUGE {refuge:.6f}; RIM {rim:.6f}; seeds {seeds}/3; oracle gap {oracle:.6f} |
| current safety | {safety} | current drop {current:.6f}; max current-class {current_class:.6f}; max seed-domain {seed_domain:.6f} |
| stability | {stability} | shared p10 {shared_p10:.6f}; historical p10 {history_p10:.6f}; current p90 {current_p90:.6f}; max seed-domain p90 {seed_p90:.6f}; feasible {feasible}/100 |
| incremental | {incremental} | C6-C3 overall {inc_overall:.6f}; historical {inc_history:.6f}; regret reduction {regret:.6f}; p90 increase {p90_inc:.6f} |

H5 recovery: **{h5}**. Segmentation training/optimizer/update counts were 0; frozen cache materialization used {forward} batches. Router fitting was closed-form with nonzero fit count and zero optimizer steps. C0-C8 exact aggregate values are in the status and metrics CSV. The five candidate-seal hashes, selected candidates, C3/C6 comparison, clustered bootstrap and ordinary-case sensitivity are in the status.

No V0.4 private test artifact was read, no old artifact was modified, no retry or main merge occurred, and no next stage was launched.
""".format(status=status, isolation=isolation_gate, value=value_gate, safety=safety_gate,
           stability=stability_gate, incremental=incremental_gate, overall=c6["three_domain_gain"],
           historical=c6["historical_gain"], refuge=c6["REFUGE_gain"], rim=c6["RIM_ONE_r3_gain"],
           seeds=c6["positive_seed_count"], oracle=domain_oracle_gap, current=c6["current_domain_drop"],
           current_class=c6["maximum_current_class_drop"], seed_domain=c6["maximum_seed_domain_drop"],
           shared_p10=stability["shared_gain_p10"], history_p10=stability["historical_gain_p10"],
           current_p90=stability["current_domain_drop_p90"], seed_p90=stability["maximum_seed_domain_drop_p90"],
           feasible=stability["feasible_replicates"], inc_overall=incremental["C6_minus_C3_overall_gain"],
           inc_history=incremental["C6_minus_C3_historical_gain"], regret=regret_reduction,
           p90_inc=incremental["C6_minus_C3_current_drop_p90"], h5=stability_gate,
           forward=metadata["frozen_cache_materialization_batches"])
    write_text_new(public / PUBLIC_FILES[1], report)
    entries = [{"path": name, "bytes": (public / name).stat().st_size,
                "sha256": sha256_file(public / name)} for name in PUBLIC_FILES]
    write_json_new(public / "RC_SHOR_V0_5_MANIFEST.json", {
        "schema_version": 1, "status": "PASS_PUBLIC_REPORT_BUNDLE_COMPLETE",
        "scientific_status": status, "execution_code_commit": args.code_commit,
        "entries": entries, "candidate_seal_sha256": [sha256_file(path) for path in result["seals"]],
        "private_large_artifacts_root": str(args.output), "private_large_artifacts_published": False})
    write_json_new(args.output / "EXECUTION_RECEIPT.json", {"status": "COMMAND_COMPLETED",
                   "scientific_status": status, "code_commit": args.code_commit,
                   "segmentation_training_steps": 0, "router_optimizer_steps": 0})
    inventory = []
    for path in sorted(p for p in args.output.rglob("*") if p.is_file()
                       and p.name != "RC_SHOR_V0_5_PRIVATE_MANIFEST.json"):
        inventory.append({"path": path.relative_to(args.output).as_posix(), "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    write_json_new(args.output / "RC_SHOR_V0_5_PRIVATE_MANIFEST.json", {
        "files": len(inventory), "bytes": sum(row["bytes"] for row in inventory),
        "content_sha256": canonical_hash(inventory), "entries": inventory})
    print(json.dumps({"scientific_status": status, "output": str(args.output)}, sort_keys=True))
    return status


def refuse_occupied_output(output):
    if (Path(output) / "public/RC_SHOR_V0_5_STATUS.json").exists():
        raise ProtocolViolation("REFUSED_AFTER_STATUS_EXISTS")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--test-report", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv); args.output = args.output.resolve()
    refuse_occupied_output(args.output)
    protocol, audit = load_protocol()
    require(socket.gethostname() == "zmic44" and os.getuid() == 1006, "wrong execution host/uid")
    nas = Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg").resolve()
    require(args.output.is_dir() and not args.output.is_symlink()
            and str(args.output).startswith(str(nas) + os.sep), "NAS create-only output required")
    require(os.environ.get("SSLCL_STORAGE_ROOT") == str(nas), "NAS wrapper is not active")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and args.device == "cuda:0",
            "exactly one visible CUDA device required")
    publication = source_gate(protocol, args.code_commit, args.test_report)
    deterministic = v4.configure_determinism()
    checkpoints, routers, thresholds, threshold_hashes, bootstrap_thresholds, baseline_path, threshold_path, router_path = load_frozen(protocol)
    rows = blind_legal_rows(protocol); folds = fold_assignments(rows)
    metadata = {"registration_id": protocol["registration_id"], "publication": publication,
                "deterministic": deterministic, "started_at": v4.now(), "v0_4_formal_03_reads": 0,
                "outer_GT_reads_before_seal": 0, "outer_domain_reads_before_seal": 0,
                "baseline_manifest_sha256": sha256_file(baseline_path),
                "threshold_manifest_sha256": sha256_file(threshold_path),
                "ridge_manifest_sha256": sha256_file(router_path), "threshold_hashes": threshold_hashes,
                "segmentation_training_steps": 0, "segmentation_optimizer_steps": 0,
                "segmentation_parameter_updates": 0, "router_optimizer_steps": 0}
    write_json_new(args.output / "RC_SHOR_V0_5_RUN_METADATA.json", metadata)
    descriptors, alpha, features, hard, probabilities, input_hashes, batches = materialize_blind_inputs(
        args, protocol, rows, checkpoints, routers, metadata)
    metadata["frozen_cache_materialization_batches"] = batches
    metadata["segmentation_expert_forward_batches"] = batches - sum(
        int(math.ceil(sum(row["seed"] == seed for row in rows) / BATCH_SIZE)) for seed in range(3))
    metadata["materialized_input_hashes"] = input_hashes
    write_json_new(args.output / "RC_SHOR_V0_5_MATERIALIZATION.json", metadata)
    result = execute_outer_folds(args, protocol, rows, folds, alpha, features, hard, probabilities,
                                 thresholds, bootstrap_thresholds)
    return 0 if decide_and_report(args, protocol, audit, publication, metadata, rows, result, args.test_report) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProtocolViolation, RequiredInputMissing, v4.ProtocolViolation, v4.RequiredInputMissing) as error:
        print(json.dumps({"status": getattr(error, "status", "BLOCKED_PROTOCOL_OR_LEAKAGE"),
                          "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
