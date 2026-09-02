#!/usr/bin/env python3
"""One-shot held-out test of the exact frozen SHOR V0.3.1 S3 policy."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import h5py
import numpy as np
import torch

from di_dmpa_gate1.binding import safe_asset
from di_dmpa_gate1_v2.features import ImmutableModels
from di_dmpa_gate1c_v2.binding import no_updates
from di_dmpa_gate1c_v3.inputs import load_models
from pres_dsr_sf_v0_2.core import raw_style_descriptors, router_probabilities
from shor_jascl_v0_3.core import shor_routes, top1_lowest


LCRSEG_ROOT = Path(__file__).resolve().parent
REPO = LCRSEG_ROOT.parents[1]
PROTOCOL_PATH = LCRSEG_ROOT / "docs/shor_v0_4_fixed_policy_test/SHOR_V0_4_TEST_PROTOCOL.json"
REMOTE = "https://github.com/DLwbm123/SSL_CL_seg.git"
DOMAINS = ("REFUGE", "RIM_ONE_r3", "Drishti_GS")
POLICIES = ("S0_SHARED", "S1_RIDGE_HARD", "S2_RIDGE_SOFT", "S3_SHOR", "S4_ORACLE")
BATCH_SIZE = 8
PUBLIC_FILES = ("SHOR_V0_4_TEST_STATUS.json", "SHOR_V0_4_TEST_REPORT.md", "shor_v0_4_test_metrics.csv")


class RequiredInputMissing(RuntimeError):
    status = "BLOCKED_REQUIRED_FROZEN_INPUT_MISSING"


class ProtocolViolation(RuntimeError):
    status = "BLOCKED_PROTOCOL_OR_LEAKAGE"


def require(condition, message, error=ProtocolViolation):
    if not condition:
        raise error(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_text_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def write_csv_new(path, rows):
    rows = list(rows)
    require(bool(rows), "empty public metrics")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def require_file(path, expected):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise RequiredInputMissing(str(path))
    observed = sha256_file(path)
    require(observed == expected, "frozen input SHA mismatch: %s" % path, RequiredInputMissing)
    return observed


def load_protocol():
    protocol = read_json(PROTOCOL_PATH)
    require(protocol["registration_id"] == "SHOR_V0_4_FIXED_POLICY_HELD_OUT_TEST", "wrong protocol")
    require(protocol["predecessor"]["scientific_status"] == "FAIL_SELECTIVE_OVERRIDE_STABILITY"
            and protocol["predecessor"]["H5"] is False
            and protocol["predecessor"]["mutable_by_this_protocol"] is False,
            "V0.3.1 status was changed")
    require(protocol["scope"]["stage_index"] == 2 and tuple(protocol["scope"]["domains"]) == DOMAINS,
            "test scope changed")
    return protocol


def source_gate(protocol, code_commit):
    branch = subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()
    require(branch == protocol["branch"] and head == code_commit and not dirty, "wrong or dirty execution source")
    count = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-list", "--count", protocol["base_commit"] + ".." + code_commit],
        text=True).strip()
    require(count == "1", "execution must be the first and only pre-test commit")
    expected = {
        "experiments/lcrseg/docs/shor_v0_4_fixed_policy_test/SHOR_V0_4_TEST_PROTOCOL.json",
        "experiments/lcrseg/shor_v0_4_test.py",
        "experiments/lcrseg/tests/shor_v0_4_test/test_protocol.py",
    }
    changed = set(subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "--name-only", protocol["base_commit"], code_commit],
        text=True).splitlines())
    require(changed == expected, "unregistered pre-test source delta")
    remote = subprocess.check_output(["git", "ls-remote", REMOTE, "refs/heads/" + protocol["branch"]], text=True).split()
    require(bool(remote) and remote[0] == code_commit, "pre-test commit is not published")
    return {"branch": branch, "code_commit": code_commit, "remote_sha": remote[0]}


def configure_determinism():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


class TestAccess:
    """The only transition that permits this run to materialize test truth/domain fields."""

    def __init__(self):
        self.sealed = False
        self.test_GT_reads = 0
        self.test_domain_reads = 0

    def mark_sealed(self, seal_path):
        seal = read_json(seal_path)
        require(seal["test_GT_reads"] == seal["test_domain_reads"] == 0, "candidate seal is contaminated")
        require(seal["status"] == "PASS_TEST_CANDIDATES_SEALED_BEFORE_GT", "candidate seal failed")
        self.sealed = True

    def require_evaluator(self):
        require(self.sealed, "test GT/domain requested before candidate seal")


def blind_test_rows(data_root, seed, expected_manifest_sha):
    """Materialize only role, ID and image fields; domain and label columns are never indexed."""
    manifest = Path(data_root) / "manifests/training" / ("lcrseg_v1_seed%d.csv" % seed)
    require_file(manifest, expected_manifest_sha)
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        required = ("dataset", "primary_20pct_split", "case_id", "image_h5_relpath", "image_sha256")
        require(all(name in header for name in required), "blind manifest fields missing", RequiredInputMissing)
        index = {name: header.index(name) for name in required}
        rows = []
        for values in reader:
            if values[index["dataset"]] == "fundus" and values[index["primary_20pct_split"]] == "test":
                rows.append({name: values[index[name]] for name in ("case_id", "image_h5_relpath", "image_sha256")})
    rows.sort(key=lambda row: row["case_id"])
    require(bool(rows) and len(rows) == len({row["case_id"] for row in rows}), "invalid blind test inventory",
            RequiredInputMissing)
    return rows, manifest


def full_test_rows(data_root, seed, expected_manifest_sha, expected_case_ids, access):
    access.require_evaluator()
    manifest = Path(data_root) / "manifests/training" / ("lcrseg_v1_seed%d.csv" % seed)
    require_file(manifest, expected_manifest_sha)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["dataset"] == "fundus" and row["primary_20pct_split"] == "test"]
    rows.sort(key=lambda row: row["case_id"])
    require([row["case_id"] for row in rows] == list(expected_case_ids), "Phase A/B test cases differ")
    require(all(row["site_or_vendor"] in DOMAINS and row["label_h5_relpath"] and row["label_sha256"] for row in rows),
            "test evaluator fields missing", RequiredInputMissing)
    access.test_domain_reads += len(rows)
    return rows


def read_images(rows, data_root):
    images = []
    for row in rows:
        path = safe_asset(data_root, row["image_h5_relpath"])
        with h5py.File(path, "r") as handle:
            image = handle["image"][...]
        require(image.shape == (3, 384, 384), "test image geometry changed", RequiredInputMissing)
        images.append(np.asarray(image, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(images))


def read_label(row, data_root, access):
    access.require_evaluator()
    path = safe_asset(data_root, row["label_h5_relpath"])
    require_file(path, row["label_sha256"])
    with h5py.File(path, "r") as handle:
        label = handle["label"][...]
    require(label.shape == (384, 384), "test label geometry changed", RequiredInputMissing)
    access.test_GT_reads += 1
    return np.asarray(label, dtype=np.int64), path


def load_frozen_policy(protocol):
    frozen = protocol["frozen_inputs"]
    threshold_path = Path(frozen["v0_3_1_threshold_manifest"]["path"])
    router_path = Path(frozen["ridge_state_manifest"]["path"])
    require_file(threshold_path, frozen["v0_3_1_threshold_manifest"]["sha256"])
    require_file(router_path, frozen["ridge_state_manifest"]["sha256"])
    threshold_manifest = read_json(threshold_path)
    router_manifest = read_json(router_path)
    thresholds, routers, threshold_hashes = {}, {}, {}
    for seed in range(3):
        threshold_rows = [row for row in threshold_manifest["formal"]
                          if row["seed"] == seed and row["stage_index"] == 2]
        require(len(threshold_rows) == 2 and {row["historical_domain"] for row in threshold_rows} == {0, 1}
                and all(row["feasible"] and np.isfinite(row["threshold"]) for row in threshold_rows),
                "required frozen threshold missing", RequiredInputMissing)
        thresholds[seed] = {row["historical_domain"]: float(row["threshold"]) for row in threshold_rows}
        threshold_hashes[str(seed)] = {str(row["historical_domain"]): canonical_hash(row) for row in threshold_rows}
        state_rows = [row for row in router_manifest["formal"] if row["seed"] == seed and row["stage_index"] == 2]
        require(len(state_rows) == 1, "required frozen ridge state missing", RequiredInputMissing)
        row = state_rows[0]
        require(row["selected_temperature"] == row["M1_temperature"] == 0.5
                and row["selected_lambda"] in (0.0001, 0.001, 0.01, 0.1, 1.0),
                "frozen ridge selection changed", RequiredInputMissing)
        constant = np.asarray(row["constant"], dtype=bool)
        std = np.asarray(row["std"], dtype=np.float64)
        routers[seed] = {
            "mean": np.asarray(row["mean"], dtype=np.float64),
            "std": std,
            "scale": np.where(constant, 1.0, std),
            "constant": constant,
            "weights": np.asarray(row["weights"], dtype=np.float64),
            "selected_temperature": float(row["selected_temperature"]),
        }
        require(routers[seed]["mean"].shape == routers[seed]["std"].shape == (102,)
                and routers[seed]["weights"].shape == (103, 3), "frozen ridge tensor schema changed",
                RequiredInputMissing)
    return routers, thresholds, threshold_hashes, threshold_path, router_path


def route_s3(alpha, thresholds):
    """The S3 API intentionally has no test-domain argument."""
    return shor_routes(np.asarray(alpha, dtype=np.float64), stage=2, thresholds=thresholds)


def new_memmap(path, shape, dtype):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "create-only cache exists: %s" % path)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def extract_test_descriptors(output, seed, rows, checkpoint, data_root, device, metadata):
    models, payload = load_models(LCRSEG_ROOT, checkpoint, device=device, sources=("ema_teacher",))
    model = models["ema_teacher"]
    batches = []
    with ImmutableModels(models, checkpoint, output / ("immutability/router_seed%d" % seed), metadata):
        with torch.no_grad():
            for start in range(0, len(rows), BATCH_SIZE):
                images = read_images(rows[start:start + BATCH_SIZE], data_root).to(device)
                with torch.autocast(device_type="cuda", enabled=False):
                    enc1 = model.enc1(images)
                    enc2 = model.enc2(model.pool(enc1))
                batches.append(raw_style_descriptors(images, enc1, enc2))
    value = np.concatenate(batches)
    require(value.shape == (len(rows), 102), "test descriptor schema changed")
    path = output / "descriptor_cache" / ("seed%d.npy" % seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
    del models, payload, model
    torch.cuda.empty_cache()
    return value, path


def predict_expert(output, seed, expert, rows, checkpoint, data_root, device, metadata):
    models, payload = load_models(LCRSEG_ROOT, checkpoint, device=device, sources=("student",))
    model = models["student"]
    path = output / "expert_probability_cache" / ("seed%d_expert%d.npy" % (seed, expert))
    target = new_memmap(path, (len(rows), 3, 384, 384), np.float32)
    with ImmutableModels(models, checkpoint, output / ("immutability/expert_seed%d_stage%d" % (seed, expert)), metadata):
        with torch.no_grad():
            for start in range(0, len(rows), BATCH_SIZE):
                images = read_images(rows[start:start + BATCH_SIZE], data_root).to(device)
                with torch.autocast(device_type="cuda", enabled=False):
                    logits, _ = model(images, stochastic_classifier=False)
                    probability = logits.float().softmax(1)
                require(probability.shape == (len(images), 3, 384, 384), "expert probability schema changed")
                target[start:start + len(images)] = probability.cpu().numpy()
    target.flush()
    del target, models, payload, model
    torch.cuda.empty_cache()
    return path


def materialize_candidates(output, seed, alpha, thresholds, expert_paths):
    experts = [np.load(path, mmap_mode="r", allow_pickle=False) for path in expert_paths]
    count = len(alpha)
    top = top1_lowest(alpha)
    s3 = route_s3(alpha, thresholds)
    routes = {
        "S0_SHARED": np.full(count, 2, dtype=np.int64),
        "S1_RIDGE_HARD": top.astype(np.int64),
        "S3_SHOR": s3.astype(np.int64),
    }
    paths = {}
    for policy in ("S0_SHARED", "S1_RIDGE_HARD", "S2_RIDGE_SOFT", "S3_SHOR"):
        path = output / "candidate_prediction_cache" / ("seed%d_%s.npy" % (seed, policy))
        target = new_memmap(path, (count, 384, 384), np.uint8)
        for index in range(count):
            if policy == "S2_RIDGE_SOFT":
                fused = sum(float(alpha[index, expert]) * experts[expert][index] for expert in range(3))
                target[index] = np.argmax(fused, axis=0).astype(np.uint8)
            else:
                target[index] = np.argmax(experts[int(routes[policy][index])][index], axis=0).astype(np.uint8)
        target.flush()
        del target
        paths[policy] = path
    route_path = output / "route_cache" / ("seed%d.npz" % seed)
    route_path.parent.mkdir(parents=True, exist_ok=True)
    with route_path.open("xb") as handle:
        np.savez_compressed(handle, alpha=alpha, top1=top, s3=s3)
    return routes, paths, route_path


def validate_eval_alignment(candidate_ids, evaluator_ids):
    require(list(candidate_ids) == list(evaluator_ids) and len(set(candidate_ids)) == len(candidate_ids),
            "S0/S3 evaluator cases are not identical")


def case_metrics(prediction, label):
    valid = label != 255
    dice, iou = [], []
    for class_id in (0, 1, 2):
        pred = (prediction == class_id) & valid
        truth = (label == class_id) & valid
        intersection = int((pred & truth).sum())
        denom = int(pred.sum() + truth.sum())
        union = int((pred | truth).sum())
        dice.append(1.0 if denom == 0 else 2.0 * intersection / denom)
        iou.append(1.0 if union == 0 else intersection / union)
    return {
        "foreground_dice": float((dice[1] + dice[2]) / 2.0),
        "rim_dice": float(dice[1]),
        "cup_dice": float(dice[2]),
        "mean_iou": float(np.mean(iou)),
    }


def balanced_case_mean(rows, value):
    groups = {}
    for row in rows:
        groups.setdefault((row["seed"], row["domain_index"]), []).append(float(value(row)))
    require(bool(groups), "empty balanced aggregate")
    return float(np.mean([np.mean(group) for group in groups.values()]))


def summarize(case_rows, level, key, value):
    selected = [row for row in case_rows if key is None or row[key] == value]
    output = []
    for policy in POLICIES:
        rows = [row for row in selected if row["policy"] == policy]
        require(bool(rows), "empty aggregate: %s %s" % (level, policy))
        item = {
            "row_type": "metric",
            "level": level,
            "key": "all" if value is None else value,
            "policy": policy,
            "cases": len(rows),
            "foreground_dice": balanced_case_mean(rows, lambda row: row["foreground_dice"]),
            "rim_dice": balanced_case_mean(rows, lambda row: row["rim_dice"]),
            "cup_dice": balanced_case_mean(rows, lambda row: row["cup_dice"]),
            "mean_iou": balanced_case_mean(rows, lambda row: row["mean_iou"]),
        }
        if policy == "S3_SHOR":
            item.update(
                route0_frequency=balanced_case_mean(rows, lambda row: row["s3_route"] == 0),
                route1_frequency=balanced_case_mean(rows, lambda row: row["s3_route"] == 1),
                route2_frequency=balanced_case_mean(rows, lambda row: row["s3_route"] == 2),
                historical_override_frequency=balanced_case_mean(rows, lambda row: row["s3_route"] < 2),
                false_historical_override_frequency=balanced_case_mean(
                    rows, lambda row: row["s3_route"] < 2 and row["s3_route"] != row["domain_index"]),
            )
        output.append(item)
    return output


def paired_groups(case_rows):
    lookup = {}
    for row in case_rows:
        if row["policy"] in ("S0_SHARED", "S3_SHOR"):
            lookup[(row["seed"], row["domain_index"], row["case_index"], row["policy"])] = row["foreground_dice"]
    groups = {}
    for seed in range(3):
        for domain in range(3):
            keys = sorted(key for key in lookup if key[0] == seed and key[1] == domain and key[3] == "S0_SHARED")
            require(bool(keys), "empty paired seed/domain")
            gains = []
            for key in keys:
                paired = (key[0], key[1], key[2], "S3_SHOR")
                require(paired in lookup, "S0/S3 pairing changed")
                gains.append(lookup[paired] - lookup[key])
            groups[(seed, domain)] = np.asarray(gains, dtype=np.float64)
    return groups


def paired_bootstrap(groups, replicates=2000, seed=20260902):
    require(set(groups) == {(seed_, domain) for seed_ in range(3) for domain in range(3)},
            "paired bootstrap groups incomplete")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty((replicates, 3), dtype=np.float64)
    for replicate in range(replicates):
        means = {}
        for key, values in groups.items():
            sampled = rng.integers(0, len(values), size=len(values))
            means[key] = float(values[sampled].mean())
        draws[replicate, 0] = np.mean(list(means.values()))
        draws[replicate, 1] = np.mean([means[(model_seed, domain)] for model_seed in range(3) for domain in (0, 1)])
        draws[replicate, 2] = np.mean([-means[(model_seed, 2)] for model_seed in range(3)])
    names = ("delta_all", "delta_history", "current_drop")
    point = {
        "delta_all": float(np.mean([values.mean() for values in groups.values()])),
        "delta_history": float(np.mean([groups[(seed_, domain)].mean() for seed_ in range(3) for domain in (0, 1)])),
        "current_drop": float(np.mean([-groups[(seed_, 2)].mean() for seed_ in range(3)])),
    }
    intervals = {name: {"lower95": float(np.quantile(draws[:, index], 0.025)),
                        "upper95": float(np.quantile(draws[:, index], 0.975))}
                 for index, name in enumerate(names)}
    seed_delta = {str(seed_): float(np.mean([groups[(seed_, domain)].mean() for domain in range(3)]))
                  for seed_ in range(3)}
    return {"replicates": replicates, "seed": seed, "point": point, "intervals": intervals,
            "seed_delta_all": seed_delta, "draw_sha256": array_hash(draws)}


def phase_a(args, protocol, checkpoints, routers, thresholds, threshold_hashes, access, metadata):
    data_root = Path(protocol["frozen_inputs"]["data_root"])
    frozen = protocol["frozen_inputs"]
    baseline_path = REPO / frozen["baseline_snapshot_manifest"]["path"]
    input_hashes = {
        str(PROTOCOL_PATH): sha256_file(PROTOCOL_PATH),
        str(baseline_path): frozen["baseline_snapshot_manifest"]["sha256"],
        frozen["v0_3_1_threshold_manifest"]["path"]: frozen["v0_3_1_threshold_manifest"]["sha256"],
        frozen["ridge_state_manifest"]["path"]: frozen["ridge_state_manifest"]["sha256"],
    }
    blind, orders = {}, {}
    for asset in protocol["frozen_inputs"]["seed_assets"]:
        seed = asset["seed"]
        rows, manifest = blind_test_rows(data_root, seed, asset["manifest_sha256"])
        split = data_root / "splits" / ("fundus_seed%d.json" % seed)
        input_hashes[str(manifest)] = require_file(manifest, asset["manifest_sha256"])
        input_hashes[str(split)] = require_file(split, asset["split_sha256"])
        for row in rows:
            path = safe_asset(data_root, row["image_h5_relpath"])
            input_hashes[str(path)] = require_file(path, row["image_sha256"])
        blind[seed] = rows
        orders[seed] = [row["case_id"] for row in rows]
    descriptor_hashes, snapshot_hashes, route_hashes, candidate_hashes = {}, {}, {}, {}
    probability_hashes = {}
    with no_updates():
        for seed in range(3):
            rows = blind[seed]
            router_cp = checkpoints[(seed, 0)]
            snapshot_hashes[router_cp["checkpoint_id"]] = require_file(router_cp["path"], router_cp["sha256"])
            descriptors, descriptor_path = extract_test_descriptors(
                args.output, seed, rows, router_cp, data_root, args.device, metadata)
            descriptor_hashes[str(seed)] = sha256_file(descriptor_path)
            alpha = router_probabilities(descriptors, routers[seed])
            require(alpha.shape == (len(rows), 3), "test ridge alpha schema changed")
            expert_paths = []
            for expert in range(3):
                cp = checkpoints[(seed, expert)]
                snapshot_hashes[cp["checkpoint_id"]] = require_file(cp["path"], cp["sha256"])
                path = predict_expert(args.output, seed, expert, rows, cp, data_root, args.device, metadata)
                expert_paths.append(path)
                probability_hashes["seed%d_expert%d" % (seed, expert)] = sha256_file(path)
            routes, candidate_paths, route_path = materialize_candidates(
                args.output, seed, alpha, thresholds[seed], expert_paths)
            route_hashes[str(seed)] = {name: array_hash(value) for name, value in routes.items()}
            route_hashes[str(seed)]["cache_sha256"] = sha256_file(route_path)
            candidate_hashes[str(seed)] = {policy: sha256_file(path) for policy, path in candidate_paths.items()}
    seal = {
        "schema_version": 1,
        "status": "PASS_TEST_CANDIDATES_SEALED_BEFORE_GT",
        "sealed_at": now(),
        "code_commit": args.code_commit,
        "input_hashes": input_hashes,
        "snapshot_hashes": snapshot_hashes,
        "threshold_hashes": threshold_hashes,
        "case_order_hashes": {str(seed): canonical_hash(orders[seed]) for seed in range(3)},
        "case_order_hash": canonical_hash({str(seed): orders[seed] for seed in range(3)}),
        "descriptor_hashes": descriptor_hashes,
        "probability_hashes": probability_hashes,
        "route_hashes": route_hashes,
        "candidate_prediction_hashes": candidate_hashes,
        "test_GT_reads": access.test_GT_reads,
        "test_domain_reads": access.test_domain_reads,
        "training_steps": 0,
        "optimizer_constructions": 0,
        "model_updates": 0,
        "S3_test_domain_inputs": 0,
    }
    seal_path = args.output / "TEST_CANDIDATE_SEAL.json"
    write_json_new(seal_path, seal)
    access.mark_sealed(seal_path)
    return orders, seal_path


def phase_b(args, protocol, orders, access, seal_path):
    data_root = Path(protocol["frozen_inputs"]["data_root"])
    case_rows, label_hashes = [], {}
    private_metrics = args.output / "test_case_metrics.jsonl"
    with private_metrics.open("x", encoding="utf-8") as case_handle:
        for asset in protocol["frozen_inputs"]["seed_assets"]:
            seed = asset["seed"]
            rows = full_test_rows(data_root, seed, asset["manifest_sha256"], orders[seed], access)
            validate_eval_alignment(orders[seed], [row["case_id"] for row in rows])
            predictions = {policy: np.load(
                args.output / "candidate_prediction_cache" / ("seed%d_%s.npy" % (seed, policy)),
                mmap_mode="r", allow_pickle=False) for policy in POLICIES[:4]}
            experts = [np.load(args.output / "expert_probability_cache" / ("seed%d_expert%d.npy" % (seed, expert)),
                               mmap_mode="r", allow_pickle=False) for expert in range(3)]
            with np.load(args.output / "route_cache" / ("seed%d.npz" % seed), allow_pickle=False) as route_cache:
                s3_routes = np.asarray(route_cache["s3"], dtype=np.int64)
            for index, row in enumerate(rows):
                domain = DOMAINS.index(row["site_or_vendor"])
                label, label_path = read_label(row, data_root, access)
                label_hashes[str(label_path)] = row["label_sha256"]
                for policy in POLICIES:
                    prediction = (np.argmax(experts[domain][index], axis=0).astype(np.uint8)
                                  if policy == "S4_ORACLE" else predictions[policy][index])
                    item = {
                        "seed": seed,
                        "domain_index": domain,
                        "domain": DOMAINS[domain],
                        "case_index": index,
                        "case_id": row["case_id"],
                        "policy": policy,
                        "s3_route": int(s3_routes[index]),
                    }
                    item.update(case_metrics(prediction, label))
                    case_rows.append(item)
                    case_handle.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
        case_handle.flush()
        os.fsync(case_handle.fileno())
    evaluation_inputs = {
        "status": "PASS_EVALUATOR_INPUTS_AFTER_CANDIDATE_SEAL",
        "candidate_seal_sha256": sha256_file(seal_path),
        "label_file_sha256": label_hashes,
        "test_GT_reads": access.test_GT_reads,
        "test_domain_reads": access.test_domain_reads,
        "test_domain_uses": ["domain_aggregation", "S4_ORACLE", "post_hoc_attribution"],
        "S3_test_domain_inputs": 0,
    }
    evaluation_path = args.output / "TEST_EVALUATION_INPUTS.json"
    write_json_new(evaluation_path, evaluation_inputs)

    metrics = []
    metrics.extend(summarize(case_rows, "overall", None, None))
    for seed in range(3):
        metrics.extend(summarize(case_rows, "seed", "seed", seed))
    for domain in DOMAINS:
        metrics.extend(summarize(case_rows, "domain", "domain", domain))
    bootstrap = paired_bootstrap(paired_groups(case_rows),
                                 replicates=protocol["bootstrap"]["replicates"],
                                 seed=protocol["bootstrap"]["seed"])
    conditions = {
        "lower95_delta_all_gt_0": bootstrap["intervals"]["delta_all"]["lower95"] > 0.0,
        "lower95_delta_history_gt_0": bootstrap["intervals"]["delta_history"]["lower95"] > 0.0,
        "upper95_current_drop_le_0_02": bootstrap["intervals"]["current_drop"]["upper95"] <= 0.02,
        "positive_seed_count_ge_2": sum(value > 0.0 for value in bootstrap["seed_delta_all"].values()) >= 2,
    }
    scientific_status = ("PASS_FIXED_POLICY_TEST_EFFECTIVENESS" if all(conditions.values())
                         else "FAIL_FIXED_POLICY_TEST_EFFECTIVENESS")
    overall = {row["policy"]: row for row in metrics if row["level"] == "overall"}
    oracle_gap = overall["S4_ORACLE"]["foreground_dice"] - overall["S3_SHOR"]["foreground_dice"]
    for name in ("delta_all", "delta_history", "current_drop"):
        metrics.append({
            "row_type": "comparison",
            "level": "overall",
            "key": name,
            "policy": "S3_SHOR_minus_S0_SHARED" if name != "current_drop" else "S0_SHARED_minus_S3_SHOR",
            "estimate": bootstrap["point"][name],
            "lower95": bootstrap["intervals"][name]["lower95"],
            "upper95": bootstrap["intervals"][name]["upper95"],
        })
    metrics.append({"row_type": "comparison", "level": "overall", "key": "oracle_gap",
                    "policy": "S4_ORACLE_minus_S3_SHOR", "estimate": oracle_gap})

    public = args.output / "public"
    public.mkdir()
    status = {
        "schema_version": 1,
        "registration_id": protocol["registration_id"],
        "scientific_status": scientific_status,
        "completed_at": now(),
        "execution_code_commit": args.code_commit,
        "formal_attempt": 1,
        "additional_attempt_authorized": False,
        "decision_conditions": conditions,
        "bootstrap": bootstrap,
        "oracle_gap": oracle_gap,
        "S3_route_frequency": {
            "route0": overall["S3_SHOR"]["route0_frequency"],
            "route1": overall["S3_SHOR"]["route1_frequency"],
            "route2": overall["S3_SHOR"]["route2_frequency"],
            "historical_override": overall["S3_SHOR"]["historical_override_frequency"],
            "false_historical_override": overall["S3_SHOR"]["false_historical_override_frequency"],
        },
        "isolation": {
            "candidate_sealed_before_test_GT_or_domain": True,
            "candidate_seal_sha256": sha256_file(seal_path),
            "evaluation_input_record_sha256": sha256_file(evaluation_path),
            "test_GT_reads": access.test_GT_reads,
            "test_domain_reads": access.test_domain_reads,
            "S3_test_domain_inputs": 0,
            "training_steps": 0,
            "optimizer_constructions": 0,
            "model_updates": 0,
        },
        "predecessor": {
            "scientific_status": "FAIL_SELECTIVE_OVERRIDE_STABILITY",
            "H1": True, "H2": True, "H3": True, "H4": True, "H5": False, "H6": True,
            "unchanged_by_this_test": True,
        },
        "repeat_final_evaluation": "REFUSED_AFTER_STATUS_EXISTS",
        "large_private_artifacts": str(args.output),
        "report_commit": None,
        "report_commit_resolution": "second Git commit adding these exact report bytes",
    }
    status_path = public / PUBLIC_FILES[0]
    write_json_new(status_path, status)
    metrics_path = public / PUBLIC_FILES[2]
    write_csv_new(metrics_path, metrics)
    report = """# SHOR V0.4 fixed-policy held-out test report

## Outcome

**{status}**

This was a new, one-shot held-out effectiveness question. It was not a V0.3.1 retry, repair, or re-adjudication. SHOR V0.3.1 remains permanently `FAIL_SELECTIVE_OVERRIDE_STABILITY`: H1/H2/H3/H4/H6 passed and H5 failed.

## Direct S3 versus S0 result

| Quantity | Point estimate | 95% paired hierarchical bootstrap CI |
|---|---:|---:|
| delta_all | {da:.12f} | [{dal:.12f}, {dau:.12f}] |
| delta_history | {dh:.12f} | [{dhl:.12f}, {dhu:.12f}] |
| current_drop | {cd:.12f} | [{cdl:.12f}, {cdu:.12f}] |

Positive seed deltas: {positive}/3. The fixed raw-value conditions were: `{conditions}`.

S3 overall foreground Dice was {s3:.12f}; S0 was {s0:.12f}. S3 historical-route frequency was {route:.12f}, false historical override frequency was {false_route:.12f}, and the S4 oracle gap was {oracle:.12f}.

## Isolation and scope

All test images were processed with the exact frozen descriptors, ridge state, temperature, thresholds, tie rule, score, S3 rule, snapshots, preprocessing, and deterministic inference settings. `TEST_CANDIDATE_SEAL.json` was durably written with zero test-GT reads, zero test-domain reads, and zero training steps before Phase B opened evaluator-only label/domain fields. S3 never received test-domain identity. No fitting, optimizer, backward, training, snapshot update, validation reuse, or H1-H6 rerun occurred.

The private NAS retains case-level metrics, image/label input hashes, descriptors, probabilities, routes, and masks. GitHub receives only aggregate metrics and small reviewer-facing evidence. The formal test is final; repeated evaluation is refused and no retry, redesign, training, or main-branch merge is authorized.
""".format(
        status=scientific_status,
        da=bootstrap["point"]["delta_all"], dal=bootstrap["intervals"]["delta_all"]["lower95"],
        dau=bootstrap["intervals"]["delta_all"]["upper95"],
        dh=bootstrap["point"]["delta_history"], dhl=bootstrap["intervals"]["delta_history"]["lower95"],
        dhu=bootstrap["intervals"]["delta_history"]["upper95"],
        cd=bootstrap["point"]["current_drop"], cdl=bootstrap["intervals"]["current_drop"]["lower95"],
        cdu=bootstrap["intervals"]["current_drop"]["upper95"],
        positive=sum(value > 0 for value in bootstrap["seed_delta_all"].values()),
        conditions=all(conditions.values()), s3=overall["S3_SHOR"]["foreground_dice"],
        s0=overall["S0_SHARED"]["foreground_dice"],
        route=overall["S3_SHOR"]["historical_override_frequency"],
        false_route=overall["S3_SHOR"]["false_historical_override_frequency"], oracle=oracle_gap)
    report_path = public / PUBLIC_FILES[1]
    write_text_new(report_path, report)
    entries = [{"path": name, "bytes": (public / name).stat().st_size, "sha256": sha256_file(public / name)}
               for name in PUBLIC_FILES]
    manifest = {
        "schema_version": 1,
        "status": "PASS_PUBLIC_REPORT_BUNDLE_COMPLETE",
        "scientific_status": scientific_status,
        "execution_code_commit": args.code_commit,
        "entries": entries,
        "candidate_seal_sha256": sha256_file(seal_path),
        "private_case_metrics_sha256": sha256_file(private_metrics),
        "private_large_artifacts_published": False,
        "private_large_artifacts_root": str(args.output),
        "public_scope": ["protocol", "execution_code", "targeted_tests", "aggregate_metrics", "status", "report"],
    }
    write_json_new(public / "SHOR_V0_4_TEST_MANIFEST.json", manifest)
    return scientific_status


def refuse_occupied_output(output):
    status = Path(output) / "public/SHOR_V0_4_TEST_STATUS.json"
    if status.exists():
        raise ProtocolViolation("REFUSED_AFTER_STATUS_EXISTS")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    refuse_occupied_output(args.output)
    protocol = load_protocol()
    require(socket.gethostname() == "zmic44" and os.getuid() == 1006, "wrong execution host/uid")
    nas = Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg").resolve()
    require(str(args.output).startswith(str(nas) + os.sep) and args.output.is_dir() and not args.output.is_symlink(),
            "NAS create-only output required")
    require(os.environ.get("SSLCL_STORAGE_ROOT") == str(nas), "NAS wrapper not active")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and args.device == "cuda:0",
            "exactly one visible CUDA GPU required")
    publication = source_gate(protocol, args.code_commit)
    deterministic = configure_determinism()
    frozen = protocol["frozen_inputs"]
    baseline_path = REPO / frozen["baseline_snapshot_manifest"]["path"]
    require_file(baseline_path, frozen["baseline_snapshot_manifest"]["sha256"])
    baseline = read_json(baseline_path)
    require(baseline["status"] == "PASS_ALL_THREE_REGENERATED_B0_SEEDS", "snapshot baseline unavailable",
            RequiredInputMissing)
    checkpoints = {(row["seed"], row["stage_index"]): row for row in baseline["checkpoints"]}
    require(set(checkpoints) == {(seed, stage) for seed in range(3) for stage in range(3)},
            "required frozen snapshots missing", RequiredInputMissing)
    routers, thresholds, threshold_hashes, threshold_path, router_path = load_frozen_policy(protocol)
    metadata = {
        "registration_id": protocol["registration_id"],
        "publication": publication,
        "deterministic_backend": deterministic,
        "threshold_manifest_sha256": sha256_file(threshold_path),
        "ridge_state_manifest_sha256": sha256_file(router_path),
        "baseline_snapshot_manifest_sha256": sha256_file(baseline_path),
        "test_GT_reads_before_candidate_seal": 0,
        "test_domain_reads_before_candidate_seal": 0,
        "training_steps": 0,
        "started_at": now(),
    }
    write_json_new(args.output / "SHOR_V0_4_TEST_RUN_METADATA.json", metadata)
    access = TestAccess()
    orders, seal_path = phase_a(
        args, protocol, checkpoints, routers, thresholds, threshold_hashes, access, metadata)
    status = phase_b(args, protocol, orders, access, seal_path)
    print(json.dumps({"scientific_status": status, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RequiredInputMissing, ProtocolViolation) as error:
        print(json.dumps({"status": error.status, "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
