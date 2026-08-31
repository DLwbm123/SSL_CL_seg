#!/usr/bin/env python3
"""The one registered Fundus baseline test readout; no training or retries."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.common import sha256_path, utc_now, write_csv, write_json
from lcrseg.data.h5_dataset import load_training_records
from lcrseg.engine.checkpoint import load_checkpoint
from lcrseg.engine.evaluator import evaluate_site
from lcrseg.models import UNet2D

TRAIN_COMMIT = "4d4c2e4333fd0c75733b58d0c44227b15beedc6b"
REG_SHA = "70dc562b87c8d49740253d9c91b5497cdce637dcd1320f2edaf5341b27b07c09"
PLAN_SHA = "97a26c3db35ff1db68fcb0b7ec70c34fdf135ee876c47d860b2d6f59cf48cddb"
TEST_COUNTS = {"REFUGE": 100, "RIM_ONE_r3": 40, "Drishti_GS": 25}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def adjudicate(cells, reg):
    domains = reg["domain_order"]
    expected = {(seed, arm, stage, site) for seed in reg["seeds"] for arm in reg["arms"]
                for stage in range(3) for site in domains[:stage + 1]}
    lookup = {}
    for row in cells:
        key = (row["seed"], row["arm"], row["stage"], row["site"])
        require(key in expected and key not in lookup, "unexpected or duplicate test cell")
        values = [float(row[k]) for k in ("dice_class_1", "dice_class_2", "mean_foreground_dice")]
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), "invalid required Dice")
        require(math.isclose(values[2], (values[0] + values[1]) / 2, abs_tol=1e-12, rel_tol=0), "class mean differs")
        lookup[key] = values[2]
    require(set(lookup) == expected and len(cells) == reg["test_cells"], "incomplete test matrix")
    summaries, paired = [], []
    for seed in reg["seeds"]:
        scores = {}
        for arm in reg["arms"]:
            final = [lookup[seed, arm, 2, site] for site in domains]
            diagonal = [lookup[seed, arm, stage, site] for stage, site in enumerate(domains)]
            scores[arm] = dict(F=sum(final) / 3, I=sum(diagonal) / 3,
                               BWT=sum(final[i] - diagonal[i] for i in (0, 1)) / 2)
            summaries.append(dict(seed=seed, arm=arm, **scores[arm]))
        paired.append(dict(seed=seed, **{k: scores["uniform_kd"][k] - scores["sequential_ssl"][k] for k in ("F", "I", "BWT")}))
    means = {key: sum(row[key] for row in paired) / len(paired) for key in ("F", "I", "BWT")}
    bounds = reg["success"]
    observed = dict(mean_final_dice_improvement=means["F"], positive_final_dice_seeds=sum(r["F"] > 0 for r in paired),
                    per_seed_final_dice_improvement=min(r["F"] for r in paired), mean_bwt_improvement=means["BWT"],
                    mean_incoming_dice_improvement=means["I"])
    gates = {key: dict(observed=value, minimum=bounds[key + "_min"], passed=value >= bounds[key + "_min"])
             for key, value in observed.items()}
    return dict(status="PASS_BASELINE_FEASIBILITY" if all(x["passed"] for x in gates.values()) else "FAIL_BASELINE_FEASIBILITY",
                cells=len(cells), per_seed_arm=summaries, paired=paired, gates=gates,
                failed_gates=[key for key, value in gates.items() if not value["passed"]])


def audit_training(root, reg):
    """Complete this zero-forward audit before requesting any test-role asset."""
    require(read(root / "operations/ENGINEERING_ADMISSION.json")["status"] == "PASS_ENGINEERING", "engineering gate missing")
    plan_path = root / "operations/FORMAL_PLAN.json"
    require(sha256_path(plan_path) == PLAN_SHA, "formal plan changed")
    plan = {(r["seed"], r["arm"]): r for r in read(plan_path)}
    # All process exits are checked before opening checkpoints or test views.
    for seed in reg["seeds"]:
        require(read(root / f"operations/queue_seed{seed}/exit.json")["exit_code"] == 0, "queue not complete")
        for arm in reg["arms"]:
            op = root / f"operations/seed{seed}_{arm}"
            require(read(op / "exit.json")["exit_code"] == 0, "training child not complete")
            require(read(op / "training_verified.json")["status"] == "PASS_TRAINING_ARTIFACTS", "training artifact gate missing")
    training_code = root / "code/SSL_CL_seg"
    require(subprocess.check_output(["git", "-C", str(training_code), "rev-parse", "HEAD"], text=True).strip() == TRAIN_COMMIT, "training source changed")
    require(not subprocess.check_output(["git", "-C", str(training_code), "status", "--porcelain"], text=True).strip(), "dirty training source")
    hashes, checkpoints = {}, []
    for entry in reg["frozen_inputs"]:
        seed = entry["seed"]
        for path, digest in [(Path(reg["data_root"]) / f"manifests/training/lcrseg_v1_seed{seed}.csv", entry["manifest_sha256"]),
                             (Path(reg["data_root"]) / f"splits/fundus_seed{seed}.json", entry["split_sha256"])]:
            require(sha256_path(path) == digest, "frozen metadata changed")
            hashes[str(path)] = digest
        for arm in reg["arms"]:
            op, run = root / f"operations/seed{seed}_{arm}", root / f"runs/seed{seed}_{arm}"
            require(sha256_path(op / "expected_config.json") == plan[seed, arm]["config_sha256"], "planned config changed")
            config = read(op / "expected_config.json")
            require(read(run / "config.yaml") == config, "runtime config mismatch")
            summary = read(run / "run_summary.json")
            require(summary["status"] == "complete" and summary["completed_global_steps"] == 13400, "incomplete training budget")
            rows = list(csv.DictReader((run / "train_log.csv").open()))
            require(len(rows) == 13400 and [int(x["global_step"]) for x in rows] == list(range(1, 13401)), "noncontiguous training log")
            require(all(float(x["optimizer_step_skipped"]) == 0 and all(math.isfinite(float(v)) for k, v in x.items() if k.startswith("loss_")) for x in rows), "nonfinite/skipped training")
            previous = None
            for stage, (site, steps) in enumerate(zip(reg["domain_order"], reg["steps_by_domain"])):
                require(sum(x["site_id"] == site for x in rows) == steps, "domain budget differs")
                path = run / f"checkpoint_final_site{stage}_{site}.pt"
                hashes[str(path)] = sha256_path(path)
                cp = load_checkpoint(path)
                require(cp["method_name"] == arm and cp["git_commit"] == TRAIN_COMMIT and cp["config_resolved"] == config, "checkpoint provenance differs")
                require(cp["site_index"] == stage and cp["site_id"] == site and cp["site_step"] == steps and cp["global_step"] == sum(reg["steps_by_domain"][:stage + 1]), "checkpoint counters differ")
                require(cp["manifest_hash"] == entry["manifest_sha256"] and cp["data_split_hash"] == entry["split_sha256"], "checkpoint data lineage differs")
                state = cp["current_model_state"]
                require(all(torch.isfinite(x).all().item() for x in state.values()), "nonfinite checkpoint")
                require(not cp["current_anchor_state"] and not cp["historical_anchor_state"], "unexpected prototype state")
                old = cp["method_statistics"]["old_model_state"]
                if arm == "uniform_kd" and stage > 0:
                    require(old.keys() == previous.keys() and all(torch.equal(old[k], previous[k]) for k in old), "old model changed")
                else:
                    require(not old, "unexpected old model")
                previous = state
                checkpoints.append(dict(seed=seed, arm=arm, stage=stage, path=str(path), sha256=hashes[str(path)]))
    audit_path = root / "operations/input_audit/audit.json"
    require(sha256_path(audit_path) == "12e36dd7043d894ae401c25d625998ff33faefddb77852085ec10bf09eab5728", "initial input audit changed")
    original = read(audit_path)
    for path, item in original["files"].items():
        require(sha256_path(Path(path)) == item["sha256"], "training/validation asset changed")
        hashes[path] = item["sha256"]
    checksum_path = Path(reg["data_root"]) / "checksums/checksums.sha256"
    require(sha256_path(checksum_path) == reg["checksums_sha256"], "checksum inventory changed")
    hashes[str(checksum_path)] = reg["checksums_sha256"]
    return dict(status="PASS_TEST_ADMISSION", at=utc_now(), formal_steps=80400, checkpoints=checkpoints, protected_hashes=hashes)


def main():
    reg_path = PROJECT_ROOT / "docs/fundus_lwf_v1/registration.json"
    require(sha256_path(reg_path) == REG_SHA, "registration changed")
    reg = read(reg_path)
    root = Path(reg["nas_root"])
    output = root / "test_evaluation"
    require(not output.exists(), "test attempt already exists; no retry")
    require(subprocess.check_output(["findmnt", "-rn", "-T", str(root), "-o", "FSTYPE"], text=True).strip() in {"nfs", "nfs4"}, "NAS unavailable")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in {str(x) for x in reg["allowed_gpus"]} and torch.cuda.is_available(), "one authorized GPU required")
    readout_commit = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
    require(not subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"], text=True).strip(), "dirty readout source")
    admission = audit_training(root, reg)
    output.mkdir(exist_ok=False)  # This consumes the sole test attempt, before test-role access.
    write_json(output / "ADMISSION.json", admission)
    report = dict(status="RUNNING", started_at=utc_now(), training_commit=TRAIN_COMMIT,
                  readout_commit=readout_commit, registration_sha256=REG_SHA, forwards=0, images=0, optimizer_steps=0)
    cells, cases, predictions = [], [], []
    hashes = dict(admission["protected_hashes"])
    try:
        records = {}
        for seed in reg["seeds"]:
            rows = load_training_records(Path(reg["data_root"]), seed=seed, dataset="fundus", roles=("test",))
            for site, count in TEST_COUNTS.items():
                selected = [row for row in rows if row["site_or_vendor"] == site]
                require(len(selected) == count and len({x["patient_id"] for x in selected}) == count
                        and len({x["case_id"] for x in selected}) == count, "test case coverage differs")
                records[seed, site] = selected
                for row in selected:
                    for kind in ("image", "label"):
                        path = Path(reg["data_root"]) / "h5/v1" / row[kind + "_h5_relpath"]
                        digest = row[kind + "_sha256"]
                        require(sha256_path(path) == digest, "test asset hash differs")
                        hashes[str(path)] = digest
        write_json(output / "INPUT_HASHES.json", hashes)
        (output / "predictions").mkdir()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        for checkpoint in admission["checkpoints"]:
            seed, arm, stage = (checkpoint[key] for key in ("seed", "arm", "stage"))
            cp = load_checkpoint(Path(checkpoint["path"]))
            model = UNet2D(3, 3).cuda().eval().requires_grad_(False)
            model.load_state_dict(cp["current_model_state"], strict=True)
            for site in reg["domain_order"][:stage + 1]:
                selected = records[seed, site]
                cursor = 0
                def before_forward(module, inputs):
                    require(report["forwards"] < 612, "test forward budget exceeded")
                    report["forwards"] += 1
                def capture_prediction(module, inputs, result):
                    nonlocal cursor
                    require(torch.isfinite(result.logits).all().item(), "nonfinite test logits")
                    value = result.logits.detach().argmax(1).cpu().numpy().astype(np.uint8)
                    current = selected[cursor:cursor + len(value)]
                    require(len(current) == len(value), "prediction case mapping overflow")
                    filename = f"predictions/{seed}_{arm}_{stage}_{site}_{cursor:03d}.npy"
                    with (output / filename).open("xb") as stream:
                        np.save(stream, value)
                    predictions.append(dict(seed=seed, arm=arm, stage=stage, site=site, path=filename,
                                            case_ids=[x["case_id"] for x in current], sha256=sha256_path(output / filename)))
                    cursor += len(value)
                    report["images"] += len(value)
                hooks = [model.register_forward_pre_hook(before_forward), model.register_forward_hook(capture_prediction)]
                result = evaluate_site(model, data_root=Path(reg["data_root"]), seed=seed, dataset="fundus", site=site,
                                       num_classes=3, role="test", device="cuda", batch_size=4, num_workers=0)
                for hook in hooks:
                    hook.remove()
                require(cursor == len(selected), "incomplete prediction coverage")
                require(len(result.per_case) == len(selected) and result.per_site[0]["patients"] == len(selected), "patient denominator differs")
                require({x["patient_id"] for x in result.per_case} == {x["patient_id"] for x in selected}, "patient coverage differs")
                require(all(math.isfinite(float(row[k])) and 0 <= float(row[k]) <= 1 for row in result.per_case
                            for k in ("dice_class_1", "dice_class_2", "mean_foreground_dice")), "invalid case Dice")
                meta = dict(seed=seed, arm=arm, stage=stage, site=site)
                cells.append({**meta, **{k: result.per_site[0][k] for k in ("patients", "dice_class_1", "dice_class_2", "mean_foreground_dice")}})
                cases.extend({**meta, **row} for row in result.per_case)
                write_csv(output / "cells.csv", cells)
                write_csv(output / "per_case.csv", cases)
                write_json(output / "PREDICTIONS.json", predictions)
            del model, cp
        require(report["forwards"] == 612 and report["images"] == 2430, "test execution coverage differs")
        for path, digest in hashes.items():
            require(sha256_path(Path(path)) == digest, "input or checkpoint changed during readout")
        report.update(adjudicate(cells, reg), input_checkpoint_bytes_unchanged=True, test_role_usage="evaluator_only", hidden_training_labels_used=False)
    except BaseException as exc:
        report.update(status="FAIL_ENGINEERING", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at"] = utc_now()
        write_json(output / "RESULT.json", report)
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
