#!/usr/bin/env python3
"""Audit saved Fundus predictions and arithmetic; never load or run a model."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from di_dmpa_gate1c_v3.durable import now, read, sha256, write_new


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def key(row):
    return int(row["seed"]), row["arm"], int(row["stage"]), row["site"]


def equal(actual, expected):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=0), "arithmetic mismatch")


def main():
    root = Path(sys.argv[1])
    output = root / "test_evaluation"
    reg_path = root / "evaluation_code/SSL_CL_seg/experiments/lcrseg/docs/fundus_lwf_v1/registration.json"
    require(sha256(reg_path) == "70dc562b87c8d49740253d9c91b5497cdce637dcd1320f2edaf5341b27b07c09", "registration changed")
    reg = read(reg_path)
    require(root == Path(reg["nas_root"]), "unexpected study root")
    require(read(root / "operations/test_evaluation/exit.json")["exit_code"] == 0, "readout process not complete")
    result = read(output / "RESULT.json")
    require(result["status"] in {"PASS_BASELINE_FEASIBILITY", "FAIL_BASELINE_FEASIBILITY"}, "invalid scientific completion")
    require(result["forwards"] == 612 and result["images"] == 2430 and result["optimizer_steps"] == 0, "execution budget differs")
    protected = read(output / "INPUT_HASHES.json")
    for path, digest in protected.items():
        require(sha256(path) == digest, "protected bytes changed")
    domains = reg["domain_order"]
    expected = {(seed, arm, stage, site) for seed in reg["seeds"] for arm in reg["arms"]
                for stage in range(3) for site in domains[:stage + 1]}
    records = {}
    labels = {}
    for seed in reg["seeds"]:
        manifest = Path(reg["data_root"]) / f"manifests/training/lcrseg_v1_seed{seed}.csv"
        rows = list(csv.DictReader(manifest.open()))
        for site, count in zip(domains, (100, 40, 25)):
            selected = [row for row in rows if row["dataset"] == "fundus" and row["site_or_vendor"] == site and row["primary_20pct_split"] == "test"]
            require(len(selected) == count and len({x["case_id"] for x in selected}) == count
                    and len({x["patient_id"] for x in selected}) == count, "test role coverage differs")
            records[seed, site] = selected
            for row in selected:
                path = Path(reg["data_root"]) / "h5/v1" / row["label_h5_relpath"]
                require(str(path) in protected and protected[str(path)] == row["label_sha256"], "unaudited test label")
                if str(path) not in labels:
                    with h5py.File(path, "r") as handle:
                        truth = handle["label"][...]
                    require(truth.ndim == 2 and np.isin(truth, [0, 1, 2]).all(), "invalid test label")
                    labels[str(path)] = truth
    predictions = read(output / "PREDICTIONS.json")
    require(len(predictions) == 612, "prediction batch count differs")
    case_rows = list(csv.DictReader((output / "per_case.csv").open()))
    saved_cases = {(key(row), row["patient_id"]): row for row in case_rows}
    require(len(saved_cases) == len(case_rows) == 2430, "case metric coverage differs")
    seen_paths, used_cases = set(), set()
    grouped = {k: [] for k in expected}
    for batch in predictions:
        k = key(batch)
        require(k in expected, "unexpected prediction cell")
        relative = Path(batch["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and relative.parts[0] == "predictions"
                and batch["path"] not in seen_paths and not (output / relative).is_symlink(), "unsafe or duplicate prediction path")
        seen_paths.add(batch["path"])
        require(sha256(output / relative) == batch["sha256"], "prediction bytes changed")
        values = np.load(output / relative, allow_pickle=False)
        require(values.dtype == np.uint8 and values.ndim == 3 and len(values) == len(batch["case_ids"])
                and 1 <= len(values) <= 4 and np.isin(values, [0, 1, 2]).all(), "invalid prediction array")
        selected = records[k[0], k[3]][len(grouped[k]):len(grouped[k]) + len(values)]
        require([x["case_id"] for x in selected] == batch["case_ids"], "prediction order or case mapping differs")
        for pred, row in zip(values, selected):
            ck = (k, row["patient_id"])
            require(ck not in used_cases and ck in saved_cases, "duplicate or missing case")
            used_cases.add(ck)
            truth = labels[str(Path(reg["data_root"]) / "h5/v1" / row["label_h5_relpath"])]
            require(pred.shape == truth.shape, "prediction geometry differs")
            dice = []
            for index in (1, 2):
                p, t = pred == index, truth == index
                denominator = int(p.sum()) + int(t.sum())
                dice.append(1.0 if denominator == 0 else 2 * int(np.logical_and(p, t).sum()) / denominator)
            saved = saved_cases[ck]
            require(saved["case_ids"] == row["case_id"] and int(saved["slices_or_images"]) == 1, "patient grouping differs")
            for metric, value in zip(("dice_class_1", "dice_class_2", "mean_foreground_dice"), [*dice, sum(dice) / 2]):
                equal(saved[metric], value)
            grouped[k].append(dice)
    require(used_cases == set(saved_cases), "unconsumed case evidence")
    require(seen_paths == {p.relative_to(output).as_posix() for p in (output / "predictions").iterdir()}, "untracked prediction files")
    cell_rows = list(csv.DictReader((output / "cells.csv").open()))
    cells = {key(row): row for row in cell_rows}
    require(set(cells) == expected and len(cell_rows) == 36, "test matrix coverage differs")
    scores = {}
    for k, values in grouped.items():
        require(len(values) == len(records[k[0], k[3]]) == int(cells[k]["patients"]), "changed patient denominator")
        means = np.asarray(values, dtype=np.float64).mean(axis=0)
        for metric, value in zip(("dice_class_1", "dice_class_2", "mean_foreground_dice"), [*means, means.mean()]):
            equal(cells[k][metric], value)
        scores[k] = float(means.mean())
    summary_rows = {(row["seed"], row["arm"]): row for row in result["per_seed_arm"]}
    require(len(summary_rows) == len(result["per_seed_arm"]) == 6, "summary coverage differs")
    paired_rows = {row["seed"]: row for row in result["paired"]}
    require(set(paired_rows) == set(reg["seeds"]) and len(result["paired"]) == 3, "paired seed coverage differs")
    deltas = []
    for seed in reg["seeds"]:
        arm_scores = []
        for arm in reg["arms"]:
            final = np.array([scores[seed, arm, 2, site] for site in domains])
            diagonal = np.array([scores[seed, arm, stage, site] for stage, site in enumerate(domains)])
            values = [final.mean(), diagonal.mean(), (final[:2] - diagonal[:2]).mean()]
            for metric, value in zip(("F", "I", "BWT"), values):
                equal(summary_rows[seed, arm][metric], value)
            arm_scores.append(np.array(values))
        delta = arm_scores[1] - arm_scores[0]
        for metric, value in zip(("F", "I", "BWT"), delta):
            equal(paired_rows[seed][metric], value)
        deltas.append(delta)
    values = np.stack(deltas)
    observed = dict(mean_final_dice_improvement=float(values[:, 0].mean()), positive_final_dice_seeds=int((values[:, 0] > 0).sum()),
                    per_seed_final_dice_improvement=float(values[:, 0].min()), mean_bwt_improvement=float(values[:, 2].mean()),
                    mean_incoming_dice_improvement=float(values[:, 1].mean()))
    require(set(result["gates"]) == set(observed), "gate coverage differs")
    failed = []
    for name, value in observed.items():
        minimum = reg["success"][name + "_min"]
        equal(result["gates"][name]["observed"], value)
        equal(result["gates"][name]["minimum"], minimum)
        require(result["gates"][name]["passed"] == bool(value >= minimum), "gate decision differs")
        if value < minimum:
            failed.append(name)
    require(set(result["failed_gates"]) == set(failed), "failed gate list differs")
    require(result["status"] == ("FAIL_BASELINE_FEASIBILITY" if failed else "PASS_BASELINE_FEASIBILITY"), "scientific decision differs")
    for path, digest in protected.items():
        require(sha256(path) == digest, "protected bytes changed during artifact audit")
    receipt = dict(status="PASS_ARTIFACT_AUDIT", at=now(), auditor_sha256=sha256(__file__), model_forwards=0,
                   optimizer_steps=0, batches=612, case_predictions=2430, test_cells=36, protected_paths=len(protected),
                   result_sha256=sha256(output / "RESULT.json"), max_arithmetic_tolerance=1e-12,
                   scientific_result=result["status"], independent_scientific_peer_review=False)
    write_new(root / "operations/artifact_audit/AUDIT.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
