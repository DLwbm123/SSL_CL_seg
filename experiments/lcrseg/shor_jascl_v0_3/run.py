"""One create-only, zero-model-forward SHOR-JASCL V0.3 validation execution."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import sys
import traceback

import h5py
import numpy as np

from di_dmpa_gate1.binding import safe_asset
from di_dmpa_gate1c_v2 import binding as b
from di_dmpa_gate1c_v3 import durable as d
from pres_dsr_sf_v0_2.core import apply_standardizer, probability_fusion, router_probabilities, select_memory
from pres_jascl_v0_1.core import DOMAINS, array_sha256, pixel_confusion, require, segmentation_metrics

from .core import (adjudicate, bootstrap_weights, historical_score, one_hot, reconstruct_oof,
                   select_threshold, shor_routes, top1_lowest)
from .protocol import (PRIVATE_BUNDLE, PRIVATE_CONTENT_SHA, PRIVATE_ROOT, PHASES, compile_call_graph,
                       execution_gate, gate1c_contract, input_audit, isolation_guard, phase_barrier,
                       verify_call_graph, verify_private_bundle)

POLICIES = ("S0_SHARED", "S1_RIDGE_HARD", "S2_RIDGE_SOFT", "S3_SHOR", "S4_ORACLE")


def text_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(value)


def csv_new(path, rows):
    rows = list(rows)
    require(rows, f"empty CSV: {path}", "BLOCKED_INCOMPLETE_EVIDENCE")
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def npz_new(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def new_memmap(path, shape, dtype):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"create-only cache exists: {path}", "BLOCKED_INCOMPLETE_EVIDENCE")
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def link_test_evidence(output, test_report):
    report = d.read(test_report)
    for source, name in ((report["junit_path"], "pytest.xml"),
                         (report["pytest_output_path"], "pytest_output.txt")):
        os.link(source, Path(output) / name)


def load_router_rows(private_root):
    rows = {}
    with (Path(private_root) / "pres_dsr_router_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seed"]), int(row["stage_index"]))
            rows.setdefault(key, []).append(dict(case_id=row["case_id"],
                                                  ridge_alpha=np.asarray(json.loads(row["ridge_alpha"]),
                                                                         dtype=np.float64)))
    for key in rows:
        rows[key].sort(key=lambda row: row["case_id"])
    require(set(rows) == {(seed, stage) for seed in range(3) for stage in (1, 2)}
            and all(len(rows[(seed, 1)]) == 110 and len(rows[(seed, 2)]) == 165 for seed in range(3)),
            "frozen router score coverage changed", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
    return rows


def load_sealed_inputs(private_root):
    root = Path(private_root)
    released = d.read(root / "PRES_DSR_SF_ROUTING_METADATA.json")["seeds"]
    router_rows = load_router_rows(root)
    router_manifest = d.read(root / "PRES_DSR_SF_ROUTER_MANIFEST.json")
    old_models = {(row["seed"], row["stage_index"]): row for row in router_manifest["formal"]}
    require(set(old_models) == set(router_rows), "frozen router manifest changed", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
    descriptors, memories, validation, experts = {}, {}, {}, {}
    for seed in range(3):
        with np.load(root / "descriptor_cache" / f"seed{seed}.npz", allow_pickle=False) as source:
            ids = source["case_ids"].astype(str).tolist()
            raw = np.asarray(source["raw_descriptors"], dtype=np.float64)
        require(raw.shape == (495, 102) and len(ids) == len(set(ids)) == 495,
                "descriptor cache changed", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        descriptors[seed] = dict(case_ids=ids, raw=raw, index={case: i for i, case in enumerate(ids)})
        metadata = released[str(seed)]
        require([row["case_id"] for row in metadata] == ids, "routing metadata/descriptor order changed",
                "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        memories[seed] = {}
        for domain in range(3):
            candidates = [row for row in metadata if row["domain_index"] == domain
                          and row["role"] in ("train_labeled", "train_unlabeled")]
            selected, hashes = select_memory(candidates)
            with np.load(root / "memory_cache" / f"seed{seed}_domain{domain}.npz", allow_pickle=False) as source:
                value = np.asarray(source["descriptors"], dtype=np.float64)
                observed_hashes = source["case_hashes"].astype(str).tolist()
                labels = np.asarray(source["domain_indices"], dtype=np.int64)
            require(observed_hashes == hashes and bool((labels == domain).all()) and len(value) == len(selected),
                    "train memory alignment changed", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
            memories[seed][domain] = dict(case_ids=[row["case_id"] for row in selected], descriptors=value,
                                                  case_hashes=hashes)
        validation[seed], experts[seed] = {}, {}
        for stage in (1, 2):
            stage_rows = router_rows[(seed, stage)]
            val_ids = [row["case_id"] for row in stage_rows]
            raw_val = raw[[descriptors[seed]["index"][case] for case in val_ids]]
            old = old_models[(seed, stage)]
            std = np.asarray(old["std"], dtype=np.float64)
            constant = np.asarray(old["constant"], dtype=bool)
            state = dict(mean=np.asarray(old["mean"], dtype=np.float64), std=std,
                         scale=np.where(constant, 1.0, std), constant=constant,
                         weights=np.asarray(old["weights"], dtype=np.float64),
                         selected_temperature=old["selected_temperature"])
            alpha = router_probabilities(raw_val, state)
            recorded = np.stack([row["ridge_alpha"] for row in stage_rows])
            require(bool(np.allclose(alpha, recorded, atol=1e-12, rtol=1e-12)),
                    "frozen validation ridge alpha changed", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
            validation[seed][stage] = dict(case_ids=val_ids, raw=raw_val, alpha=alpha,
                                           old_model=old, top1=top1_lowest(alpha))
        require(validation[seed][2]["case_ids"] == sorted(validation[seed][2]["case_ids"]),
                "expert validation ordering changed", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        for expert in range(3):
            value = np.load(root / "expert_probability_cache" / f"seed{seed}_expert{expert}.npy",
                            mmap_mode="r", allow_pickle=False)
            require(value.shape == (165, 3, 384, 384) and value.dtype == np.float32,
                    "expert probability cache changed", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
            experts[seed][expert] = value
    return descriptors, memories, validation, experts


def training_unit(memories, seed, stage):
    case_ids = [case for domain in range(stage + 1) for case in memories[seed][domain]["case_ids"]]
    labels = np.concatenate([np.full(len(memories[seed][domain]["case_ids"]), domain, dtype=np.int64)
                             for domain in range(stage + 1)])
    value = np.concatenate([memories[seed][domain]["descriptors"] for domain in range(stage + 1)])
    return case_ids, labels, value


def threshold_record(seed, stage, domain, selected, row_count, oof_sha):
    return dict(seed=seed, stage_index=stage, historical_domain=domain, feasible=selected is not None,
                threshold=None if selected is None else selected["threshold"],
                accepted_count=0.0 if selected is None else selected["accepted_count"],
                precision=0.0 if selected is None else selected["precision"],
                historical_recall=0.0 if selected is None else selected["historical_recall"],
                current_false_override=0.0 if selected is None else selected["current_false_override"],
                candidate_threshold_count=row_count, oof_alpha_sha256=oof_sha)


def build_thresholds(output, memories, validation, counters):
    oof_root = Path(output) / "oof_cache"
    boot_root = Path(output) / "bootstrap_oof_cache"
    oof_root.mkdir()
    boot_root.mkdir()
    formal, boot = {}, {}
    threshold_rows, calibration_rows, formal_entries, boot_entries = [], [], [], []
    for seed in range(3):
        formal[seed], boot[seed] = {}, {}
        for stage in (1, 2):
            case_ids, labels, value = training_unit(memories, seed, stage)
            model, oof = reconstruct_oof(value, labels, case_ids)
            counters["ridge_closed_form_fits"] += 31
            old = validation[seed][stage]["old_model"]
            require(model["selected_lambda"] == old["selected_lambda"]
                    and model["selected_temperature"] == old["selected_temperature"]
                    and np.allclose(model["weights"], np.asarray(old["weights"]), atol=1e-12, rtol=1e-12),
                    "formal ridge formula parity changed", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
            path = oof_root / f"seed{seed}_stage{stage}.npz"
            npz_new(path, case_ids=np.asarray(case_ids), domain_indices=labels, multiplicity=np.ones(len(labels)),
                    folds=model["folds"], alpha=oof)
            thresholds = {}
            for domain in range(stage):
                selected, rows = select_threshold(oof, labels, stage=stage, domain=domain)
                thresholds[domain] = selected
                for row in rows:
                    calibration_rows.append(dict(seed=seed, stage_index=stage, historical_domain=domain, **row))
                record = threshold_record(seed, stage, domain, selected, len(rows), array_sha256(oof))
                threshold_rows.append(record)
                formal_entries.append(record)
            formal[seed][stage] = dict(model=model, oof=oof, thresholds=thresholds,
                                       alpha=validation[seed][stage]["alpha"],
                                       routes=shor_routes(validation[seed][stage]["alpha"], stage=stage,
                                                          thresholds=thresholds))
            boot[seed][stage] = {}
            ids_by_domain = {domain: memories[seed][domain]["case_ids"] for domain in range(stage + 1)}
            for replicate in range(5):
                mult, draw_seeds = bootstrap_weights(ids_by_domain, seed=seed, stage=stage, replicate=replicate)
                boot_model, boot_oof = reconstruct_oof(value, labels, case_ids, multiplicity=mult)
                counters["ridge_closed_form_fits"] += 31
                counters["bootstrap_operations"] += 1
                boot_thresholds = {}
                unit_records = []
                for domain in range(stage):
                    selected, rows = select_threshold(boot_oof, labels, stage=stage, domain=domain, multiplicity=mult)
                    boot_thresholds[domain] = selected
                    record = threshold_record(seed, stage, domain, selected, len(rows), array_sha256(boot_oof))
                    record.update(replicate=replicate, draw_seeds=draw_seeds)
                    boot_entries.append(record)
                    unit_records.append(record)
                alpha = router_probabilities(validation[seed][stage]["raw"], boot_model)
                routes = shor_routes(alpha, stage=stage, thresholds=boot_thresholds)
                npz_new(boot_root / f"seed{seed}_stage{stage}_rep{replicate}.npz",
                        case_ids=np.asarray(case_ids), domain_indices=labels, multiplicity=mult,
                        folds=boot_model["folds"], alpha=boot_oof)
                boot[seed][stage][replicate] = dict(model=boot_model, oof=boot_oof, thresholds=boot_thresholds,
                                                    alpha=alpha, routes=routes, threshold_records=unit_records)
    require(len(threshold_rows) == 9 and len(formal_entries) == 9 and len(boot_entries) == 45,
            "threshold unit key set changed", "BLOCKED_OUTPUT_KEYSET_MISMATCH")
    csv_new(Path(output) / "shor_oof_thresholds.csv", threshold_rows)
    csv_new(Path(output) / "shor_oof_calibration.csv", calibration_rows)
    formal_seal, boot_seal = d.seal(oof_root), d.seal(boot_root)
    manifest = dict(status="PASS_ALL_TRAIN_OOF_AND_THRESHOLDS_SEALED_BEFORE_VALIDATION_GT",
                    formal=formal_entries, bootstrap=boot_entries, formal_units=9, bootstrap_units=45,
                    formal_oof_content_sha256=formal_seal["content_sha256"],
                    bootstrap_oof_content_sha256=boot_seal["content_sha256"],
                    validation_data_used_for_threshold_selection=False, segmentation_GT_fields=0,
                    model_forwards=0, created_at=d.now())
    d.write_new(Path(output) / "SHOR_OOF_THRESHOLD_MANIFEST.json", manifest)
    counters["formal_threshold_units"] = 9
    counters["bootstrap_threshold_units"] = 45
    return formal, boot, threshold_rows


def padded(alpha, stage):
    result = np.zeros((len(alpha), 3), dtype=np.float64)
    result[:, :stage + 1] = alpha
    return result


def hard_predictions(experts, positions, routes):
    result = np.empty((len(routes), 384, 384), dtype=np.uint8)
    for start in range(0, len(routes), 4):
        stop = min(start + 4, len(routes))
        batch = positions[start:stop]
        selected = np.stack([experts[int(routes[i])][batch[i - start]] for i in range(start, stop)])
        result[start:stop] = selected.argmax(axis=1).astype(np.uint8)
    return result


def materialize_candidates(output, validation, experts, formal, boot, counters):
    root = Path(output) / "candidate_cache"
    formal_root, boot_root = root / "formal", root / "bootstrap"
    formal_root.mkdir(parents=True)
    boot_root.mkdir()
    route_rows, formal_predictions, boot_predictions, entries = [], {}, {}, []
    for seed in range(3):
        formal_predictions[seed], boot_predictions[seed] = {}, {}
        global_ids = validation[seed][2]["case_ids"]
        global_index = {case: i for i, case in enumerate(global_ids)}
        for stage in (1, 2):
            ids = validation[seed][stage]["case_ids"]
            positions = np.asarray([global_index[case] for case in ids], dtype=np.int64)
            alpha = formal[seed][stage]["alpha"]
            top = top1_lowest(alpha)
            routes = formal[seed][stage]["routes"]
            policies = {"S0_SHARED": np.full(len(ids), stage, dtype=np.int64),
                        "S1_RIDGE_HARD": top, "S3_SHOR": routes}
            formal_predictions[seed][stage] = {}
            for policy, routed in policies.items():
                path = formal_root / f"seed{seed}_stage{stage}_{policy}.npy"
                target = new_memmap(path, (len(ids), 384, 384), np.uint8)
                target[:] = hard_predictions(experts[seed], positions, routed)
                target.flush(); del target
                formal_predictions[seed][stage][policy] = np.load(path, mmap_mode="r", allow_pickle=False)
                entries.append(dict(kind="formal", seed=seed, stage_index=stage, policy=policy,
                                    cases=len(ids), route_sha256=array_sha256(routed), cache=path.name))
            path = formal_root / f"seed{seed}_stage{stage}_S2_RIDGE_SOFT.npy"
            target = new_memmap(path, (len(ids), 384, 384), np.uint8)
            for start in range(0, len(ids), 4):
                stop = min(start + 4, len(ids)); batch = positions[start:stop]
                values = np.stack([experts[seed][expert][batch] for expert in range(3)], axis=1)
                target[start:stop] = probability_fusion(padded(alpha[start:stop], stage), values).argmax(1).astype(np.uint8)
            target.flush(); del target
            formal_predictions[seed][stage]["S2_RIDGE_SOFT"] = np.load(path, mmap_mode="r", allow_pickle=False)
            entries.append(dict(kind="formal", seed=seed, stage_index=stage, policy="S2_RIDGE_SOFT",
                                cases=len(ids), alpha_sha256=array_sha256(alpha), cache=path.name))
            counters["formal_candidate_case_predictions"] += len(ids) * 4
            scores = {domain: historical_score(alpha, stage, domain) for domain in range(stage)}
            sorted_alpha = np.sort(alpha, axis=1)
            for i, case in enumerate(ids):
                chosen = int(top[i]) if top[i] < stage else None
                threshold = formal[seed][stage]["thresholds"].get(chosen) if chosen is not None else None
                route_rows.append(dict(seed=seed, stage_index=stage, case_id=case, ridge_alpha=alpha[i].tolist(),
                                       ridge_top1=int(top[i]), historical_alpha_mass=float(alpha[i, :stage].sum()),
                                       current_alpha=float(alpha[i, stage]), top1_margin=float(sorted_alpha[i, -1] - sorted_alpha[i, -2]),
                                       SHOR_scores={str(domain): float(scores[domain][i]) for domain in range(stage)},
                                       selected_top1_threshold=None if threshold is None else threshold["threshold"],
                                       S0_route=stage, S1_route=int(top[i]), S2_alpha=padded(alpha[i:i+1], stage)[0].tolist(),
                                       S3_route=int(routes[i]), S4_route_commitment="evaluator_true_domain"))
            boot_predictions[seed][stage] = {}
            for replicate in range(5):
                state = boot[seed][stage][replicate]
                path = boot_root / f"seed{seed}_stage{stage}_rep{replicate}.npy"
                target = new_memmap(path, (len(ids), 384, 384), np.uint8)
                target[:] = hard_predictions(experts[seed], positions, state["routes"])
                target.flush(); del target
                boot_predictions[seed][stage][replicate] = np.load(path, mmap_mode="r", allow_pickle=False)
                entries.append(dict(kind="bootstrap", seed=seed, stage_index=stage, replicate=replicate,
                                    cases=len(ids), route_sha256=array_sha256(state["routes"]), cache=path.name))
                counters["bootstrap_candidate_case_predictions"] += len(ids)
    require(len(route_rows) == 825 and len(entries) == 54, "candidate key set changed",
            "BLOCKED_OUTPUT_KEYSET_MISMATCH")
    csv_new(Path(output) / "shor_routes.csv", route_rows)
    seal = d.seal(root)
    manifest = dict(status="PASS_ALL_SHOR_CANDIDATES_SEALED_BEFORE_VALIDATION_GT", entries=entries,
                    formal_policies_materialized=["S0_SHARED", "S1_RIDGE_HARD", "S2_RIDGE_SOFT", "S3_SHOR"],
                    S4_ORACLE_commitment="fixed evaluator true-domain expert over already-sealed expert probabilities",
                    no_probability_or_logit_mixing_in_S3=True, validation_domain_reads=0,
                    validation_segmentation_GT_reads=0, model_forwards=0, route_rows=len(route_rows),
                    content_sha256=seal["content_sha256"], manifest_sha256=d.sha256(root / "PRIVATE_BUNDLE_MANIFEST.json"))
    d.write_new(Path(output) / "SHOR_CANDIDATE_MANIFEST.json", manifest)
    counters["formal_route_rows"] = len(route_rows)
    return route_rows, formal_predictions, boot_predictions


def read_validation_records(contract):
    data_root = Path(contract["destination"]["data_root"])
    records = {}
    for seed in range(3):
        records[seed] = {}
        for domain in range(3):
            records[seed][domain] = b.records(data_root, contract, seed, domain, "val")
    return data_root, records


def read_labels(rows, data_root):
    labels = []
    for row in rows:
        require(row.get("label_h5_relpath") and row.get("label_sha256"), "validation label unavailable")
        path = safe_asset(data_root, row["label_h5_relpath"])
        b.check_hash(path, row["label_sha256"])
        with h5py.File(path, "r") as handle:
            value = handle["label"][...]
        require(value.shape == (384, 384), "label geometry changed", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        labels.append(np.asarray(value, dtype=np.int64))
    return np.stack(labels)


def metric(prediction, target):
    return segmentation_metrics(pixel_confusion(prediction, target))


def evaluate(output, contract, validation, experts, formal, boot, formal_predictions, boot_predictions, counters):
    require((Path(output) / "PHASE_candidate_prediction_seal_MANIFEST.json").is_file(),
            "candidate seal missing before validation GT", "BLOCKED_PROTOCOL_OR_LEAKAGE")
    data_root, records = read_validation_records(contract)
    segmentation_rows, attribution_rows, utility_rows, bootstrap_rows = [], [], [], []
    for seed in range(3):
        all_rows = sorted((row for domain in range(3) for row in records[seed][domain]), key=lambda row: row["case_id"])
        require([row["case_id"] for row in all_rows] == validation[seed][2]["case_ids"],
                "sealed/evaluator case order changed", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        labels = read_labels(all_rows, data_root)
        counters["validation_GT_case_reads"] += len(labels)
        global_index = {row["case_id"]: i for i, row in enumerate(all_rows)}
        for stage in (1, 2):
            ids = validation[seed][stage]["case_ids"]
            stage_rows = [all_rows[global_index[case]] for case in ids]
            truth = np.asarray([DOMAINS.index(row["site_or_vendor"]) for row in stage_rows], dtype=np.int64)
            stage_labels = labels[[global_index[case] for case in ids]]
            predictions = dict(formal_predictions[seed][stage])
            positions = np.asarray([global_index[case] for case in ids], dtype=np.int64)
            predictions["S4_ORACLE"] = hard_predictions(experts[seed], positions, truth)
            per_case = {policy: [metric(predictions[policy][i], stage_labels[i]) for i in range(len(ids))]
                        for policy in POLICIES}
            for policy in POLICIES:
                for domain in range(stage + 1):
                    mask = truth == domain
                    segmentation_rows.append(dict(seed=seed, stage_index=stage, true_domain=domain,
                                                  domain=DOMAINS[domain], policy=policy,
                                                  **metric(predictions[policy][mask], stage_labels[mask])))
            alpha = formal[seed][stage]["alpha"]
            top = top1_lowest(alpha)
            routes = formal[seed][stage]["routes"]
            for i, case in enumerate(ids):
                dice = {policy: per_case[policy][i]["mean_foreground_dice"] for policy in POLICIES}
                attribution_rows.append(dict(seed=seed, stage_index=stage, case_id=case, true_domain=int(truth[i]),
                                             ridge_top1=int(top[i]), top1_correct=bool(top[i] == truth[i]),
                                             SHOR_route=int(routes[i]), SHOR_route_correct=bool(routes[i] == truth[i]),
                                             historical_alpha_mass=float(alpha[i, :stage].sum()),
                                             soft_vs_current_regret=float(dice["S0_SHARED"] - dice["S2_RIDGE_SOFT"]),
                                             soft_vs_hard_regret=float(dice["S1_RIDGE_HARD"] - dice["S2_RIDGE_SOFT"]),
                                             current_domain_regret=None if truth[i] != stage else float(dice["S0_SHARED"] - dice["S3_SHOR"]),
                                             correct_route_current_case=bool(truth[i] == stage and routes[i] == stage),
                                             misrouted_current_case=bool(truth[i] == stage and routes[i] != stage),
                                             policy_foreground_dice=dice))
            for domain in range(stage + 1):
                mask = truth == domain
                lookup = {(row["policy"], row["true_domain"]): row for row in segmentation_rows
                          if row["seed"] == seed and row["stage_index"] == stage}
                utility_rows.append(dict(seed=seed, stage_index=stage, true_domain=domain, domain=DOMAINS[domain],
                                         cases=int(mask.sum()), overrides=int(np.sum(mask & (routes < stage))),
                                         correct_overrides=int(np.sum(mask & (routes == truth) & (routes < stage))),
                                         gain_over_S0=float(lookup[("S3_SHOR", domain)]["mean_foreground_dice"]
                                                            - lookup[("S0_SHARED", domain)]["mean_foreground_dice"]),
                                         gain_over_S2=float(lookup[("S3_SHOR", domain)]["mean_foreground_dice"]
                                                            - lookup[("S2_RIDGE_SOFT", domain)]["mean_foreground_dice"])))
            s0 = {domain: next(row for row in segmentation_rows if row["seed"] == seed and row["stage_index"] == stage
                               and row["true_domain"] == domain and row["policy"] == "S0_SHARED")
                  for domain in range(stage + 1)}
            s4 = {domain: next(row for row in segmentation_rows if row["seed"] == seed and row["stage_index"] == stage
                               and row["true_domain"] == domain and row["policy"] == "S4_ORACLE")
                  for domain in range(stage + 1)}
            for replicate in range(5):
                prediction = boot_predictions[seed][stage][replicate]
                for domain in range(stage + 1):
                    mask = truth == domain
                    value = metric(prediction[mask], stage_labels[mask])
                    bootstrap_rows.append(dict(seed=seed, stage_index=stage, true_domain=domain,
                                               domain=DOMAINS[domain], replicate=replicate,
                                               feasible_thresholds=all(row["feasible"] for row in
                                                                       boot[seed][stage][replicate]["threshold_records"]),
                                               mean_foreground_dice=value["mean_foreground_dice"],
                                               gain_over_S0=float(value["mean_foreground_dice"]
                                                                  - s0[domain]["mean_foreground_dice"]),
                                               oracle_gap=float(s4[domain]["mean_foreground_dice"]
                                                                - value["mean_foreground_dice"])))
        del labels
    require(len(segmentation_rows) == 75 and len(attribution_rows) == 825 and len(utility_rows) == 15
            and len(bootstrap_rows) == 75 and counters["validation_GT_case_reads"] == 495,
            "evaluator output key set changed", "BLOCKED_OUTPUT_KEYSET_MISMATCH")
    csv_new(Path(output) / "shor_failure_attribution.csv", attribution_rows)
    csv_new(Path(output) / "shor_segmentation.csv", segmentation_rows)
    csv_new(Path(output) / "shor_override_utility.csv", utility_rows)
    csv_new(Path(output) / "shor_bootstrap.csv", bootstrap_rows)
    counters.update(segmentation_rows=len(segmentation_rows), failure_attribution_rows=len(attribution_rows),
                    override_utility_rows=len(utility_rows), bootstrap_metric_rows=len(bootstrap_rows))
    return segmentation_rows, attribution_rows, utility_rows, bootstrap_rows


def aggregate_evidence(segmentation_rows, bootstrap_rows, threshold_rows, boot):
    lookup = {(row["seed"], row["true_domain"], row["policy"]): row for row in segmentation_rows
              if row["stage_index"] == 2}
    def values(policy):
        return {(seed, domain): lookup[(seed, domain, policy)]["mean_foreground_dice"]
                for seed in range(3) for domain in range(3)}
    s0, s2, s3, s4 = (values(policy) for policy in ("S0_SHARED", "S2_RIDGE_SOFT", "S3_SHOR", "S4_ORACLE"))
    gains = {key: s3[key] - s0[key] for key in s0}
    soft_gains = {key: s2[key] - s0[key] for key in s0}
    current_class_drop = []
    for class_index in (1, 2):
        current_class_drop.append(float(np.mean([
            lookup[(seed, 2, "S0_SHARED")]["per_class_dice"][class_index]
            - lookup[(seed, 2, "S3_SHOR")]["per_class_dice"][class_index] for seed in range(3)])))
    safety = dict(current_domain_drop=float(max(0.0, np.mean([s0[(seed, 2)] - s3[(seed, 2)] for seed in range(3)]))),
                  maximum_current_class_drop=float(max(0.0, max(current_class_drop))),
                  maximum_seed_domain_drop=float(max(0.0, max(-value for value in gains.values()))))
    seed_gain = [float(np.mean([gains[(seed, domain)] for domain in range(3)])) for seed in range(3)]
    value = dict(three_domain_gain=float(np.mean(list(gains.values()))),
                 historical_gain=float(np.mean([gains[(seed, domain)] for seed in range(3) for domain in (0, 1)])),
                 oracle_gap=float(np.mean([s4[key] - s3[key] for key in s0])),
                 positive_seed_count=sum(item > 0 for item in seed_gain),
                 REFUGE_mean_gain=float(np.mean([gains[(seed, 0)] for seed in range(3)])),
                 RIM_ONE_r3_mean_gain=float(np.mean([gains[(seed, 1)] for seed in range(3)])))
    soft_current = float(max(0.0, max(s0[(seed, 2)] - s2[(seed, 2)] for seed in range(3))))
    shor_current = float(max(0.0, max(s0[(seed, 2)] - s3[(seed, 2)] for seed in range(3))))
    soft_max = float(max(0.0, max(-item for item in soft_gains.values())))
    shor_max = float(max(0.0, max(-item for item in gains.values())))
    soft_shared, shor_shared = float(np.mean(list(soft_gains.values()))), value["three_domain_gain"]
    soft_hist = float(np.mean([soft_gains[(seed, domain)] for seed in range(3) for domain in (0, 1)]))
    repair = dict(current_domain_drop_reduction=soft_current - shor_current,
                  maximum_seed_domain_drop_reduction=soft_max - shor_max,
                  shared_gain_loss=soft_shared - shor_shared,
                  historical_gain_loss=soft_hist - value["historical_gain"],
                  S2_current_domain_drop=soft_current, S3_current_domain_drop=shor_current,
                  S2_maximum_seed_domain_drop=soft_max, S3_maximum_seed_domain_drop=shor_max)
    by_rep = {}
    for replicate in range(5):
        rows = [row for row in bootstrap_rows if row["stage_index"] == 2 and row["replicate"] == replicate]
        gain = [row["gain_over_S0"] for row in rows]
        by_rep[replicate] = dict(shared_gain=float(np.mean(gain)),
                                 historical_gain=float(np.mean([row["gain_over_S0"] for row in rows
                                                                if row["true_domain"] in (0, 1)])),
                                 current_domain_drop=float(max(0.0, max(-row["gain_over_S0"] for row in rows
                                                                         if row["true_domain"] == 2))),
                                 maximum_seed_domain_drop=float(max(0.0, max(-row["gain_over_S0"] for row in rows))))
    unit_counts = {}
    for seed in range(3):
        for stage in (1, 2):
            for domain in range(stage):
                unit_counts[(seed, stage, domain)] = sum(
                    boot[seed][stage][rep]["thresholds"][domain] is not None for rep in range(5))
    stability = dict(shared_gain_p10=float(np.quantile([row["shared_gain"] for row in by_rep.values()], .1, method="linear")),
                     historical_gain_p10=float(np.quantile([row["historical_gain"] for row in by_rep.values()], .1, method="linear")),
                     current_domain_drop_p90=float(np.quantile([row["current_domain_drop"] for row in by_rep.values()], .9, method="linear")),
                     maximum_seed_domain_drop_p90=float(np.quantile([row["maximum_seed_domain_drop"] for row in by_rep.values()], .9, method="linear")),
                     every_unit_feasible_in_at_least_4_of_5=all(value >= 4 for value in unit_counts.values()),
                     feasible_replicates={f"seed{key[0]}_stage{key[1]}_domain{key[2]}": value
                                          for key, value in unit_counts.items()}, all_finite=True,
                     aggregate_replicates=by_rep)
    calibration_ = dict(all_units_feasible=all(row["feasible"] for row in threshold_rows),
                        feasible_units=sum(row["feasible"] for row in threshold_rows), total_units=9,
                        all_finite=all(np.isfinite(row[key]) for row in threshold_rows
                                      for key in ("accepted_count", "precision", "historical_recall",
                                                  "current_false_override")))
    return dict(calibration=calibration_, current_safety=safety, value=value, repair=repair, stability=stability)


def report(output, metadata, counters, decision, evidence):
    status = dict(metadata=metadata, **decision, evidence=evidence, counters=counters,
                  new_model_forwards=0, model_constructions=0, checkpoint_tensor_reads=0,
                  model_autograd_calls=0, model_backward_calls=0, model_optimizer_steps=0,
                  router_optimizer_steps=0, parameter_grad_writes=0, training_launched=False,
                  method_registered=False, threshold_builder_segmentation_GT_usage="none",
                  validation_GT_usage="evaluator_only_after_candidate_seal", test_GT_reads=0,
                  controls_cannot_rescue_S3=True, report_commit=None,
                  report_commit_resolution="first Git commit adding these exact public report bytes")
    d.write_new(Path(output) / "SHOR_STATUS.json", status)
    lines = ["# SHOR-JASCL V0.3 final report", "", f"Scientific status: `{decision['scientific_status']}`.", "",
             "## H1-H6", "", "; ".join(f"H{i}={decision[f'H{i}']}" for i in range(1, 7)) + ".", "",
             f"Calibration: {evidence['calibration']}.", "", f"Current safety: {evidence['current_safety']}.", "",
             f"Value: {evidence['value']}.", "", f"Repair of soft failure: {evidence['repair']}.", "",
             f"Bootstrap stability: {evidence['stability']}.", "", "## Isolation", "",
             "The run constructed no model and performed zero model forwards, checkpoint tensor loads, autograd, backward, optimizer or router-optimizer steps, parameter-gradient writes, training, and test-GT reads. Thresholds used train-only memory OOF probabilities. Validation domain and segmentation GT entered only after candidate sealing.", "",
             "The run hard-stops for independent review. No test evaluation, second attempt, threshold modification, refit, training, other benchmark, sweep, or main merge is authorized.", ""]
    text_new(Path(output) / "SHOR_FINAL_REPORT.md", "\n".join(lines))
    failed = [f"H{i}" for i in range(1, 7) if not decision[f"H{i}"]]
    warnings = ["# SHOR-JASCL V0.3 failures and warnings", "", f"Final status: `{decision['scientific_status']}`.", "",
                f"Failed gates: {', '.join(failed) if failed else 'none'}.", "",
                "S0, S1, S2, and S4 are controls and cannot rescue primary S3.", "",
                "Private paths, case IDs, alphas, predictions, labels, and raw CSV rows are omitted from public reporting.", ""]
    text_new(Path(output) / "SHOR_FAILURES_AND_WARNINGS.md", "\n".join(warnings))
    text_new(Path(output) / "SHOR_EXACT_COMMANDS.md", "\n".join([
        "# SHOR-JASCL V0.3 exact commands", "", "## Tests", "", "```sh", metadata["exact_test_command"], "```", "",
        "## Durable validation child", "", "```sh", shlex.join(metadata["exact_command"]), "```", "",
        "Both commands ran through the NAS wrapper. No model, forward, optimizer, backward, or training command ran.", ""]))


def artifact_manifest(output):
    excluded = {"SHOR_ARTIFACT_MANIFEST.json", "controller.log", "supervisor.log", "LAUNCH_REQUEST.json",
                "LAUNCH_RECEIPT.json", "PROCESS_START.json", "PROCESS_PID.json", "PROCESS_EXIT.json",
                "EXECUTION_COMPLETION.json", "PHASE_shor_jascl_v0_3.json", "PHASE_shor_jascl_v0_3_MANIFEST.json"}
    entries = d.file_entries(output, exclude=tuple(excluded))
    required = {"SHOR_INPUT_AUDIT.json", "SHOR_CALL_GRAPH.json", "SHOR_OOF_THRESHOLD_MANIFEST.json",
                "shor_oof_thresholds.csv", "shor_oof_calibration.csv", "SHOR_CANDIDATE_MANIFEST.json",
                "shor_routes.csv", "shor_failure_attribution.csv", "shor_segmentation.csv",
                "shor_override_utility.csv", "shor_bootstrap.csv", "SHOR_H1_H6.json", "SHOR_STATUS.json",
                "SHOR_FINAL_REPORT.md", "SHOR_FAILURES_AND_WARNINGS.md", "SHOR_EXACT_COMMANDS.md",
                "pytest.xml", "pytest_output.txt"}
    names = {entry["path"] for entry in entries}
    require(required.issubset(names), "required SHOR artifact missing", "BLOCKED_INCOMPLETE_EVIDENCE")
    result = dict(status="PASS_SHOR_ARTIFACT_MANIFEST", entries=entries, files=len(entries),
                  bytes=sum(row["bytes"] for row in entries), required_outputs_complete=True,
                  excludes_live_supervisor_receipts=True, created_at=d.now())
    d.write_new(Path(output) / "SHOR_ARTIFACT_MANIFEST.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--test-report", required=True, type=Path)
    parser.add_argument("--private-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        require(args.output.is_dir(), "durable create-only output was not initialized")
        metadata = execution_gate(args.output, args.code_commit, args.test_report, args.private_root)
        metadata["exact_command"] = sys.argv
        d.write_new(args.output / "SHOR_RUN_METADATA.json", metadata)
        link_test_evidence(args.output, args.test_report)
        counters = dict(new_model_forwards=0, model_constructions=0, checkpoint_tensor_reads=0,
                        formal_threshold_units=0, bootstrap_threshold_units=0, ridge_closed_form_fits=0,
                        bootstrap_operations=0, formal_route_rows=0, formal_candidate_case_predictions=0,
                        bootstrap_candidate_case_predictions=0, validation_GT_case_reads=0,
                        segmentation_rows=0, failure_attribution_rows=0, override_utility_rows=0,
                        bootstrap_metric_rows=0)
        with isolation_guard():
            audit = input_audit(args.output, metadata, args.private_root)
            descriptors, memories, validation, experts = load_sealed_inputs(args.private_root)
            graph = compile_call_graph(args.output, {stage: len(validation[0][stage]["case_ids"]) for stage in (1, 2)},
                                       args.code_commit)
            phase_barrier(args.output, "input_audit", ("SHOR_RUN_METADATA.json", "SHOR_INPUT_AUDIT.json",
                                                        "SHOR_CALL_GRAPH.json", "pytest.xml", "pytest_output.txt"))
            formal, boot, threshold_rows = build_thresholds(args.output, memories, validation, counters)
            phase_barrier(args.output, "oof_threshold_seal", ("SHOR_OOF_THRESHOLD_MANIFEST.json",
                                                               "shor_oof_thresholds.csv", "shor_oof_calibration.csv"))
            route_rows, formal_predictions, boot_predictions = materialize_candidates(
                args.output, validation, experts, formal, boot, counters)
            phase_barrier(args.output, "candidate_prediction_seal", ("SHOR_CANDIDATE_MANIFEST.json", "shor_routes.csv"))
            contract = gate1c_contract()
            segmentation_rows, attribution_rows, utility_rows, bootstrap_rows = evaluate(
                args.output, contract, validation, experts, formal, boot, formal_predictions, boot_predictions, counters)
            phase_barrier(args.output, "validation_evaluation", ("shor_failure_attribution.csv",
                                                                  "shor_segmentation.csv", "shor_override_utility.csv"))
            phase_barrier(args.output, "bootstrap_evaluation", ("shor_bootstrap.csv",))
            evidence = aggregate_evidence(segmentation_rows, bootstrap_rows, threshold_rows, boot)
            current_private = d.read(args.private_root / PRIVATE_BUNDLE.name)
            evidence["isolation"] = bool(current_private["content_sha256"] == PRIVATE_CONTENT_SHA
                                         and counters["new_model_forwards"] == counters["model_constructions"]
                                         == counters["checkpoint_tensor_reads"] == 0
                                         and counters["validation_GT_case_reads"] == 495)
            decision = adjudicate(evidence)
            verify_call_graph(graph, counters)
            d.write_new(args.output / "SHOR_H1_H6.json",
                        dict(status="PASS_H1_H6_COMPILED", decision=decision, evidence=evidence,
                             controls_cannot_rescue_S3=True, compiled_at=d.now()))
            phase_barrier(args.output, "H1_H6_compile", ("SHOR_H1_H6.json",))
            report(args.output, metadata, counters, decision, evidence)
            manifest = artifact_manifest(args.output)
            phase_barrier(args.output, "artifact_audit", ("SHOR_ARTIFACT_MANIFEST.json",))
            phase_barrier(args.output, "NAS_archive", ("SHOR_ARTIFACT_MANIFEST.json", "SHOR_STATUS.json"))
            phase_barrier(args.output, "report", ("SHOR_STATUS.json", "SHOR_FINAL_REPORT.md",
                                                   "SHOR_FAILURES_AND_WARNINGS.md", "SHOR_EXACT_COMMANDS.md"))
        print(json.dumps(dict(status=decision["scientific_status"], artifacts=manifest["files"], counters=counters),
                         sort_keys=True), flush=True)
    except BaseException as error:
        status = getattr(error, "status", "BLOCKED_INCOMPLETE_EVIDENCE")
        allowed = {"BLOCKED_BASE_COMMIT_AMBIGUOUS", "BLOCKED_PRIVATE_BUNDLE_MISMATCH",
                   "BLOCKED_PROTOCOL_OR_LEAKAGE", "BLOCKED_OUTPUT_KEYSET_MISMATCH",
                   "BLOCKED_NUMERICAL_FAILURE", "BLOCKED_INCOMPLETE_EVIDENCE"}
        if status not in allowed:
            status = "BLOCKED_PROTOCOL_OR_LEAKAGE"
        failure = args.output / f"SHOR_FAILURE_{os.getpid()}.json"
        if args.output.is_dir() and not failure.exists():
            d.write_new(failure, dict(status=status, error=f"{type(error).__name__}: {error}",
                                      traceback=traceback.format_exc(), command=sys.argv, recorded_at=d.now(),
                                      new_attempt_authorized=False, new_model_forwards=0, model_optimizer_steps=0,
                                      router_optimizer_steps=0, training_launched=False))
        raise


if __name__ == "__main__":
    main()
