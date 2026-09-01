"""One create-only PRES validation execution; no optimizer, autograd or training."""
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
import torch

from di_dmpa_gate1.binding import safe_asset
from di_dmpa_gate1c_v2 import binding as b
from di_dmpa_gate1c_v3 import durable as d
from di_dmpa_gate1c_v3.inputs import load_models as _load_models
from di_dmpa_gate1_v2.features import ImmutableModels

from .core import (Blocked, DOMAINS, ORACLE_EXPERT, adjudicate, array_sha256, bootstrap_draw,
                   fit_prototypes, multiplicity, pixel_confusion, prototype_stability, require,
                   route, routing_rows, routing_summary, segmentation_metrics, style_descriptors)
from .protocol import (BATCH_SIZE, ROOT, compile_call_graph, execution_gate, gate1c_contract,
                       input_audit, isolation_guard, verify_call_graph)


def deterministic_backend_state():
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                cudnn_benchmark_disabled=not torch.backends.cudnn.benchmark,
                matmul_tf32_disabled=not torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32_disabled=not torch.backends.cudnn.allow_tf32,
                autocast_disabled=not torch.is_autocast_enabled())


def enforce_deterministic_backend():
    """Restore the registered backend after the pinned JASCL import mutates cuDNN globals."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return deterministic_backend_state()


def load_readonly_models(*args, **kwargs):
    result = _load_models(*args, **kwargs)
    enforce_deterministic_backend()
    return result


def text_new(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def csv_new(path, rows):
    rows = list(rows)
    require(rows, f"empty CSV: {path}", "BLOCKED_INCOMPLETE_EVIDENCE")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def checkpoint(p, seed, stage):
    value = next(value for value in p["immutable_baseline"]["checkpoint_inputs"]
                 if value["checkpoint_id"] == f"B0/seed{seed}/stage{stage}")
    keys = ("baseline", "checkpoint_id", "path", "sha256", "stage_index", "seed", "domain", "legacy_pas_capture")
    return {key: value[key] for key in keys}


def image_only(row):
    return {key: row[key] for key in ("case_id", "image_h5_relpath", "image_sha256")}


def read_images(rows, data_root):
    images = []
    for row in rows:
        path = safe_asset(data_root, row["image_h5_relpath"])
        with h5py.File(path, "r") as handle:
            value = handle["image"][...]
        require(value.shape == (3, 384, 384), "image geometry changed", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        images.append(np.asarray(value, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(images))


def read_labels(rows, data_root):
    labels = []
    for row in rows:
        require(row.get("label_h5_relpath") and row.get("label_sha256"), "validation label unavailable")
        path = safe_asset(data_root, row["label_h5_relpath"])
        with h5py.File(path, "r") as handle:
            value = handle["label"][...]
        require(value.shape == (384, 384), "label geometry changed", "BLOCKED_PRIVATE_BUNDLE_MISMATCH")
        labels.append(np.asarray(value, dtype=np.int64))
    return np.stack(labels)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return dict(path=str(path), sha256=d.sha256(path), bytes=path.stat().st_size,
                arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=array_sha256(value))
                        for key, value in arrays.items()})


def descriptor_image_plan(records, seed):
    rows = []
    for stage in range(3):
        for role in ("train_unlabeled", "val"):
            rows.extend(image_only(row) for row in records[seed][stage][role])
    rows.sort(key=lambda row: row["case_id"])
    require(len(rows) == len({row["case_id"] for row in rows}) == 429, "descriptor case identity mismatch",
            "BLOCKED_INCOMPLETE_EVIDENCE")
    return rows


def extract_descriptors(output, image_plans, data_root, router_checkpoints, metadata, device, counters):
    cache_root = Path(output) / "descriptor_cache"
    cache_root.mkdir()
    values, entries = {}, []
    for seed in range(3):
        rows = image_plans[seed]
        cp = router_checkpoints[seed]
        models, payload = load_readonly_models(ROOT, cp, device=device, sources=("ema_teacher",))
        model = models["ema_teacher"]
        descriptors, validity = [], []
        with ImmutableModels(models, cp, Path(output)/"router_models"/f"seed{seed}", metadata):
            with torch.no_grad():
                for start in range(0, len(rows), BATCH_SIZE):
                    images = read_images(rows[start:start+BATCH_SIZE], data_root).to(device)
                    require(images.dtype == torch.float32 and not model.training, "router forward mode/dtype changed")
                    with torch.autocast(device_type=device.type, enabled=False):
                        enc1 = model.enc1(images)
                        enc2 = model.enc2(model.pool(enc1))
                    desc, valid = style_descriptors(images, enc1, enc2)
                    descriptors.append(desc)
                    validity.append(valid)
                    counters["router_extraction_forwards"] += 1
                    counters["router_extraction_case_passes"] += len(images)
        counters["model_guards"] += 1
        require(all(parameter.grad is None for parameter in model.parameters()), "router wrote parameter.grad",
                "BLOCKED_MODEL_MUTATION")
        value = np.concatenate(descriptors)
        valid = np.concatenate(validity)
        require(value.shape == (429, 102) and valid.shape == (429, 3), "descriptor schema mismatch")
        values[seed] = value
        entries.append(dict(seed=seed, checkpoint_id=cp["checkpoint_id"], checkpoint_sha256=cp["sha256"],
                            feature_source="ema_teacher", case_count=len(rows), descriptor_dim=102,
                            cache=save_npz(cache_root/f"seed{seed}.npz",
                                           case_ids=np.asarray([row["case_id"] for row in rows]),
                                           descriptors=value, block_valid=valid)))
        del models, payload, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    seal = d.seal(cache_root)
    manifest = dict(status="PASS_DESCRIPTORS_SEALED_BEFORE_DOMAIN_METADATA", entries=entries,
                    total_cases=1287, one_descriptor_per_case=True, descriptor_dtype="float64",
                    descriptor_bundle_content_sha256=seal["content_sha256"],
                    descriptor_bundle_manifest_sha256=d.sha256(cache_root/"PRIVATE_BUNDLE_MANIFEST.json"),
                    router_API_received_GT=False, router_API_received_domain_metadata=False,
                    router_API_received_filename_or_path_features=False, model_optimizer_steps=0, router_optimizer_steps=0)
    d.write_new(Path(output)/"PRES_JASCL_ROUTER_DESCRIPTOR_MANIFEST.json", manifest)
    return values


def release_routing_metadata(output, records, image_plans):
    plans = {}
    for seed in range(3):
        plans[seed] = []
        for stage in range(3):
            for role in ("train_unlabeled", "val"):
                plans[seed].extend(dict(case_id=row["case_id"], role=role, domain_index=stage,
                                        image_h5_relpath=row["image_h5_relpath"], image_sha256=row["image_sha256"])
                                   for row in records[seed][stage][role])
        plans[seed].sort(key=lambda row: row["case_id"])
        require([row["case_id"] for row in plans[seed]] == [row["case_id"] for row in image_plans[seed]],
                "routing metadata does not bind sealed descriptors", "BLOCKED_INCOMPLETE_EVIDENCE")
    # Domain and role metadata become available only after the descriptor bundle is closed.
    d.write_new(Path(output)/"PRES_JASCL_ROUTING_METADATA.json",
                dict(status="PASS_METADATA_RELEASED_AFTER_DESCRIPTOR_SEAL", seeds={str(seed): plans[seed] for seed in range(3)},
                     segmentation_GT_fields=0, test_records=0))
    return plans


def build_prototypes(output, descriptors, plans):
    banks, rows = {}, []
    for seed in range(3):
        banks[seed] = {}
        index = {row["case_id"]: i for i, row in enumerate(plans[seed])}
        for M in (1, 2):
            banks[seed][M] = {}
            for domain in range(3):
                ids = sorted(row["case_id"] for row in plans[seed]
                             if row["role"] == "train_unlabeled" and row["domain_index"] == domain)
                x = descriptors[seed][[index[case] for case in ids]]
                fitted = fit_prototypes(x, M, seed=seed, domain_index=domain)
                banks[seed][M][domain] = fitted
                rows.append(dict(seed=seed, M=M, domain_index=domain, domain=DOMAINS[domain], case_count=len(ids),
                                 centers=fitted["centers"].tolist(), active=fitted["active"].tolist(),
                                 occupancy=fitted["occupancy"].tolist(), centers_sha256=fitted["centers_sha256"],
                                 active_sha256=fitted["active_sha256"], selected_restart=fitted["selected_restart"],
                                 within_domain_prototype_cosine=fitted["within_domain_prototype_cosine"],
                                 within_domain_prototype_cosine_distance=fitted["within_domain_prototype_cosine_distance"],
                                 restarts=fitted["restarts"]))
            stage1 = {domain: (banks[seed][M][domain]["centers_sha256"], banks[seed][M][domain]["active_sha256"])
                      for domain in (0, 1)}
            stage2 = {domain: (banks[seed][M][domain]["centers_sha256"], banks[seed][M][domain]["active_sha256"])
                      for domain in (0, 1)}
            require(stage1 == stage2, "old prototype bytes changed", "BLOCKED_MODEL_MUTATION")
    manifest = dict(status="PASS_PROTOTYPE_BANK", prototypes=rows, candidate_M=[1, 2],
                    old_domain_prototypes_byte_unchanged=True, no_domain_mixing=True,
                    train_unlabeled_only=True, router_trainable_parameters=0)
    d.write_new(Path(output)/"PRES_JASCL_DOMAIN_PROTOTYPE_MANIFEST.json", manifest)
    return banks


def evaluate_routing(output, descriptors, plans, banks, counters):
    score_rows, confusion_rows, summaries, routes = [], [], {}, {}
    for seed in range(3):
        summaries[seed], routes[seed] = {}, {}
        index = {row["case_id"]: i for i, row in enumerate(plans[seed])}
        for M in (1, 2):
            summaries[seed][M], routes[seed][M] = {}, {}
            for stage in (1, 2):
                selected = [row for row in plans[seed] if row["role"] == "val" and row["domain_index"] <= stage]
                selected.sort(key=lambda row: row["case_id"])
                x = descriptors[seed][[index[row["case_id"]] for row in selected]]
                truth = [row["domain_index"] for row in selected]
                routed, scores, entropy = route(x, banks[seed][M], tuple(range(stage + 1)))
                rows = routing_rows([row["case_id"] for row in selected], truth, routed, scores, entropy,
                                    seed=seed, stage=stage, M=M)
                summary = routing_summary(rows, stage + 1)
                summaries[seed][M][stage] = summary
                score_rows.extend(rows)
                for true in range(stage + 1):
                    for predicted in range(stage + 1):
                        confusion_rows.append(dict(seed=seed, stage_index=stage, M=M, true_domain=true,
                                                   routed_domain=predicted, count=int(summary["confusion_matrix"][true, predicted])))
                if stage == 2:
                    routes[seed][M] = {row["case_id"]: row["routed_domain"] for row in rows}
    require(len(score_rows) == 1830 and len(confusion_rows) == 78, "routing row coverage mismatch",
            "BLOCKED_CALL_GRAPH_MISMATCH")
    csv_new(Path(output)/"pres_router_scores.csv", score_rows)
    csv_new(Path(output)/"pres_router_confusion.csv", confusion_rows)
    d.write_new(Path(output)/"PRES_JASCL_STAGE2_ROUTES.json",
                dict(status="PASS_ROUTES_SEALED_BEFORE_GT", routes={str(s): {str(M): r for M, r in routes[s].items()}
                                                                    for s in range(3)}, validation_GT_reads=0))
    counters["output_rows"]["router_scores"] = len(score_rows)
    counters["output_rows"]["router_confusion"] = len(confusion_rows)
    return score_rows, summaries, routes


def run_bootstraps(output, descriptors, plans, formal_banks, counters):
    rows, stability = [], {1: [], 2: []}
    index = {seed: {row["case_id"]: i for i, row in enumerate(plans[seed])} for seed in range(3)}
    for seed in range(3):
        for M in (1, 2):
            for stage in (1, 2):
                for replicate in range(5):
                    boot_banks, cosine_values, occupancies, train_seeds = {}, [], [], []
                    for domain in range(stage + 1):
                        train_ids = sorted(row["case_id"] for row in plans[seed]
                                           if row["role"] == "train_unlabeled" and row["domain_index"] == domain)
                        draws, train_seed = bootstrap_draw(train_ids, seed=seed, stage=stage,
                                                           role="train_unlabeled", domain=domain, replicate=replicate)
                        train_seeds.append(train_seed)
                        weights = multiplicity(train_ids, draws)
                        x = descriptors[seed][[index[seed][case] for case in train_ids]]
                        fitted = fit_prototypes(x, M, seed=seed, domain_index=domain,
                                                weights=weights, replicate=replicate)
                        boot_banks[domain] = fitted
                        values = prototype_stability(formal_banks[seed][M][domain], fitted)
                        cosine_values.extend(values)
                        stability[M].extend(values)
                        occupancies.extend(fitted["occupancy"].tolist())
                    correct, totals, validation_seeds = [0] * (stage + 1), [0] * (stage + 1), []
                    for domain in range(stage + 1):
                        val_ids = sorted(row["case_id"] for row in plans[seed]
                                         if row["role"] == "val" and row["domain_index"] == domain)
                        draws, validation_seed = bootstrap_draw(val_ids, seed=seed, stage=stage,
                                                                role="val", domain=domain, replicate=replicate)
                        validation_seeds.append(validation_seed)
                        x = descriptors[seed][[index[seed][case] for case in draws]]
                        routed, _, _ = route(x, boot_banks, tuple(range(stage + 1)))
                        correct[domain] = int((routed == domain).sum())
                        totals[domain] = len(draws)
                    per_domain = [correct[d] / totals[d] for d in range(stage + 1)]
                    rows.append(dict(seed=seed, stage_index=stage, M=M, replicate=replicate,
                                     train_draw_seeds=train_seeds,
                                     validation_draw_seeds=validation_seeds,
                                     per_domain_accuracy=per_domain, macro_accuracy=float(np.mean(per_domain)),
                                     prototype_cosines=cosine_values, prototype_cosine_median=float(np.median(cosine_values)),
                                     minimum_bootstrap_occupancy=float(min(occupancies)), all_finite=True))
                    counters["bootstrap_operations"] += 1
    require(len(rows) == 60 and all(np.isfinite(stability[M]).all() for M in (1, 2)), "bootstrap coverage/nonfinite",
            "BLOCKED_CALL_GRAPH_MISMATCH")
    csv_new(Path(output)/"pres_router_bootstrap.csv", rows)
    counters["output_rows"]["router_bootstrap"] = len(rows)
    return rows, stability


def predict_experts(output, plans, data_root, expert_checkpoints, metadata, device, counters):
    cache_root = Path(output)/"prediction_cache"
    cache_root.mkdir()
    predictions, orders, entries = {}, {}, []
    for seed in range(3):
        rows = [row for row in plans[seed] if row["role"] == "val"]
        rows.sort(key=lambda row: row["case_id"])
        require(len(rows) == len({row["case_id"] for row in rows}) == 165, "validation prediction identity mismatch")
        orders[seed] = [row["case_id"] for row in rows]
        predictions[seed] = {}
        images_only = [image_only(row) for row in rows]
        for expert in range(3):
            cp = expert_checkpoints[seed][expert]
            models, payload = load_readonly_models(ROOT, cp, device=device, sources=("student",))
            model = models["student"]
            batches = []
            with ImmutableModels(models, cp, Path(output)/"expert_models"/f"seed{seed}_expert{expert}", metadata):
                with torch.no_grad():
                    for start in range(0, len(rows), BATCH_SIZE):
                        images = read_images(images_only[start:start+BATCH_SIZE], data_root).to(device)
                        with torch.autocast(device_type=device.type, enabled=False):
                            logits, _ = model(images, stochastic_classifier=False)
                        require(logits.dtype == torch.float32 and tuple(logits.shape[1:]) == (3, 384, 384),
                                "segmentation forward contract changed")
                        batches.append(logits.argmax(dim=1).to(torch.uint8).cpu().numpy())
                        counters["cross_expert_segmentation_forwards"] += 1
                        counters["cross_expert_segmentation_case_passes"] += len(images)
            counters["model_guards"] += 1
            require(all(parameter.grad is None for parameter in model.parameters()), "expert wrote parameter.grad",
                    "BLOCKED_MODEL_MUTATION")
            value = np.concatenate(batches)
            require(value.shape == (165, 384, 384), "prediction cache schema mismatch")
            predictions[seed][expert] = value
            entries.append(dict(seed=seed, expert=expert, checkpoint_id=cp["checkpoint_id"],
                                checkpoint_sha256=cp["sha256"], source="student", stochastic_classifier=False,
                                evaluation_classifier="posterior_mean", case_count=165,
                                cache=save_npz(cache_root/f"seed{seed}_expert{expert}.npz",
                                               case_ids=np.asarray([row["case_id"] for row in rows]), predictions=value)))
            del models, payload, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    seal = d.seal(cache_root)
    d.write_new(Path(output)/"PRES_JASCL_PREDICTION_CACHE_MANIFEST.json",
                dict(status="PASS_PREDICTIONS_SEALED_BEFORE_GT", entries=entries, case_expert_predictions=1485,
                     content_sha256=seal["content_sha256"],
                     manifest_sha256=d.sha256(cache_root/"PRIVATE_BUNDLE_MANIFEST.json"), validation_GT_reads=0,
                     deterministic_preprocessing=True, evaluation_classifier="posterior_mean"))
    return predictions, orders


def evaluate_segmentation(output, predictions, orders, routes, records, data_root, counters):
    cross_rows, strategy_rows = [], []
    for seed in range(3):
        available = {row["case_id"]: row for stage in range(3) for row in records[seed][stage]["val"]}
        rows = [available[case_id] for case_id in orders[seed]]
        labels = read_labels(rows, data_root)
        counters["validation_GT_case_reads"] += len(labels)
        case_index = {row["case_id"]: i for i, row in enumerate(rows)}
        for domain in range(3):
            cases = sorted(row["case_id"] for row in rows if row["site_or_vendor"] == DOMAINS[domain])
            positions = [case_index[case] for case in cases]
            for expert in range(3):
                confusion = np.zeros((3, 3), dtype=np.int64)
                for pos in positions:
                    confusion += pixel_confusion(predictions[seed][expert][pos], labels[pos])
                cross_rows.append(dict(seed=seed, true_domain=domain, domain=DOMAINS[domain], expert=expert,
                                       **segmentation_metrics(confusion)))
            for strategy, M in (("Shared-final", None), ("Oracle-snapshot", None),
                                ("Prototype-routed-M1", 1), ("Prototype-routed-M2", 2)):
                confusion = np.zeros((3, 3), dtype=np.int64)
                for case, pos in zip(cases, positions):
                    expert = 2 if strategy == "Shared-final" else (ORACLE_EXPERT[DOMAINS[domain]] if strategy == "Oracle-snapshot"
                              else routes[seed][M][case])
                    confusion += pixel_confusion(predictions[seed][expert][pos], labels[pos])
                strategy_rows.append(dict(seed=seed, true_domain=domain, domain=DOMAINS[domain], strategy=strategy,
                                          routed_M=M, **segmentation_metrics(confusion)))
    require(len(cross_rows) == 27 and len(strategy_rows) == 36 and counters["validation_GT_case_reads"] == 495,
            "segmentation evaluator coverage mismatch", "BLOCKED_CALL_GRAPH_MISMATCH")
    csv_new(Path(output)/"pres_cross_expert_matrix.csv", cross_rows)
    csv_new(Path(output)/"pres_oracle_vs_routed.csv", strategy_rows)
    counters["output_rows"]["cross_expert"] = len(cross_rows)
    counters["output_rows"]["oracle_vs_routed"] = len(strategy_rows)
    return cross_rows, strategy_rows


def _strategy_map(rows):
    return {(row["strategy"], row["seed"], row["true_domain"]): row for row in rows}


def gate_inputs(routing, bootstrap, stability, banks, strategies):
    mapped = _strategy_map(strategies)
    seed_oracle_gain, all_oracle_gain = [], []
    for seed in range(3):
        gains = []
        for domain in range(3):
            oracle = mapped[("Oracle-snapshot", seed, domain)]["mean_foreground_dice"]
            shared = mapped[("Shared-final", seed, domain)]["mean_foreground_dice"]
            gains.append(oracle - shared)
            all_oracle_gain.append(oracle - shared)
        seed_oracle_gain.append(float(np.mean(gains)))
    oracle_by_domain = {DOMAINS[domain]: float(np.mean([
        mapped[("Oracle-snapshot", seed, domain)]["mean_foreground_dice"] for seed in range(3)]))
        for domain in range(3)}
    shared_by_domain = {DOMAINS[domain]: float(np.mean([
        mapped[("Shared-final", seed, domain)]["mean_foreground_dice"] for seed in range(3)]))
        for domain in range(3)}
    snapshot_gain_by_domain = {domain: oracle_by_domain[domain] - shared_by_domain[domain] for domain in DOMAINS}
    d1 = dict(three_domain_gain=float(np.mean(all_oracle_gain)),
              historical_gain=float(np.mean([all_oracle_gain[3*seed+domain] for seed in range(3) for domain in (0, 1)])),
              positive_seed_count=sum(value > 0 for value in seed_oracle_gain),
              maximum_domain_drop=float(max(0.0, max(-value for value in all_oracle_gain))), seed_gains=seed_oracle_gain,
              oracle_foreground_dice_by_domain=oracle_by_domain, shared_foreground_dice_by_domain=shared_by_domain,
              gain_by_domain=snapshot_gain_by_domain,
              oracle_three_domain_foreground_dice=float(np.mean(list(oracle_by_domain.values()))),
              shared_three_domain_foreground_dice=float(np.mean(list(shared_by_domain.values()))),
              oracle_historical_foreground_dice=float(np.mean([oracle_by_domain[d] for d in DOMAINS[:2]])),
              shared_historical_foreground_dice=float(np.mean([shared_by_domain[d] for d in DOMAINS[:2]])),
              historical_forgetting_by_domain={d: snapshot_gain_by_domain[d] for d in DOMAINS[:2]},
              historical_forgetting_average=float(np.mean([snapshot_gain_by_domain[d] for d in DOMAINS[:2]])),
              current_domain_performance=dict(domain=DOMAINS[2], oracle=oracle_by_domain[DOMAINS[2]],
                                              shared_final=shared_by_domain[DOMAINS[2]]))
    candidates = {}
    for M in (1, 2):
        route_gate = {}
        for stage in (1, 2):
            per_domain = [float(np.mean([routing[seed][M][stage]["per_domain_accuracy"][domain] for seed in range(3)]))
                          for domain in range(stage + 1)]
            route_gate[f"stage{stage}_per_domain"] = per_domain
            route_gate[f"stage{stage}_macro"] = float(np.mean(per_domain))
            route_gate[f"stage{stage}_accuracy"] = float(np.mean([
                routing[seed][M][stage]["accuracy"] for seed in range(3)]))
            for metric in ("margin_p05", "margin_p10", "margin_median", "route_entropy_mean",
                           "route_entropy_p05", "route_entropy_p10", "route_entropy_median"):
                route_gate[f"stage{stage}_{metric}"] = float(np.mean([
                    routing[seed][M][stage][metric] for seed in range(3)]))
        routed_name = f"Prototype-routed-M{M}"
        oracle_gap, shared_gain, historical_gain, seed_gain, domain_gain = [], [], [], [], []
        for seed in range(3):
            seed_values = []
            for domain in range(3):
                routed = mapped[(routed_name, seed, domain)]["mean_foreground_dice"]
                oracle = mapped[("Oracle-snapshot", seed, domain)]["mean_foreground_dice"]
                shared = mapped[("Shared-final", seed, domain)]["mean_foreground_dice"]
                oracle_gap.append(oracle - routed)
                shared_gain.append(routed - shared)
                domain_gain.append(routed - shared)
                seed_values.append(routed - shared)
                if domain in (0, 1):
                    historical_gain.append(routed - shared)
            seed_gain.append(float(np.mean(seed_values)))
        routed_by_domain = {DOMAINS[domain]: float(np.mean([
            mapped[(routed_name, seed, domain)]["mean_foreground_dice"] for seed in range(3)]))
            for domain in range(3)}
        segmentation = dict(oracle_gap=float(np.mean(oracle_gap)), shared_gain=float(np.mean(shared_gain)),
                            historical_gain=float(np.mean(historical_gain)),
                            positive_seed_count=sum(value > 0 for value in seed_gain),
                            maximum_domain_drop=float(max(0.0, max(-value for value in domain_gain))), seed_gains=seed_gain,
                            routed_foreground_dice_by_domain=routed_by_domain,
                            gain_vs_shared_by_domain={d: routed_by_domain[d] - shared_by_domain[d] for d in DOMAINS},
                            gap_to_oracle_by_domain={d: oracle_by_domain[d] - routed_by_domain[d] for d in DOMAINS},
                            routed_three_domain_foreground_dice=float(np.mean(list(routed_by_domain.values()))),
                            routed_historical_foreground_dice=float(np.mean([routed_by_domain[d] for d in DOMAINS[:2]])),
                            current_domain_performance=dict(domain=DOMAINS[2], routed=routed_by_domain[DOMAINS[2]],
                                                            oracle=oracle_by_domain[DOMAINS[2]],
                                                            shared_final=shared_by_domain[DOMAINS[2]]))
        stage2 = [row for row in bootstrap if row["M"] == M and row["stage_index"] == 2]
        replicate_macro = [float(np.mean([row["macro_accuracy"] for row in stage2 if row["replicate"] == replicate]))
                           for replicate in range(5)]
        formal_occupancies = [float(value) for seed in range(3) for domain in range(3)
                              for value in banks[seed][M][domain]["occupancy"]]
        bootstrap_occupancies = [float(row["minimum_bootstrap_occupancy"]) for row in bootstrap if row["M"] == M]
        prototype_separations = [float(banks[seed][M][domain]["within_domain_prototype_cosine_distance"])
                                 for seed in range(3) for domain in range(3) if M == 2]
        stable = dict(prototype_cosine_median=float(np.median(stability[M])) if M == 1 else None,
                      matched_cosine_median=float(np.median(stability[M])) if M == 2 else None,
                      occupancies=[*formal_occupancies, *bootstrap_occupancies],
                      formal_occupancies=formal_occupancies, bootstrap_minimum_occupancies=bootstrap_occupancies,
                      within_domain_prototype_cosine_distances=prototype_separations,
                      bootstrap_macro_by_replicate=replicate_macro,
                      bootstrap_macro_p10=float(np.quantile(replicate_macro, .10, method="linear")), all_finite=True)
        candidates[M] = dict(complete=True, routing=route_gate, segmentation=segmentation, stability=stable)
    return d1, candidates


def immutability_pass(output, counters):
    guards = sorted(Path(output).glob("*_models/**/immutability/*.json"))
    values = [d.read(path) for path in guards]
    return (len(values) == counters["model_guards"] == 12
            and all(value["bitwise_unchanged"] and value["extraction_completed"] for value in values))


def link_test_evidence(output, test_report):
    report = d.read(test_report)
    for source, name in ((report["junit_path"], "pytest.xml"), (report["pytest_output_path"], "pytest_output.txt")):
        os.link(source, Path(output)/name)


def router_seal(output):
    names = ("PRES_JASCL_ROUTER_DESCRIPTOR_MANIFEST.json", "PRES_JASCL_ROUTING_METADATA.json",
             "PRES_JASCL_DOMAIN_PROTOTYPE_MANIFEST.json", "pres_router_scores.csv",
             "pres_router_confusion.csv", "pres_router_bootstrap.csv", "PRES_JASCL_STAGE2_ROUTES.json")
    entries = [dict(path=name, sha256=d.sha256(Path(output)/name), bytes=(Path(output)/name).stat().st_size) for name in names]
    d.write_new(Path(output)/"PRES_JASCL_ROUTER_SEAL.json",
                dict(status="PASS_ROUTER_SEALED_BEFORE_VALIDATION_GT", entries=entries,
                     validation_GT_reads_before_seal=0, router_builder_GT_argument=False,
                     test_role_constructions=0, sealed_at=d.now()))


def report(output, metadata, counters, d1, candidates, decision):
    status = dict(metadata=metadata, **decision, D1_values=d1,
                  candidate_values={str(M): candidates[M] for M in (1, 2)}, counters=counters,
                  model_optimizer_steps=0, router_optimizer_steps=0, autograd_calls=0, backward_calls=0,
                  parameter_grad_writes=0, method_registered=False, training_launched=False,
                  validation_GT_usage="segmentation_evaluator_only", hidden_GT_usage="none", test_GT_usage="none",
                  expert_inference_source="student", router_extractor_source="stage0_ema_teacher",
                  evaluation_classifier="posterior_mean", deterministic_evaluation=True,
                  call_graph_sha256=d.sha256(Path(output)/"PRES_JASCL_CALL_GRAPH.json"),
                  report_commit=None, report_commit_resolution="first Git commit adding these exact public report bytes")
    d.write_new(Path(output)/"PRES_JASCL_STATUS.json", status)
    selected = "none" if decision["selected_M"] is None else str(decision["selected_M"])
    lines = [
        "# PRES-JASCL V0.1 final report", "",
        f"Scientific status: `{decision['scientific_status']}`. Passing M: `{decision['passing_M']}`; selected M: `{selected}`.", "",
        "This was one no-training validation-only execution of fixed regenerated-B0 snapshot experts. The router used each seed's stage0 EMA encoder and RGB/enc1/enc2 style statistics; segmentation used the frozen evaluator's student posterior mean (`stochastic_classifier=False`). Expert selection was never tuned on validation.", "",
        "## D1-D5", "",
        f"D1={decision['D1']}; D2={decision['D2']}; D3={decision['D3']}; D4={decision['D4']}; D5={decision['D5']}.",
        f"D1 three-domain Oracle gain={d1['three_domain_gain']:.12f}, historical gain={d1['historical_gain']:.12f}, positive seeds={d1['positive_seed_count']}, maximum domain drop={d1['maximum_domain_drop']:.12f}.", "",
        f"Snapshot Oracle/Shared foreground Dice by domain: {d1['oracle_foreground_dice_by_domain']} / {d1['shared_foreground_dice_by_domain']}. Historical forgetting average={d1['historical_forgetting_average']:.12f}; current-domain performance={d1['current_domain_performance']}.", "",
    ]
    for M in (1, 2):
        value = candidates[M]
        lines.extend([f"### M{M}", "",
                      f"Stage1 routing accuracy={value['routing']['stage1_accuracy']:.12f}, macro={value['routing']['stage1_macro']:.12f}, per-domain={value['routing']['stage1_per_domain']}; Stage2 accuracy={value['routing']['stage2_accuracy']:.12f}, macro={value['routing']['stage2_macro']:.12f}, per-domain={value['routing']['stage2_per_domain']}.",
                      f"Stage2 true-domain margin p05/p10/median={value['routing']['stage2_margin_p05']:.12f}/{value['routing']['stage2_margin_p10']:.12f}/{value['routing']['stage2_margin_median']:.12f}; route entropy mean/median={value['routing']['stage2_route_entropy_mean']:.12f}/{value['routing']['stage2_route_entropy_median']:.12f} nats.",
                      f"Routed Stage2 Oracle gap={value['segmentation']['oracle_gap']:.12f}, Shared gain={value['segmentation']['shared_gain']:.12f}, historical gain={value['segmentation']['historical_gain']:.12f}, positive seeds={value['segmentation']['positive_seed_count']}, maximum domain drop={value['segmentation']['maximum_domain_drop']:.12f}.",
                      f"Routed foreground Dice by domain={value['segmentation']['routed_foreground_dice_by_domain']}; current-domain performance={value['segmentation']['current_domain_performance']}.",
                      f"Bootstrap routing macro p10={value['stability']['bootstrap_macro_p10']:.12f}; prototype stability median={value['stability']['prototype_cosine_median'] if M == 1 else value['stability']['matched_cosine_median']}; minimum occupancy={min(value['stability']['occupancies']):.12f}; within-domain prototype cosine distances={value['stability']['within_domain_prototype_cosine_distances']}.", ""])
    lines.extend(["## Isolation and coverage", "",
                  f"The manifest-frozen graph completed {counters['router_extraction_forwards']} router encoder forwards over {counters['router_extraction_case_passes']} cases and {counters['cross_expert_segmentation_forwards']} expert forwards over {counters['cross_expert_segmentation_case_passes']} case-expert passes. All {counters['model_guards']} model/checkpoint guards passed; {counters['bootstrap_operations']} bootstrap operations and {counters['total_output_rows']} registered CSV rows completed.", "",
                  "Optimizer, autograd.grad, backward, parameter-grad-write, EMA/GAS/PAS/K2-update and training counters are all zero. Validation GT entered only the independent pixel-confusion evaluator after router and prediction seals. No test object or GT was constructed. Model and checkpoint states remained bitwise unchanged.", "",
                  "The complete Gate1C private bundle, its manifest/content identity, all nine checkpoint SHAs, and 2962 frozen data checksums passed the registered input audit. Raw descriptors, predictions, case-level scores, prototypes and private manifests remain on NAS.", "",
                  "Execution now hard-stops for independent review; no test evaluation, regeneration, retraining, adapter, MILE, other benchmark, sweep, or main merge is authorized.", ""])
    text_new(Path(output)/"PRES_JASCL_FINAL_REPORT.md", "\n".join(lines))
    warnings = ["# PRES-JASCL failures and warnings", "", f"Final scientific status: `{decision['scientific_status']}`.", ""]
    if decision["scientific_status"] == "PASS_PRES_ROUTING_FEASIBILITY":
        warnings.append("No scientific gate failed. PASS remains a validation feasibility result, not a training-method or test-set claim.")
    else:
        warnings.append("One or more registered scientific gates failed; controls and the other M candidate cannot rescue that decision.")
    warnings.extend(["", "The server access path, case IDs, raw tensors, descriptors, predictions, and private artifact paths are intentionally omitted from the public report.", ""])
    text_new(Path(output)/"PRES_JASCL_FAILURES_AND_WARNINGS.md", "\n".join(warnings))
    text_new(Path(output)/"PRES_JASCL_EXACT_COMMANDS.md", "\n".join([
        "# PRES-JASCL exact commands", "", "## Tests", "", "```sh", metadata["exact_test_command"], "```", "",
        "## Durable child argv", "", "```sh", shlex.join(metadata["exact_command"]), "```", "",
        "The server-local durable launch receipt records the parent command, cwd, environment, PID and actual exit code. Both commands ran through `bash experiments/lcrseg/scripts/with_nas_storage.sh`; no package, optimizer or training command was used.", "",
    ]))


def artifact_manifest(output):
    excluded = ("PRES_JASCL_ARTIFACT_MANIFEST.json", "controller.log", "supervisor.log", "LAUNCH_REQUEST.json",
                "LAUNCH_RECEIPT.json", "PROCESS_START.json", "PROCESS_PID.json", "PROCESS_EXIT.json",
                "EXECUTION_COMPLETION.json", "PHASE_pres_jascl.json", "PHASE_pres_jascl_MANIFEST.json")
    entries = d.file_entries(output, exclude=excluded)
    required = {"PRES_JASCL_INPUT_AUDIT.json", "PRES_JASCL_CALL_GRAPH.json",
                "PRES_JASCL_ROUTER_DESCRIPTOR_MANIFEST.json", "PRES_JASCL_DOMAIN_PROTOTYPE_MANIFEST.json",
                "pres_router_scores.csv", "pres_router_confusion.csv", "pres_router_bootstrap.csv",
                "pres_cross_expert_matrix.csv", "pres_oracle_vs_routed.csv", "PRES_JASCL_STATUS.json",
                "PRES_JASCL_FINAL_REPORT.md", "PRES_JASCL_FAILURES_AND_WARNINGS.md",
                "PRES_JASCL_EXACT_COMMANDS.md", "pytest.xml", "pytest_output.txt"}
    names = {entry["path"] for entry in entries}
    require(required.issubset(names), "required artifact missing", "BLOCKED_INCOMPLETE_EVIDENCE")
    result = dict(status="PASS_CONTROLLER_ARTIFACT_MANIFEST", created_at=d.now(), artifacts=entries,
                  file_count=len(entries), total_bytes=sum(entry["bytes"] for entry in entries),
                  required_outputs_complete=True, excludes_live_supervisor_receipts=True)
    d.write_new(Path(output)/"PRES_JASCL_ARTIFACT_MANIFEST.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(args.output.is_dir(), "durable create-only output was not initialized")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)
        enforce_deterministic_backend()
        metadata = execution_gate(args.output, args.code_commit, args.test_report)
        metadata["exact_command"] = sys.argv
        d.write_new(args.output/"PRES_JASCL_RUN_METADATA.json", metadata)
        link_test_evidence(args.output, args.test_report)
        p = gate1c_contract()
        records = input_audit(args.output, p, metadata)
        graph = compile_call_graph(args.output, records, args.code_commit)
        data_root = p["destination"]["data_root"]
        router_checkpoints = {seed: checkpoint(p, seed, 0) for seed in range(3)}
        expert_checkpoints = {seed: {expert: checkpoint(p, seed, expert) for expert in range(3)} for seed in range(3)}
        counters = dict(router_extraction_forwards=0, router_extraction_case_passes=0,
                        cross_expert_segmentation_forwards=0, cross_expert_segmentation_case_passes=0,
                        bootstrap_operations=0, model_guards=0, validation_GT_case_reads=0,
                        output_rows={key: 0 for key in graph["output_rows"]})
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one visible CUDA GPU required")
        device = torch.device("cuda:0")
        with isolation_guard():
            image_plans = {seed: descriptor_image_plan(records, seed) for seed in range(3)}
            descriptors = extract_descriptors(args.output, image_plans, data_root, router_checkpoints,
                                              metadata, device, counters)
            plans = release_routing_metadata(args.output, records, image_plans)
            banks = build_prototypes(args.output, descriptors, plans)
            score_rows, routing, routes = evaluate_routing(args.output, descriptors, plans, banks, counters)
            bootstrap, stability = run_bootstraps(args.output, descriptors, plans, banks, counters)
            router_seal(args.output)
            predictions, orders = predict_experts(args.output, plans, data_root, expert_checkpoints,
                                                  metadata, device, counters)
            cross_rows, strategies = evaluate_segmentation(args.output, predictions, orders, routes,
                                                           records, data_root, counters)
        counters["total_output_rows"] = sum(counters["output_rows"].values())
        d1, candidates = gate_inputs(routing, bootstrap, stability, banks, strategies)
        backend = deterministic_backend_state()
        counters["deterministic_backend"] = backend
        deterministic = all(backend.values())
        d5 = (immutability_pass(args.output, counters) and counters["validation_GT_case_reads"] == 495
              and counters["total_output_rows"] == graph["total_output_rows"] and deterministic)
        decision = adjudicate(d1, candidates, d5)
        observed = dict(counters)
        verify_call_graph(graph, observed)
        report(args.output, metadata, counters, d1, candidates, decision)
        manifest = artifact_manifest(args.output)
        print(json.dumps(dict(status=decision["scientific_status"], selected_M=decision["selected_M"],
                              artifacts=manifest["file_count"], counters=counters), sort_keys=True), flush=True)
    except BaseException as error:
        status = getattr(error, "status", "BLOCKED_INCOMPLETE_EVIDENCE")
        if status not in ("BLOCKED_BASE_COMMIT_AMBIGUOUS", "BLOCKED_PRIVATE_BUNDLE_MISMATCH",
                          "BLOCKED_PROTOCOL_OR_LEAKAGE", "BLOCKED_MODEL_MUTATION", "BLOCKED_CALL_GRAPH_MISMATCH",
                          "BLOCKED_INCOMPLETE_EVIDENCE", "BLOCKED_NUMERICAL_FAILURE"):
            status = "BLOCKED_PROTOCOL_OR_LEAKAGE"
        failure = args.output/f"PRES_JASCL_FAILURE_{os.getpid()}.json"
        if args.output.is_dir() and not failure.exists():
            d.write_new(failure, dict(status=status, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                                      command=sys.argv, recorded_at=d.now(), new_attempt_authorized=False,
                                      model_optimizer_steps=0, router_optimizer_steps=0, autograd_calls=0,
                                      method_registered=False, training_launched=False))
        raise


if __name__ == "__main__":
    main()
