"""One create-only PRES-DSR-SF validation execution; no model training or autograd."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import traceback

import numpy as np
import torch

from di_dmpa_gate1.binding import safe_asset
from di_dmpa_gate1c_v3 import durable as d
from di_dmpa_gate1c_v3.inputs import load_models
from di_dmpa_gate1_v2.features import ImmutableModels
from pres_jascl_v0_1 import run as v1run
from pres_jascl_v0_1.core import (DOMAINS, array_sha256, bootstrap_draw, fit_prototypes, multiplicity,
                                  pixel_confusion, prototype_stability, require, route, routing_summary,
                                  segmentation_metrics, style_descriptors)

from .core import (Blocked, LAMBDAS, TEMPERATURES, adjudicate, bootstrap_multiplicity, case_folds,
                   domain_metrics, fit_router, hard_routes, probability_fusion, raw_style_descriptors,
                   router_probabilities, salted_hash, select_memory, softmax)
from .protocol import (BATCH_SIZE, ROOT, backend_import_gate, compile_call_graph, execution_gate,
                       gate1c_contract, input_audit, isolation_guard, require_backend, verify_call_graph)

POLICIES = ("C0_SHARED", "C1_ORACLE", "C2_M1_HARD", "C3_M2_HARD",
            "C4_M1_SOFT", "C5_RIDGE_HARD", "C6_RIDGE_SOFT", "C7_UNIFORM")
V1_STATUS_SHA = "25d2e96bd16f379dbcf13910a0264d6fd86df36ed40a6838a81117b3520a894f"


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


def descriptor_plan(records, seed):
    rows = []
    for domain in range(3):
        for role in ("train_labeled", "train_unlabeled", "val"):
            rows.extend(v1run.image_only(row) for row in records[seed][domain][role])
    rows.sort(key=lambda row: row["case_id"])
    require(len(rows) == len({row["case_id"] for row in rows}), "descriptor case identity changed",
            "BLOCKED_INCOMPLETE_EVIDENCE")
    return rows


def load_models_checked(checkpoint, *, device, sources, phase):
    require_backend(phase + "_before_model_load")
    models, payload = load_models(ROOT, checkpoint, device=device, sources=sources)
    require_backend(phase + "_after_model_load")
    return models, payload


def extract_descriptors(output, plans, data_root, router_checkpoints, metadata, device, counters):
    cache_root = Path(output) / "descriptor_cache"
    cache_root.mkdir()
    values, entries = {}, []
    for seed in range(3):
        rows = plans[seed]
        checkpoint = router_checkpoints[seed]
        models, payload = load_models_checked(checkpoint, device=device, sources=("ema_teacher",),
                                              phase=f"descriptor_seed{seed}")
        model = models["ema_teacher"]
        raw_batches, legacy_batches, validity_batches = [], [], []
        with ImmutableModels(models, checkpoint, Path(output) / "router_models" / f"seed{seed}", metadata):
            with torch.no_grad():
                for start in range(0, len(rows), BATCH_SIZE):
                    images = v1run.read_images(rows[start:start + BATCH_SIZE], data_root).to(device)
                    with torch.autocast(device_type=device.type, enabled=False):
                        enc1 = model.enc1(images)
                        enc2 = model.enc2(model.pool(enc1))
                    raw_batches.append(raw_style_descriptors(images, enc1, enc2))
                    legacy, valid = style_descriptors(images, enc1, enc2)
                    legacy_batches.append(legacy)
                    validity_batches.append(valid)
                    counters["descriptor_forwards"] += 1
                    counters["descriptor_case_passes"] += len(images)
        counters["model_guards"] += 1
        require(all(parameter.grad is None for parameter in model.parameters()), "descriptor model wrote parameter.grad",
                "BLOCKED_MODEL_MUTATION")
        require_backend(f"descriptor_seed{seed}_complete")
        raw = np.concatenate(raw_batches)
        legacy = np.concatenate(legacy_batches)
        validity = np.concatenate(validity_batches)
        require(raw.shape == legacy.shape == (len(rows), 102) and validity.shape == (len(rows), 3),
                "descriptor cache schema changed")
        values[seed] = dict(raw=raw, legacy=legacy)
        name = f"seed{seed}.npz"
        npz_new(cache_root / name, case_ids=np.asarray([row["case_id"] for row in rows]),
                raw_descriptors=raw, legacy_descriptors=legacy, legacy_block_valid=validity)
        entries.append(dict(seed=seed, checkpoint_id=checkpoint["checkpoint_id"], case_count=len(rows),
                            descriptor_dim=102, raw_dtype="float64", legacy_dtype="float64", cache=name))
        del models, payload, model
        torch.cuda.empty_cache()
    seal = d.seal(cache_root)
    d.write_new(Path(output) / "PRES_DSR_SF_DESCRIPTOR_MANIFEST.json",
                dict(status="PASS_DESCRIPTORS_SEALED_BEFORE_DOMAIN_METADATA", entries=entries,
                     total_cases=sum(len(plans[seed]) for seed in range(3)), descriptor_dim=102,
                     formula="mean+log(population_std+1e-6)", per_block_L2_normalization=False,
                     train_statistics_not_yet_fit=True, router_API_received_GT=False,
                     router_API_received_domain_metadata=False, content_sha256=seal["content_sha256"],
                     manifest_sha256=d.sha256(cache_root / "PRIVATE_BUNDLE_MANIFEST.json")))
    return values


def release_metadata(output, records, plans):
    released = {}
    for seed in range(3):
        rows = []
        for domain in range(3):
            for role in ("train_labeled", "train_unlabeled", "val"):
                rows.extend(dict(case_id=row["case_id"], role=role, domain_index=domain,
                                 image_h5_relpath=row["image_h5_relpath"], image_sha256=row["image_sha256"])
                            for row in records[seed][domain][role])
        rows.sort(key=lambda row: row["case_id"])
        require([row["case_id"] for row in rows] == [row["case_id"] for row in plans[seed]],
                "metadata does not bind sealed descriptors")
        released[seed] = rows
    d.write_new(Path(output) / "PRES_DSR_SF_ROUTING_METADATA.json",
                dict(status="PASS_METADATA_RELEASED_AFTER_DESCRIPTOR_SEAL",
                     seeds={str(seed): released[seed] for seed in range(3)}, segmentation_GT_fields=0, test_records=0))
    return released


def build_memory(output, descriptors, metadata, data_root):
    cache_root = Path(output) / "memory_cache"
    cache_root.mkdir()
    memories, entries, cost_rows = {}, [], []
    for seed in range(3):
        memories[seed] = {}
        index = {row["case_id"]: i for i, row in enumerate(metadata[seed])}
        old_hashes = None
        for domain in range(3):
            candidates = [row for row in metadata[seed]
                          if row["domain_index"] == domain and row["role"] in ("train_labeled", "train_unlabeled")]
            selected, hashes = select_memory(candidates)
            raw = descriptors[seed]["raw"][[index[row["case_id"]] for row in selected]]
            name = f"seed{seed}_domain{domain}.npz"
            npz_new(cache_root / name, descriptors=raw,
                    domain_indices=np.full(len(selected), domain, dtype=np.int64), case_hashes=np.asarray(hashes))
            memories[seed][domain] = dict(case_ids=[row["case_id"] for row in selected], descriptors=raw,
                                          case_hashes=hashes, path=cache_root / name)
            image_bytes = sum(safe_asset(data_root, row["image_h5_relpath"]).stat().st_size for row in selected)
            entry = dict(seed=seed, domain_index=domain, domain=DOMAINS[domain], candidate_rows=len(candidates),
                         selected_rows=len(selected), cap=512, case_hashes=hashes, cache=name,
                         descriptor_bytes=raw.nbytes, source_image_asset_bytes=image_bytes,
                         stored_images=0, stored_masks=0, segmentation_labels_read=0)
            entries.append(entry)
            cost_rows.append({key: entry[key] for key in ("seed", "domain_index", "domain", "candidate_rows",
                                                           "selected_rows", "descriptor_bytes", "source_image_asset_bytes")})
            if domain == 1:
                old_hashes = [d.sha256(memories[seed][old]["path"]) for old in (0, 1)]
            if domain == 2:
                require(old_hashes == [d.sha256(memories[seed][old]["path"]) for old in (0, 1)],
                        "historical memory bytes changed", "BLOCKED_MODEL_MUTATION")
    v1run.csv_new(Path(output) / "pres_dsr_memory_cost.csv", cost_rows)
    seal = d.seal(cache_root)
    d.write_new(Path(output) / "PRES_DSR_SF_MEMORY_MANIFEST.json",
                dict(status="PASS_TRAIN_ONLY_MEMORY", entries=entries, rows_per_domain_max=512,
                     train_labeled_images_used=True, train_unlabeled_images_used=True,
                     segmentation_labels_read=0, stored_images=0, stored_masks=0,
                     historical_memory_byte_unchanged=True, content_sha256=seal["content_sha256"],
                     manifest_sha256=d.sha256(cache_root / "PRIVATE_BUNDLE_MANIFEST.json")))
    return memories, cost_rows


def fit_m1_temperature(legacy, rows, *, seed, stage, counters):
    selected = [row for row in rows if row["role"] == "train_unlabeled" and row["domain_index"] <= stage]
    selected.sort(key=lambda row: row["case_id"])
    ids = [row["case_id"] for row in selected]
    labels = np.asarray([row["domain_index"] for row in selected], dtype=np.int64)
    index = {row["case_id"]: i for i, row in enumerate(rows)}
    value = legacy[[index[case] for case in ids]]
    folds = case_folds(ids, labels)
    oof = np.full((len(ids), stage + 1), np.nan, dtype=np.float64)
    for fold in range(5):
        banks = {}
        for domain in range(stage + 1):
            train = (labels == domain) & (folds != fold)
            banks[domain] = fit_prototypes(value[train], 1, seed=seed, domain_index=domain)
            counters["m1_cv_prototype_fits"] += 1
        held = folds == fold
        _, scores, _ = route(value[held], banks, tuple(range(stage + 1)))
        oof[held] = scores
    require(bool(np.isfinite(oof).all()), "M1 OOF scores incomplete")
    rows_out = []
    for temperature in TEMPERATURES:
        rows_out.append(dict(kind="m1_temperature", value=temperature,
                             **domain_metrics(oof, labels, temperature=temperature)))
    selected_temperature = min(TEMPERATURES, key=lambda value: (
        next(row["domain_nll"] for row in rows_out if row["value"] == value), abs(np.log(value)), value))
    return selected_temperature, rows_out


def clean_controls(descriptors, metadata, counters):
    banks, states, routing, temperatures, cv_rows = {}, {}, {}, {}, []
    for seed in range(3):
        index = {row["case_id"]: i for i, row in enumerate(metadata[seed])}
        banks[seed], states[seed], routing[seed], temperatures[seed] = {}, {}, {}, {}
        for M in (1, 2):
            banks[seed][M] = {}
            for domain in range(3):
                ids = sorted(row["case_id"] for row in metadata[seed]
                             if row["role"] == "train_unlabeled" and row["domain_index"] == domain)
                banks[seed][M][domain] = fit_prototypes(
                    descriptors[seed]["legacy"][[index[case] for case in ids]], M, seed=seed, domain_index=domain)
                counters["clean_control_prototype_fits"] += 1
        for stage in (1, 2):
            temperatures[seed][stage], temp_rows = fit_m1_temperature(
                descriptors[seed]["legacy"], metadata[seed], seed=seed, stage=stage, counters=counters)
            cv_rows.extend(dict(seed=seed, stage_index=stage, router="M1", **row) for row in temp_rows)
        for M in (1, 2):
            states[seed][M], routing[seed][M] = {}, {}
            for stage in (1, 2):
                selected = [row for row in metadata[seed] if row["role"] == "val" and row["domain_index"] <= stage]
                selected.sort(key=lambda row: row["case_id"])
                ids = [row["case_id"] for row in selected]
                truth = np.asarray([row["domain_index"] for row in selected], dtype=np.int64)
                value = descriptors[seed]["legacy"][[index[case] for case in ids]]
                routed, scores, entropy = route(value, banks[seed][M], tuple(range(stage + 1)))
                summary_rows = [dict(true_domain=int(truth[i]), routed_domain=int(routed[i]),
                                     true_domain_margin=float(scores[i, truth[i]] - np.delete(scores[i], truth[i]).max()),
                                     route_entropy=float(entropy[i])) for i in range(len(ids))]
                routing[seed][M][stage] = routing_summary(summary_rows, stage + 1)
                states[seed][M][stage] = dict(case_ids=ids, truth=truth, routed=routed, scores=scores)
    return banks, states, routing, temperatures, cv_rows


def clean_control_bootstraps(descriptors, metadata, formal_banks, counters):
    rows, stability = [], {1: [], 2: []}
    indices = {seed: {row["case_id"]: i for i, row in enumerate(metadata[seed])} for seed in range(3)}
    for seed in range(3):
        for M in (1, 2):
            for stage in (1, 2):
                for replicate in range(5):
                    boot_banks, cosine_values, occupancies = {}, [], []
                    for domain in range(stage + 1):
                        ids = sorted(row["case_id"] for row in metadata[seed]
                                     if row["role"] == "train_unlabeled" and row["domain_index"] == domain)
                        draws, _ = bootstrap_draw(ids, seed=seed, stage=stage, role="train_unlabeled",
                                                  domain=domain, replicate=replicate)
                        fitted = fit_prototypes(descriptors[seed]["legacy"][[indices[seed][case] for case in ids]], M,
                                                seed=seed, domain_index=domain, weights=multiplicity(ids, draws),
                                                replicate=replicate)
                        counters["clean_control_prototype_fits"] += 1
                        boot_banks[domain] = fitted
                        values = prototype_stability(formal_banks[seed][M][domain], fitted)
                        stability[M].extend(values)
                        cosine_values.extend(values)
                        occupancies.extend(fitted["occupancy"].tolist())
                    per_domain = []
                    for domain in range(stage + 1):
                        ids = sorted(row["case_id"] for row in metadata[seed]
                                     if row["role"] == "val" and row["domain_index"] == domain)
                        draws, _ = bootstrap_draw(ids, seed=seed, stage=stage, role="val",
                                                  domain=domain, replicate=replicate)
                        routed, _, _ = route(descriptors[seed]["legacy"][[indices[seed][case] for case in draws]],
                                             boot_banks, tuple(range(stage + 1)))
                        per_domain.append(float(np.mean(routed == domain)))
                    rows.append(dict(kind="clean_control", seed=seed, stage_index=stage, M=M, replicate=replicate,
                                     macro_accuracy=float(np.mean(per_domain)), per_domain_accuracy=per_domain,
                                     prototype_cosines=cosine_values,
                                     minimum_bootstrap_occupancy=float(min(occupancies)), all_finite=True))
                    counters["clean_control_bootstrap_operations"] += 1
    return rows, stability


def fit_ridge_routers(memories, descriptors, metadata, clean_states, m1_temperatures, output, counters):
    models, states, cv_rows, score_rows, confusion_rows, manifest_rows = {}, {}, [], [], [], []
    for seed in range(3):
        models[seed], states[seed] = {}, {}
        index = {row["case_id"]: i for i, row in enumerate(metadata[seed])}
        for stage in (1, 2):
            ids = [case for domain in range(stage + 1) for case in memories[seed][domain]["case_ids"]]
            labels = np.concatenate([np.full(len(memories[seed][domain]["case_ids"]), domain, dtype=np.int64)
                                     for domain in range(stage + 1)])
            train = np.concatenate([memories[seed][domain]["descriptors"] for domain in range(stage + 1)])
            model = fit_router(train, labels, ids)
            counters["ridge_closed_form_fits"] += 26
            models[seed][stage] = model
            cv_rows.extend(dict(seed=seed, stage_index=stage, router="ridge", **row) for row in model["cv_rows"])
            selected = [row for row in metadata[seed] if row["role"] == "val" and row["domain_index"] <= stage]
            selected.sort(key=lambda row: row["case_id"])
            val_ids = [row["case_id"] for row in selected]
            truth = np.asarray([row["domain_index"] for row in selected], dtype=np.int64)
            raw = descriptors[seed]["raw"][[index[case] for case in val_ids]]
            alpha = router_probabilities(raw, model)
            routed = hard_routes(alpha, range(stage + 1))
            states[seed][stage] = dict(case_ids=val_ids, truth=truth, alpha=alpha, routed=routed)
            manifest_rows.append(dict(seed=seed, stage_index=stage, selected_lambda=model["selected_lambda"],
                                      selected_temperature=model["selected_temperature"],
                                      M1_temperature=m1_temperatures[seed][stage],
                                      mean_sha256=array_sha256(model["mean"]), std_sha256=array_sha256(model["std"]),
                                      constant_sha256=array_sha256(model["constant"]),
                                      weights_sha256=array_sha256(model["weights"]), folds_sha256=array_sha256(model["folds"]),
                                      weights=model["weights"].tolist(), mean=model["mean"].tolist(),
                                      std=model["std"].tolist(), constant=model["constant"].tolist()))
            for i, case in enumerate(val_ids):
                m1 = clean_states[seed][1][stage]
                m2 = clean_states[seed][2][stage]
                score_rows.append(dict(seed=seed, stage_index=stage, case_id=case, true_domain=int(truth[i]),
                                       M1_hard=int(m1["routed"][i]), M2_hard=int(m2["routed"][i]),
                                       ridge_hard=int(routed[i]), M1_scores=m1["scores"][i].tolist(),
                                       M2_scores=m2["scores"][i].tolist(),
                                       ridge_alpha=alpha[i].tolist()))
            for name, predicted in (("M1_HARD", clean_states[seed][1][stage]["routed"]),
                                    ("M2_HARD", clean_states[seed][2][stage]["routed"]),
                                    ("RIDGE_HARD", routed)):
                for true in range(stage + 1):
                    for route_to in range(stage + 1):
                        confusion_rows.append(dict(seed=seed, stage_index=stage, router=name, true_domain=true,
                                                   routed_domain=route_to,
                                                   count=int(np.sum((truth == true) & (predicted == route_to)))))
    require(len(cv_rows) == 78 and len(score_rows) == 915 and len(confusion_rows) == 117,
            "router output coverage changed", "BLOCKED_INCOMPLETE_EVIDENCE")
    v1run.csv_new(Path(output) / "pres_dsr_cv.csv", cv_rows)
    v1run.csv_new(Path(output) / "pres_dsr_router_scores.csv", score_rows)
    v1run.csv_new(Path(output) / "pres_dsr_routing_confusion.csv", confusion_rows)
    return models, states, cv_rows, score_rows, confusion_rows, manifest_rows


def fit_ridge_bootstraps(memories, descriptors, metadata, counters):
    states, rows, manifests = {}, [], []
    for seed in range(3):
        states[seed] = {}
        index = {row["case_id"]: i for i, row in enumerate(metadata[seed])}
        for stage in (1, 2):
            states[seed][stage] = {}
            ids = [case for domain in range(stage + 1) for case in memories[seed][domain]["case_ids"]]
            labels = np.concatenate([np.full(len(memories[seed][domain]["case_ids"]), domain, dtype=np.int64)
                                     for domain in range(stage + 1)])
            train = np.concatenate([memories[seed][domain]["descriptors"] for domain in range(stage + 1)])
            val_rows = [row for row in metadata[seed] if row["role"] == "val" and row["domain_index"] <= stage]
            val_rows.sort(key=lambda row: row["case_id"])
            val_ids = [row["case_id"] for row in val_rows]
            truth = np.asarray([row["domain_index"] for row in val_rows], dtype=np.int64)
            raw = descriptors[seed]["raw"][[index[case] for case in val_ids]]
            for replicate in range(5):
                parts, draw_seeds = [], []
                for domain in range(stage + 1):
                    mult, draw_seed = bootstrap_multiplicity(memories[seed][domain]["case_ids"], seed=seed,
                                                             stage=stage, domain=domain, replicate=replicate)
                    parts.append(mult)
                    draw_seeds.append(draw_seed)
                multiplicity_ = np.concatenate(parts)
                model = fit_router(train, labels, ids, multiplicity=multiplicity_)
                counters["ridge_closed_form_fits"] += 26
                counters["bootstrap_operations"] += 1
                alpha = router_probabilities(raw, model)
                routed = hard_routes(alpha, range(stage + 1))
                per_domain = [float(np.mean(routed[truth == domain] == domain)) for domain in range(stage + 1)]
                states[seed][stage][replicate] = dict(case_ids=val_ids, truth=truth, alpha=alpha, routed=routed)
                row = dict(kind="ridge", seed=seed, stage_index=stage, replicate=replicate,
                           hard_macro_accuracy=float(np.mean(per_domain)), hard_per_domain_accuracy=per_domain,
                           selected_lambda=model["selected_lambda"], selected_temperature=model["selected_temperature"],
                           train_draw_seeds=draw_seeds, soft_three_domain_gain=None, soft_oracle_gap=None)
                rows.append(row)
                manifests.append(dict(seed=seed, stage_index=stage, replicate=replicate,
                                      selected_lambda=model["selected_lambda"],
                                      selected_temperature=model["selected_temperature"], draw_seeds=draw_seeds,
                                      weights_sha256=array_sha256(model["weights"]), alpha_sha256=array_sha256(alpha)))
    return states, rows, manifests


def predict_expert_probabilities(output, records, data_root, checkpoints, metadata, device, counters):
    cache_root = Path(output) / "expert_probability_cache"
    cache_root.mkdir()
    arrays, orders, entries = {}, {}, []
    for seed in range(3):
        rows = sorted((row for domain in range(3) for row in records[seed][domain]["val"]),
                      key=lambda row: row["case_id"])
        require(len(rows) == len({row["case_id"] for row in rows}) == 165, "validation probability coverage changed")
        orders[seed] = [row["case_id"] for row in rows]
        arrays[seed] = {}
        for expert in range(3):
            checkpoint = checkpoints[seed][expert]
            models, payload = load_models_checked(checkpoint, device=device, sources=("student",),
                                                  phase=f"expert_seed{seed}_expert{expert}")
            model = models["student"]
            path = cache_root / f"seed{seed}_expert{expert}.npy"
            target = new_memmap(path, (165, 3, 384, 384), np.float32)
            with ImmutableModels(models, checkpoint, Path(output) / "expert_models" / f"seed{seed}_expert{expert}", metadata):
                with torch.no_grad():
                    for start in range(0, len(rows), BATCH_SIZE):
                        images = v1run.read_images([v1run.image_only(row) for row in rows[start:start + BATCH_SIZE]], data_root).to(device)
                        with torch.autocast(device_type=device.type, enabled=False):
                            logits, _ = model(images, stochastic_classifier=False)
                            probability = logits.float().softmax(1)
                        require(probability.dtype == torch.float32 and tuple(probability.shape[1:]) == (3, 384, 384),
                                "expert probability schema changed")
                        target[start:start + len(images)] = probability.cpu().numpy()
                        counters["expert_probability_forwards"] += 1
                        counters["expert_probability_case_passes"] += len(images)
            target.flush()
            del target
            counters["model_guards"] += 1
            require(all(parameter.grad is None for parameter in model.parameters()), "expert wrote parameter.grad",
                    "BLOCKED_MODEL_MUTATION")
            require_backend(f"expert_seed{seed}_expert{expert}_complete")
            arrays[seed][expert] = np.load(path, mmap_mode="r", allow_pickle=False)
            entries.append(dict(seed=seed, expert=expert, checkpoint_id=checkpoint["checkpoint_id"],
                                source="student", posterior_mean=True, stochastic_classifier=False,
                                case_count=165, cache=path.name))
            del models, payload, model
            torch.cuda.empty_cache()
    seal = d.seal(cache_root)
    d.write_new(Path(output) / "PRES_DSR_SF_EXPERT_PROBABILITY_MANIFEST.json",
                dict(status="PASS_EXPERT_PROBABILITIES_SEALED_BEFORE_GT", entries=entries,
                     case_expert_probabilities=1485, validation_GT_reads=0, content_sha256=seal["content_sha256"],
                     manifest_sha256=d.sha256(cache_root / "PRIVATE_BUNDLE_MANIFEST.json")))
    return arrays, orders


def padded_alpha(alpha, stage):
    output = np.zeros((len(alpha), 3), dtype=np.float64)
    output[:, :stage + 1] = alpha
    return output


def one_hot_routes(routes):
    output = np.zeros((len(routes), 3), dtype=np.float64)
    output[np.arange(len(routes)), np.asarray(routes, dtype=np.int64)] = 1.0
    return output


def materialize_candidates(output, expert_probability, expert_orders, clean_states, m1_temperatures,
                           ridge_states, bootstrap_states, counters):
    cache_root = Path(output) / "candidate_prediction_cache"
    formal_root, boot_root = cache_root / "formal", cache_root / "bootstrap"
    formal_root.mkdir(parents=True)
    boot_root.mkdir()
    formal, boot, formal_entries, boot_entries = {}, {}, [], []
    for seed in range(3):
        formal[seed], boot[seed] = {}, {}
        expert_index = {case: i for i, case in enumerate(expert_orders[seed])}
        for stage in (1, 2):
            ids = ridge_states[seed][stage]["case_ids"]
            positions = np.asarray([expert_index[case] for case in ids], dtype=np.int64)
            truth = ridge_states[seed][stage]["truth"]
            m1 = clean_states[seed][1][stage]
            m2 = clean_states[seed][2][stage]
            alpha = {
                "C0_SHARED": one_hot_routes(np.full(len(ids), 2)),
                "C1_ORACLE": one_hot_routes(truth),
                "C2_M1_HARD": one_hot_routes(m1["routed"]),
                "C3_M2_HARD": one_hot_routes(m2["routed"]),
                "C4_M1_SOFT": padded_alpha(softmax(m1["scores"], m1_temperatures[seed][stage]), stage),
                "C5_RIDGE_HARD": one_hot_routes(ridge_states[seed][stage]["routed"]),
                "C6_RIDGE_SOFT": padded_alpha(ridge_states[seed][stage]["alpha"], stage),
                "C7_UNIFORM": padded_alpha(np.full((len(ids), stage + 1), 1.0 / (stage + 1)), stage),
            }
            formal[seed][stage] = {}
            targets = {policy: new_memmap(formal_root / f"seed{seed}_stage{stage}_{policy}.npy",
                                           (len(ids), 384, 384), np.uint8) for policy in POLICIES}
            for start in range(0, len(ids), 4):
                batch = positions[start:start + 4]
                experts = np.stack([expert_probability[seed][expert][batch] for expert in range(3)], axis=1)
                for policy in POLICIES:
                    mixed = probability_fusion(alpha[policy][start:start + len(batch)], experts)
                    targets[policy][start:start + len(batch)] = mixed.argmax(axis=1).astype(np.uint8)
            for policy, target in targets.items():
                target.flush()
                path = Path(target.filename)
                del target
                formal[seed][stage][policy] = np.load(path, mmap_mode="r", allow_pickle=False)
                formal_entries.append(dict(seed=seed, stage_index=stage, policy=policy, cases=len(ids), cache=path.name,
                                           alpha_sha256=array_sha256(alpha[policy])))
            counters["formal_candidate_case_predictions"] += len(ids) * len(POLICIES)
            boot[seed][stage] = {}
            for replicate in range(5):
                state = bootstrap_states[seed][stage][replicate]
                target = new_memmap(boot_root / f"seed{seed}_stage{stage}_rep{replicate}.npy",
                                    (len(ids), 384, 384), np.uint8)
                boot_alpha = padded_alpha(state["alpha"], stage)
                for start in range(0, len(ids), 4):
                    batch = positions[start:start + 4]
                    experts = np.stack([expert_probability[seed][expert][batch] for expert in range(3)], axis=1)
                    mixed = probability_fusion(boot_alpha[start:start + len(batch)], experts)
                    target[start:start + len(batch)] = mixed.argmax(axis=1).astype(np.uint8)
                target.flush()
                path = Path(target.filename)
                del target
                boot[seed][stage][replicate] = np.load(path, mmap_mode="r", allow_pickle=False)
                boot_entries.append(dict(seed=seed, stage_index=stage, replicate=replicate, cases=len(ids), cache=path.name,
                                         alpha_sha256=array_sha256(boot_alpha)))
                counters["bootstrap_soft_case_predictions"] += len(ids)
    seal = d.seal(cache_root)
    d.write_new(Path(output) / "PRES_DSR_SF_CANDIDATE_OUTPUT_MANIFEST.json",
                dict(status="PASS_ALL_CANDIDATE_OUTPUTS_SEALED_BEFORE_GT", formal=formal_entries,
                     bootstrap=boot_entries, validation_GT_reads=0, probability_fusion=True, logit_fusion=False,
                     content_sha256=seal["content_sha256"],
                     manifest_sha256=d.sha256(cache_root / "PRIVATE_BUNDLE_MANIFEST.json")))
    return formal, boot


def evaluate_segmentation(output, records, data_root, expert_probability, expert_orders, formal, boot,
                          bootstrap_rows, counters):
    cross_rows, policy_rows = [], []
    boot_lookup = {(row["seed"], row["stage_index"], row["replicate"]): row for row in bootstrap_rows}
    for seed in range(3):
        all_rows = sorted((row for domain in range(3) for row in records[seed][domain]["val"]),
                          key=lambda row: row["case_id"])
        require([row["case_id"] for row in all_rows] == expert_orders[seed], "expert/label order changed")
        labels = v1run.read_labels(all_rows, data_root)
        counters["validation_GT_case_reads"] += len(labels)
        global_index = {row["case_id"]: i for i, row in enumerate(all_rows)}
        for domain in range(3):
            positions = [i for i, row in enumerate(all_rows) if row["site_or_vendor"] == DOMAINS[domain]]
            for expert in range(3):
                confusion = np.zeros((3, 3), dtype=np.int64)
                for start in range(0, len(positions), 4):
                    batch = positions[start:start + 4]
                    predicted = np.asarray(expert_probability[seed][expert][batch]).argmax(axis=1)
                    confusion += pixel_confusion(predicted, labels[batch])
                cross_rows.append(dict(seed=seed, true_domain=domain, domain=DOMAINS[domain], expert=expert,
                                       **segmentation_metrics(confusion)))
        for stage in (1, 2):
            ids = sorted(row["case_id"] for row in all_rows if DOMAINS.index(row["site_or_vendor"]) <= stage)
            stage_labels = labels[[global_index[case] for case in ids]]
            truth = np.asarray([DOMAINS.index(all_rows[global_index[case]]["site_or_vendor"]) for case in ids])
            for policy in POLICIES:
                predictions = formal[seed][stage][policy]
                for domain in range(stage + 1):
                    mask = truth == domain
                    policy_rows.append(dict(seed=seed, stage_index=stage, true_domain=domain, domain=DOMAINS[domain],
                                            policy=policy, **segmentation_metrics(pixel_confusion(predictions[mask],
                                                                                                 stage_labels[mask]))))
            for replicate in range(5):
                predictions = boot[seed][stage][replicate]
                soft_metrics = []
                for domain in range(stage + 1):
                    mask = truth == domain
                    soft_metrics.append(segmentation_metrics(pixel_confusion(predictions[mask], stage_labels[mask]))[
                        "mean_foreground_dice"])
                shared = [next(row["mean_foreground_dice"] for row in policy_rows
                               if row["seed"] == seed and row["stage_index"] == stage
                               and row["true_domain"] == domain and row["policy"] == "C0_SHARED")
                          for domain in range(stage + 1)]
                oracle = [next(row["mean_foreground_dice"] for row in policy_rows
                               if row["seed"] == seed and row["stage_index"] == stage
                               and row["true_domain"] == domain and row["policy"] == "C1_ORACLE")
                          for domain in range(stage + 1)]
                row = boot_lookup[(seed, stage, replicate)]
                row["soft_three_domain_gain"] = float(np.mean(np.asarray(soft_metrics) - shared))
                row["soft_oracle_gap"] = float(np.mean(np.asarray(oracle) - soft_metrics))
        del labels
    require(len(cross_rows) == 27 and len(policy_rows) == 120 and counters["validation_GT_case_reads"] == 495,
            "segmentation output coverage changed", "BLOCKED_INCOMPLETE_EVIDENCE")
    v1run.csv_new(Path(output) / "pres_dsr_cross_expert.csv", cross_rows)
    v1run.csv_new(Path(output) / "pres_dsr_soft_fusion.csv", policy_rows)
    return cross_rows, policy_rows, bootstrap_rows


def aggregate_routing(states):
    stages = {}
    for stage in (1, 2):
        per_seed_domain = []
        for seed in range(3):
            truth, routed = states[seed][stage]["truth"], states[seed][stage]["routed"]
            per_seed_domain.append([float(np.mean(routed[truth == domain] == domain)) for domain in range(stage + 1)])
        per_domain = np.mean(per_seed_domain, axis=0).tolist()
        stages[stage] = dict(macro=float(np.mean(per_domain)), per_domain=per_domain)
    return dict(stage1_macro=stages[1]["macro"], stage1_per_domain=stages[1]["per_domain"],
                stage2_macro=stages[2]["macro"], stage2_per_domain=stages[2]["per_domain"])


def aggregate_policy(policy_rows, policy):
    return {(row["seed"], row["true_domain"]): row["mean_foreground_dice"] for row in policy_rows
            if row["stage_index"] == 2 and row["policy"] == policy}


def gate_evidence(policy_rows, ridge_states, bootstrap_rows):
    shared, oracle = aggregate_policy(policy_rows, "C0_SHARED"), aggregate_policy(policy_rows, "C1_ORACLE")
    m1, soft = aggregate_policy(policy_rows, "C2_M1_HARD"), aggregate_policy(policy_rows, "C6_RIDGE_SOFT")
    oracle_gains = [oracle[key] - shared[key] for key in shared]
    seed_oracle = [float(np.mean([oracle[(seed, domain)] - shared[(seed, domain)] for domain in range(3)]))
                   for seed in range(3)]
    oracle_values = dict(three_domain_gain=float(np.mean(oracle_gains)),
                         historical_gain=float(np.mean([oracle[(seed, domain)] - shared[(seed, domain)]
                                                              for seed in range(3) for domain in (0, 1)])),
                         positive_seed_count=sum(value > 0 for value in seed_oracle),
                         maximum_domain_drop=float(max(0.0, max(-value for value in oracle_gains))))
    gains = [soft[key] - shared[key] for key in shared]
    seed_gain = [float(np.mean([soft[(seed, domain)] - shared[(seed, domain)] for domain in range(3)]))
                 for seed in range(3)]
    soft_values = dict(oracle_gap=float(np.mean([oracle[key] - soft[key] for key in shared])),
                       shared_gain=float(np.mean(gains)),
                       historical_gain=float(np.mean([soft[(seed, domain)] - shared[(seed, domain)]
                                                          for seed in range(3) for domain in (0, 1)])),
                       gain_over_m1_hard=float(np.mean([soft[key] - m1[key] for key in shared])),
                       positive_seed_count=sum(value > 0 for value in seed_gain),
                       maximum_seed_domain_drop=float(max(0.0, max(-value for value in gains))),
                       current_domain_drop=float(max(0.0, max(shared[(seed, 2)] - soft[(seed, 2)] for seed in range(3)))))
    by_replicate = {}
    for replicate in range(5):
        rows = [row for row in bootstrap_rows if row["stage_index"] == 2 and row["replicate"] == replicate]
        by_replicate[replicate] = dict(hard=float(np.mean([row["hard_macro_accuracy"] for row in rows])),
                                       gain=float(np.mean([row["soft_three_domain_gain"] for row in rows])),
                                       gap=float(np.mean([row["soft_oracle_gap"] for row in rows])))
    stability = dict(hard_macro_p10=float(np.quantile([v["hard"] for v in by_replicate.values()], .1, method="linear")),
                     soft_gain_p10=float(np.quantile([v["gain"] for v in by_replicate.values()], .1, method="linear")),
                     soft_oracle_gap_p90=float(np.quantile([v["gap"] for v in by_replicate.values()], .9, method="linear")),
                     all_domains_nonempty=True, all_finite=True, aggregate_replicates=by_replicate)
    return oracle_values, aggregate_routing(ridge_states), soft_values, stability


def control_reproduction(routing, clean_bootstrap, stability, banks, policy_rows):
    strategies = []
    names = {"C0_SHARED": "Shared-final", "C1_ORACLE": "Oracle-snapshot",
             "C2_M1_HARD": "Prototype-routed-M1", "C3_M2_HARD": "Prototype-routed-M2"}
    for row in policy_rows:
        if row["stage_index"] == 2 and row["policy"] in names:
            strategies.append(dict(seed=row["seed"], true_domain=row["true_domain"],
                                   strategy=names[row["policy"]], mean_foreground_dice=row["mean_foreground_dice"]))
    d1, candidates = v1run.gate_inputs(routing, clean_bootstrap, stability, banks, strategies)
    observed = {
        "D1": {key: d1[key] for key in ("three_domain_gain", "historical_gain", "positive_seed_count", "maximum_domain_drop")},
        "1": dict(stage1_macro=candidates[1]["routing"]["stage1_macro"],
                  stage1_per_domain=candidates[1]["routing"]["stage1_per_domain"],
                  stage2_macro=candidates[1]["routing"]["stage2_macro"],
                  stage2_per_domain=candidates[1]["routing"]["stage2_per_domain"],
                  oracle_gap=candidates[1]["segmentation"]["oracle_gap"],
                  shared_gain=candidates[1]["segmentation"]["shared_gain"],
                  historical_gain=candidates[1]["segmentation"]["historical_gain"],
                  positive_seed_count=candidates[1]["segmentation"]["positive_seed_count"],
                  maximum_domain_drop=candidates[1]["segmentation"]["maximum_domain_drop"],
                  bootstrap_macro_p10=candidates[1]["stability"]["bootstrap_macro_p10"],
                  prototype_cosine_median=candidates[1]["stability"]["prototype_cosine_median"]),
        "2": dict(stage1_macro=candidates[2]["routing"]["stage1_macro"],
                  stage1_per_domain=candidates[2]["routing"]["stage1_per_domain"],
                  stage2_macro=candidates[2]["routing"]["stage2_macro"],
                  stage2_per_domain=candidates[2]["routing"]["stage2_per_domain"],
                  oracle_gap=candidates[2]["segmentation"]["oracle_gap"],
                  shared_gain=candidates[2]["segmentation"]["shared_gain"],
                  historical_gain=candidates[2]["segmentation"]["historical_gain"],
                  positive_seed_count=candidates[2]["segmentation"]["positive_seed_count"],
                  maximum_domain_drop=candidates[2]["segmentation"]["maximum_domain_drop"],
                  bootstrap_macro_p10=candidates[2]["stability"]["bootstrap_macro_p10"],
                  matched_cosine_median=candidates[2]["stability"]["matched_cosine_median"],
                  minimum_occupancy=min(candidates[2]["stability"]["occupancies"])),
    }
    path = ROOT / "docs/pres_jascl_v0_1/PRES_JASCL_STATUS.json"
    require(d.sha256(path) == V1_STATUS_SHA, "V0.1 public control changed", "BLOCKED_CONTROL_REPRODUCTION_MISMATCH")
    reference_status = d.read(path)
    reference = {"D1": reference_status["D1_observations"],
                 "1": reference_status["candidate_observations"]["1"],
                 "2": reference_status["candidate_observations"]["2"]}
    differences = []
    for group in ("D1", "1", "2"):
        for key, expected in reference[group].items():
            if key not in observed[group]:
                continue
            actual = observed[group][key]
            if isinstance(expected, list):
                differences.extend(np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)).tolist())
            else:
                differences.append(abs(float(actual) - float(expected)))
    maximum = max(differences)
    require(maximum <= 1e-6, f"clean control difference {maximum} exceeds 1e-6",
            "BLOCKED_CONTROL_REPRODUCTION_MISMATCH")
    return dict(status="PASS_CLEAN_M1_M2_REPRODUCTION", maximum_absolute_metric_difference=maximum,
                tolerance=1e-6, observed=observed, reference=reference)


def model_immutability(output, counters):
    guards = [d.read(path) for path in sorted(Path(output).glob("*_models/**/immutability/*.json"))]
    return len(guards) == counters["model_guards"] == 12 and all(
        row["bitwise_unchanged"] and row["extraction_completed"] for row in guards)


def report(output, metadata, counters, decision, evidence, control, memory_rows, backend_checks):
    status = dict(metadata=metadata, **decision, evidence=evidence, clean_control=control, counters=counters,
                  backend_checks=backend_checks, memory_rows_per_domain=memory_rows,
                  model_optimizer_steps=0, model_autograd_calls=0, model_backward_calls=0,
                  parameter_grad_writes=0, router_is_closed_form=True, method_registered=False,
                  training_launched=False, validation_GT_usage="evaluator_only", test_GT_reads=0,
                  report_commit=None, report_commit_resolution="first Git commit adding these exact public report bytes")
    d.write_new(Path(output) / "PRES_DSR_SF_STATUS.json", status)
    lines = ["# PRES-DSR-SF V0.2 final report", "",
             f"Scientific status: `{decision['scientific_status']}`.", "",
             "## E1-E6", "",
             f"E1={decision['E1']}; E2={decision['E2']}; E3={decision['E3']}; E4={decision['E4']}; E5={decision['E5']}; E6={decision['E6']}.", "",
             f"Oracle value: {evidence['oracle']}.", "",
             f"Ridge-hard routing: {evidence['ridge_hard']}.", "",
             f"Primary ridge-soft segmentation: {evidence['ridge_soft']}.", "",
             f"Bootstrap stability: {evidence['stability']}.", "",
             "## Isolation and coverage", "",
             f"The frozen graph completed {counters['descriptor_forwards']} descriptor forwards over {counters['descriptor_case_passes']} cases and {counters['expert_probability_forwards']} expert forwards over {counters['expert_probability_case_passes']} case-expert passes. All {counters['model_guards']} model guards passed. Validation GT was read for {counters['validation_GT_case_reads']} cases only after all candidate outputs were sealed.", "",
             f"Clean M1/M2 reproduction maximum absolute difference was {control['maximum_absolute_metric_difference']:.12g} (limit 1e-6).", "",
             "Model optimizer, autograd, backward, parameter-grad-write, EMA/GAS/PAS-update and training counts are zero. Router fitting was closed-form CPU float64. No test object or GT was constructed.", "",
             "Execution hard-stops for independent review. No test evaluation, regeneration, retraining, validation refit, expert fine-tuning, adapter, other benchmark, sweep, or main merge is authorized.", ""]
    v1run.text_new(Path(output) / "PRES_DSR_SF_FINAL_REPORT.md", "\n".join(lines))
    warnings = ["# PRES-DSR-SF failures and warnings", "", f"Final status: `{decision['scientific_status']}`.", ""]
    if decision["scientific_status"] == "PASS_PRES_DSR_SF_FEASIBILITY":
        warnings.append("PASS is validation feasibility only; it is not a trained-method or test-set claim.")
    else:
        warnings.append("The first failed registered gate determines the scientific failure; controls cannot rescue C6.")
    warnings.extend(["", "Private paths, case IDs, descriptors, probabilities, masks, checkpoints, and raw CSV rows are omitted from public reporting.", ""])
    v1run.text_new(Path(output) / "PRES_DSR_SF_FAILURES_AND_WARNINGS.md", "\n".join(warnings))
    v1run.text_new(Path(output) / "PRES_DSR_SF_EXACT_COMMANDS.md", "\n".join([
        "# PRES-DSR-SF exact commands", "", "## Tests", "", "```sh", metadata["exact_test_command"], "```", "",
        "## Durable child argv", "", "```sh", shlex.join(metadata["exact_command"]), "```", "",
        "Both commands ran through the NAS storage wrapper. No package installation, optimizer, backward, or training command ran.", ""]))


def artifact_manifest(output):
    excluded = ("PRES_DSR_SF_ARTIFACT_MANIFEST.json", "controller.log", "supervisor.log", "LAUNCH_REQUEST.json",
                "LAUNCH_RECEIPT.json", "PROCESS_START.json", "PROCESS_PID.json", "PROCESS_EXIT.json",
                "EXECUTION_COMPLETION.json", "PHASE_pres_dsr_sf.json", "PHASE_pres_dsr_sf_MANIFEST.json")
    entries = d.file_entries(output, exclude=excluded)
    required = {"PRES_DSR_SF_INPUT_AUDIT.json", "PRES_DSR_SF_BACKEND_AUDIT.json", "PRES_DSR_SF_CALL_GRAPH.json",
                "PRES_DSR_SF_DESCRIPTOR_MANIFEST.json", "PRES_DSR_SF_MEMORY_MANIFEST.json",
                "PRES_DSR_SF_ROUTER_MANIFEST.json", "pres_dsr_cv.csv", "pres_dsr_router_scores.csv",
                "pres_dsr_routing_confusion.csv", "pres_dsr_cross_expert.csv", "pres_dsr_soft_fusion.csv",
                "pres_dsr_bootstrap.csv", "pres_dsr_memory_cost.csv", "PRES_DSR_SF_STATUS.json",
                "PRES_DSR_SF_FINAL_REPORT.md", "PRES_DSR_SF_FAILURES_AND_WARNINGS.md",
                "PRES_DSR_SF_EXACT_COMMANDS.md", "pytest.xml", "pytest_output.txt"}
    names = {entry["path"] for entry in entries}
    require(required.issubset(names), "required artifact missing", "BLOCKED_INCOMPLETE_EVIDENCE")
    result = dict(status="PASS_CONTROLLER_ARTIFACT_MANIFEST", artifacts=entries, file_count=len(entries),
                  total_bytes=sum(row["bytes"] for row in entries), required_outputs_complete=True,
                  excludes_live_supervisor_receipts=True, created_at=d.now())
    d.write_new(Path(output) / "PRES_DSR_SF_ARTIFACT_MANIFEST.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--test-report", required=True, type=Path)
    args = parser.parse_args()
    try:
        require(args.output.is_dir(), "durable create-only output was not initialized")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)
        metadata = execution_gate(args.output, args.code_commit, args.test_report)
        metadata["exact_command"] = sys.argv
        backend = backend_import_gate(args.output)
        d.write_new(args.output / "PRES_DSR_SF_RUN_METADATA.json", metadata)
        v1run.link_test_evidence(args.output, args.test_report)
        contract = gate1c_contract()
        records = input_audit(args.output, contract, metadata)
        graph = compile_call_graph(args.output, records, args.code_commit)
        data_root = Path(contract["destination"]["data_root"])
        router_checkpoints = {seed: v1run.checkpoint(contract, seed, 0) for seed in range(3)}
        expert_checkpoints = {seed: {expert: v1run.checkpoint(contract, seed, expert) for expert in range(3)}
                              for seed in range(3)}
        counters = dict(descriptor_forwards=0, descriptor_case_passes=0, expert_probability_forwards=0,
                        expert_probability_case_passes=0, ridge_closed_form_fits=0, m1_cv_prototype_fits=0,
                        clean_control_prototype_fits=0, bootstrap_operations=0,
                        clean_control_bootstrap_operations=0, model_guards=0,
                        formal_candidate_case_predictions=0, bootstrap_soft_case_predictions=0,
                        validation_GT_case_reads=0,
                        output_rows=dict(cv=0, router_scores=0, routing_confusion=0, cross_expert=0,
                                         soft_fusion=0, bootstrap=0, memory_cost=0))
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one visible CUDA GPU required")
        device = torch.device("cuda:0")
        backend_checks = [require_backend("before_real_forward")]
        with isolation_guard():
            plans = {seed: descriptor_plan(records, seed) for seed in range(3)}
            descriptors = extract_descriptors(args.output, plans, data_root, router_checkpoints, metadata, device, counters)
            backend_checks.append(require_backend("after_descriptors"))
            released = release_metadata(args.output, records, plans)
            memories, memory_cost = build_memory(args.output, descriptors, released, data_root)
            banks, clean_states, clean_routing, m1_temperatures, m1_cv = clean_controls(descriptors, released, counters)
            clean_bootstrap, clean_stability = clean_control_bootstraps(descriptors, released, banks, counters)
            ridge_models, ridge_states, ridge_cv, router_scores, routing_confusion, router_manifest = fit_ridge_routers(
                memories, descriptors, released, clean_states, m1_temperatures, args.output, counters)
            bootstrap_states, bootstrap_rows, bootstrap_manifest = fit_ridge_bootstraps(
                memories, descriptors, released, counters)
            d.write_new(args.output / "PRES_DSR_SF_ROUTER_MANIFEST.json",
                        dict(status="PASS_ROUTER_AND_SELECTION_SEALED_BEFORE_GT", formal=router_manifest,
                             bootstrap=bootstrap_manifest, ridge_closed_form=True, CPU_float64=True,
                             validation_used_for_lambda_or_temperature=False, segmentation_GT_fields=0,
                             M1_temperature_train_unlabeled_only=True))
            probabilities, orders = predict_expert_probabilities(args.output, records, data_root, expert_checkpoints,
                                                                  metadata, device, counters)
            backend_checks.append(require_backend("after_expert_probabilities"))
            formal_predictions, bootstrap_predictions = materialize_candidates(
                args.output, probabilities, orders, clean_states, m1_temperatures, ridge_states, bootstrap_states, counters)
        backend_checks.append(require_backend("after_all_forwards"))
        cross_rows, policy_rows, bootstrap_rows = evaluate_segmentation(
            args.output, records, data_root, probabilities, orders, formal_predictions, bootstrap_predictions,
            bootstrap_rows, counters)
        all_bootstrap_rows = [*clean_bootstrap, *bootstrap_rows]
        v1run.csv_new(args.output / "pres_dsr_bootstrap.csv", all_bootstrap_rows)
        control = control_reproduction(clean_routing, clean_bootstrap, clean_stability, banks, policy_rows)
        oracle, ridge_hard, ridge_soft, stability = gate_evidence(policy_rows, ridge_states, bootstrap_rows)
        memory_counts = [len(memories[seed][domain]["case_ids"]) for seed in range(3) for domain in range(3)]
        counters["output_rows"].update(cv=len(m1_cv) + len(ridge_cv), router_scores=len(router_scores),
                                       routing_confusion=len(routing_confusion), cross_expert=len(cross_rows),
                                       soft_fusion=len(policy_rows), bootstrap=len(all_bootstrap_rows),
                                       memory_cost=len(memory_cost))
        counters["total_output_rows"] = sum(counters["output_rows"].values())
        immutable = model_immutability(args.output, counters)
        E1 = all(all(row["state"].values()) for row in backend_checks) and immutable and (
            control["maximum_absolute_metric_difference"] <= 1e-6)
        E6 = (max(memory_counts) <= 512 and counters["validation_GT_case_reads"] == 495 and immutable
              and counters["model_guards"] == 12)
        evidence = dict(E1=E1, E6=E6, oracle=oracle, ridge_hard=ridge_hard,
                        ridge_soft=ridge_soft, stability=stability)
        decision = adjudicate(evidence)
        verify_call_graph(graph, counters)
        report(args.output, metadata, counters, decision, evidence, control, memory_counts, backend_checks)
        manifest = artifact_manifest(args.output)
        print(json.dumps(dict(status=decision["scientific_status"], artifacts=manifest["file_count"],
                              counters=counters), sort_keys=True), flush=True)
    except BaseException as error:
        status = getattr(error, "status", "BLOCKED_INCOMPLETE_EVIDENCE")
        allowed = {"BLOCKED_BASE_COMMIT_AMBIGUOUS", "BLOCKED_BACKEND_STATE_MUTATION",
                   "BLOCKED_CONTROL_REPRODUCTION_MISMATCH", "BLOCKED_PRIVATE_INPUT_MISMATCH",
                   "BLOCKED_PROTOCOL_OR_LEAKAGE", "BLOCKED_MODEL_MUTATION",
                   "BLOCKED_NUMERICAL_FAILURE", "BLOCKED_INCOMPLETE_EVIDENCE"}
        if status not in allowed:
            status = "BLOCKED_PROTOCOL_OR_LEAKAGE"
        failure = args.output / f"PRES_DSR_SF_FAILURE_{os.getpid()}.json"
        if args.output.is_dir() and not failure.exists():
            d.write_new(failure, dict(status=status, error=f"{type(error).__name__}: {error}",
                                      traceback=traceback.format_exc(), command=sys.argv, recorded_at=d.now(),
                                      new_attempt_authorized=False, model_optimizer_steps=0, model_autograd_calls=0,
                                      method_registered=False, training_launched=False))
        raise


if __name__ == "__main__":
    main()
