#!/usr/bin/env python3
"""Derived-artifact-only, non-adjudicative audit of RC-SHOR V0.5."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


EXPECTED = {"C3": 0.21708218085929243, "C4": 0.11330836336206518,
            "C5": 0.02074762773975747, "C6": 0.0}
POLICIES = ("C3", "C4", "C5", "C6")


class AuditBlocked(RuntimeError):
    status = "BLOCKED_V0_5_DERIVED_ARTIFACT_INCONSISTENCY"


def require(condition, message):
    if not condition:
        raise AuditBlocked(message)


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
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


class VerifiedBundle:
    def __init__(self, root):
        self.root = Path(root)
        path = self.root / "RC_SHOR_V0_5_PRIVATE_MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        entries = manifest.get("entries", [])
        require(len(entries) == manifest.get("files"), "private manifest file count mismatch")
        require(sum(row["bytes"] for row in entries) == manifest.get("bytes"),
                "private manifest byte count mismatch")
        require(canonical_hash(entries) == manifest.get("content_sha256"),
                "private manifest content seal mismatch")
        paths = [row["path"] for row in entries]
        require(len(paths) == len(set(paths)) and all(Path(p).as_posix() == p and not Path(p).is_absolute()
                and ".." not in Path(p).parts for p in paths), "unsafe or duplicate manifest path")
        self.manifest = manifest
        self.entries = {row["path"]: row for row in entries}
        self.verified = {}

    def verify(self, relative):
        require(relative in self.entries, "unsealed derived artifact: " + relative)
        path = self.root / relative
        row = self.entries[relative]
        require(path.is_file() and not path.is_symlink(), "missing or symlink artifact: " + relative)
        require(path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"],
                "derived artifact SHA mismatch: " + relative)
        self.verified[relative] = dict(row)
        return path

    def json(self, relative):
        return json.loads(self.verify(relative).read_text(encoding="utf-8"))

    def npz(self, relative, keys):
        path = self.verify(relative)
        with np.load(path, allow_pickle=False) as archive:
            require(set(keys) <= set(archive.files), "missing array in " + relative)
            return {key: archive[key] for key in keys}


def aligned_rows(case_rows, policy, count, sort=True):
    rows = [row for row in case_rows if row["policy"] == policy]
    require(len(rows) == count, "%s row count mismatch" % policy)
    if sort:
        rows.sort(key=lambda row: row["row_index"])
    require([row["row_index"] for row in rows] == list(range(count)),
            "%s route is not in global row_index order" % policy)
    return rows


def balanced_mean(values, seeds, domains, selected_domains=(0, 1, 2)):
    groups = []
    for seed in range(3):
        for domain in selected_domains:
            selected = (seeds == seed) & (domains == domain)
            require(selected.any(), "empty seed/domain metric group")
            groups.append(float(np.mean(values[selected])))
    return float(np.mean(groups))


def route_summary(routes, utility_fg, utility_class, seeds, domains):
    routes = np.asarray(routes, dtype=np.int64)
    require(routes.shape == (len(seeds),), "route length mismatch")
    delta = np.zeros(len(routes), dtype=np.float64)
    class_delta = np.zeros((len(routes), 2), dtype=np.float64)
    for historical in (0, 1):
        selected = routes == historical
        delta[selected] = utility_fg[selected, historical]
        class_delta[selected] = utility_class[selected, historical]
    selected = routes < 2
    selected_utility = delta[selected]
    beneficial_historical = (domains < 2) & (np.max(utility_fg, axis=1) > 0)
    precision_num = int(np.sum(selected_utility > 0))
    precision_den = int(selected.sum())
    current_gain = balanced_mean(delta, seeds, domains, (2,))
    group_gain = [float(np.mean(delta[(seeds == seed) & (domains == domain)]))
                  for seed in range(3) for domain in range(3)]
    current_class = [balanced_mean(class_delta[:, c], seeds, domains, (2,)) for c in range(2)]
    return {
        "three_domain_gain": balanced_mean(delta, seeds, domains),
        "historical_gain": balanced_mean(delta, seeds, domains, (0, 1)),
        "REFUGE_gain": balanced_mean(delta, seeds, domains, (0,)),
        "RIM_ONE_r3_gain": balanced_mean(delta, seeds, domains, (1,)),
        "current_domain_drop": max(0.0, -current_gain),
        "maximum_current_class_drop": max([0.0] + [-value for value in current_class]),
        "maximum_seed_domain_drop": max([0.0] + [-value for value in group_gain]),
        "route_frequency": float(np.mean(selected)),
        "route_count": precision_den,
        "route_precision": None if precision_den == 0 else precision_num / precision_den,
        "route_precision_numerator": precision_num,
        "route_precision_denominator": precision_den,
        "historical_recall": (float(np.mean(selected[beneficial_historical]))
                              if beneficial_historical.any() else None),
        "historical_recall_numerator": int(np.sum(selected & beneficial_historical)),
        "historical_recall_denominator": int(beneficial_historical.sum()),
    }


def route_decisions(score, confidence, epsilon, rho, ood):
    """One decision rule shared by final and policy-realization routes."""
    chosen = np.argmax(score, axis=1)
    route = np.full(len(score), 2, dtype=np.int64)
    for index, historical in enumerate(chosen):
        if (np.isfinite(score[index]).all() and score[index, historical] > epsilon
                and confidence[index, historical] >= rho and not ood[index, historical]):
            route[index] = int(historical)
    return route


def full_policy_route(lcb, feasible, support, epsilon, rho, ood, replicate=None):
    """C6 route or one full-policy realization; both retain the same gates."""
    lcb = np.asarray(lcb, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)
    support = np.asarray(support)
    count = lcb.shape[1]
    if feasible.sum() < 90 or np.sum(feasible & (support >= 15)) < 90:
        return np.full(count, 2, dtype=np.int64)
    if replicate is not None:
        if not feasible[replicate]:
            return np.full(count, 2, dtype=np.int64)
        score = lcb[replicate]
        confidence = np.zeros_like(score)
        active = np.isfinite(score).all(axis=1)
        confidence[np.flatnonzero(active), np.argmax(score[active], axis=1)] = 1.0
        return route_decisions(score, confidence, epsilon, rho, ood)
    active = feasible[:, None] & np.isfinite(lcb).all(axis=2)
    score = np.full((count, 2), np.nan)
    confidence = np.zeros((count, 2))
    for index in range(count):
        valid = active[:, index]
        if not valid.any():
            continue
        score[index] = np.median(lcb[valid, index], axis=0)
        winners = np.argmax(lcb[valid, index], axis=1)
        for historical in (0, 1):
            confidence[index, historical] = np.sum(
                (winners == historical) & (lcb[valid, index, historical] > epsilon)) / 100.0
    return route_decisions(score, confidence, epsilon, rho, ood)


def ungated_ensemble_route(lcb, feasible, epsilon, rho, ood):
    """Counterfactual candidate-local route used only to expose early-return degeneracy."""
    lcb = np.asarray(lcb)
    count = lcb.shape[1]
    score = np.full((count, 2), np.nan)
    confidence = np.zeros((count, 2))
    for index in range(count):
        valid = feasible & np.isfinite(lcb[:, index]).all(axis=1)
        if not valid.any():
            continue
        score[index] = np.median(lcb[valid, index], axis=0)
        winners = np.argmax(lcb[valid, index], axis=1)
        for historical in (0, 1):
            confidence[index, historical] = np.sum(
                (winners == historical) & (lcb[valid, index, historical] > epsilon)) / 100.0
    return route_decisions(score, confidence, epsilon, rho, ood)


def old_auxiliary_draw_routes(lcb, feasible, epsilon):
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


def feasibility_report(feasible_arrays):
    sets = [set(np.flatnonzero(np.asarray(value, dtype=bool)).tolist()) for value in feasible_arrays]
    intersection = sorted(set.intersection(*sets)) if sets else []
    return {"per_fold": [len(value) for value in sets],
            "global_replicate_index_intersection": len(intersection),
            "intersection_indices": intersection}


def model_ood(model, features):
    z = (features.transpose(1, 0, 2) - model["base_mean"][:, None, :]) / model["base_scale"][:, None, :]
    return np.max(np.abs(z), axis=2).T > 8.0


def load_case_rows(bundle):
    path = bundle.verify("case_metrics.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_candidates(bundle):
    path = bundle.verify("calibration_curves.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_foreground(case_rows, policy, seeds, domains):
    rows = aligned_rows(case_rows, policy, len(seeds))
    return balanced_mean(np.asarray([row["foreground_dice"] for row in rows]), seeds, domains)


def design_fold(bundle, fold, folds, metadata, features, candidates):
    train_indices = np.flatnonzero(folds != fold)
    eval_indices = np.flatnonzero(folds == fold)
    boot = bundle.npz("bootstrap_draws/fold%d.npz" % fold,
                      ("multiplicity", "support", "feasible", "train_oof_lcb", "eval_lcb"))
    routes = bundle.npz("routes/fold%d.npz" % fold,
                        ("C3", "C4", "C5", "C6", "base_lcb", "median_lcb", "consensus", "ood"))
    model = bundle.npz("model_states/fold%d.npz" % fold,
                       ("base_mean", "base_scale", "selected_lambdas"))
    conformal = bundle.json("conformal_states/fold%d.json" % fold)
    bundle.json("candidate_seals/fold%d.json" % fold)
    multiplicity = boot["multiplicity"]
    feasible = boot["feasible"]
    train_lcb = boot["train_oof_lcb"]
    require(multiplicity.shape == train_lcb.shape[:2] == (100, len(train_indices)),
            "fold %d training shape mismatch" % fold)
    train = [metadata[index] for index in train_indices]
    train_seeds = np.asarray([row["seed"] for row in train])
    train_domains = np.asarray([row["domain_index"] for row in train])
    train_patients = np.asarray([row["patient_id"] for row in train])
    cell_counts = []
    for seed in range(3):
        for domain in range(3):
            mask = (train_seeds == seed) & (train_domains == domain)
            for historical in (0, 1):
                fit_ok = []
                for replicate in range(100):
                    active = mask & (multiplicity[replicate] > 0)
                    fit_ok.append(bool(active.any() and feasible[replicate]
                                       and np.isfinite(train_lcb[replicate, active, historical]).all()))
                cell_counts.append({
                    "seed": seed, "domain_index": domain, "historical_expert": historical,
                    "rows": int(mask.sum()), "unique_patients": len(set(train_patients[mask].tolist())),
                    "feasible_replicates": int(sum(fit_ok)), "denominator": 100,
                    "feasible_rate": float(np.mean(fit_ok)),
                })
    inactive_finite = []
    for index in range(len(train_indices)):
        for historical in (0, 1):
            inactive_finite.append(int(np.sum((multiplicity[:, index] == 0)
                                                & np.isfinite(train_lcb[:, index, historical]))))
    selected = conformal["selected_candidate"]
    epsilon, rho = selected["epsilon"], selected["rho"]
    auxiliary = old_auxiliary_draw_routes(boot["eval_lcb"], feasible, epsilon)
    differences = np.sum(auxiliary != routes["C6"][None, :], axis=1)
    votes = np.zeros((len(eval_indices), 2))
    for replicate in np.flatnonzero(feasible):
        score = boot["eval_lcb"][replicate]
        active = np.isfinite(score).all(axis=1)
        chosen = np.argmax(score[active], axis=1)
        best = score[np.flatnonzero(active), chosen]
        accepted = best > epsilon
        for index, historical in zip(np.flatnonzero(active)[accepted], chosen[accepted]):
            votes[index, historical] += 1
    votes /= 100.0
    proper = np.stack([full_policy_route(boot["eval_lcb"], feasible, boot["support"], epsilon,
                                        rho, routes["ood"], replicate=r) for r in range(100)])
    require(np.array_equal(proper, np.broadcast_to(routes["C6"], proper.shape)),
            "full-policy realization differs from final C6")
    fold_candidates = [row for row in candidates if int(row["fold"]) == fold]
    require(len(fold_candidates) == 9, "fold candidate count mismatch")
    base_ood = model_ood(model, features[train_indices])
    before, after = {}, {}
    for row in fold_candidates:
        candidate_id = row["candidate_id"]
        route = ungated_ensemble_route(train_lcb, feasible, float(row["epsilon"]), float(row["rho"]), base_ood)
        before[candidate_id] = {"route_sha256": array_hash(route),
                                "route_count": int(np.sum(route < 2))}
        gated = full_policy_route(train_lcb, feasible, boot["support"], float(row["epsilon"]),
                                  float(row["rho"]), base_ood)
        after[candidate_id] = {"route_sha256": array_hash(gated),
                               "route_count": int(np.sum(gated < 2))}
    group_sizes = []
    for seed in range(3):
        for domain in range(3):
            size = int(np.sum((train_seeds == seed) & (train_domains == domain)))
            higher_index = math.ceil(0.90 * (size - 1))
            group_sizes.append({"seed": seed, "domain_index": domain, "rows": size,
                                "higher_q_is_group_max": higher_index == size - 1})
    seed_domain_counts = []
    for seed in range(3):
        for domain in range(3):
            mask = (train_seeds == seed) & (train_domains == domain)
            seed_domain_counts.append({"seed": seed, "domain_index": domain, "rows": int(mask.sum()),
                                       "unique_patients": len(set(train_patients[mask].tolist()))})
    unique_train = len(set(train_patients.tolist()))
    return {
        "fold": fold,
        "inner_training_rows": len(train_indices), "inner_training_unique_patients": unique_train,
        "outer_rows": len(eval_indices),
        "outer_unique_patients": len(set(metadata[index]["patient_id"] for index in eval_indices)),
        "seed_domain_counts": seed_domain_counts,
        "feature_dimension": int(features.shape[2]),
        "feature_dimension_per_unique_patient": features.shape[2] / unique_train,
        "conformal_group_sizes": group_sizes,
        "conformal": [{"historical_expert": row["historical_expert"],
                        "max_over_nine_q": row["q"], "group_q": row["group_q"]}
                       for row in conformal["calibration"]],
        "selected_lambdas": model["selected_lambdas"].tolist(),
        "C5_route_count": int(np.sum(routes["C5"] < 2)),
        "C5_route_denominator": len(routes["C5"]),
        "C5_route_coverage": float(np.mean(routes["C5"] < 2)),
        "feasible_replicates": int(feasible.sum()),
        "unit_feasibility": cell_counts,
        "inactive_finite_prediction_count_distribution": dict(sorted(Counter(inactive_finite).items())),
        "theoretical_max_consensus": float(feasible.sum() / 100.0),
        "actual_max_auxiliary_consensus": float(votes.max(initial=0.0)),
        "final_C6_vs_auxiliary_single_draw": {
            "replicates_with_any_difference": int(np.sum(differences > 0)),
            "difference_count_min": int(differences.min()),
            "difference_count_median": float(np.median(differences)),
            "difference_count_max": int(differences.max()),
        },
        "route_from_ensemble_early_returns": {"candidate_comparison": 9, "final_C6": 1, "total": 10},
        "candidate_routes_before_global_gate": before,
        "candidate_routes_after_global_gate": after,
        "unique_candidate_routes_before_global_gate": len({row["route_sha256"] for row in before.values()}),
        "unique_candidate_routes_after_global_gate": len({row["route_sha256"] for row in after.values()}),
    }


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)


def audit(root, output):
    bundle = VerifiedBundle(root)
    case_rows = load_case_rows(bundle)
    utility = bundle.npz("utility_targets/outer_oof.npz", ("utility_fg", "utility_class", "expert_fg"))
    assignment = bundle.json("cross_fit_assignments.json")
    candidates = read_candidates(bundle)
    feature = bundle.npz("feature_cache/RC_SHOR_FEATURES.npz", ("features",))
    count = len(utility["utility_fg"])
    metadata = aligned_rows(case_rows, "C0", count)
    seeds = np.asarray([row["seed"] for row in metadata], dtype=np.int64)
    domains = np.asarray([row["domain_index"] for row in metadata], dtype=np.int64)
    folds = np.asarray(assignment["folds"], dtype=np.int64)
    require(folds.shape == (count,), "fold assignment length mismatch")

    corrected, old = {}, {}
    for policy in POLICIES:
        route = np.asarray([row["route"] for row in aligned_rows(case_rows, policy, count)])
        corrected[policy] = route_summary(route, utility["utility_fg"], utility["utility_class"], seeds, domains)
        fold_order = [row for row in case_rows if row["policy"] == policy]
        old[policy] = route_summary(np.asarray([row["route"] for row in fold_order]),
                                    utility["utility_fg"], utility["utility_class"], seeds, domains)
        require(abs(corrected[policy]["three_domain_gain"] - EXPECTED[policy]) < 1e-15,
                "%s corrected gain invariant failed" % policy)
    require(abs(corrected["C3"]["historical_gain"] - 0.3256232712889387) < 1e-15,
            "C3 historical gain invariant failed")
    require(corrected["C3"]["current_domain_drop"] == 0.0, "C3 current drop invariant failed")
    c0_metric = metric_foreground(case_rows, "C0", seeds, domains)
    c3_metric = metric_foreground(case_rows, "C3", seeds, domains)
    require(abs((c3_metric - c0_metric) - EXPECTED["C3"]) < 1e-15,
            "C3 metric difference does not equal routing gain")
    c0_current = [row for row in aligned_rows(case_rows, "C0", count) if row["domain_index"] == 2]
    c3_current = [row for row in aligned_rows(case_rows, "C3", count) if row["domain_index"] == 2]
    require([r["foreground_dice"] for r in c0_current] == [r["foreground_dice"] for r in c3_current],
            "current-domain C3 is not C0")

    fold_reports = [design_fold(bundle, fold, folds, metadata, feature["features"], candidates)
                    for fold in range(5)]
    feasible_sets = []
    for fold in range(5):
        data = bundle.npz("bootstrap_draws/fold%d.npz" % fold, ("feasible",))
        feasible_sets.append(set(np.flatnonzero(data["feasible"]).tolist()))
    feasibility = feasibility_report([np.asarray([index in value for index in range(100)])
                                      for value in feasible_sets])
    intersection = feasibility["intersection_indices"]
    require(feasibility["per_fold"] == [59, 38, 74, 83, 44],
            "per-fold feasible counts changed")
    require(len(intersection) == 5, "replicate-index intersection changed")
    stability_fields = ("historical_gain_p10", "current_domain_drop_p90",
                        "maximum_seed_domain_drop_p90", "shared_gain_p10", "feasible_replicates")
    rho_same = {}
    for fold in range(5):
        rows = [row for row in candidates if int(row["fold"]) == fold]
        for epsilon in sorted({row["epsilon"] for row in rows}):
            subset = [row for row in rows if row["epsilon"] == epsilon]
            rho_same["fold%d_epsilon%s" % (fold, epsilon)] = all(
                len({row[field] for row in subset}) == 1 for field in stability_fields)
    require(all(rho_same.values()), "old rho-invariance audit changed")

    routing_order = {
        "bug_confirmed": True,
        "old_route_order": "outer-fold append order",
        "utility_order": "global row_index order",
        "mismatched_positions": int(sum(row["row_index"] != i for i, row in enumerate(
            [row for row in case_rows if row["policy"] == "C3"]))),
        "old_summaries": old,
        "corrected_summaries": corrected,
        "metric_check": {"C0_overall_foreground_dice": c0_metric,
                         "C3_overall_foreground_dice": c3_metric,
                         "difference": c3_metric - c0_metric},
    }
    stability = {
        "bug_confirmed": True,
        "auxiliary_draw_uses_rho": False,
        "auxiliary_draw_applies_feasible_ge_90_gate": False,
        "auxiliary_draw_uses_full_ensemble_consensus": False,
        "auxiliary_draw_is_final_C6_bootstrap_realization": False,
        "per_fold_feasible": feasibility["per_fold"],
        "replicate_index_intersection": {"numerator": len(intersection), "denominator": 100,
                                         "indices": intersection,
                                         "meaning": "outer-fold replicate-index intersection; not per-unit feasibility"},
        "rho_candidates_have_identical_reported_stability": rho_same,
        "full_policy_realizations_match_final_C6": True,
        "folds": [{key: row[key] for key in ("fold", "unit_feasibility",
                  "inactive_finite_prediction_count_distribution", "theoretical_max_consensus",
                  "actual_max_auxiliary_consensus", "final_C6_vs_auxiliary_single_draw")}
                  for row in fold_reports],
    }
    design = {"status": "AUDIT_COMPLETE", "feature_schema": "141-dimensional RC utility feature",
              "folds": fold_reports,
              "early_return_totals": {"candidate_comparison": 45, "final_C6": 5, "total": 50}}
    audit_json = {
        "status": "PASS_NON_ADJUDICATIVE_IMPLEMENTATION_AUDIT",
        "scientific_status_before": "FAIL_RC_SHOR_VALUE",
        "scientific_status_after": "FAIL_RC_SHOR_VALUE",
        "scientific_status_changed": False,
        "outer_evaluation_rerun": False,
        "forbidden_reads": {"model_forwards": 0, "checkpoint_loads": 0, "image_reads": 0,
                            "label_H5_reads": 0, "domain_manifest_reads": 0,
                            "v0_4_formal_03_reads": 0},
        "routing_order_audit": routing_order,
        "stability_semantics_audit": stability,
        "private_artifact_inventory": {"files": bundle.manifest["files"], "bytes": bundle.manifest["bytes"],
                                       "content_sha256": bundle.manifest["content_sha256"],
                                       "verified_files_read": len(bundle.verified),
                                       "verified_bytes_read": sum(row["bytes"] for row in bundle.verified.values()),
                                       "verified_entries": [bundle.verified[key] for key in sorted(bundle.verified)]},
        "confirmed_code_bugs": [
            "routing_rows paired fold-append-order routes with global-row-order utility targets",
            "routes_for_bootstrap_draws ignored rho, the feasible>=90 ensemble gate, and full consensus",
        ],
    }

    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), "erratum output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    write_new(output / "RC_SHOR_V0_5_IMPLEMENTATION_AUDIT.json",
              json.dumps(audit_json, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_new(output / "RC_SHOR_V0_5_DESIGN_DEGENERACY.json",
              json.dumps(design, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with (output / "RC_SHOR_V0_5_ROUTING_CORRECTED.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = ["policy", "three_domain_gain", "historical_gain", "REFUGE_gain", "RIM_ONE_r3_gain",
                  "current_domain_drop", "maximum_current_class_drop", "maximum_seed_domain_drop",
                  "route_frequency", "route_count", "route_precision", "route_precision_numerator",
                  "route_precision_denominator", "historical_recall", "historical_recall_numerator",
                  "historical_recall_denominator"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for policy in POLICIES:
            writer.writerow({"policy": policy, **corrected[policy]})
    audit_md = """# RC-SHOR V0.5 implementation audit

This is a derived-artifact-only, non-adjudicative audit. It performed no model forward,
checkpoint load, image read, label-H5 read, domain-manifest read, router refit, candidate
selection, or outer evaluation.

## Findings

1. `routing_rows()` filtered routes in outer-fold append order and paired them with utility
   arrays in global `row_index` order. Sorting every policy by `row_index` repairs the summary.
2. `routes_for_bootstrap_draws()` did not use `rho`, did not apply the feasible >= 90 ensemble
   gate, and did not execute the full consensus procedure. Its p10/p90 values therefore do not
   characterize bootstrap realizations of final C6.
3. All five folds had fewer than 90 feasible replicates (%s); final C6 consequently remained C0.
   The five common replicate indices are an outer-fold intersection, not per-unit feasibility.

Corrected gains are C3 %.17g, C4 %.17g, C5 %.17g, and C6 %.17g. C3 historical gain is
%.17g and its current-domain drop is %.17g. The machine-readable audit contains per-fold,
per-expert, per-seed/domain feasibility, inactive-row finite-prediction counts, attainable
consensus, conformal degeneracy, lambdas, C5 coverage, and candidate-route identities.

RC-SHOR V0.5 remains `FAIL_RC_SHOR_VALUE`; no original V0.5 artifact or conclusion changed.
""" % ([len(value) for value in feasible_sets], corrected["C3"]["three_domain_gain"],
       corrected["C4"]["three_domain_gain"], corrected["C5"]["three_domain_gain"],
       corrected["C6"]["three_domain_gain"], corrected["C3"]["historical_gain"],
       corrected["C3"]["current_domain_drop"])
    write_new(output / "RC_SHOR_V0_5_IMPLEMENTATION_AUDIT.md", audit_md)
    erratum = """# RC-SHOR V0.5 public erratum

RC-SHOR V0.5's scientific status remains **FAIL_RC_SHOR_VALUE**. C6 remains exactly C0;
all previously published C0-C8 Dice values and the C3-C6 direct comparison remain unchanged.

The C3/C4/C5 routing summaries in the original `RC_SHOR_V0_5_ROUTING.csv` are incorrect
because fold-order route arrays were paired with global-row-order utilities. The corrected
three-domain gains are C3 %.17g, C4 %.17g, and C5 %.17g; C6 remains %.17g.

The original stability p10/p90 values were computed from auxiliary single-bootstrap routes,
not the full final-C6 consensus procedure: those routes ignored `rho`, the feasible >= 90 gate,
and ensemble consensus. The reported 5/100 is the intersection of feasible replicate indices
across five outer folds, not the preregistered per-unit feasibility measure.

This erratum does not rerun or re-adjudicate V0.5 and does not alter any original V0.5 byte.
""" % (corrected["C3"]["three_domain_gain"], corrected["C4"]["three_domain_gain"],
       corrected["C5"]["three_domain_gain"], corrected["C6"]["three_domain_gain"])
    write_new(output / "RC_SHOR_V0_5_ERRATUM.md", erratum)
    manifest_entries = []
    for path in sorted(output.iterdir()):
        manifest_entries.append({"path": path.name, "bytes": path.stat().st_size,
                                 "sha256": sha256_file(path)})
    manifest = {"schema_version": 1, "status": "PASS_ERRATUM_BUNDLE_COMPLETE",
                "scientific_status": "FAIL_RC_SHOR_VALUE", "scientific_status_changed": False,
                "entries": manifest_entries, "files": len(manifest_entries),
                "bytes": sum(row["bytes"] for row in manifest_entries),
                "content_sha256": canonical_hash(manifest_entries),
                "private_large_artifacts_published": False}
    write_new(output / "RC_SHOR_V0_5_ERRATUM_MANIFEST.json",
              json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return audit_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.input, args.output)
    except AuditBlocked as error:
        print(error.status + ": " + str(error))
        raise SystemExit(2)
    print(json.dumps({"status": result["status"], "scientific_status_changed": False,
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
