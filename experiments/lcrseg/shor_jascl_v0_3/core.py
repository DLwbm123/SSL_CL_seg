"""Train-only SHOR thresholding, hard routing, bootstrap, and frozen gates."""
from __future__ import annotations

import numpy as np

from pres_dsr_sf_v0_2.core import (apply_standardizer, balanced_weights, bootstrap_multiplicity,
                                   fit_router, fit_standardizer, ridge_fit, ridge_logits, softmax)
from pres_jascl_v0_1.core import Blocked, require

EPSILON = 1e-12


def finite(value, name="value"):
    require(bool(np.isfinite(np.asarray(value)).all()), f"nonfinite {name}", "BLOCKED_NUMERICAL_FAILURE")


def active_support(alpha, labels, multiplicity=None):
    alpha = np.asarray(alpha, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    mult = np.ones(len(labels), dtype=np.float64) if multiplicity is None else np.asarray(multiplicity, dtype=np.float64)
    require(alpha.ndim == 2 and alpha.shape[0] == len(labels) and mult.shape == labels.shape
            and bool(np.isfinite(mult).all()) and bool((mult >= 0).all()), "invalid active-support arrays")
    active = mult > 0
    require(bool(active.any()), "empty active support", "BLOCKED_INCOMPLETE_EVIDENCE")
    finite(alpha[active], "active ridge alpha")
    return alpha[active], labels[active], mult[active]


def top1_lowest(alpha):
    alpha = np.asarray(alpha, dtype=np.float64)
    require(alpha.ndim == 2 and alpha.shape[1] in (2, 3) and len(alpha) > 0, "invalid ridge alpha")
    finite(alpha, "ridge alpha")
    require(bool((alpha >= 0).all()) and bool(np.allclose(alpha.sum(1), 1.0, atol=1e-12, rtol=1e-12)),
            "invalid ridge probability")
    return np.argmax(alpha, axis=1)


def historical_score(alpha, stage, domain):
    alpha = np.asarray(alpha, dtype=np.float64)
    require(stage in (1, 2) and 0 <= domain < stage and alpha.shape == (len(alpha), stage + 1),
            "invalid SHOR score unit")
    value = np.log(alpha[:, domain] + EPSILON) - np.log(alpha[:, stage] + EPSILON)
    finite(value, "SHOR score")
    return value


def calibration(alpha, labels, *, stage, domain, threshold, multiplicity=None):
    alpha, labels, mult = active_support(alpha, labels, multiplicity)
    require(alpha.shape == (len(labels), stage + 1), "invalid calibration arrays")
    require(np.isfinite(threshold) or threshold == np.inf, "invalid threshold")
    require(set(np.unique(labels)) == set(range(stage + 1)),
            "calibration domains incomplete", "BLOCKED_INCOMPLETE_EVIDENCE")
    top = top1_lowest(alpha)
    score = historical_score(alpha, stage, domain)
    accepted = (top == domain) & (score >= threshold)
    accepted_count = float(mult[accepted].sum())
    true_count = float(mult[labels == domain].sum())
    correct_count = float(mult[accepted & (labels == domain)].sum())
    precision = correct_count / accepted_count if accepted_count else 0.0
    recall = correct_count / true_count
    false_rates = {}
    for other in range(stage + 1):
        total = float(mult[labels == other].sum())
        false_rates[str(other)] = float(mult[accepted & (labels == other)].sum()) / total
    result = dict(threshold=float(threshold), accepted_count=accepted_count, precision=precision,
                  historical_recall=recall, current_false_override=false_rates[str(stage)],
                  other_domain_false_override=false_rates,
                  feasible=bool(np.isfinite(threshold) and precision >= .98 and false_rates[str(stage)] <= .02
                                and accepted_count >= 15 and recall >= .35))
    finite([accepted_count, precision, recall, *false_rates.values()], "calibration metric")
    return result


def select_threshold(alpha, labels, *, stage, domain, multiplicity=None):
    alpha, labels, mult = active_support(alpha, labels, multiplicity)
    top = top1_lowest(alpha)
    score = historical_score(alpha, stage, domain)
    candidates = sorted(set(float(x) for x in score[(top == domain) & np.isfinite(score)]))
    candidates.append(float("inf"))
    rows = [calibration(alpha, labels, stage=stage, domain=domain, threshold=value,
                        multiplicity=mult) for value in candidates]
    feasible = [row for row in rows if row["feasible"]]
    selected = max(feasible, key=lambda row: (row["historical_recall"], row["precision"], row["threshold"])) \
        if feasible else None
    return selected, rows


def shor_routes(alpha, *, stage, thresholds):
    alpha = np.asarray(alpha, dtype=np.float64)
    top = top1_lowest(alpha)
    require(set(thresholds) == set(range(stage)), "SHOR threshold key set changed",
            "BLOCKED_OUTPUT_KEYSET_MISMATCH")
    routed = np.full(len(alpha), stage, dtype=np.int64)
    for domain in range(stage):
        selected = thresholds[domain]
        if selected is None:
            continue
        threshold = selected["threshold"] if isinstance(selected, dict) else float(selected)
        require(np.isfinite(threshold), "selected SHOR threshold must be finite")
        score = historical_score(alpha, stage, domain)
        accepted = (top == domain) & (score >= threshold)
        routed[accepted] = domain
    require(bool((routed <= stage).all()) and bool((routed >= 0).all()), "future expert route")
    return routed


def one_hot(routes, width=3):
    routes = np.asarray(routes, dtype=np.int64)
    require(routes.ndim == 1 and bool((routes >= 0).all()) and bool((routes < width).all()), "invalid hard route")
    result = np.zeros((len(routes), width), dtype=np.float64)
    result[np.arange(len(routes)), routes] = 1.0
    return result


def reconstruct_oof(value, labels, case_ids, *, multiplicity=None):
    value = np.asarray(value, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    mult = np.ones(len(labels), dtype=np.float64) if multiplicity is None else np.asarray(multiplicity, dtype=np.float64)
    model = fit_router(value, labels, case_ids, multiplicity=mult)
    oof = np.full((len(labels), len(np.unique(labels))), np.nan, dtype=np.float64)
    for fold in range(5):
        train = (model["folds"] != fold) & (mult > 0)
        held = (model["folds"] == fold) & (mult > 0)
        require(bool(train.any()) and bool(held.any()), "empty OOF fold", "BLOCKED_INCOMPLETE_EVIDENCE")
        state = fit_standardizer(value[train], mult[train])
        weights = ridge_fit(apply_standardizer(value[train], state), labels[train], model["selected_lambda"],
                            weights=balanced_weights(labels[train], mult[train]))
        logits = ridge_logits(apply_standardizer(value[held], state), weights)
        oof[held] = softmax(logits, model["selected_temperature"])
    require(bool(np.isfinite(oof[mult > 0]).all()), "incomplete OOF reconstruction",
            "BLOCKED_INCOMPLETE_EVIDENCE")
    return model, oof


def bootstrap_weights(case_ids_by_domain, *, seed, stage, replicate):
    parts, draw_seeds = [], []
    for domain in range(stage + 1):
        mult, draw_seed = bootstrap_multiplicity(case_ids_by_domain[domain], seed=seed, stage=stage,
                                                 domain=domain, replicate=replicate)
        parts.append(mult)
        draw_seeds.append(draw_seed)
    return np.concatenate(parts), draw_seeds


def adjudicate(evidence):
    calibration_ = evidence["calibration"]
    safety = evidence["current_safety"]
    value = evidence["value"]
    repair = evidence["repair"]
    stability = evidence["stability"]
    H1 = bool(calibration_["all_units_feasible"] and calibration_["all_finite"])
    H2 = bool(safety["current_domain_drop"] <= .010 and safety["maximum_current_class_drop"] <= .015
              and safety["maximum_seed_domain_drop"] <= .020)
    H3 = bool(value["three_domain_gain"] >= .100 and value["historical_gain"] >= .150
              and value["oracle_gap"] <= .060 and value["positive_seed_count"] == 3
              and value["REFUGE_mean_gain"] > 0 and value["RIM_ONE_r3_mean_gain"] > 0)
    H4 = bool(repair["current_domain_drop_reduction"] >= .020
              and repair["maximum_seed_domain_drop_reduction"] >= .020
              and repair["shared_gain_loss"] <= .060 and repair["historical_gain_loss"] <= .080)
    H5 = bool(stability["shared_gain_p10"] >= .080 and stability["historical_gain_p10"] >= .120
              and stability["current_domain_drop_p90"] <= .015
              and stability["maximum_seed_domain_drop_p90"] <= .025
              and stability["every_unit_feasible_in_at_least_4_of_5"] and stability["all_finite"])
    H6 = bool(evidence["isolation"])
    if not H6:
        status = "BLOCKED_PROTOCOL_OR_LEAKAGE"
    elif not H1:
        status = "FAIL_SELECTIVE_OVERRIDE_CALIBRATION"
    elif not H2:
        status = "FAIL_SELECTIVE_OVERRIDE_CURRENT_SAFETY"
    elif not H3 or not H4:
        status = "FAIL_SELECTIVE_OVERRIDE_VALUE"
    elif not H5:
        status = "FAIL_SELECTIVE_OVERRIDE_STABILITY"
    else:
        status = "PASS_SHOR_JASCL_VALIDATION_FEASIBILITY"
    return dict(scientific_status=status, H1=H1, H2=H2, H3=H3, H4=H4, H5=H5, H6=H6)


__all__ = ["Blocked", "active_support", "adjudicate", "bootstrap_weights", "calibration", "historical_score", "one_hot",
           "reconstruct_oof", "select_threshold", "shor_routes", "top1_lowest"]
