"""Raw style descriptors, closed-form ridge routing, fusion, bootstrap, and gates."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from di_dmpa_gate1.binding import S
from pres_jascl_v0_1.core import (Blocked, DOMAINS, ORACLE_EXPERT, array_sha256,
                                  pixel_confusion, require, segmentation_metrics)

LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
TEMPERATURES = (0.5, 1.0, 2.0, 4.0)


def finite(value, name="value"):
    require(bool(np.isfinite(np.asarray(value)).all()), f"nonfinite {name}", "BLOCKED_NUMERICAL_FAILURE")


def raw_style_block(features):
    require(isinstance(features, torch.Tensor) and features.ndim == 4, "style block must be BCHW")
    value = features.detach().double()
    require(bool(torch.isfinite(value).all()), "nonfinite style feature", "BLOCKED_NUMERICAL_FAILURE")
    mean = value.mean(dim=(-2, -1))
    std = ((value - mean[:, :, None, None]).square().mean(dim=(-2, -1))).sqrt()
    result = torch.cat((mean, torch.log(std + 1e-6)), dim=1)
    require(bool(torch.isfinite(result).all()), "nonfinite raw style block", "BLOCKED_NUMERICAL_FAILURE")
    return result


def raw_style_descriptors(rgb, enc1, enc2):
    require(rgb.shape[0] == enc1.shape[0] == enc2.shape[0] > 0, "style batch mismatch")
    value = torch.cat([raw_style_block(x) for x in (rgb, enc1, enc2)], dim=1).cpu().numpy()
    require(value.shape[1] == 102, "raw descriptor dimension changed", "BLOCKED_INCOMPLETE_EVIDENCE")
    finite(value, "raw descriptor")
    return value


def salted_hash(prefix, case_id):
    return hashlib.sha256((prefix + "\0" + str(case_id)).encode()).hexdigest()


def select_memory(rows, cap=512):
    require(cap == 512 and rows, "invalid memory selection")
    require(len({row["case_id"] for row in rows}) == len(rows), "duplicate memory case")
    ordered = sorted(rows, key=lambda row: (salted_hash("pres-dsr-sf-v0.2-memory", row["case_id"]), row["case_id"]))
    selected = ordered[:cap]
    return selected, [salted_hash("pres-dsr-sf-v0.2-memory", row["case_id"]) for row in selected]


def case_folds(case_ids, labels=None):
    case_ids = list(case_ids)
    require(len(case_ids) == len(set(case_ids)) and len(case_ids) >= 5, "invalid fold cases")
    labels = np.zeros(len(case_ids), dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64)
    require(labels.shape == (len(case_ids),), "fold labels changed")
    mapping = {}
    for domain in np.unique(labels):
        ordered = sorted((case_ids[i] for i in np.flatnonzero(labels == domain)),
                         key=lambda case: (salted_hash("pres-dsr-sf-v0.2-fold", case), case))
        require(len(ordered) >= 5, "domain has fewer than five fold cases", "BLOCKED_INCOMPLETE_EVIDENCE")
        mapping.update({case: rank % 5 for rank, case in enumerate(ordered)})
    return np.asarray([mapping[case] for case in case_ids], dtype=np.int64)


def balanced_weights(labels, multiplicity=None):
    labels = np.asarray(labels, dtype=np.int64)
    mult = np.ones(len(labels), dtype=np.float64) if multiplicity is None else np.asarray(multiplicity, dtype=np.float64)
    require(labels.ndim == mult.ndim == 1 and len(labels) == len(mult) > 0 and bool((mult >= 0).all()),
            "invalid domain weights")
    domains = np.unique(labels[mult > 0])
    require(len(domains) in (2, 3), "seen domains incomplete", "BLOCKED_INCOMPLETE_EVIDENCE")
    weights = np.zeros(len(labels), dtype=np.float64)
    for domain in domains:
        mask = labels == domain
        total = float(mult[mask].sum())
        require(total > 0, "empty weighted domain", "BLOCKED_INCOMPLETE_EVIDENCE")
        weights[mask] = mult[mask] / (len(domains) * total)
    require(np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=1e-12), "domain mass changed")
    return weights


def fit_standardizer(value, weights=None):
    value = np.asarray(value, dtype=np.float64)
    require(value.ndim == 2 and value.shape[1] == 102 and len(value) > 0, "invalid standardizer input")
    weights = np.ones(len(value), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    require(weights.shape == (len(value),) and float(weights.sum()) > 0 and bool((weights >= 0).all()),
            "invalid standardizer weights")
    weights = weights / weights.sum()
    mean = np.sum(value * weights[:, None], axis=0)
    std = np.sqrt(np.sum((value - mean) ** 2 * weights[:, None], axis=0))
    constant = std <= 1e-12
    scale = std.copy()
    scale[constant] = 1.0
    finite(mean, "standardizer mean")
    finite(std, "standardizer std")
    return dict(mean=mean, std=std, scale=scale, constant=constant)


def apply_standardizer(value, state):
    value = np.asarray(value, dtype=np.float64)
    result = (value - state["mean"]) / state["scale"]
    result[:, state["constant"]] = 0.0
    finite(result, "standardized descriptor")
    return result


def ridge_fit(value, labels, regularization, *, weights=None):
    value = np.asarray(value, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    require(regularization in LAMBDAS and value.ndim == 2 and value.shape[0] == len(labels), "invalid ridge fit")
    domains = np.unique(labels)
    require(np.array_equal(domains, np.arange(len(domains))) and len(domains) in (2, 3), "domain labels changed")
    weights = balanced_weights(labels) if weights is None else np.asarray(weights, dtype=np.float64)
    require(weights.shape == (len(labels),) and bool((weights >= 0).all()) and np.isclose(weights.sum(), 1.0),
            "invalid ridge weights")
    augmented = np.column_stack((value, np.ones(len(value), dtype=np.float64)))
    target = np.eye(len(domains), dtype=np.float64)[labels]
    penalty = np.eye(augmented.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    matrix = augmented.T @ (weights[:, None] * augmented) + regularization * penalty
    rhs = augmented.T @ (weights[:, None] * target)
    try:
        solved = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as error:
        raise Blocked(str(error), "BLOCKED_NUMERICAL_FAILURE") from error
    finite(solved, "ridge weights")
    return solved


def ridge_logits(value, weights):
    value = np.asarray(value, dtype=np.float64)
    result = np.column_stack((value, np.ones(len(value), dtype=np.float64))) @ np.asarray(weights, dtype=np.float64)
    finite(result, "ridge logits")
    return result


def softmax(logits, temperature=1.0):
    require(temperature in TEMPERATURES or temperature == 1.0, "unregistered temperature")
    value = np.asarray(logits, dtype=np.float64) / float(temperature)
    value -= value.max(axis=1, keepdims=True)
    probability = np.exp(value)
    probability /= probability.sum(axis=1, keepdims=True)
    finite(probability, "router probability")
    require(bool((probability >= 0).all()) and bool(np.allclose(probability.sum(1), 1.0, atol=1e-12, rtol=1e-12)),
            "invalid router probability")
    return probability


def domain_metrics(logits, labels, *, temperature=1.0, multiplicity=None):
    labels = np.asarray(labels, dtype=np.int64)
    mult = np.ones(len(labels), dtype=np.float64) if multiplicity is None else np.asarray(multiplicity, dtype=np.float64)
    probability = softmax(logits, temperature)
    routed = np.argmax(probability, axis=1)
    accuracies, losses = [], []
    for domain in np.unique(labels[mult > 0]):
        mask = (labels == domain) & (mult > 0)
        weight = mult[mask]
        accuracies.append(float(np.average(routed[mask] == domain, weights=weight)))
        losses.append(float(np.average(-np.log(np.maximum(probability[mask, domain], np.finfo(np.float64).tiny)),
                                               weights=weight)))
    return dict(macro_accuracy=float(np.mean(accuracies)), domain_nll=float(np.mean(losses)),
                per_domain_accuracy=accuracies)


def fit_router(value, labels, case_ids, *, multiplicity=None):
    value = np.asarray(value, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    case_ids = list(case_ids)
    mult = np.ones(len(labels), dtype=np.float64) if multiplicity is None else np.asarray(multiplicity, dtype=np.float64)
    require(value.shape == (len(labels), 102) and len(case_ids) == len(labels), "router fit coverage changed")
    folds = case_folds(case_ids, labels)
    oof_by_lambda, rows = {}, []
    for regularization in LAMBDAS:
        oof = np.full((len(value), len(np.unique(labels))), np.nan, dtype=np.float64)
        for fold in range(5):
            train = (folds != fold) & (mult > 0)
            held = (folds == fold) & (mult > 0)
            require(bool(train.any()) and bool(held.any()), "empty CV fold", "BLOCKED_INCOMPLETE_EVIDENCE")
            state = fit_standardizer(value[train], mult[train])
            fit_x = apply_standardizer(value[train], state)
            fit_w = balanced_weights(labels[train], mult[train])
            weights = ridge_fit(fit_x, labels[train], regularization, weights=fit_w)
            oof[held] = ridge_logits(apply_standardizer(value[held], state), weights)
        positive = mult > 0
        require(bool(np.isfinite(oof[positive]).all()), "incomplete OOF logits", "BLOCKED_INCOMPLETE_EVIDENCE")
        metrics = domain_metrics(oof[positive], labels[positive], multiplicity=mult[positive])
        rows.append(dict(kind="lambda", value=regularization, **metrics))
        oof_by_lambda[regularization] = oof
    selected_lambda = min(LAMBDAS, key=lambda value: (-next(r["macro_accuracy"] for r in rows
                                                             if r["kind"] == "lambda" and r["value"] == value),
                                                        next(r["domain_nll"] for r in rows
                                                             if r["kind"] == "lambda" and r["value"] == value),
                                                        -value))
    positive = mult > 0
    for temperature in TEMPERATURES:
        metrics = domain_metrics(oof_by_lambda[selected_lambda][positive], labels[positive],
                                 temperature=temperature, multiplicity=mult[positive])
        rows.append(dict(kind="temperature", value=temperature, **metrics))
    selected_temperature = min(TEMPERATURES, key=lambda value: (
        next(r["domain_nll"] for r in rows if r["kind"] == "temperature" and r["value"] == value),
        abs(np.log(value)), value))
    state = fit_standardizer(value[positive], mult[positive])
    final_x = apply_standardizer(value[positive], state)
    weights = ridge_fit(final_x, labels[positive], selected_lambda,
                        weights=balanced_weights(labels[positive], mult[positive]))
    return dict(selected_lambda=selected_lambda, selected_temperature=selected_temperature,
                mean=state["mean"], std=state["std"], scale=state["scale"], constant=state["constant"],
                weights=weights, folds=folds, cv_rows=rows)


def router_probabilities(value, model):
    standardized = apply_standardizer(value, model)
    return softmax(ridge_logits(standardized, model["weights"]), model["selected_temperature"])


def hard_routes(probability, seen_domains):
    probability = np.asarray(probability, dtype=np.float64)
    seen = np.asarray(tuple(seen_domains), dtype=np.int64)
    require(probability.shape[1] == len(seen) and tuple(seen) in ((0, 1), (0, 1, 2)), "invalid hard router")
    return seen[np.argmax(probability, axis=1)]


def probability_fusion(alpha, expert_probability):
    alpha = np.asarray(alpha, dtype=np.float64)
    experts = np.asarray(expert_probability)
    require(experts.ndim >= 3 and experts.shape[:2] == alpha.shape, "fusion shape mismatch")
    require(bool((alpha >= 0).all()) and bool(np.allclose(alpha.sum(1), 1.0, atol=1e-12, rtol=1e-12)),
            "invalid fusion weights")
    result = np.einsum("nd,nd...->n...", alpha, experts, optimize=True)
    finite(result, "fused probability")
    return result


def bootstrap_multiplicity(case_ids, *, seed, stage, domain, replicate):
    case_ids = list(case_ids)
    require(len(case_ids) == len(set(case_ids)) > 0 and replicate in range(5), "invalid bootstrap cases")
    draw_seed = S(["pres-dsr-sf-v0.2-bootstrap", seed, stage, domain, replicate])
    rng = np.random.Generator(np.random.PCG64(draw_seed))
    draw = rng.integers(0, len(case_ids), size=len(case_ids))
    return np.bincount(draw, minlength=len(case_ids)).astype(np.float64), draw_seed


def adjudicate(evidence):
    e1 = bool(evidence["E1"])
    oracle = evidence["oracle"]
    e2 = (oracle["three_domain_gain"] >= .015 and oracle["historical_gain"] >= .020
          and oracle["positive_seed_count"] >= 2 and oracle["maximum_domain_drop"] <= .005)
    routing = evidence["ridge_hard"]
    e3 = (routing["stage1_macro"] >= .95 and min(routing["stage1_per_domain"]) >= .90
          and routing["stage2_macro"] >= .90 and min(routing["stage2_per_domain"]) >= .85)
    soft = evidence["ridge_soft"]
    e4 = (soft["oracle_gap"] <= .020 and soft["shared_gain"] >= .130 and soft["historical_gain"] >= .200
          and soft["gain_over_m1_hard"] >= .010 and soft["positive_seed_count"] == 3
          and soft["maximum_seed_domain_drop"] <= .020 and soft["current_domain_drop"] <= .010)
    stable = evidence["stability"]
    e5 = (stable["hard_macro_p10"] >= .85 and stable["soft_gain_p10"] >= .10
          and stable["soft_oracle_gap_p90"] <= .03 and stable["all_domains_nonempty"] and stable["all_finite"])
    e6 = bool(evidence["E6"])
    if not e1 or not e6:
        status = "BLOCKED_PROTOCOL_OR_LEAKAGE"
    elif not e2:
        status = "FAIL_SNAPSHOT_EXPERT_VALUE"
    elif not e3:
        status = "FAIL_DISCRIMINATIVE_DOMAIN_ROUTING"
    elif not e4:
        status = "FAIL_SOFT_EXPERT_FUSION_VALUE"
    elif not e5:
        status = "FAIL_ROUTER_STABILITY"
    else:
        status = "PASS_PRES_DSR_SF_FEASIBILITY"
    return dict(scientific_status=status, E1=e1, E2=bool(e2), E3=bool(e3), E4=bool(e4), E5=bool(e5), E6=e6)
