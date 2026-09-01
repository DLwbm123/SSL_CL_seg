"""Registered descriptor, prototype, routing, evaluator, bootstrap and gates."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from di_dmpa_gate1.binding import S
from di_dmpa_gate1.bootstrap import matched_cosines
from di_dmpa_gate1.geometry_metrics import nearest
from di_dmpa_gate1.spherical_kmeans import fit as spherical_fit
from di_dmpa_jascl.metrics import ConfusionMetrics

DOMAINS = ("REFUGE", "RIM_ONE_r3", "Drishti_GS")
ORACLE_EXPERT = {domain: index for index, domain in enumerate(DOMAINS)}


class Blocked(RuntimeError):
    def __init__(self, message, status="BLOCKED_PROTOCOL_OR_LEAKAGE"):
        super().__init__(message)
        self.status = status


def require(condition, message, status="BLOCKED_PROTOCOL_OR_LEAKAGE"):
    if not condition:
        raise Blocked(message, status)


def _finite(value, name):
    require(bool(np.isfinite(value).all()), f"nonfinite {name}", "BLOCKED_NUMERICAL_FAILURE")


def _unit_rows(value, name):
    value = np.asarray(value, dtype=np.float64)
    _finite(value, name)
    norm = np.linalg.norm(value, axis=1)
    require(bool((norm > 1e-12).all()), f"zero {name}", "BLOCKED_NUMERICAL_FAILURE")
    return value / norm[:, None]


def style_block(features):
    """Float64 population mean/std block for a BCHW tensor."""
    require(isinstance(features, torch.Tensor) and features.ndim == 4, "style block must be BCHW")
    value = features.detach().double()
    require(bool(torch.isfinite(value).all()), "nonfinite style feature", "BLOCKED_NUMERICAL_FAILURE")
    mean = value.mean(dim=(-2, -1))
    std = ((value - mean[:, :, None, None]).square().mean(dim=(-2, -1))).sqrt()
    block = torch.cat((mean, std), dim=1)
    norm = torch.linalg.vector_norm(block, dim=1)
    valid = norm > 1e-12
    output = torch.zeros_like(block)
    output[valid] = block[valid] / norm[valid, None]
    return output, valid


def style_descriptors(rgb, enc1, enc2):
    require(rgb.shape[0] == enc1.shape[0] == enc2.shape[0] > 0, "style batch mismatch")
    blocks, validity = [], []
    for value in (rgb, enc1, enc2):
        block, valid = style_block(value)
        blocks.append(block)
        validity.append(valid)
    joined = torch.cat(blocks, dim=1)
    norm = torch.linalg.vector_norm(joined, dim=1)
    require(bool((norm > 1e-12).all()), "final style norm <= 1e-12", "BLOCKED_NUMERICAL_FAILURE")
    result = (joined / norm[:, None]).cpu().numpy()
    _finite(result, "style descriptor")
    return result, torch.stack(validity, dim=1).cpu().numpy()


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def fit_prototypes(descriptors, M, *, seed, domain_index, weights=None, replicate=-1):
    x = _unit_rows(descriptors, "descriptor")
    weights = np.ones(len(x), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    require(weights.shape == (len(x),) and bool(np.isfinite(weights).all()) and bool((weights >= 0).all())
            and float(weights.sum()) > 0, "invalid case weights")
    if M == 1:
        center = np.sum(x * weights[:, None], axis=0)
        norm = float(np.linalg.norm(center))
        require(np.isfinite(norm) and norm > 1e-12, "zero M1 resultant", "BLOCKED_NUMERICAL_FAILURE")
        centers = (center / norm)[None]
        active = np.array([True])
        restarts = []
        selected_restart = None
    else:
        require(M == 2, "only M1/M2 registered")
        try:
            fitted = spherical_fit(x, weights, 2, seed=seed, stage=domain_index,
                                   class_id=domain_index, replicate=replicate)
        except Exception as error:
            status = getattr(error, "status", "BLOCKED_NUMERICAL_FAILURE")
            raise Blocked(str(error), status) from error
        centers = np.asarray(fitted["centers"], dtype=np.float64)
        active = np.asarray(fitted["active"], dtype=bool)
        restarts = fitted["restarts"]
        selected_restart = fitted["selected_restart"]
        require(active.shape == (2,) and bool(active.all()), "inactive M2 prototype slot",
                "BLOCKED_NUMERICAL_FAILURE")
    assignments, _ = nearest(x, centers, active)
    occupancy = np.bincount(assignments, weights=weights, minlength=M) / weights.sum()
    _finite(centers, "prototype centers")
    _finite(occupancy, "prototype occupancy")
    if active.any():
        require(bool(np.allclose(np.linalg.norm(centers[active], axis=1), 1.0, atol=1e-12, rtol=1e-12)),
                "prototype is not unit norm", "BLOCKED_NUMERICAL_FAILURE")
    pair_cosine = None if M == 1 else float(np.clip(centers[0] @ centers[1], -1, 1))
    return dict(M=M, centers=centers, active=active, occupancy=occupancy,
                selected_restart=selected_restart, restarts=restarts,
                within_domain_prototype_cosine=pair_cosine,
                within_domain_prototype_cosine_distance=None if pair_cosine is None else 1.0 - pair_cosine,
                centers_sha256=array_sha256(centers), active_sha256=array_sha256(active))


def route(descriptors, prototypes, seen_domains):
    x = _unit_rows(descriptors, "routing descriptor")
    seen = tuple(int(d) for d in seen_domains)
    require(seen in ((0, 1), (0, 1, 2)), "unregistered seen-domain bank")
    columns = []
    for domain in seen:
        bank = prototypes[domain]
        centers, active = np.asarray(bank["centers"]), np.asarray(bank["active"], dtype=bool)
        require(active.any(), f"domain {domain} has no active prototype")
        score = x @ centers[active].T
        _finite(score, "domain cosine score")
        columns.append(score.max(axis=1))
    scores = np.stack(columns, axis=1)
    routed = np.asarray(seen, dtype=np.int64)[np.argmax(scores, axis=1)]
    shifted = scores - scores.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    entropy = -(probability * np.log(probability)).sum(axis=1)
    _finite(entropy, "route entropy")
    return routed, scores, entropy


def routing_rows(case_ids, true_domains, routed, scores, entropy, *, seed, stage, M):
    true_domains = np.asarray(true_domains, dtype=np.int64)
    rows = []
    for index, case_id in enumerate(case_ids):
        alternatives = np.delete(scores[index], true_domains[index])
        margin = float(scores[index, true_domains[index]] - alternatives.max())
        row = dict(seed=seed, stage_index=stage, M=M, case_id=case_id,
                   true_domain=int(true_domains[index]), routed_domain=int(routed[index]),
                   true_domain_score=float(scores[index, true_domains[index]]),
                   true_domain_margin=margin, route_entropy=float(entropy[index]))
        row.update({f"score_domain{d}": float(scores[index, d]) for d in range(stage + 1)})
        rows.append(row)
    return rows


def routing_summary(rows, domain_count):
    require(rows and domain_count in (2, 3), "empty/invalid routing rows")
    per_domain = []
    matrix = np.zeros((domain_count, domain_count), dtype=np.int64)
    for row in rows:
        matrix[row["true_domain"], row["routed_domain"]] += 1
    for domain in range(domain_count):
        total = int(matrix[domain].sum())
        require(total > 0, "empty routing domain", "BLOCKED_INCOMPLETE_EVIDENCE")
        per_domain.append(float(matrix[domain, domain] / total))
    margins = np.asarray([row["true_domain_margin"] for row in rows], dtype=np.float64)
    entropies = np.asarray([row["route_entropy"] for row in rows], dtype=np.float64)
    _finite(margins, "routing margins")
    _finite(entropies, "routing entropies")
    return dict(accuracy=float(np.trace(matrix) / matrix.sum()), macro_accuracy=float(np.mean(per_domain)),
                per_domain_accuracy=per_domain, confusion_matrix=matrix,
                margin_p05=float(np.quantile(margins, .05, method="linear")),
                margin_p10=float(np.quantile(margins, .10, method="linear")),
                margin_median=float(np.median(margins)), route_entropy_mean=float(np.mean(entropies)),
                route_entropy_p05=float(np.quantile(entropies, .05, method="linear")),
                route_entropy_p10=float(np.quantile(entropies, .10, method="linear")),
                route_entropy_median=float(np.median(entropies)))


def bootstrap_draw(case_ids, *, seed, stage, role, domain, replicate):
    ids = np.asarray(sorted(case_ids))
    require(len(ids) > 0 and replicate in range(5), "invalid bootstrap draw")
    draw_seed = S(["pres-bootstrap-v1", seed, stage, role, domain, replicate])
    rng = np.random.Generator(np.random.PCG64(draw_seed))
    return ids[rng.integers(0, len(ids), size=len(ids))].tolist(), draw_seed


def multiplicity(case_ids, draws):
    case_ids = list(case_ids)
    require(len(case_ids) == len(set(case_ids)), "duplicate formal case id")
    counts = {case: 0 for case in case_ids}
    for case in draws:
        require(case in counts, "bootstrap case outside domain")
        counts[case] += 1
    return np.asarray([counts[case] for case in case_ids], dtype=np.float64)


def prototype_stability(formal, boot):
    if formal["M"] == 1:
        return [float(np.clip(formal["centers"][0] @ boot["centers"][0], -1, 1))]
    return matched_cosines(formal["centers"], formal["active"], boot["centers"], boot["active"]).tolist()


def pixel_confusion(prediction, target, num_classes=3, ignore_label=255):
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    require(prediction.shape == target.shape, "prediction/target geometry mismatch")
    evaluator = ConfusionMetrics(num_classes, ignore_label)
    evaluator.update(torch.from_numpy(prediction), torch.from_numpy(target))
    return evaluator.matrix.numpy().copy()


def segmentation_metrics(confusion):
    matrix = np.asarray(confusion, dtype=np.int64)
    require(matrix.shape == (3, 3) and bool((matrix >= 0).all()) and matrix.sum() > 0, "invalid segmentation confusion")
    evaluator = ConfusionMetrics(3, 255)
    evaluator.matrix.copy_(torch.from_numpy(matrix))
    result = evaluator.summary()
    require(all(np.isfinite(result[key]) for key in ("mean_iou", "mean_dice", "mean_foreground_dice")),
            "nonfinite segmentation metric", "BLOCKED_NUMERICAL_FAILURE")
    return result


def adjudicate(d1, candidates, d5):
    require(set(candidates) == {1, 2}, "both M1/M2 evidence required", "BLOCKED_INCOMPLETE_EVIDENCE")
    require(all(value.get("complete") for value in candidates.values()), "incomplete M1/M2 evidence",
            "BLOCKED_INCOMPLETE_EVIDENCE")
    D1 = (d1["three_domain_gain"] >= .015 and d1["historical_gain"] >= .020
          and d1["positive_seed_count"] >= 2 and d1["maximum_domain_drop"] <= .005)
    decisions = {}
    for M, value in candidates.items():
        routing = value["routing"]
        D2 = (routing["stage1_macro"] >= .95 and min(routing["stage1_per_domain"]) >= .90
              and routing["stage2_macro"] >= .90 and min(routing["stage2_per_domain"]) >= .85)
        seg = value["segmentation"]
        D3 = (seg["oracle_gap"] <= .010 and seg["shared_gain"] >= .010 and seg["historical_gain"] >= .015
              and seg["positive_seed_count"] >= 2 and seg["maximum_domain_drop"] <= .010)
        stable = value["stability"]
        specific = stable["prototype_cosine_median"] >= .95 if M == 1 else (
            min(stable["occupancies"]) >= .10 and stable["matched_cosine_median"] >= .90)
        D4 = (specific and stable["bootstrap_macro_p10"] >= routing["stage2_macro"] - .05
              and stable["all_finite"] is True)
        decisions[M] = dict(D2=bool(D2), D3=bool(D3), D4=bool(D4))
    passing = [M for M in (1, 2) if all(decisions[M].values())]
    selected = passing[0] if passing else None
    if not d5:
        status = "BLOCKED_PROTOCOL_OR_LEAKAGE"
    elif not D1:
        status = "FAIL_SNAPSHOT_EXPERT_VALUE"
    elif not any(v["D2"] for v in decisions.values()):
        status = "FAIL_DOMAIN_PROTOTYPE_ROUTING"
    elif not any(v["D2"] and v["D3"] for v in decisions.values()):
        status = "FAIL_ROUTED_SEGMENTATION_VALUE"
    elif not passing:
        status = "FAIL_ROUTER_STABILITY"
    else:
        status = "PASS_PRES_ROUTING_FEASIBILITY"
    return dict(scientific_status=status, D1=bool(D1), D2={str(M): decisions[M]["D2"] for M in (1, 2)},
                D3={str(M): decisions[M]["D3"] for M in (1, 2)}, D4={str(M): decisions[M]["D4"] for M in (1, 2)},
                D5=bool(d5), passing_M=passing, selected_M=selected)
