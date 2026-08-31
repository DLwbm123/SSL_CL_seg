"""GT-free exact-count selection and autograd.grad-only gradient projection."""
import numpy as np
import torch

from di_dmpa_gate1c_v2.binding import array_hash, finite
from di_dmpa_gate1c_v2.gradients import BLOCKS, objective
from di_dmpa_gate1c_v2.metrics import tie_keys


class Blocked(RuntimeError):
    def __init__(self, message, status="BLOCKED_PROTOCOL_OR_LEAKAGE"):
        super().__init__(message)
        self.status = status


def require(condition, message, status="BLOCKED_PROTOCOL_OR_LEAKAGE"):
    if not condition:
        raise Blocked(message, status)


def mass_match(rank, predicted, active, r1, *, seed, stage, cases, height=384, width=384):
    """No GT/ignore mask, cross-image or cross-class allocation is possible."""
    rank, predicted, active, r1 = map(np.asarray, (rank, predicted, active, r1))
    n = height * width
    require(height > 0 and width > 0 and len(set(cases)) == len(cases) > 0, "image UID geometry")
    require(all(a.shape == (n * len(cases),) for a in (rank, predicted, active, r1)), "selection shape")
    require(rank.dtype == np.float64 and active.dtype == r1.dtype == bool, "unrounded score/binary mask dtype")
    require(np.issubdtype(predicted.dtype, np.integer) and np.isin(predicted, [0, 1, 2]).all(), "predicted classes")
    finite(rank)
    require((rank >= 0).all() and (rank <= 1).all() and (rank[~active] == 0).all(), "score/null contract")
    weights = r1.copy()
    rows = []
    for j, case in enumerate(cases):
        start, end = j * n, (j + 1) * n
        keys = tie_keys(seed, stage, [case], height, width)
        for c in range(3):
            stratum = np.flatnonzero(predicted[start:end] == c) + start
            indices = stratum[active[stratum]]
            nulls = stratum[~active[stratum]]
            target = int(r1[indices].sum())
            k = keys[indices - start]
            # Coordinates are row-major and consulted only after all 256 hash bits.
            order = np.lexsort((indices - start, k[:, 3], k[:, 2], k[:, 1], k[:, 0], -rank[indices]))
            chosen = indices[order[:target]]
            weights[indices] = False
            weights[chosen] = True
            _, multiplicity = np.unique(rank[indices], return_counts=True)
            cutoff = float(rank[chosen[-1]]) if target else None
            total_r1, total_mmpr = int(r1[stratum].sum()), int(weights[stratum].sum())
            row = dict(seed=seed, stage_index=stage, case_id=case, class_id=c,
                       pixels=len(stratum), active_pixels=len(indices), null_pixels=len(nulls),
                       target_active_count=target, selected_active_count=int(weights[indices].sum()),
                       R1_mass=total_r1, MMPR_mass=total_mmpr, mass_difference=total_mmpr-total_r1,
                       R1_null_mass=int(r1[nulls].sum()), MMPR_null_mass=int(weights[nulls].sum()),
                       newly_selected=int((weights[stratum] & ~r1[stratum]).sum()),
                       deselected=int((r1[stratum] & ~weights[stratum]).sum()),
                       tied_score_rows=int(multiplicity[multiplicity > 1].sum()),
                       tied_score_groups=int((multiplicity > 1).sum()), cutoff_score=cutoff,
                       cutoff_tie_rows=int((rank[indices] == cutoff).sum()) if cutoff is not None else 0,
                       hash_collisions=len(k)-len(np.unique(k, axis=0)),
                       stratum_mask_sha256=array_hash(weights[stratum]), GT_received_by_builder=False)
            require(row["mass_difference"] == 0 and row["R1_null_mass"] == row["MMPR_null_mass"], "mass not conserved")
            rows.append(row)
    require(np.array_equal(weights[~active], r1[~active]), "null mask changed")
    return weights, rows


def consistency(probability, target, weights, *, class_component=None):
    """Reuse the frozen pixel-normalized B0 probability-MSE, including zero mass."""
    return objective(probability, target, weights, target.detach().argmax(1),
                     "pixel_normalized", class_component=class_component)


def parameters(student):
    named = list(student.named_parameters())
    require(named and all(p.requires_grad for _, p in named), "include every student trainable parameter")
    require(all(p.grad is None for _, p in named), "parameter.grad already populated")
    return named


def gradient(loss, named, *, retain=True):
    finite(loss)
    values = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True,
                                 retain_graph=retain, create_graph=False)
    result, none = [], []
    for (_, p), value in zip(named, values):
        none.append(value is None)
        result.append(torch.zeros_like(p) if value is None else value.detach())
        finite(result[-1])
    require(all(p.grad is None for _, p in named), "autograd populated parameter.grad")
    return tuple(result), tuple(none)


def inventory(named, none_masks):
    require(none_masks and all(len(x) == len(named) for x in none_masks.values()), "gradient inventory coverage")
    rows = []
    for i, (name, p) in enumerate(named):
        matches = [b for b, prefixes in BLOCKS.items() if name.startswith(prefixes)]
        active = not all(mask[i] for mask in none_masks.values())
        require(len(matches) <= 1 and (not active or len(matches) == 1), "active parameter not in six blocks: " + name)
        rows.append(dict(name=name, shape=list(p.shape), elements=p.numel(), dtype=str(p.dtype),
                         trainable=p.requires_grad, active=active, block=matches[0] if matches else None,
                         gradient_is_None={k: v[i] for k, v in none_masks.items()},
                         None_gradient_zero_placeholder=True, parameter_grad_is_None=p.grad is None))
    return rows


def vectors(values, named):
    flat = [g.detach().double().cpu().numpy().reshape(-1) for g in values]
    result = {"global": np.concatenate(flat)}
    for block, prefixes in BLOCKS.items():
        selected = [a for a, (name, _) in zip(flat, named) if name.startswith(prefixes)]
        require(bool(selected), "missing block: " + block)
        result[block] = np.concatenate(selected)
    finite(*result.values())
    return result


def alignment(supervised, unsupervised):
    finite(supervised, unsupervised)
    require(supervised.shape == unsupervised.shape and supervised.ndim == 1, "gradient vector shape")
    dot = float(np.dot(supervised, unsupervised))
    sn, un = float(np.linalg.norm(supervised)), float(np.linalg.norm(unsupervised))
    cosine = float(np.clip(dot / (sn * un), -1, 1)) if sn and un else None
    return dict(dot=dot, cosine=cosine, supervised_norm=sn, unsupervised_norm=un,
                supervised_zero=sn == 0, unsupervised_zero=un == 0,
                negative_cosine=cosine < 0 if cosine is not None else None,
                undefined_reason=None if cosine is not None else "ZERO_GRADIENT_NORM")


def project(supervised, raw):
    supervised, raw = np.asarray(supervised, np.float64), np.asarray(raw, np.float64)
    before = alignment(supervised, raw)
    alpha = min(0.0, before["dot"]) / (float(np.dot(supervised, supervised)) + 1e-12)
    projected = raw - alpha * supervised
    finite(projected)
    after = alignment(supervised, projected)
    return projected, dict(raw_dot=before["dot"], raw_cosine=before["cosine"],
                           projected_dot=after["dot"], projected_cosine=after["cosine"],
                           raw_norm=before["unsupervised_norm"], projected_norm=after["unsupervised_norm"],
                           supervised_norm=before["supervised_norm"],
                           norm_ratio=after["unsupervised_norm"] / before["unsupervised_norm"] if before["unsupervised_norm"] else None,
                           projection_active=alpha < 0, projection_coefficient=alpha,
                           projected_zero=after["unsupervised_norm"] == 0,
                           projected_dot_pass=after["dot"] >= -1e-10)
