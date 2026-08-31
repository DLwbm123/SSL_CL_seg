"""Frozen full-vector gradients, certified small cones, stateless displacements."""
from contextlib import contextmanager
from itertools import combinations
from unittest.mock import patch

import numpy as np
import torch

from di_dmpa_gate1.feature_extraction import seed_after_load, state_hash
from di_dmpa_gate1c_v2 import binding as b
from di_dmpa_gate1c_v2.precision import student_forward, replay_draw, compare, comparable
from di_dmpa_jascl.checkpoint import capture_rng_state, restore_rng_state
from di_dmpa_jascl.modeling import pas_probability_objective
from mmpr_gs_v0_1.core import Blocked, require, gradient, parameters, consistency, inventory
from mmpr_gs_v0_1.diagnostic import supervised

CANDIDATES = ("P0", "P1", "P2", "P3", "P4")
COUNT_KEYS = ("native_forwards", "fp64_forwards", "native_autograd", "fp64_autograd")


def full_inventory(named):
    require(len(named) == 51 and sum(p.numel() for _, p in named) == 484016,
            "student inventory must be 51 tensors / 484016 elements", "BLOCKED_GRADIENT_PARTITION_ERROR")
    require(len({n for n, _ in named}) == 51, "duplicate parameter name")
    return [dict(name=n, shape=list(p.shape), elements=p.numel(), dtype=str(p.dtype)) for n, p in named]


def vector(loss, named, *, retain):
    values, none = gradient(loss, named, retain=retain)
    return np.concatenate([v.double().cpu().numpy().reshape(-1) for v in values]), none


@contextmanager
def measured(models, expected=None):
    """Count actual root forwards/VJPs, including functional_call, before dispatch."""
    counts = dict.fromkeys(COUNT_KEYS, 0)
    trace, hooks = [], []

    def record(key, detail):
        if expected is not None:
            require(counts[key] < expected[key], "forward/autograd budget exceeded: " + key,
                    "BLOCKED_CALL_GRAPH_MISMATCH")
        counts[key] += 1
        trace.append(dict(kind=key, **detail))

    def hook(name):
        def before(model, args, kwargs):
            key = "fp64_forwards" if next(model.parameters()).dtype == torch.float64 else "native_forwards"
            record(key, dict(model=name, shape=list(args[0].shape), stochastic=kwargs["stochastic_classifier"]))
        return before

    require(len({id(m) for m in models.values()}) == len(models), "aliased model counters")
    for name, model in models.items():
        hooks.append(model.register_forward_pre_hook(hook(name), with_kwargs=True))
    original = torch.autograd.grad

    def vjp(outputs, inputs, *args, **kwargs):
        inputs = tuple(inputs)
        require(inputs and not kwargs.get("create_graph", False), "invalid diagnostic VJP")
        key = "fp64_autograd" if inputs[0].dtype == torch.float64 else "native_autograd"
        record(key, dict(parameters=len(inputs), elements=sum(p.numel() for p in inputs)))
        return original(outputs, inputs, *args, **kwargs)

    try:
        with b.no_updates(), patch.object(torch.autograd, "grad", vjp):
            yield counts, trace
        if expected is not None:
            require(counts == {k: expected[k] for k in COUNT_KEYS}, "incomplete real call graph",
                    "BLOCKED_CALL_GRAPH_MISMATCH")
    finally:
        for handle in hooks:
            handle.remove()


@contextmanager
def immutable(models, bank):
    rng = capture_rng_state()
    before = {n: state_hash(m.state_dict()) for n, m in models.items()}
    bank_before = b.tensor_hash(bank)
    require(all(p.grad is None for m in models.values() for p in m.parameters()), "preexisting parameter.grad")
    receipt = {}
    try:
        yield receipt
    finally:
        restore_rng_state(rng)
        after = {n: state_hash(m.state_dict()) for n, m in models.items()}
        require(before == after and b.tensor_hash(bank) == bank_before, "model/bank mutation", "BLOCKED_MODEL_MUTATION")
        require(all(p.grad is None for m in models.values() for p in m.parameters()), "parameter.grad changed", "BLOCKED_MODEL_MUTATION")
        require(not bank.requires_grad and bank.grad is None, "bank gradients enabled")
        require(all(not p.requires_grad for n, m in models.items() if n in ("old", "ema_teacher") for p in m.parameters()), "teacher gradient enabled")
        receipt.update(model_before=before, model_after=after, models_bitwise_unchanged=True,
                       bank_sha256=bank_before, banks_unchanged=True, rng_restored=True,
                       parameter_grad_fields="None", teacher_bank_gradients="None",
                       optimizer_constructed=False, backward_called=False, model_optimizer_steps=0,
                       transport_optimizer_steps=0, hidden_gt_training_usage="none", test_gt_usage="none")


def raw_pair(models, legacy, pair, inputs):
    xu, xl, labels = inputs
    sl, sf, du, ud = student_forward(models, xu, pair["forward_seeds"]["student_unlabeled"])
    seed_after_load(pair["teacher_draw_seeds"][0])
    with torch.no_grad():
        tl, tf = models["ema_teacher"](xu, stochastic_classifier=True)
    ll, lf, dl, ld = student_forward(models, xl, pair["forward_seeds"]["student_labeled"])
    b.finite(sl, sf, du, tl, tf, ll, lf, dl)
    native_u, sv, tv, mask = pas_probability_objective(lambda *a, **k: (sl, sf), lambda *a, **k: (tl, tf), None, legacy)
    target = tl.float().softmax(1).detach()
    shadow_u = consistency(du.softmax(1), target.double(), mask.double().detach())
    vectors, masks, losses = {}, {}, {}
    for kind, logits, loss_u, model in (("native", ll, native_u, models["student"]),
                                      ("fp64", dl, shadow_u, models["gradient_student"])):
        named = parameters(model)
        loss_s = supervised(logits, labels)
        total = loss_s + 0.5 * loss_u
        gs, ns = vector(loss_s, named, retain=True)
        g0, n0 = vector(total, named, retain=False)
        vectors[kind + "_supervised"], vectors[kind + "_g0"] = gs, g0
        masks[kind] = dict(supervised=ns, total=n0)
        losses[kind] = dict(supervised=float(loss_s.detach()), consistency=float(loss_u.detach()), total=float(total.detach()))
    comparisons = {name: compare(vectors["native_" + name], vectors["fp64_" + name]) for name in ("supervised", "g0")}
    require(all(comparable(v) for v in comparisons.values()), "native/FP64 B0 gradient parity failed", "BLOCKED_NUMERICAL_FAILURE")
    inv = inventory(parameters(models["gradient_student"]), masks["fp64"])
    require(masks["native"] == masks["fp64"], "native/shadow inactive parameter set differs")
    result = dict(pair=pair, losses=losses, precision=comparisons, parameter_inventory=inv,
                  none_gradients={k: {loss: list(v) for loss, v in masks[k].items()} for k in masks},
                  native_student_draws=dict(unlabeled=ud, labeled=ld), direct_PAS_joint_parity=bool(torch.equal(mask, sv.valid_mask & tv.valid_mask)),
                  PAS_valid_count=int(mask.sum()), PAS_mask_sha256=b.tensor_hash(mask),
                  teacher_probability_sha256=b.tensor_hash(target),
                  native_labeled_logits_sha256=b.tensor_hash(ll), native_unlabeled_logits_sha256=b.tensor_hash(sl))
    return result, dict(vectors, PAS_mask=mask.cpu().numpy())


def constraint_sets(global_guard, sup, old, support):
    require(sup.shape == old.shape == (6, len(global_guard)) and len(support) == 6, "mode gradient geometry")
    b.finite(global_guard, sup, old)
    cur = [("sup_" + str(i), sup[i]) for i, r in enumerate(support) if r["active_pixels"] > 0 and r["center_active"]]
    kd = [("old_" + str(i), old[i]) for i, r in enumerate(support) if r["KD_active"]]
    classes = []
    for c in range(3):
        for prefix, array, key, minimum in (("class_sup_", sup, "active_pixels", 1), ("class_old_", old, "old_correct_pixels", 32)):
            counts = np.array([support[2*c+k][key] for k in range(2)], np.float64)
            if counts.sum() >= minimum:
                classes.append((prefix + str(c), (array[2*c:2*c+2] * counts[:, None]).sum(0) / counts.sum()))
    global_row = [("global_supervised", np.asarray(global_guard, np.float64))]
    return dict(P0=[], P1=global_row, P2=global_row + classes, P3=global_row + cur, P4=global_row + cur + kd)


def solve_cone(g0, guards):
    """At most thirteen constraints: enumerate supports, certify KKT, no fallback."""
    g0 = np.asarray(g0, np.float64)
    require(g0.ndim == 1 and g0.size > 0 and len(guards) <= 13, "cone dimension/constraint limit")
    H = np.stack([np.asarray(h, np.float64) for _, h in guards]) if guards else np.empty((0, len(g0)), np.float64)
    require(H.shape == (len(guards), len(g0)), "guard vector length")
    b.finite(g0, H)
    norms = np.linalg.norm(H, axis=1)
    active = norms > 0
    N = H[active] / norms[active, None]
    A, q = N @ N.T, N @ g0
    spectrum = np.linalg.eigvalsh(A) if len(A) else np.empty(0)
    cutoff = 1e-12 * max(float(spectrum[-1]) if len(spectrum) else 0.0, 0.0)
    positive = spectrum[spectrum > cutoff]
    rank = len(positive)
    condition = float(positive[-1] / positive[0]) if rank else None
    gnorm = float(np.linalg.norm(g0))
    tested = 0
    # ponytail: <=13 guards bounds exhaustive enumeration at 8192 tiny systems.
    for size in range(len(N) + 1):
        for subset in combinations(range(len(N)), size):
            tested += 1
            dual = np.zeros(len(N), np.float64)
            if size:
                index = np.array(subset)
                B = A[np.ix_(index, index)]
                eigen, basis = np.linalg.eigh(B)
                keep = eigen > 1e-12 * max(float(eigen[-1]), 0.0)
                value = basis[:, keep] @ ((basis[:, keep].T @ -q[index]) / eigen[keep])
                if np.max(np.abs(B @ value + q[index])) > 1e-10 * (1 + float(np.max(np.abs(q)))):
                    continue
                if (value < -1e-12).any():
                    continue
                dual[index] = np.maximum(value, 0)
            # Reject clearly infeasible tiny Gram systems before a full-vector product.
            if len(N) and float(np.min(q + A @ dual)) < -1e-10*(1+gnorm):
                continue
            direction = g0 + N.T @ dual
            normalized_dots, raw_dots = N @ direction, H @ direction
            displacement = direction - g0
            objective = 0.5 * float(displacement @ displacement)
            dual_objective = -0.5 * float(dual @ A @ dual) - float(q @ dual)
            gap = abs(objective - dual_objective)
            complementarity = float(np.max(np.abs(dual * normalized_dots))) if len(N) else 0.0
            stationarity = float(np.linalg.norm(displacement - N.T @ dual))
            if ((len(raw_dots) and float(raw_dots.min()) < -1e-10)
                    or (len(normalized_dots) and float(normalized_dots.min()) < -1e-12*(1+gnorm))
                    or gap > 1e-10*(1+objective) or complementarity > 1e-10*(1+objective)
                    or stationarity > 1e-10*(1+gnorm)):
                continue
            original_dual = np.zeros(len(H), np.float64)
            original_dual[active] = dual
            return direction, dict(status="CERTIFIED", guard_names=[n for n, _ in guards],
                constraint_count=len(H), nonzero_constraint_count=int(active.sum()), zero_constraint_indices=np.flatnonzero(~active).tolist(),
                guard_norms=norms.tolist(), raw_guard_dots=raw_dots.tolist(), normalized_guard_dots=normalized_dots.tolist(),
                dual=original_dual.tolist(), primal_feasible=True, dual_feasible=bool((dual >= 0).all()),
                complementarity_max=complementarity, objective=objective, dual_objective=dual_objective, duality_gap=gap,
                stationarity_residual=stationarity, active_set=[int(np.flatnonzero(active)[i]) for i in np.flatnonzero(dual > 0)],
                gram_rank=rank, rank_deficient=rank < len(N), condition_number=condition if rank == len(N) else None,
                effective_condition_number=condition, condition_undefined_reason="rank_deficient" if rank < len(N) else ("no_nonzero_constraints" if not rank else None),
                enumerated_supports=tested, fallback_used=False, direction_sha256=b.array_hash(direction))
    raise Blocked("no sufficiently certified KKT active set", "BLOCKED_NUMERICAL_FAILURE")


def projections(g0, sets):
    result, arrays = {}, {}
    norm0 = float(np.linalg.norm(g0))
    for candidate in CANDIDATES:
        if candidate == "P0":
            direction = np.array(g0, dtype=np.float64, copy=True)
            row = dict(status="RAW_IDENTITY", guard_names=[], raw_guard_dots=[], constraint_count=0, fallback_used=False)
        else:
            direction, row = solve_cone(g0, sets[candidate])
            repeat, certificate = solve_cone(g0, sets[candidate])
            require(np.array_equal(direction, repeat) and row == certificate, "nondeterministic cone result", "BLOCKED_NUMERICAL_FAILURE")
        norm = float(np.linalg.norm(direction))
        row.update(candidate=candidate, norm=norm, raw_norm=norm0, norm_ratio=norm/norm0 if norm0 else None,
                   zero_direction=norm <= 1e-12*max(1.0, norm0), deterministic_repeat=True,
                   direction_sha256=b.array_hash(direction))
        result[candidate], arrays[candidate] = row, direction
    return result, arrays


def native_gaussian(model, seed, expected_hash):
    rng = capture_rng_state()
    try:
        seed_after_load(seed)
        draw = torch.randn_like(model.decoder.conv_logit.mu.weight).detach()
        require(b.tensor_hash(draw) == expected_hash, "guard classifier replay draw differs")
        return draw
    finally:
        restore_rng_state(rng)


def displacement(model, direction, *, raw_norm):
    named = parameters(model)
    direction = np.asarray(direction, np.float64)
    require(direction.shape == (sum(p.numel() for _, p in named),), "virtual direction inventory")
    b.finite(direction)
    norm = float(np.linalg.norm(direction))
    valid = norm > 1e-12 * max(1.0, raw_norm)
    delta = -0.001 * direction / norm if valid else np.zeros_like(direction)
    mapping, offset = {}, 0
    for name, p in named:
        require(p.dtype == torch.float64, "stateless step must use FP64 parameters")
        count = p.numel()
        value = torch.from_numpy(delta[offset:offset+count].copy()).reshape(p.shape).to(p.device)
        mapping[name] = p.detach() + value
        offset += count
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    step_norm = float(np.linalg.norm(delta))
    require(not valid or abs(step_norm-0.001) <= 1e-15, "virtual step norm differs", "BLOCKED_NUMERICAL_FAILURE")
    actual_delta = np.concatenate([(mapping[name]-p.detach()).cpu().numpy().reshape(-1) for name, p in named])
    require(np.linalg.norm(actual_delta-delta) <= 1e-12, "FP64 parameter addition lost virtual step", "BLOCKED_NUMERICAL_FAILURE")
    return (mapping, buffers), dict(step_valid=valid, zero_direction=not valid, requested_step_norm=0.001,
                                  step_norm=step_norm, realized_step_norm=float(np.linalg.norm(actual_delta)),
                                  delta_sha256=b.array_hash(delta), realized_delta_sha256=b.array_hash(actual_delta),
                                  stateless=True, optimizer_constructed=False, checkpoint_written=False)


def functional_forward(model, mapping, images, *, draw=None):
    with torch.no_grad():
        if draw is not None:
            with replay_draw(draw):
                return torch.func.functional_call(model, mapping, (images.double(),), {"stochastic_classifier": True}, strict=True)
        return torch.func.functional_call(model, mapping, (images.double(),), {"stochastic_classifier": False}, strict=True)
