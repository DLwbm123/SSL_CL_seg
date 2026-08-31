"""One shared synthetic/real draw0 kernel, five forwards and nine grad calls."""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from di_dmpa_gate1.feature_extraction import seed_after_load, state_hash
from di_dmpa_jascl.checkpoint import capture_rng_state, restore_rng_state
from di_dmpa_jascl.modeling import pas_probability_objective
from di_dmpa_gate1c_v2 import binding as b, reliability as r
from di_dmpa_gate1c_v2.precision import student_forward, compare, comparable, rng_hash
from .core import (BLOCKS, require, mass_match, consistency, parameters, gradient,
                   inventory, vectors, alignment, project)

LIMITS = dict(native_forwards=3, fp64_forwards=2, native_autograd=3, fp64_autograd=6)
ROW_COUNTS = dict(raw_mask_case_class_rows=6, gradient_alignment_rows=2,
                  gradient_blockwise_rows=12, gradient_class_component_rows=21,
                  projection_retention_rows=7, precision_comparison_rows=21, model_guard=1)


@contextmanager
def budget(models):
    """Count top-level models, not their internal convolution submodules."""
    counts = dict.fromkeys(LIMITS, 0)
    trace = []
    handles = []
    original_grad = torch.autograd.grad

    def count(key, detail):
        require(counts[key] < LIMITS[key], "compute budget exceeded: " + key, "BLOCKED_CALL_GRAPH_MISMATCH")
        counts[key] += 1
        trace.append(dict(operation=key, detail=detail))

    def hook(name):
        def before(model, args):
            dtype = next(model.parameters()).dtype
            require(dtype in (torch.float32, torch.float64), "unexpected forward precision")
            count("native_forwards" if dtype == torch.float32 else "fp64_forwards", name)
        return before

    def grad(loss, inputs, **kwargs):
        dtype = inputs[0].dtype
        require(dtype in (torch.float32, torch.float64), "unexpected autograd precision")
        count("native_autograd" if dtype == torch.float32 else "fp64_autograd", str(dtype))
        return original_grad(loss, inputs, **kwargs)

    try:
        for name, model in models.items():
            handles.append(model.register_forward_pre_hook(hook(name)))
        with patch.object(torch.autograd, "grad", grad), b.no_updates():
            yield counts, trace
        require(counts == LIMITS, "incomplete call graph", "BLOCKED_CALL_GRAPH_MISMATCH")
    finally:
        for handle in handles:
            handle.remove()


def supervised(logits, labels):
    require(labels.shape == logits.shape[:1] + logits.shape[2:] and bool((labels != 255).any()), "labeled batch support")
    return F.cross_entropy(logits, labels, ignore_index=255, reduction="none").sum() / (labels != 255).sum()


def compute_pair(models, legacy, current, history, pair, inputs):
    """Pure computation on supplied inputs; no checkpoint/cache/GT loader here."""
    require(set(models) == {"student", "ema_teacher", "gradient_student"}, "exact native/teacher/shadow set")
    require(all(not m.training for model in models.values() for m in model.modules()), "eval mode required")
    require(all(not p.requires_grad for p in models["ema_teacher"].parameters()), "teacher gradients enabled")
    require(not legacy.requires_grad and legacy.grad is None, "PAS bank gradients enabled")
    rng = capture_rng_state()
    rng_before = rng_hash()
    before = {k: state_hash(v.state_dict()) for k, v in models.items()}
    banks_before = (b.tensor_hash(legacy), b.array_hash(current), b.array_hash(history))
    counts = dict.fromkeys(LIMITS, 0)
    try:
        with budget(models) as (counts, trace):
            result, arrays = _kernel(models, legacy, current, history, pair, inputs)
        result.update(counts=counts, call_trace=trace)
    finally:
        restore_rng_state(rng)
        after = {k: state_hash(v.state_dict()) for k, v in models.items()}
        require(before == after, "model tensors changed", "BLOCKED_MODEL_MUTATION")
        require(banks_before == (b.tensor_hash(legacy), b.array_hash(current), b.array_hash(history)),
                "prototype bank changed", "BLOCKED_MODEL_MUTATION")
        require(all(p.grad is None for model in models.values() for p in model.parameters()), "parameter.grad mutated")
        require(rng_before == rng_hash(), "RNG not restored", "BLOCKED_MODEL_MUTATION")
    result["isolation"] = dict(model_before=before, model_after=after, model_bitwise_unchanged=True,
                               bank_before=list(banks_before), banks_unchanged=True,
                               rng_before=rng_before, rng_after=rng_hash(), rng_restored=True,
                               teacher_gradients="None", bank_gradients="None", parameter_grad_fields="None",
                               backward_called=False, optimizer_constructed=False, model_optimizer_steps=0,
                               transport_optimizer_steps=0, hidden_gt_training_usage="none", test_gt_usage="none")
    expected = dict(ROW_COUNTS)
    expected["raw_mask_case_class_rows"] = len(pair["unlabeled_case_ids"]) * 3
    actual = dict(raw_mask_case_class_rows=len(result["mass_rows"]), gradient_alignment_rows=len(result["alignment"]),
                  gradient_blockwise_rows=len(result["blockwise"]), gradient_class_component_rows=len(result["components"]),
                  projection_retention_rows=len(result["retention"]), precision_comparison_rows=len(result["precision"]), model_guard=1)
    require(actual == expected, "output row budget mismatch", "BLOCKED_CALL_GRAPH_MISMATCH")
    result["output_rows"] = actual
    return result, arrays


def _kernel(models, legacy, current, history, pair, inputs):
    xu, xl, labels = inputs
    sl, sf, du, udraw = student_forward(models, xu, pair["forward_seeds"]["student_unlabeled"])
    native_p, shadow_p = sl.float().softmax(1), du.softmax(1)
    seed_after_load(pair["teacher_draw_seeds"][0])
    with torch.no_grad():
        tl, tf = models["ema_teacher"](xu, stochastic_classifier=True)
    ll, lf, dl, ldraw = student_forward(models, xl, pair["forward_seeds"]["student_labeled"])
    b.finite(sl, sf, du, tl, tf, ll, lf, dl)
    scores = r.build(sl.detach(), sf.detach(), tl, tf, legacy, current, history)
    # The shared original PAS path uses cached outputs; these lambdas add no model forwards.
    with torch.no_grad():
        _, sv, tv, original_r1 = pas_probability_objective(lambda *a, **k: (sl, sf),
                                                          lambda *a, **k: (tl, tf), None, legacy)
    require(np.array_equal(original_r1.cpu().numpy().reshape(-1), scores["R1"]), "R1 joint PAS differs from B0")
    predicted = scores["teacher_probability"].argmax(1)
    weights, mass = mass_match(scores["R3"], predicted, scores["active_mask"], scores["R1"],
                              seed=pair["seed"], stage=pair["stage_index"], cases=pair["unlabeled_case_ids"],
                              height=xu.shape[-2], width=xu.shape[-1])
    shape = xu.shape[:1] + xu.shape[2:]
    target = tl.float().softmax(1).detach()
    native_named, shadow_named = parameters(models["student"]), parameters(models["gradient_student"])
    require([n for n, _ in native_named] == [n for n, _ in shadow_named], "shadow parameter inventory differs")
    full = {"native": {}, "fp64": {}}
    none = {"native": {}, "fp64": {}}
    losses = {}
    for mode, logits, named in (("native", ll, native_named), ("fp64", dl, shadow_named)):
        loss = supervised(logits, labels)
        values, none[mode]["supervised"] = gradient(loss, named, retain=False)
        full[mode]["supervised"] = vectors(values, named)
        losses[mode + "_supervised"] = float(loss.detach())
    for candidate, mask in (("Q0", scores["R1"]), ("Q1", weights)):
        w = torch.as_tensor(mask.astype(np.float64), device=xu.device).reshape(shape).detach()
        for mode, probability, named in (("native", native_p, native_named), ("fp64", shadow_p, shadow_named)):
            loss = consistency(probability, target.to(probability), w)
            values, none[mode][candidate] = gradient(loss, named, retain=not (mode == "native" and candidate == "Q1"))
            full[mode][candidate] = vectors(values, named)
            losses[mode + "_" + candidate] = float(loss.detach())
    w = torch.as_tensor(weights.astype(np.float64), device=xu.device).reshape(shape).detach()
    class_vectors = []
    for c in range(3):
        values, none["fp64"]["class" + str(c)] = gradient(
            consistency(shadow_p, target.double(), w, class_component=c), shadow_named, retain=c < 2)
        class_vectors.append(vectors(values, shadow_named))
    inv = inventory(shadow_named, none["fp64"])
    native_inventory = inventory(native_named, none["native"])
    require([r["active"] for r in inv] == [r["active"] for r in native_inventory], "native/shadow active set differs")
    ctx = {k: pair[k] for k in ("batch_id", "seed", "stage_index", "domain", "pair_index")}
    global_rows, blocks, precision = [], [], []
    gs, gu = full["fp64"]["supervised"], full["fp64"]["Q1"]
    projected, projection = project(gs["global"], gu["global"])
    alpha = projection["projection_coefficient"]
    retention, components = [], []
    for block in ("global", *BLOCKS):
        for candidate in ("Q0", "Q1"):
            row = dict(ctx, candidate=candidate, block=block, **alignment(gs[block], full["fp64"][candidate][block]))
            (global_rows if block == "global" else blocks).append(row)
        for candidate in ("supervised", "Q0", "Q1"):
            values = compare(full["native"][candidate][block], full["fp64"][candidate][block])
            precision.append(dict(ctx, candidate=candidate, block=block, **values, precision_comparable=comparable(values)))
            if block == "global":
                require(comparable(values), "native/FP64 global VJP not comparable", "BLOCKED_NUMERICAL_FAILURE")
        block_proj = gu[block] - alpha * gs[block]
        raw_stats, proj_stats = alignment(gs[block], gu[block]), alignment(gs[block], block_proj)
        retention.append(dict(ctx, block=block, raw_dot=raw_stats["dot"], raw_cosine=raw_stats["cosine"],
                              projected_dot=proj_stats["dot"], projected_cosine=proj_stats["cosine"],
                              raw_norm=raw_stats["unsupervised_norm"], projected_norm=proj_stats["unsupervised_norm"],
                              norm_ratio=proj_stats["unsupervised_norm"] / raw_stats["unsupervised_norm"] if raw_stats["unsupervised_norm"] else None,
                              projection_active=alpha < 0, projection_coefficient=alpha,
                              projected_zero=proj_stats["unsupervised_norm"] == 0,
                              projected_dot_pass=proj_stats["dot"] >= -1e-10))
        part_sum = sum(v[block] for v in class_vectors)
        residual = float(np.abs(part_sum-gu[block]).max())
        require(np.allclose(part_sum, gu[block], atol=1e-6, rtol=1e-4), "class vector sum mismatch", "BLOCKED_NUMERICAL_FAILURE")
        for c, v in enumerate(class_vectors):
            fraction = float(np.dot(gs["global"], v["global"])) / projection["raw_dot"] if alpha < 0 else 0.0
            pv = v[block] - alpha * fraction * gs[block]
            ps = alignment(gs[block], pv)
            components.append(dict(ctx, class_id=c, block=block, **alignment(gs[block], v[block]),
                                   projected_dot=ps["dot"], projected_cosine=ps["cosine"],
                                   projected_norm=ps["unsupervised_norm"], component_projection_fraction=fraction,
                                   component_sum_max_abs_error=residual, component_sum_pass=True,
                                   component_vector_sha256=b.array_hash(v[block])))
    arrays = dict(student_logits=sl.detach().cpu().numpy(), student_features=sf.detach().cpu().numpy(),
                  teacher_logits=tl.cpu().numpy(), teacher_features=tf.cpu().numpy(),
                  student_probability=native_p.detach().cpu().numpy(), teacher_probability=target.cpu().numpy(),
                  R1=scores["R1"], R2=scores["R2"], R3=scores["R3"], active_mask=scores["active_mask"],
                  raw_norms=scores["raw_norms"], MMPR=weights, predicted=predicted,
                  g_supervised=gs["global"], g_R1=full["fp64"]["Q0"]["global"], g_MMPR=gu["global"], g_projected=projected,
                  **{"g_class" + str(c): v["global"] for c, v in enumerate(class_vectors)})
    for source, pas in (("student", sv), ("teacher", tv)):
        for field in ("valid_mask", "predicted_class", "confidence", "similarity"):
            arrays[source + "_PAS_" + field] = getattr(pas, field).cpu().numpy()
    result = dict(pair=pair, alignment=global_rows, blockwise=blocks, retention=retention, components=components,
                  precision=precision, parameter_inventory=inv, native_parameter_inventory=native_inventory,
                  losses=losses, mass_rows=mass, projection=projection, direct_R1_parity=True,
                  student_draw_replay=dict(unlabeled=udraw, labeled=ldraw),
                  student_logits_sha256=b.tensor_hash(sl), student_features_sha256=b.tensor_hash(sf),
                  labeled_logits_sha256=b.tensor_hash(ll), teacher_features_sha256=b.tensor_hash(tf),
                  native_student_probability_sha256=b.tensor_hash(native_p), teacher_probability_sha256=b.array_hash(target.cpu().numpy()),
                  R1_validity_sha256=b.array_hash(scores["R1"]), score_R3_sha256=b.array_hash(scores["R3"]),
                  R2_R3_exact_equal=bool(np.array_equal(scores["R2"], scores["R3"])),
                  no_optimizer=True, no_backward=True, no_parameter_grad_writes=True)
    return result, arrays
