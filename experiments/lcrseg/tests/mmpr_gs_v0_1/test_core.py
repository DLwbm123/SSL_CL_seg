import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from di_dmpa_gate1c_v2 import binding as b, reliability as r
from di_dmpa_gate1c_v2.metrics import tie_keys
from di_dmpa_gate1c_v3 import durable as d
from di_dmpa_gate1c_v3.archive import promote
from mmpr_gs_v0_1 import core as c, diagnostic, evaluator, report, run, testing


def selection(rank=None, predicted=None, active=None, r1=None, cases=("a",), **kwargs):
    n = len(cases)*8
    return c.mass_match(np.asarray(rank if rank is not None else np.linspace(.1, .8, n), np.float64),
                        np.asarray(predicted if predicted is not None else np.tile([0, 0, 0, 1, 1, 1, 2, 2], len(cases)), np.int64),
                        np.asarray(active if active is not None else np.ones(n), bool),
                        np.asarray(r1 if r1 is not None else np.tile([1, 0, 0, 1, 0, 0, 1, 0], len(cases)), bool),
                        seed=0, stage=1, cases=list(cases), height=2, width=4, **kwargs)


def test_old_line_freeze_and_published_authorities():
    reg, old = run.authority()
    closure = d.read(run.REPO/reg["closure"]["json_path"])
    assert closure["identity_history_reliability_status"] == "FAIL_IDENTITY_HISTORY_RELIABILITY_NOT_SUPPORTED"
    assert closure["transport_status"] == "FAIL_TRANSPORT_NOT_SUPPORTED"
    assert not closure["DI_DMPA_additional_attempts_authorized"] and closure["reduced_candidate"] == "NONE"
    assert len(old["gradient_diagnostic"]["batch_pairs"]) == 72


def test_private_bundle_hash_binding():
    reg, _ = run.authority()
    assert reg["private_bundle"]["logical_files"] == 14470 and reg["private_bundle"]["logical_bytes"] == 17712127650
    assert reg["private_bundle"]["content_sha256"] == "8a82c7b8f0c72eb4faf619f51d7c1eae67a5f81059bc7f283b6b8df22d563526"
    assert reg["private_bundle"]["manifest_sha256"] == "480b627e0f63839ff5430d980020ca026c45838cf5eeb345f2b4cf7c4d578bb2"


@pytest.mark.parametrize("which", ["null", "zero_target", "full_target", "all_ties", "zero_scores"])
def test_selection_edge_counts_and_nulls(which):
    ranks = np.arange(8, dtype=np.float64)/8
    active = np.ones(8, bool)
    r1 = np.array([1, 0, 0, 1, 0, 0, 1, 0], bool)
    if which == "null":
        active[[0, 4]] = False
        ranks[~active] = 0
    if which == "zero_target":
        r1[:] = False
    if which == "full_target":
        r1[:] = True
    if which == "all_ties":
        ranks[:] = .5
    if which == "zero_scores":
        ranks[:] = 0
    weights, rows = selection(rank=ranks, active=active, r1=r1)
    assert all(x["mass_difference"] == 0 and x["target_active_count"] == x["selected_active_count"] for x in rows)
    assert np.array_equal(weights[~active], r1[~active])
    if which in ("zero_target", "full_target"):
        assert np.array_equal(weights, r1)


def test_unrounded_sort_and_coordinate_priority_after_full_hash():
    rank = np.array([.5, np.nextafter(.5, 1), .1, .2, .3, .4, .5, .6], np.float64)
    weights, _ = selection(rank=rank)
    assert weights[1] and not weights[0]
    a, _ = selection(rank=np.ones(8)*.5)
    keys = tie_keys(0, 1, ["a"], 2, 4)
    for indices in ([0, 1, 2], [3, 4, 5], [6, 7]):
        ordered = sorted(indices, key=lambda i: (*keys[i].tolist(), i))
        assert a[ordered[0]]
    bmask, _ = selection(rank=np.ones(8)*.5)
    assert np.array_equal(a, bmask)


def test_full_hash_serialization_matches_original():
    key = tie_keys(3, 1, ["case"], 1, 2)[1]
    digest = hashlib.sha256(json.dumps(["reliability-tie-v1", 3, 1, "case", 0, 1], ensure_ascii=False, separators=(",", ":")).encode()).digest()
    assert key.tobytes() == digest


def test_coordinate_fallback_only_after_hash_collision(monkeypatch):
    monkeypatch.setattr(c, "tie_keys", lambda *args: np.zeros((8, 4), dtype=">u8"))
    w, rows = selection(rank=np.ones(8)*.5)
    assert np.array_equal(np.flatnonzero(w), [0, 3, 6])
    assert sum(row["hash_collisions"] for row in rows) == 5


def test_no_cross_image_or_class_allocation():
    first, rows = selection(cases=("a", "b"))
    rank = np.r_[np.linspace(.1, .8, 8), np.linspace(.01, .02, 8)]
    second, changed = selection(rank=rank, cases=("a", "b"))
    assert np.array_equal(first[:8], second[:8])
    assert all(x["mass_difference"] == 0 for x in rows+changed)
    assert sum(second[:8]) == sum(second[8:]) == 3


@pytest.mark.parametrize("bad", ["float32", "nan", "negative", "null_score", "GT", "duplicate_case", "wrong_shape", "class3"])
def test_selection_trust_boundary(bad):
    args = dict(rank=np.zeros(8, np.float64), predicted=np.zeros(8, np.int64), active=np.ones(8, bool), r1=np.zeros(8, bool),
                seed=0, stage=0, cases=["a"], height=2, width=4)
    if bad == "float32": args["rank"] = args["rank"].astype(np.float32)
    if bad == "nan": args["rank"][0] = np.nan
    if bad == "negative": args["rank"][0] = -.1
    if bad == "null_score": args["active"][0] = False; args["rank"][0] = .1
    if bad == "GT": args["GT"] = np.zeros(8)
    if bad == "duplicate_case": args["cases"] = ["a", "a"]
    if bad == "wrong_shape": args["active"] = args["active"][:-1]
    if bad == "class3": args["predicted"][0] = 3
    with pytest.raises((c.Blocked, b.NonfiniteEvidence, TypeError)):
        c.mass_match(**args)


def test_GT_does_not_enter_builder_and_ignore_remains_selected():
    assert not {"GT", "labels", "valid", "ignore", "boundary"} & set(inspect.signature(c.mass_match).parameters)
    mask, _ = selection(r1=np.ones(8, bool))
    probability = np.eye(3)[np.array([0, 0, 0, 1, 1, 1, 2, 2])].astype(np.float32)
    scores = dict(teacher_probability=probability, active_mask=np.ones(8, bool))
    labels = np.full((2, 4), 255)
    rows, _, _ = evaluator.evaluate_case(scores, dict(Q0=mask, Q1=mask, Q2=mask, Q3=mask), labels, seed=0, stage=0, case="a")
    assert mask.all() and all(r["valid_pixels"] == 0 and r["precision"] is None for r in rows)
    assert sum(r["full_mass"] for r in rows if r["candidate"] == "Q1") == 8


def test_boundary_ignore_and_interior():
    y = np.array([[0, 0, 1], [0, 255, 1], [2, 2, 1]])
    boundary = evaluator.boundary_mask(y)
    assert not boundary[1, 1] and boundary[0, 1] and boundary[0, 2]
    assert not boundary[0, 0]


def test_boundary_unit_aggregation_keeps_case_weights():
    context = dict(seed=0, stage_index=0, class_id=1, candidate="Q1", region="boundary")
    rows = [dict(context, valid_pixels=100, selected_mass=90, correct_mass=90, case_fraction_selected=.9, case_fraction_correct=.9),
            dict(context, valid_pixels=1000, selected_mass=10, correct_mass=0, case_fraction_selected=.01, case_fraction_correct=0.)]
    unit = evaluator.aggregate_regions(rows)[0]
    assert unit["cases"] == 2 and unit["precision"] == pytest.approx(90/91)
    assert unit["selected_mass"] == 100 and unit["domain"] == "REFUGE"


@pytest.mark.parametrize("zero", [False, True])
def test_loss_formula_detach_and_graph_connected_zero(zero):
    logits = torch.randn(2, 3, 2, 4, dtype=torch.float64, requires_grad=True)
    probability = logits.softmax(1)
    target = torch.randn_like(logits).softmax(1).requires_grad_(True)
    weight = torch.zeros((2, 2, 4), dtype=torch.float64) if zero else torch.ones((2, 2, 4), dtype=torch.float64)
    loss = c.consistency(probability, target, weight)
    expected = (weight*(probability-target.detach()).square().sum(1)).sum()/(weight.sum()+1e-12)
    assert torch.equal(loss, expected)
    grad, target_grad = torch.autograd.grad(loss, [logits, target], allow_unused=True)
    assert grad is not None and target_grad is None and logits.grad is None
    if zero: assert torch.count_nonzero(grad) == 0


@pytest.mark.parametrize("gs,gu,active", [([1., 0.], [-2., 3.], True), ([1., 0.], [2., 3.], False),
                                       ([0., 0.], [2., 3.], False), ([1., 1.], [0., 0.], False)])
def test_projection_negative_positive_zero_and_norm(gs, gu, active):
    projected, row = c.project(np.array(gs), np.array(gu))
    expected = np.array(gu)-min(0, np.dot(gs, gu))/(np.dot(gs, gs)+1e-12)*np.array(gs)
    assert np.array_equal(projected, expected) and row["projection_active"] == active
    assert row["projected_dot"] >= -1e-10 and row["projected_norm"] == np.linalg.norm(projected)
    if np.linalg.norm(gu): assert row["norm_ratio"] == np.linalg.norm(projected)/np.linalg.norm(gu)
    if not active: assert np.array_equal(projected, gu)


def test_full_parameter_inventory_None_placeholders():
    models, _, _, _, _, inputs = testing.synthetic_pair()
    named = c.parameters(models["student"])
    logits, _ = models["student"](inputs[0], stochastic_classifier=True)
    values, none = c.gradient(logits.sum(), named)
    inventory = c.inventory(named, {"supervised": none, "Q1": none})
    assert len(inventory) == len(named)
    for i, row in enumerate(inventory):
        assert row["None_gradient_zero_placeholder"] and row["parameter_grad_is_None"]
        if not row["active"]:
            assert none[i] and torch.count_nonzero(values[i]) == 0
    assert {row["name"] for row in inventory if not row["active"]} == {"decoder.conv_logit.sigma.weight", "decoder.conv_logit.grad_update"}
    assert c.vectors(values, named)["global"].size == sum(p.numel() for _, p in named)


def test_active_unpartitioned_parameter_is_blocked():
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(c.Blocked, match="six blocks"):
        c.inventory([("sigma_silently_added", p)], {"Q1": (False,)})


def test_same_gaussian_compiler_and_no_grad_writes():
    result = testing.compile_call_graph("synthetic-test")
    assert result["per_pair"] == dict(diagnostic.LIMITS, total_forwards=5, **diagnostic.ROW_COUNTS)
    assert result["model_bank_rng_immutability"]["rng_restored"]
    assert result["model_bank_rng_immutability"]["teacher_gradients"] == "None"
    assert result["model_bank_rng_immutability"]["bank_gradients"] == "None"
    assert result["model_bank_rng_immutability"]["model_bitwise_unchanged"]


@pytest.mark.parametrize("which", ["forward", "autograd"])
def test_call_budget_blocks_extra_computation(which):
    models, _, _, _, _, inputs = testing.synthetic_pair()
    with pytest.raises(c.Blocked, match="budget"):
        with diagnostic.budget(models):
            if which == "forward":
                for _ in range(4): models["student"](inputs[0], stochastic_classifier=True)
            else:
                p = next(models["student"].parameters())
                for _ in range(4): torch.autograd.grad(p.sum(), [p])


@pytest.mark.parametrize("forbidden", ["backward", "optimizer"])
def test_runtime_forbidden_operations(forbidden):
    with b.no_updates(), pytest.raises(b.ProtocolError):
        if forbidden == "backward": torch.ones(1, requires_grad=True).sum().backward()
        else: torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=.1)


def test_no_update_syntax_in_new_diagnostic():
    for module in (c, diagnostic, evaluator, run, report):
        tree = ast.parse(inspect.getsource(module))
        calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
        assert not any(isinstance(n, ast.Attribute) and n.attr in ("backward", "step", "SGD", "Adam", "update_teacher") for n in calls)
        assert not any(isinstance(n, ast.Attribute) and n.attr == "grad" and isinstance(n.ctx, ast.Store) for n in ast.walk(tree))


@pytest.mark.parametrize("tamper", ["bytes", "path", "extra", "manifest"])
def test_private_archive_audit_rejects_tampering(tmp_path, tamper):
    root = tmp_path/"owned"; root.mkdir()
    (root/"a").write_bytes(b"synthetic evidence")
    d.seal(root)
    if tamper == "bytes": (root/"a").write_bytes(b"bad")
    if tamper == "extra": (root/"b").write_bytes(b"extra")
    if tamper in ("path", "manifest"):
        path = root/"PRIVATE_BUNDLE_MANIFEST.json"
        m = d.read(path)
        if tamper == "path": m["entries"][0]["path"] = "../a"
        else: m["content_sha256"] = "0"*64
        path.write_text(json.dumps(m))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        d.verify(root)


def test_create_only_state_and_private_archive_promotion(tmp_path):
    source = tmp_path/"incoming"; source.mkdir()
    d.write_new(source/"receipt.json", {"ok": True})
    with pytest.raises(FileExistsError): d.write_new(source/"receipt.json", {"ok": False})
    m = d.seal(source)
    receipt = promote(source, tmp_path/"archives", d.sha256(source/"PRIVATE_BUNDLE_MANIFEST.json"))
    assert receipt["every_byte_and_sha_verified"] and receipt["content_sha256"] == m["content_sha256"]
    assert d.verify(receipt["archive"])["files"] == 1
