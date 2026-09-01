from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from di_dmpa_gate1_v2.features import ImmutableModels
from di_dmpa_gate1c_v3 import durable as d
from pres_jascl_v0_1 import REGISTRATION
from pres_jascl_v0_1 import postflight, protocol, run, testing
from pres_jascl_v0_1.core import (
    Blocked,
    DOMAINS,
    ORACLE_EXPERT,
    adjudicate,
    array_sha256,
    bootstrap_draw,
    fit_prototypes,
    pixel_confusion,
    route,
    routing_summary,
    segmentation_metrics,
    style_block,
    style_descriptors,
)


def _vectors():
    value = np.array([[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0], [-0.99, 0.01]], dtype=np.float64)
    return value / np.linalg.norm(value, axis=1, keepdims=True)


def _passing_candidates():
    routing = dict(stage1_macro=.95, stage1_per_domain=[.90, .90],
                   stage2_macro=.90, stage2_per_domain=[.85, .85, .85])
    segmentation = dict(oracle_gap=.010, shared_gain=.010, historical_gain=.015,
                        positive_seed_count=2, maximum_domain_drop=.010)
    return {
        1: dict(complete=True, routing=copy.deepcopy(routing), segmentation=copy.deepcopy(segmentation),
                stability=dict(prototype_cosine_median=.95, matched_cosine_median=None,
                               occupancies=[1.0], bootstrap_macro_p10=.85, all_finite=True)),
        2: dict(complete=True, routing=copy.deepcopy(routing), segmentation=copy.deepcopy(segmentation),
                stability=dict(prototype_cosine_median=None, matched_cosine_median=.90,
                               occupancies=[.10, .90], bootstrap_macro_p10=.85, all_finite=True)),
    }


def _passing_d1():
    return dict(three_domain_gain=.015, historical_gain=.020, positive_seed_count=2,
                maximum_domain_drop=.005)


def _records():
    train = (160, 63, 41)
    val = (100, 40, 25)
    return {
        seed: {
            stage: {
                "train_unlabeled": [dict(case_id=f"s{seed}d{stage}u{i}") for i in range(train[stage])],
                "val": [dict(case_id=f"s{seed}d{stage}v{i}") for i in range(val[stage])],
            }
            for stage in range(3)
        }
        for seed in range(3)
    }


def test_01_pmgc_closure_binding():
    registration = protocol.authority()
    assert registration["registration_id"] == REGISTRATION


def test_02_private_bundle_hashes(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "a.bin").write_bytes(b"a")
    sealed = d.seal(bundle)
    verified = protocol.verify_bundle_at(
        bundle,
        manifest_sha256=d.sha256(bundle / "PRIVATE_BUNDLE_MANIFEST.json"),
        content_sha256=sealed["content_sha256"],
        files=1,
        bytes_=1,
    )
    assert verified["every_file_sha256_verified"] is True
    (bundle / "extra.bin").write_bytes(b"x")
    with pytest.raises(Blocked, match="coverage"):
        protocol.verify_bundle_at(bundle, manifest_sha256=verified["manifest_sha256"],
                                  content_sha256=sealed["content_sha256"], files=1, bytes_=1)


def test_03_exact_nine_checkpoint_identities():
    registration = protocol.gate1c_contract()
    checkpoints = registration["immutable_baseline"]["checkpoint_inputs"]
    assert len(checkpoints) == len({row["checkpoint_id"] for row in checkpoints}) == 9
    assert len({row["sha256"] for row in checkpoints}) == 9
    assert "independent_evaluation" not in run.checkpoint(registration, 0, 0)


def test_04_fixed_expert_domain_mapping():
    assert DOMAINS == ("REFUGE", "RIM_ONE_r3", "Drishti_GS")
    assert ORACLE_EXPERT == {domain: index for index, domain in enumerate(DOMAINS)}


def test_05_oracle_does_not_select_best_expert():
    validation_scores = [0.99, 0.60, 0.55]
    assert int(np.argmax(validation_scores)) == 0
    assert ORACLE_EXPERT["RIM_ONE_r3"] == 1


def test_06_stagewise_seen_domain_bank():
    banks = {i: dict(centers=np.eye(3)[i:i + 1], active=np.array([True])) for i in range(3)}
    assert set(route(np.eye(3)[:2], banks, (0, 1))[0]) == {0, 1}
    assert set(route(np.eye(3), banks, (0, 1, 2))[0]) == {0, 1, 2}
    with pytest.raises(Blocked, match="seen-domain"):
        route(np.eye(3), banks, (0, 2))


def test_07_frozen_router_extractor():
    source = inspect.getsource(run.extract_descriptors)
    assert 'sources=("ema_teacher",)' in source
    assert "model.enc1(images)" in source and "model.enc2(model.pool(enc1))" in source
    assert "model.decoder" not in source and "stochastic_classifier" not in source
    assert "domain_index" not in source and "role" not in source


def test_08_rgb_style_block():
    rgb = torch.tensor([[[[1.0, 3.0]], [[2.0, 2.0]], [[4.0, 0.0]]]])
    block, valid = style_block(rgb)
    raw = torch.tensor([[2.0, 2.0, 2.0, 1.0, 0.0, 2.0]], dtype=torch.float64)
    assert valid.tolist() == [True]
    assert torch.allclose(block, raw / torch.linalg.vector_norm(raw, dim=1)[:, None])


def test_09_enc1_style_block():
    block, valid = style_block(torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4))
    assert block.shape == (1, 4) and valid.tolist() == [True]


def test_10_enc2_style_block():
    block, valid = style_block(torch.ones(2, 4, 2, 2))
    assert block.shape == (2, 8) and valid.tolist() == [True, True]


def test_11_block_normalization():
    block, _ = style_block(torch.randn(3, 5, 3, 3))
    assert torch.allclose(torch.linalg.vector_norm(block, dim=1), torch.ones(3, dtype=torch.float64), atol=1e-12)


def test_12_final_descriptor_normalization():
    descriptor, validity = style_descriptors(torch.randn(2, 3, 4, 4), torch.randn(2, 16, 4, 4),
                                             torch.randn(2, 32, 2, 2))
    assert validity.shape == (2, 3)
    assert np.allclose(np.linalg.norm(descriptor, axis=1), 1.0, atol=1e-12)


def test_13_zero_descriptor_failure():
    with pytest.raises(Blocked, match="final style norm") as caught:
        style_descriptors(torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2, 2), torch.zeros(1, 4, 1, 1))
    assert caught.value.status == "BLOCKED_NUMERICAL_FAILURE"


def test_14_descriptor_determinism():
    torch.manual_seed(7)
    values = (torch.randn(2, 3, 4, 4), torch.randn(2, 5, 3, 3), torch.randn(2, 7, 2, 2))
    first = style_descriptors(*values)
    second = style_descriptors(*values)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


def test_15_no_filename_or_path_feature():
    assert tuple(inspect.signature(style_descriptors).parameters) == ("rgb", "enc1", "enc2")
    assert tuple(run.image_only(dict(case_id="c", image_h5_relpath="p", image_sha256="h"))) == (
        "case_id", "image_h5_relpath", "image_sha256")


def test_16_no_gt_in_router():
    source = "\n".join(inspect.getsource(value) for value in
                       (run.extract_descriptors, run.build_prototypes, run.evaluate_routing, run.run_bootstraps))
    assert "read_labels" not in source and "label_h5_relpath" not in source


def test_17_m1_case_equal_mean():
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    fitted = fit_prototypes(x, 1, seed=0, domain_index=0)
    assert np.allclose(fitted["centers"], [[2 ** -.5, 2 ** -.5]])
    assert np.allclose(fitted["occupancy"], [1.0])


def test_18_m2_spherical_kmeans_determinism():
    first = fit_prototypes(_vectors(), 2, seed=2, domain_index=1)
    second = fit_prototypes(_vectors(), 2, seed=2, domain_index=1)
    assert first["selected_restart"] == second["selected_restart"]
    assert np.array_equal(first["centers"], second["centers"])


def test_19_m2_occupancy():
    fitted = fit_prototypes(_vectors(), 2, seed=0, domain_index=0)
    assert fitted["active"].tolist() == [True, True]
    assert np.allclose(fitted["occupancy"], [.5, .5])
    with pytest.raises(Blocked, match="inactive M2"):
        fit_prototypes(np.array([[1.0, 0.0], [1.0, 0.0]]), 2, seed=0, domain_index=0)


def test_20_prototype_normalization():
    for M in (1, 2):
        fitted = fit_prototypes(_vectors(), M, seed=1, domain_index=2)
        assert np.allclose(np.linalg.norm(fitted["centers"][fitted["active"]], axis=1), 1.0, atol=1e-12)


def test_21_old_prototype_immutability():
    banks = {domain: fit_prototypes(np.roll(_vectors(), domain, axis=1), 2, seed=0, domain_index=domain)
             for domain in range(3)}
    before = {domain: array_sha256(banks[domain]["centers"]) for domain in (0, 1)}
    route(np.eye(2), banks, (0, 1, 2))
    assert before == {domain: array_sha256(banks[domain]["centers"]) for domain in (0, 1)}


def test_22_cosine_routing():
    banks = {0: dict(centers=np.array([[1.0, 0.0]]), active=np.array([True])),
             1: dict(centers=np.array([[0.0, 1.0]]), active=np.array([True]))}
    routed, scores, entropy = route(np.array([[.9, .1], [.1, .9]]), banks, (0, 1))
    assert routed.tolist() == [0, 1] and scores.shape == (2, 2) and np.isfinite(entropy).all()


def test_23_domain_tie_handling():
    bank = dict(centers=np.array([[1.0, 0.0]]), active=np.array([True]))
    routed, _, _ = route(np.array([[1.0, 0.0]]), {0: bank, 1: copy.deepcopy(bank)}, (0, 1))
    assert routed.tolist() == [0]


def test_24_stage1_excludes_future_domain():
    banks = {i: dict(centers=np.eye(3)[i:i + 1], active=np.array([True])) for i in range(3)}
    routed, scores, _ = route(np.array([[0.0, 0.0, 1.0]]), banks, (0, 1))
    assert routed[0] in (0, 1) and scores.shape[1] == 2


def test_25_stage2_includes_three_domains():
    banks = {i: dict(centers=np.eye(3)[i:i + 1], active=np.array([True])) for i in range(3)}
    assert route(np.array([[0.0, 0.0, 1.0]]), banks, (0, 1, 2))[0].tolist() == [2]


def test_26_routing_confusion():
    rows = [dict(true_domain=0, routed_domain=0, true_domain_margin=.2, route_entropy=.4),
            dict(true_domain=0, routed_domain=1, true_domain_margin=-.1, route_entropy=.6),
            dict(true_domain=1, routed_domain=1, true_domain_margin=.3, route_entropy=.5)]
    summary = routing_summary(rows, 2)
    assert summary["confusion_matrix"].tolist() == [[1, 1], [0, 1]]
    assert summary["per_domain_accuracy"] == [.5, 1.0]


def test_27_cross_expert_3x3_matrix():
    source = inspect.getsource(run.evaluate_segmentation)
    assert "for domain in range(3)" in source and "for expert in range(3)" in source
    assert "len(cross_rows) == 27" in source


def test_28_shared_final_definition():
    source = inspect.getsource(run.evaluate_segmentation)
    assert 'expert = 2 if strategy == "Shared-final"' in source


def test_29_oracle_fixed_mapping():
    source = inspect.getsource(run.evaluate_segmentation)
    assert "ORACLE_EXPERT[DOMAINS[domain]]" in source
    assert "argmax" not in source


def test_30_routed_metric_pixel_aggregation():
    first = pixel_confusion(np.array([0, 1, 2, 2]), np.array([0, 1, 1, 2]))
    second = pixel_confusion(np.array([1, 1]), np.array([1, 2]))
    metrics = segmentation_metrics(first + second)
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 2, 1], [0, 1, 1]]
    assert metrics["mean_foreground_dice"] == pytest.approx((2 * 2 / 6 + 2 * 1 / 4) / 2)


def test_31_bootstrap_determinism():
    first = bootstrap_draw(["c", "a", "b"], seed=2, stage=2, role="val", domain=1, replicate=4)
    second = bootstrap_draw(["b", "c", "a"], seed=2, stage=2, role="val", domain=1, replicate=4)
    assert first == second


def test_32_d1_d5_exact_boundaries():
    result = adjudicate(_passing_d1(), _passing_candidates(), True)
    assert result["scientific_status"] == "PASS_PRES_ROUTING_FEASIBILITY"
    assert result["D1"] is True and result["D5"] is True
    below = _passing_d1()
    below["three_domain_gain"] -= 1e-12
    assert adjudicate(below, _passing_candidates(), True)["scientific_status"] == "FAIL_SNAPSHOT_EXPERT_VALUE"
    assert adjudicate(_passing_d1(), _passing_candidates(), False)["scientific_status"] == "BLOCKED_PROTOCOL_OR_LEAKAGE"


def test_33_smallest_passing_m():
    result = adjudicate(_passing_d1(), _passing_candidates(), True)
    assert result["passing_M"] == [1, 2] and result["selected_M"] == 1


def test_34_controls_cannot_rescue():
    candidates = _passing_candidates()
    candidates[3] = copy.deepcopy(candidates[2])
    with pytest.raises(Blocked, match="M1/M2"):
        adjudicate(_passing_d1(), candidates, True)


def test_35_validation_gt_isolation():
    source = inspect.getsource(run.main)
    assert source.index("router_seal(args.output)") < source.index("predict_experts(") < source.index("evaluate_segmentation(")
    predictor = inspect.getsource(run.predict_experts)
    assert "read_labels" not in predictor and "records" not in tuple(inspect.signature(run.predict_experts).parameters)


def test_36_no_test_construction():
    source = inspect.getsource(protocol.input_audit)
    assert 'for role in ("train_unlabeled", "val")' in source
    assert 'b.records(data_root, p, seed, stage, role)' in source


def test_37_no_optimizer():
    with protocol.isolation_guard(), pytest.raises(Exception, match="forbidden"):
        torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=.1)


def test_38_no_autograd():
    x = torch.ones(1, requires_grad=True)
    with protocol.isolation_guard(), pytest.raises(Blocked, match="autograd.grad"):
        torch.autograd.grad(x.sum(), x)


def test_39_no_backward():
    x = torch.ones(1, requires_grad=True)
    with protocol.isolation_guard(), pytest.raises(Exception, match="forbidden"):
        x.sum().backward()


def test_40_model_checkpoint_immutability(tmp_path: Path):
    model = torch.nn.Linear(2, 1).eval().requires_grad_(False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    checkpoint = dict(path=str(checkpoint_path), sha256=d.sha256(checkpoint_path), checkpoint_id="B0/seed0/stage0")
    with ImmutableModels({"ema_teacher": model}, checkpoint, tmp_path / "guard", {"test": True}):
        with torch.no_grad():
            model(torch.ones(1, 2))
    receipt = d.read(next((tmp_path / "guard/immutability").glob("*.json")))
    assert receipt["bitwise_unchanged"] is True and receipt["extraction_completed"] is True


def test_41_call_graph_compiler(tmp_path: Path):
    graph = protocol.compile_call_graph(tmp_path, _records(), "a" * 40)
    assert (graph["router_extraction_forwards"], graph["router_extraction_case_passes"]) == (162, 1287)
    assert (graph["cross_expert_segmentation_forwards"], graph["cross_expert_segmentation_case_passes"]) == (189, 1485)
    assert graph["bootstrap_operations"] == 60 and graph["total_output_rows"] == 2031


def test_42_create_only_state_machine(tmp_path: Path):
    path = tmp_path / "receipt.json"
    d.write_new(path, {"attempt": 1})
    with pytest.raises(FileExistsError):
        d.write_new(path, {"attempt": 2})
    assert d.read(path) == {"attempt": 1}


def test_43_durable_process_exit_receipt(tmp_path: Path):
    d.write_new(tmp_path / "EXECUTION_COMPLETION.json",
                dict(status="COMMAND_COMPLETED", actual_child_exit_code=0))
    d.write_new(tmp_path / "PROCESS_EXIT.json", dict(actual_child_exit_code=0))
    completion, process = postflight.validate_durable_completion(tmp_path)
    assert completion["status"] == "COMMAND_COMPLETED" and process["actual_child_exit_code"] == 0


def test_44_artifact_manifest(tmp_path: Path):
    required = {
        "PRES_JASCL_INPUT_AUDIT.json", "PRES_JASCL_CALL_GRAPH.json",
        "PRES_JASCL_ROUTER_DESCRIPTOR_MANIFEST.json", "PRES_JASCL_DOMAIN_PROTOTYPE_MANIFEST.json",
        "pres_router_scores.csv", "pres_router_confusion.csv", "pres_router_bootstrap.csv",
        "pres_cross_expert_matrix.csv", "pres_oracle_vs_routed.csv", "PRES_JASCL_STATUS.json",
        "PRES_JASCL_FINAL_REPORT.md", "PRES_JASCL_FAILURES_AND_WARNINGS.md",
        "PRES_JASCL_EXACT_COMMANDS.md", "pytest.xml", "pytest_output.txt",
    }
    for name in required:
        (tmp_path / name).write_text(name)
    manifest = run.artifact_manifest(tmp_path)
    assert manifest["required_outputs_complete"] is True
    assert required.issubset({row["path"] for row in manifest["artifacts"]})


def test_45_private_archive_audit():
    source = inspect.getsource(postflight.main)
    assert "PASS_PRIVATE_ARCHIVE_AUDIT" in source
    assert "PRES_JASCL_PRIVATE_BUNDLE_MANIFEST.json" in source
    assert "reused_durable_phase_hashes_without_rehash=True" in source
    assert "with forbid_forwards(), isolation_guard():" in source


def test_46_report_compiler_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cases = "".join(f'<testcase classname="tests.pres_jascl_v0_1.test_protocol" name="t{i}"/>' for i in range(45))
    junit = tmp_path / "pytest.xml"
    junit.write_text(f'<testsuite tests="45" failures="0" errors="0" skipped="0">{cases}</testsuite>')
    pytest_output = tmp_path / "pytest_output.txt"
    pytest_output.write_text("45 passed")
    report = tmp_path / "report.json"
    monkeypatch.setattr(testing, "source_gate", lambda commit: {"code_commit": commit})
    monkeypatch.setattr(sys, "argv", ["testing", "--junit", str(junit), "--pytest-output", str(pytest_output),
                                      "--output", str(report), "--code-commit", "a" * 40,
                                      "--exact-command", "python -m pytest tests/pres_jascl_v0_1"])
    with pytest.raises(Blocked, match="admission"):
        testing.main()
    assert not report.exists()


def test_47_gate_aggregation_and_report_are_serializable(tmp_path: Path):
    routing = {}
    for seed in range(3):
        routing[seed] = {}
        for M in (1, 2):
            routing[seed][M] = {}
            for stage in (1, 2):
                routing[seed][M][stage] = dict(
                    accuracy=.99, per_domain_accuracy=[.99] * (stage + 1),
                    margin_p05=.2, margin_p10=.3, margin_median=.4,
                    route_entropy_mean=.5, route_entropy_p05=.3,
                    route_entropy_p10=.4, route_entropy_median=.5,
                )
    bootstrap = [dict(seed=seed, M=M, stage_index=stage, replicate=replicate,
                      macro_accuracy=.98, minimum_bootstrap_occupancy=.4)
                 for seed in range(3) for M in (1, 2) for stage in (1, 2) for replicate in range(5)]
    banks = {seed: {M: {domain: dict(occupancy=np.array([1.0]) if M == 1 else np.array([.5, .5]),
                                          within_domain_prototype_cosine_distance=None if M == 1 else .8)
                        for domain in range(3)} for M in (1, 2)} for seed in range(3)}
    strategies = []
    scores = {"Shared-final": .70, "Oracle-snapshot": .73,
              "Prototype-routed-M1": .721, "Prototype-routed-M2": .716}
    for seed in range(3):
        for domain in range(3):
            for strategy, score in scores.items():
                strategies.append(dict(seed=seed, true_domain=domain, strategy=strategy,
                                       mean_foreground_dice=score))
    d1, candidates = run.gate_inputs(routing, bootstrap, {1: [.96], 2: [.91, .92]}, banks, strategies)
    decision = adjudicate(d1, candidates, True)
    assert decision["scientific_status"] == "PASS_PRES_ROUTING_FEASIBILITY"
    d.write_new(tmp_path / "PRES_JASCL_CALL_GRAPH.json", {"status": "synthetic"})
    counters = dict(router_extraction_forwards=162, router_extraction_case_passes=1287,
                    cross_expert_segmentation_forwards=189, cross_expert_segmentation_case_passes=1485,
                    model_guards=12, bootstrap_operations=60, total_output_rows=2031)
    metadata = dict(exact_test_command="python -m pytest", exact_command=["run", "--synthetic"])
    run.report(tmp_path, metadata, counters, d1, candidates, decision)
    assert d.read(tmp_path / "PRES_JASCL_STATUS.json")["D5"] is True
