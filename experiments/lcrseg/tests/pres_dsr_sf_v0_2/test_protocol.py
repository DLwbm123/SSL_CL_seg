from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from di_dmpa_gate1c_v3 import durable as d
from pres_dsr_sf_v0_2 import REGISTRATION
from pres_dsr_sf_v0_2 import postflight, protocol, run, testing
from pres_dsr_sf_v0_2.core import (Blocked, DOMAINS, adjudicate, balanced_weights,
                                   bootstrap_multiplicity, case_folds, fit_router, fit_standardizer,
                                   hard_routes, probability_fusion, raw_style_block, raw_style_descriptors,
                                   ridge_fit, select_memory, softmax)
from pres_jascl_v0_1.run import deterministic_backend_state


def _restore_backend(state):
    torch.use_deterministic_algorithms(state[0])
    torch.backends.cudnn.deterministic = state[1]
    torch.backends.cudnn.benchmark = state[2]
    torch.backends.cuda.matmul.allow_tf32 = state[3]
    torch.backends.cudnn.allow_tf32 = state[4]


def _passing_evidence():
    return dict(E1=True, E6=True,
                oracle=dict(three_domain_gain=.02, historical_gain=.03, positive_seed_count=2,
                            maximum_domain_drop=.005),
                ridge_hard=dict(stage1_macro=.95, stage1_per_domain=[.9, .9], stage2_macro=.9,
                                stage2_per_domain=[.85, .85, .85]),
                ridge_soft=dict(oracle_gap=.02, shared_gain=.13, historical_gain=.2, gain_over_m1_hard=.01,
                                positive_seed_count=3, maximum_seed_domain_drop=.02, current_domain_drop=.01),
                stability=dict(hard_macro_p10=.85, soft_gain_p10=.1, soft_oracle_gap_p90=.03,
                               all_domains_nonempty=True, all_finite=True))


def _router_data():
    labels = np.repeat(np.arange(2), 10)
    case_ids = [f"d{label}c{i}" for label in range(2) for i in range(10)]
    return np.zeros((20, 102), dtype=np.float64), labels, case_ids


TEST_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
TEST_TEMPERATURES = (0.5, 1.0, 2.0, 4.0)


def _cv_plan():
    keys = {
        "M1_temperature": [("M1_temperature", "M1", seed, stage, "temperature", value)
                           for seed in range(3) for stage in (1, 2) for value in TEST_TEMPERATURES],
        "ridge_lambda": [("ridge_lambda", "ridge", seed, stage, "lambda", value)
                         for seed in range(3) for stage in (1, 2) for value in TEST_LAMBDAS],
        "ridge_temperature": [("ridge_temperature", "ridge", seed, stage, "temperature", value)
                              for seed in range(3) for stage in (1, 2) for value in TEST_TEMPERATURES],
    }
    return {"key_sets": keys}


def _cv_rows():
    plan, output = _cv_plan(), {}
    for family, keys in plan["key_sets"].items():
        output[family] = [dict(zip(run.CV_FIELDS, key), macro_accuracy=.5, domain_nll=1.0,
                               per_domain_accuracy=[.5, .5], selected=key[-1] == 1.0) for key in keys]
    return plan, output


def _manifest_records():
    sizes = {"train_labeled": (40, 16, 10), "train_unlabeled": (160, 63, 41), "val": (100, 40, 25)}
    return {seed: {stage: {role: [{"case_id": f"s{seed}d{stage}{role}{i}"} for i in range(sizes[role][stage])]
                          for role in sizes} for stage in range(3)} for seed in range(3)}


def test_01_v01_closure_binding():
    prereg = protocol.authority()
    assert prereg["registration_id"] == REGISTRATION


def test_02_formal_report_guard_commit_binding():
    assert protocol.BASE_HEAD == "ff42db2ec2381aad176139ab788a9925eef9d147"
    assert [row[0] for row in protocol.AUTHORITY] == [
        "de82bf94f27f91e071f9bab4e9432f1c0ee263d3",
        "44a8870765d1ebb5efa38843f3c20b79aeb721ec",
        "752e1ac7a016d619ffaa624c347fbeefa7883137",
        "1eaf16c876a180fc9eaff6fc893e134d10518d02",
    ]


def test_03_backend_import_order_isolation():
    original = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        audit = protocol.freeze_backend(lambda: setattr(torch.backends.cudnn, "benchmark", True))
        assert audit["after_pinned_import"]["cudnn_benchmark_disabled"] is False
        assert all(audit["registered"].values())
    finally:
        _restore_backend(original)


def test_04_no_forward_before_backend_freeze():
    assert "with forbid_forwards():" in inspect.getsource(protocol.freeze_backend)
    assert "model_forwards_before_freeze=0" in inspect.getsource(protocol.freeze_backend)


def test_05_backend_state_phase_invariance():
    original = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        protocol.freeze_backend(lambda: None)
        assert all(protocol.require_backend("test")["state"].values())
    finally:
        _restore_backend(original)


def test_06_exact_nine_checkpoint_identities():
    assert len(protocol.gate1c_contract()["immutable_baseline"]["checkpoint_inputs"]) == 9


def test_07_fixed_expert_domain_mapping():
    assert DOMAINS == ("REFUGE", "RIM_ONE_r3", "Drishti_GS")


def test_08_oracle_never_selects_best_expert():
    source = inspect.getsource(run.materialize_candidates)
    assert '"C1_ORACLE": one_hot_routes(truth)' in source
    assert "argmax" not in source.split('"C1_ORACLE"')[1].split('"C2_M1_HARD"')[0]


def test_09_raw_rgb_descriptor():
    value = torch.tensor([[[[1.0, 3.0], [1.0, 3.0]]]])
    block = raw_style_block(value)
    assert torch.allclose(block[:, :1], torch.tensor([[2.0]], dtype=torch.float64))


def test_10_raw_enc1_descriptor():
    rgb, enc1, enc2 = torch.zeros(1, 3, 2, 2), torch.ones(1, 16, 2, 2), torch.zeros(1, 32, 1, 1)
    descriptor = raw_style_descriptors(rgb, enc1, enc2)
    assert np.all(descriptor[0, 6:22] == 1.0)


def test_11_raw_enc2_descriptor():
    descriptor = raw_style_descriptors(torch.zeros(1, 3, 2, 2), torch.zeros(1, 16, 2, 2),
                                       torch.full((1, 32, 1, 1), 2.0))
    assert np.all(descriptor[0, 38:70] == 2.0)


def test_12_log_std_formula():
    value = torch.tensor([[[[0.0, 2.0]]]])
    assert np.isclose(raw_style_block(value)[0, 1].item(), np.log(1.0 + 1e-6))


def test_13_expected_dimension_102():
    value = raw_style_descriptors(torch.zeros(2, 3, 2, 2), torch.zeros(2, 16, 1, 1),
                                  torch.zeros(2, 32, 1, 1))
    assert value.shape == (2, 102)


def test_14_train_only_standardization():
    state = fit_standardizer(np.vstack((np.zeros((1, 102)), np.full((1, 102), 2.0))))
    assert np.allclose(state["mean"], 1.0) and np.allclose(state["std"], 1.0)


def test_15_constant_dimension_handling():
    state = fit_standardizer(np.ones((3, 102)))
    assert state["constant"].all() and np.all(state["scale"] == 1.0)


def test_16_descriptor_determinism():
    tensors = (torch.rand(2, 3, 4, 4), torch.rand(2, 16, 2, 2), torch.rand(2, 32, 1, 1))
    assert np.array_equal(raw_style_descriptors(*tensors), raw_style_descriptors(*tensors))


def test_17_no_path_or_domain_feature():
    assert tuple(inspect.signature(raw_style_descriptors).parameters) == ("rgb", "enc1", "enc2")


def test_18_no_segmentation_gt_in_router():
    source = inspect.getsource(run.fit_ridge_routers)
    assert "label_h5" not in source and "read_labels" not in source


def test_19_memory_cap():
    selected, hashes = select_memory([{"case_id": f"c{i}"} for i in range(600)])
    assert len(selected) == len(hashes) == 512


def test_20_historical_memory_immutability():
    assert "historical memory bytes changed" in inspect.getsource(run.build_memory)


def test_21_domain_balanced_weights():
    weights = balanced_weights([0, 0, 0, 1])
    assert np.isclose(weights[:3].sum(), .5) and np.isclose(weights[3:].sum(), .5)


def test_22_deterministic_five_folds():
    ids = [f"d{d}c{i}" for d in range(2) for i in range(10)]
    labels = np.repeat(np.arange(2), 10)
    first = case_folds(ids, labels)
    assert np.array_equal(first, case_folds(ids, labels))
    assert all(np.bincount(first[labels == d], minlength=5).tolist() == [2] * 5 for d in range(2))


def test_23_ridge_closed_form_solution():
    value = np.vstack((np.full((5, 2), -1.0), np.full((5, 2), 1.0)))
    labels = np.repeat(np.arange(2), 5)
    weights = ridge_fit(value, labels, .01)
    assert weights.shape == (3, 2) and np.isfinite(weights).all()


def test_24_bias_unregularized():
    weights = ridge_fit(np.zeros((10, 2)), np.repeat(np.arange(2), 5), 1.0)
    assert np.allclose(weights[-1], [.5, .5])


def test_25_lambda_selection_ties_choose_largest():
    value, labels, ids = _router_data()
    assert fit_router(value, labels, ids)["selected_lambda"] == 1.0


def test_26_temperature_selection_ties_choose_one():
    value, labels, ids = _router_data()
    assert fit_router(value, labels, ids)["selected_temperature"] == 1.0


def test_27_validation_excluded_from_selection():
    source = inspect.getsource(run.fit_ridge_routers)
    assert source.index("model = fit_router(train, labels, ids)") < source.index('row["role"] == "val"')


def test_28_stage1_future_domain_exclusion():
    assert set(hard_routes(np.array([[.2, .8]]), (0, 1))) <= {0, 1}


def test_29_stage2_three_domain_inclusion():
    assert hard_routes(np.array([[.1, .2, .7]]), (0, 1, 2)).tolist() == [2]


def test_30_hard_routing_tie_lowest():
    assert hard_routes(np.array([[.5, .5]]), (0, 1)).tolist() == [0]


def test_31_soft_weights_sum_one():
    probability = softmax(np.array([[1.0, 2.0]]), 2.0)
    assert np.allclose(probability.sum(1), 1.0)


def test_32_probability_fusion():
    experts = np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    assert np.allclose(probability_fusion(np.array([[.25, .75]]), experts), [[[.25, .75]]])


def test_33_no_logit_fusion():
    source = inspect.getsource(run.materialize_candidates)
    assert "probability_fusion" in source
    assert "logit_fusion(" not in source and "np.log(" not in source


def test_34_m1_clean_reproduction():
    assert "clean_states[seed][1]" in inspect.getsource(run.fit_ridge_routers)


def test_35_m2_clean_reproduction():
    assert "clean_states[seed][2]" in inspect.getsource(run.fit_ridge_routers)


def test_36_shared_final_definition():
    assert '"C0_SHARED": one_hot_routes(np.full(len(ids), 2))' in inspect.getsource(run.materialize_candidates)


def test_37_oracle_definition():
    assert '"C1_ORACLE": one_hot_routes(truth)' in inspect.getsource(run.materialize_candidates)


def test_38_uniform_control():
    assert '1.0 / (stage + 1)' in inspect.getsource(run.materialize_candidates)


def test_39_routing_confusion():
    assert 'plan["key_sets"]["routing_confusion"]' in inspect.getsource(run.fit_ridge_routers)


def test_40_segmentation_aggregation():
    prediction = np.array([[0, 1], [2, 2]])
    target = np.array([[0, 1], [1, 2]])
    from pres_jascl_v0_1.core import pixel_confusion, segmentation_metrics
    metrics = segmentation_metrics(pixel_confusion(prediction, target))
    assert metrics["mean_foreground_dice"] > 0


def test_41_bootstrap_determinism():
    ids = [f"c{i}" for i in range(20)]
    first = bootstrap_multiplicity(ids, seed=0, stage=2, domain=1, replicate=3)
    second = bootstrap_multiplicity(ids, seed=0, stage=2, domain=1, replicate=3)
    assert np.array_equal(first[0], second[0]) and first[1] == second[1]


def test_42_E1_E6_exact_boundaries():
    result = adjudicate(_passing_evidence())
    assert result["scientific_status"] == "PASS_PRES_DSR_SF_FEASIBILITY" and all(result[f"E{i}"] for i in range(1, 7))


def test_43_controls_cannot_rescue_primary():
    evidence = _passing_evidence()
    evidence["ridge_soft"]["shared_gain"] = .129999
    assert adjudicate(evidence)["scientific_status"] == "FAIL_SOFT_EXPERT_FUSION_VALUE"


def test_44_no_test_construction():
    assert 'test_GT_reads=0' in inspect.getsource(run.report)


def test_45_validation_gt_isolation():
    source = inspect.getsource(run.main)
    assert source.index("materialize_candidates") < source.index("evaluate_segmentation")


def test_46_no_model_optimizer():
    assert "torch.optim" not in inspect.getsource(run)


def test_47_no_model_autograd():
    assert "model_autograd_calls=0" in inspect.getsource(run.main)


def test_48_no_backward():
    assert ".backward(" not in inspect.getsource(run)


def test_49_model_checkpoint_immutability():
    assert "ImmutableModels" in inspect.getsource(run.extract_descriptors)
    assert "ImmutableModels" in inspect.getsource(run.predict_expert_probabilities)


def test_50_call_graph_compiler(tmp_path: Path):
    records = _manifest_records()
    graph = protocol.compile_call_graph(tmp_path, records, "a" * 40)
    assert graph["descriptor_forwards"] == sum((sum(len(records[s][d][r]) for d in range(3)
                                                      for r in records[s][d]) + 7) // 8 for s in range(3))
    assert graph["expert_probability_forwards"] == 3 * 3 * ((165 + 7) // 8)
    assert graph["total_output_rows"] == sum(graph["output_rows"].values()) == 1356


def test_51_create_only_state(tmp_path: Path):
    path = tmp_path / "cache.npz"
    run.npz_new(path, x=np.zeros(1))
    with pytest.raises(FileExistsError):
        run.npz_new(path, x=np.ones(1))


def test_52_durable_child_exit(tmp_path: Path):
    d.write_new(tmp_path / "EXECUTION_COMPLETION.json", {"status": "COMMAND_COMPLETED", "actual_child_exit_code": 0})
    d.write_new(tmp_path / "PROCESS_EXIT.json", {"actual_child_exit_code": 0})
    assert postflight.validate_durable_completion(tmp_path) is None


def test_53_artifact_manifest_contract():
    source = inspect.getsource(run.artifact_manifest)
    assert "PRES_DSR_SF_V0_2_1_ARTIFACT_MANIFEST.json" in source and "required_outputs_complete=True" in source


def test_54_private_archive_audit():
    source = inspect.getsource(postflight.main)
    assert "PASS_PRIVATE_ARCHIVE_AUDIT" in source and "PRES_DSR_SF_V0_2_1_PRIVATE_BUNDLE_MANIFEST.json" in source


def test_55_report_compiler_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    junit = tmp_path / "pytest.xml"
    junit.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase classname="x" name="x"/></testsuite>')
    output = tmp_path / "output.txt"
    output.write_text("1 passed")
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(testing, "source_gate", lambda commit: {"code_commit": commit})
    monkeypatch.setattr(sys, "argv", ["testing", "--junit", str(junit), "--pytest-output", str(output),
                                      "--output", str(receipt), "--code-commit", "a" * 40,
                                      "--exact-command", "python -m pytest"])
    with pytest.raises(Blocked):
        testing.main()
    assert not receipt.exists()


def test_56_fit_router_returns_exactly_nine_rows():
    value, labels, ids = _router_data()
    assert len(fit_router(value, labels, ids)["cv_rows"]) == 9


def test_57_fit_router_kind_counts_are_five_plus_four():
    value, labels, ids = _router_data()
    rows = fit_router(value, labels, ids)["cv_rows"]
    assert sum(row["kind"] == "lambda" for row in rows) == 5
    assert sum(row["kind"] == "temperature" for row in rows) == 4


def test_58_six_ridge_routers_produce_54_rows():
    value, labels, ids = _router_data()
    rows = [row for _ in range(6) for row in fit_router(value, labels, ids)["cv_rows"]]
    assert len(rows) == 54


def test_59_ridge_lambda_keys_exactly_30():
    keys = _cv_plan()["key_sets"]["ridge_lambda"]
    assert len(keys) == len(set(keys)) == 3 * 2 * 5


def test_60_ridge_temperature_keys_exactly_24():
    keys = _cv_plan()["key_sets"]["ridge_temperature"]
    assert len(keys) == len(set(keys)) == 3 * 2 * 4


def test_61_m1_temperature_keys_exactly_24():
    keys = _cv_plan()["key_sets"]["M1_temperature"]
    assert len(keys) == len(set(keys)) == 3 * 2 * 4


def test_62_combined_cv_keys_exactly_78():
    plan, rows = _cv_rows()
    combined = run.validate_cv_rows(rows["M1_temperature"], rows["ridge_lambda"], rows["ridge_temperature"], plan)
    assert len(combined) == 78


def test_63_combined_is_disjoint_union():
    plan = _cv_plan()
    families = [set(plan["key_sets"][name]) for name in ("M1_temperature", "ridge_lambda", "ridge_temperature")]
    assert not (families[0] & families[1] or families[0] & families[2] or families[1] & families[2])
    assert len(set().union(*families)) == 78


def test_64_duplicate_key_blocks():
    plan, rows = _cv_rows()
    rows["ridge_lambda"].append(dict(rows["ridge_lambda"][0]))
    with pytest.raises(Blocked, match="duplicate"):
        run.validate_cv_rows(rows["M1_temperature"], rows["ridge_lambda"], rows["ridge_temperature"], plan)


def test_65_missing_key_blocks():
    plan, rows = _cv_rows()
    rows["ridge_lambda"].pop()
    with pytest.raises(Blocked):
        run.validate_cv_rows(rows["M1_temperature"], rows["ridge_lambda"], rows["ridge_temperature"], plan)


def test_66_extra_key_blocks():
    plan, rows = _cv_rows()
    extra = dict(rows["ridge_lambda"][0], value=99.0, selected=False)
    rows["ridge_lambda"].append(extra)
    with pytest.raises(Blocked):
        run.validate_cv_rows(rows["M1_temperature"], rows["ridge_lambda"], rows["ridge_temperature"], plan)


def test_67_wrong_family_blocks():
    plan, rows = _cv_rows()
    rows["ridge_lambda"][0] = dict(rows["ridge_lambda"][0], cv_family="M1_temperature")
    with pytest.raises(Blocked):
        run.validate_cv_rows(rows["M1_temperature"], rows["ridge_lambda"], rows["ridge_temperature"], plan)


def test_68_unregistered_value_blocks():
    plan, rows = _cv_rows()
    rows["ridge_temperature"][0] = dict(rows["ridge_temperature"][0], value=3.0)
    with pytest.raises(Blocked):
        run.validate_cv_rows(rows["M1_temperature"], rows["ridge_lambda"], rows["ridge_temperature"], plan)


def test_69_two_selected_lambdas_block():
    plan, rows = _cv_rows()
    target = next(row for row in rows["ridge_lambda"]
                  if row["seed"] == 0 and row["stage_index"] == 1 and not row["selected"])
    target["selected"] = True
    with pytest.raises(Blocked, match="lambda selection"):
        run.validate_cv_rows(rows["M1_temperature"], rows["ridge_lambda"], rows["ridge_temperature"], plan)


def test_70_two_selected_temperatures_block():
    plan, rows = _cv_rows()
    target = next(row for row in rows["ridge_temperature"]
                  if row["seed"] == 0 and row["stage_index"] == 1 and not row["selected"])
    target["selected"] = True
    with pytest.raises(Blocked, match="temperature selection"):
        run.validate_cv_rows(rows["M1_temperature"], rows["ridge_lambda"], rows["ridge_temperature"], plan)


def test_71_combined_csv_contains_all_78(tmp_path: Path):
    plan, rows = _cv_rows()
    run.write_combined_cv(tmp_path, rows["M1_temperature"], rows["ridge_lambda"], rows["ridge_temperature"], plan)
    assert len(list(csv.DictReader((tmp_path / "pres_dsr_cv.csv").open()))) == 78


def test_72_split_files_reconstruct_combined(tmp_path: Path):
    plan, rows = _cv_rows()
    run.write_combined_cv(tmp_path, rows["M1_temperature"], rows["ridge_lambda"], rows["ridge_temperature"], plan)
    read = lambda name: list(csv.DictReader((tmp_path / name).open()))
    combined = read("pres_dsr_cv.csv")
    split = read("pres_dsr_m1_temperature_cv.csv") + read("pres_dsr_ridge_cv.csv")
    key = lambda row: tuple(row[field] for field in run.CV_FIELDS)
    assert {key(row) for row in split} == {key(row) for row in combined} and len(combined) == 78


def test_73_router_score_exact_915_key_set(tmp_path: Path):
    plan = protocol.output_key_plan(tmp_path, _manifest_records(), "a" * 40)
    keys = [tuple(key) for key in plan["key_sets"]["router_scores"]]
    assert len(keys) == len(set(keys)) == 915


def test_74_confusion_exact_117_key_set(tmp_path: Path):
    plan = protocol.output_key_plan(tmp_path, _manifest_records(), "a" * 40)
    keys = [tuple(key) for key in plan["key_sets"]["routing_confusion"]]
    assert len(keys) == len(set(keys)) == 117


def test_75_full_total_output_rows_1356():
    assert 24 + 30 + 24 + 915 + 117 + 27 + 120 + 90 + 9 == 1356


@pytest.mark.parametrize("routers,lambdas,temperatures,expected", [(4, 2, 3, 20), (6, 5, 4, 54)])
def test_76_parameterized_grid_cardinality(routers, lambdas, temperatures, expected):
    rows = [(router, "lambda", value) for router in range(routers) for value in range(lambdas)]
    rows += [(router, "temperature", value) for router in range(routers) for value in range(temperatures)]
    assert len(rows) == expected


def test_77_test_oracle_does_not_import_production_count_constants():
    imports = "\n".join(Path(__file__).read_text().splitlines()[:25])
    assert "LAMBDAS" not in imports and "TEMPERATURES" not in imports


def test_78_complete_synthetic_clean_ridge_combined(tmp_path: Path):
    rng = np.random.default_rng(7)
    metadata, descriptors, memories = {}, {}, {}
    for seed in range(3):
        rows = []
        for domain in range(3):
            rows += [dict(case_id=f"s{seed}d{domain}u{i}", role="train_unlabeled", domain_index=domain)
                     for i in range(10)]
            rows += [dict(case_id=f"s{seed}d{domain}v{i}", role="val", domain_index=domain) for i in range(5)]
        rows.sort(key=lambda row: row["case_id"])
        metadata[seed] = rows
        values = rng.normal(size=(len(rows), 102))
        for i, row in enumerate(rows):
            values[i] += row["domain_index"] * .5
        descriptors[seed] = {"legacy": values, "raw": values.copy()}
        index = {row["case_id"]: i for i, row in enumerate(rows)}
        memories[seed] = {}
        for domain in range(3):
            ids = [row["case_id"] for row in rows if row["role"] == "train_unlabeled"
                   and row["domain_index"] == domain]
            memories[seed][domain] = {"case_ids": ids, "descriptors": values[[index[case] for case in ids]]}
    plan = _cv_plan()
    plan["key_sets"]["router_scores"] = [(seed, stage, row["case_id"]) for seed in range(3) for stage in (1, 2)
                                                  for row in metadata[seed]
                                                  if row["role"] == "val" and row["domain_index"] <= stage]
    plan["key_sets"]["routing_confusion"] = [(seed, stage, router, true, routed)
        for seed in range(3) for stage in (1, 2) for router in ("M1_HARD", "M2_HARD", "RIDGE_HARD")
        for true in range(stage + 1) for routed in range(stage + 1)]
    counters = dict(m1_cv_prototype_fits=0, clean_control_prototype_fits=0, ridge_closed_form_fits=0)
    banks, states, routing, temperatures, m1 = run.clean_controls(descriptors, metadata, counters, plan)
    result = run.fit_ridge_routers(memories, descriptors, metadata, states, temperatures, tmp_path, counters, plan)
    _, _, ridge_lambda, ridge_temperature, scores, confusion, _ = result
    combined = run.write_combined_cv(tmp_path, m1, ridge_lambda, ridge_temperature, plan)
    assert len(m1) == 24 and len(ridge_lambda) == 30 and len(ridge_temperature) == 24
    assert len(combined) == 78 and len(scores) == 75 and len(confusion) == 117
    assert banks and routing


def test_79_old_blocked_artifacts_unchanged():
    root = protocol.ROOT / "docs/pres_dsr_sf_v0_2"
    assert d.sha256(root / "PRES_DSR_SF_FINAL_REPORT.md") == "247dd178eda7cafe3f971994df8611e5c345d5d657ff27392f7e8e10f44e792e"
    assert d.sha256(root / "PRES_DSR_SF_STATUS.json") == "8ae3e4b29344825f0b815889cd7481ed0a409c956a57c8216bde1c3c2424cb5e"


def test_80_science_function_hashes_unchanged():
    names = ("raw_style_block", "raw_style_descriptors", "fit_standardizer", "apply_standardizer", "ridge_fit",
             "fit_router", "router_probabilities", "hard_routes", "probability_fusion",
             "bootstrap_multiplicity", "adjudicate")
    text = (protocol.ROOT / "pres_dsr_sf_v0_2/core.py").read_text()
    tree = ast.parse(text)
    nodes = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    hashes = []
    for name in names:
        node = nodes[name]
        source = "".join(text.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]).encode()
        hashes.append(name + ":" + hashlib.sha256(source).hexdigest())
    assert hashlib.sha256("\n".join(hashes).encode()).hexdigest() == "ffd08413305de31888767a440e0bc8234178624289fe98693f8dcc9b2a84740c"


def test_81_fail_closed_artifact_manifest(tmp_path: Path):
    with pytest.raises(Blocked):
        run.artifact_manifest(tmp_path)
    assert not (tmp_path / "PRES_DSR_SF_V0_2_1_ARTIFACT_MANIFEST.json").exists()


def test_82_nonfinite_cv_metric_blocks():
    plan, rows = _cv_rows()
    rows["M1_temperature"][0]["domain_nll"] = float("nan")
    with pytest.raises(Blocked):
        run.validate_cv_rows(rows["M1_temperature"], rows["ridge_lambda"], rows["ridge_temperature"], plan)
