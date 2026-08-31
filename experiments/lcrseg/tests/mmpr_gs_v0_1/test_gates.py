import copy
import numpy as np
import pytest

from di_dmpa_gate1c_v3 import durable as d
from mmpr_gs_v0_1 import core, diagnostic, report, run


def fixture():
    _, p = run.authority()
    pairs = p["gradient_diagnostic"]["batch_pairs"]
    mass = []
    for seed in range(3):
        for stage in range(3):
            for j in range(55):
                for c in range(3):
                    mass.append(dict(seed=seed, stage_index=stage, case_id=str(j), class_id=c,
                                     mass_difference=0, R1_null_mass=1, MMPR_null_mass=1, GT_received_by_builder=False))
    units = [dict(seed=s, stage_index=t, class_id=c, candidate="Q1", weighted_error=.3, reference_error=.5, precision_drop=-.2)
             for s in range(3) for t in range(3) for c in (1, 2)]
    validation = dict(case_count=495, unit_count=9, mass_rows=mass, units=units, validation_GT="evaluator_only", model_forwards=0)
    results = []
    for index, pair in enumerate(pairs):
        retention = [dict(block=block, stage_index=pair["stage_index"], norm_ratio=.8, projected_zero=False,
                          raw_dot=-.1, projected_dot=0., raw_norm=1., projected_norm=.8)
                     for block in ("global", *core.BLOCKS)]
        r1 = -.05 if index < 43 else .15
        alignment = [dict(candidate=candidate, block="global", stage_index=pair["stage_index"], cosine=cosine)
                     for candidate, cosine in (("Q0", r1), ("Q1", r1+.1))]
        iso = dict(model_bitwise_unchanged=True, banks_unchanged=True, rng_restored=True, teacher_gradients="None",
                   bank_gradients="None", parameter_grad_fields="None", model_optimizer_steps=0, transport_optimizer_steps=0,
                   backward_called=False, optimizer_constructed=False, hidden_gt_training_usage="none", test_gt_usage="none")
        prows = [dict(case_id=case, class_id=c, mass_difference=0, R1_null_mass=1, MMPR_null_mass=1, GT_received_by_builder=False)
                 for case in pair["unlabeled_case_ids"] for c in range(3)]
        results.append(dict(pair=pair, alignment=alignment, mass_rows=prows, retention=retention,
                            blockwise=[{}]*12, components=[{}]*21, precision=[{}]*21, isolation=iso, checkpoint_guard_pass=True))
    return validation, results, pairs


def test_all_five_gates_pass_only_with_complete_denominators():
    val, res, pairs = fixture()
    decision = report.adjudicate(val, res, pairs)
    assert decision["status"] == "PASS_MMPR_GS_FEASIBILITY" and all(decision["gates"].values())
    assert decision["F3"]["reference_negative_count"] == 43
    assert decision["F2"]["units"] == 18 and decision["F3"]["pairs"] == 72


@pytest.mark.parametrize("missing", ["unit", "pair", "duplicate_pair", "mass", "components", "stage", "case_class"])
def test_exact_evidence_denominators_fail_closed(missing):
    val, res, pairs = fixture()
    if missing == "unit": val["units"].pop()
    if missing == "pair": res.pop()
    if missing == "duplicate_pair": res[-1] = copy.deepcopy(res[0])
    if missing == "mass": val["mass_rows"].pop()
    if missing == "components": res[0]["components"].pop()
    if missing == "stage": res[0]["alignment"][0]["stage_index"] = 1
    if missing == "case_class": res[0]["mass_rows"][0]["class_id"] = 4
    with pytest.raises(core.Blocked): report.adjudicate(val, res, pairs)


@pytest.mark.parametrize("violation", ["mass", "null", "GT", "teacher", "bank", "model", "checkpoint", "rng", "grad", "step", "test_GT"])
def test_F1_F5_isolation_blocks_science(violation):
    val, res, pairs = fixture()
    if violation == "mass": val["mass_rows"][0]["mass_difference"] = 1
    if violation == "null": val["mass_rows"][0]["MMPR_null_mass"] = 0
    if violation == "GT": val["mass_rows"][0]["GT_received_by_builder"] = True
    if violation == "teacher": res[0]["isolation"]["teacher_gradients"] = "Tensor"
    if violation == "bank": res[0]["isolation"]["bank_gradients"] = "Tensor"
    if violation == "model": res[0]["isolation"]["model_bitwise_unchanged"] = False
    if violation == "checkpoint": res[0]["checkpoint_guard_pass"] = False
    if violation == "rng": res[0]["isolation"]["rng_restored"] = False
    if violation == "grad": res[0]["isolation"]["parameter_grad_fields"] = "Tensor"
    if violation == "step": res[0]["isolation"]["model_optimizer_steps"] = 1
    if violation == "test_GT": res[0]["isolation"]["test_gt_usage"] = "used"
    assert report.adjudicate(val, res, pairs)["status"] == "BLOCKED_PROTOCOL_OR_LEAKAGE"


@pytest.mark.parametrize("violation", ["macro", "improving", "precision", "undefined"])
def test_F2_failure_cannot_be_rescued_by_controls(violation):
    val, res, pairs = fixture()
    if violation == "macro":
        for u in val["units"]: u["weighted_error"] = .49
    if violation == "improving":
        for u in val["units"][:7]: u["weighted_error"] = .5
    if violation == "precision": val["units"][0]["precision_drop"] = np.nextafter(.02, 1)
    if violation == "undefined": val["units"][0]["weighted_error"] = None
    val["units"].extend([dict(u, candidate="Q2", weighted_error=0., precision_drop=-1.) for u in val["units"]])
    assert report.adjudicate(val, res, pairs)["status"] == "FAIL_MATCHED_MASS_RANKING_NOT_SUPPORTED"


def test_F2_exact_precision_drop_boundary():
    val, res, pairs = fixture()
    val["units"][0]["precision_drop"] = .02
    assert report.adjudicate(val, res, pairs)["gates"]["F2"]


@pytest.mark.parametrize("violation", ["negative", "median", "stage", "undefined"])
def test_F3_boundaries_and_undefined(violation):
    val, res, pairs = fixture()
    if violation == "negative":
        for r in res[:44]: r["alignment"][1]["cosine"] = -.01
    if violation == "median":
        for r in res: r["alignment"][1]["cosine"] = r["alignment"][0]["cosine"]
    if violation == "stage":
        for r in res:
            if r["pair"]["stage_index"] == 1: r["alignment"][1]["cosine"] = r["alignment"][0]["cosine"]-.051
    if violation == "undefined": res[0]["alignment"][1]["cosine"] = None
    result = report.adjudicate(val, res, pairs)
    assert not result["gates"]["F3"] and result["status"] == "FAIL_RAW_GRADIENT_COMPATIBILITY"


@pytest.mark.parametrize("violation", ["dot", "zero", "median", "p10", "stage"])
def test_F4_projection_retention_boundaries(violation):
    val, res, pairs = fixture()
    if violation == "dot": res[0]["retention"][0]["projected_dot"] = np.nextafter(-1e-10, -1)
    if violation == "zero": res[0]["retention"][0]["projected_zero"] = True
    if violation == "median":
        for r in res: r["retention"][0]["norm_ratio"] = np.nextafter(.5, 0)
    if violation == "p10":
        for r in res[:9]: r["retention"][0]["norm_ratio"] = .19
    if violation == "stage":
        for r in res:
            if r["pair"]["stage_index"] == 1: r["retention"][0]["norm_ratio"] = .39
    assert report.adjudicate(val, res, pairs)["status"] == "FAIL_PROJECTED_SIGNAL_RETENTION"


def test_projected_dot_and_norm_ratio_exact_thresholds():
    val, res, pairs = fixture()
    for r in res:
        r["retention"][0]["projected_dot"] = -1e-10
        r["retention"][0]["norm_ratio"] = .5
    assert report.adjudicate(val, res, pairs)["gates"]["F4"]


def test_failures_recorded_without_control_rescue_or_precedence_change():
    val, res, pairs = fixture()
    for u in val["units"]: u["weighted_error"] = .5
    for r in res: r["alignment"][1]["cosine"] = -.9; r["retention"][0]["norm_ratio"] = .01
    result = report.adjudicate(val, res, pairs)
    assert result["status"] == "FAIL_MATCHED_MASS_RANKING_NOT_SUPPORTED"
    assert len(result["all_scientific_failures"]) == 3 and not result["control_rescue_allowed"]


def test_nonfinite_report_is_engineering_failure():
    val, res, pairs = fixture()
    res[0]["retention"][0]["projected_dot"] = np.nan
    with pytest.raises(core.Blocked) as ex: report.adjudicate(val, res, pairs)
    assert ex.value.status == "BLOCKED_NUMERICAL_FAILURE"


def test_durable_actual_exit_required_not_SSH_status(tmp_path):
    d.write_new(tmp_path/"PROCESS_EXIT.json", {"actual_child_exit_code": 1, "ssh_exit": 0})
    d.write_new(tmp_path/"report.json", {"status": "PASS"})
    with pytest.raises(core.Blocked, match="process failed"):
        run.phase_completed(tmp_path, "report.json")


def test_original_PAS_R3_reuse_and_R2_control_separation():
    from di_dmpa_gate1c_v2 import reliability
    assert diagnostic.r.build is reliability.build
    reg, p = run.authority()
    assert len(p["immutable_baseline"]["checkpoint_inputs"]) == 9
    assert all(c["legacy_pas_tensor_sha256"] for c in p["immutable_baseline"]["checkpoint_inputs"])
    assert reg["score"]["K"] == 2 and reg["validation"]["candidates"]["Q2"].startswith("R2")
    assert not reg["mass_matching"]["class_balanced_loss"]


def test_call_graph_mismatch_is_not_a_budget_increase(monkeypatch):
    monkeypatch.setitem(diagnostic.LIMITS, "native_forwards", 4)
    with pytest.raises(core.Blocked) as ex: run.authority()
    assert ex.value.status == "BLOCKED_CALL_GRAPH_MISMATCH"
