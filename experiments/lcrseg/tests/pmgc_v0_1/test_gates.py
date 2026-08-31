import copy

import numpy as np
import pytest

from pmgc_v0_1 import report as r
from pmgc_v0_1.core import Blocked, CANDIDATES
from pmgc_v0_1.protocol import authority


def evidence():
    reg, _ = authority()
    iso = dict(models_bitwise_unchanged=True, model_before={"x":"frozen"}, model_after={"x":"frozen"}, banks_unchanged=True,
               parameter_grad_fields="None", teacher_bank_gradients="None", optimizer_constructed=False, backward_called=False,
               model_optimizer_steps=0, transport_optimizer_steps=0, hidden_gt_training_usage="none", test_gt_usage="none")
    support = [dict(class_id=i//2, mode=i%2, center_active=True, active_pixels=64, case_count=2, occupancy=.5,
                    KD_active=True, finite=True, null_rule_pass=True, fit_converged=True) for i in range(6)]
    prepared = [dict(unit=u, mode_support=copy.deepcopy(support), isolation=copy.deepcopy(iso), checkpoint_hashes_unchanged=True,
                     validation_GT_in_guards=False) for u in reg["fixed_units"]]
    base = dict(CE=0., foreground_Dice=.8, class_CE=[0.]*3, Dice=[.8]*3, mode_CE=[0.]*6,
                totals=dict(mode_pixels=[64]*6))
    rows, pairs = [], []
    for unit in reg["fixed_units"]:
        for pair in unit["formal_pairs"]:
            rows.append(dict(pair=pair, unit_id=unit["unit_id"], isolation=copy.deepcopy(iso), validation_GT_received_by_projection=False,
                before={side:copy.deepcopy(base) for side in ("previous","current")},
                projections={c:dict(constraint_count=1,guard_names=["global"],raw_guard_dots=[.1],norm_ratio=.8,zero_direction=False) for c in CANDIDATES},
                evaluated={c:dict(step=dict(stateless=True,optimizer_constructed=False,checkpoint_written=False,step_valid=True,step_norm=.001),
                                  panels={side:dict(after=copy.deepcopy(base)) for side in ("previous","current")}) for c in CANDIDATES}))
            row = rows[-1]
            for c in CANDIDATES:
                if c != "P4":
                    after = row["evaluated"][c]["panels"]["previous"]["after"]
                    after["CE"] = 2e-4
                    after["class_CE"] = [2e-4]*3
                    after["mode_CE"] = [2e-4]*6
            pairs.append(pair)
    return prepared, rows, pairs


def test_all_seven_gates_required_and_hard_stop():
    verdict = r.adjudicate(*evidence())
    assert verdict["status"] == "PASS_PMGC_FEASIBILITY" and not verdict["failed_gates"]
    assert verdict["next_action"] == "STOP_FOR_INDEPENDENT_REVIEW" and not verdict["training_launched"]


@pytest.mark.parametrize("field,passing,failing", [("active_pixels",32,31),("case_count",2,1),("occupancy",.05,np.nextafter(.05,0.))])
def test_G1_support_boundaries(field, passing, failing):
    prepared, rows, pairs = evidence(); target = prepared[0]["mode_support"][2]
    target[field] = passing
    assert r.adjudicate(prepared, rows, pairs)["G1_G7"]["G1"]["pass_"]
    target[field] = failing
    assert not r.adjudicate(prepared, rows, pairs)["G1_G7"]["G1"]["pass_"]


def test_G1_KD_fraction_boundary():
    prepared, rows, pairs = evidence(); support = [m for p in prepared for m in p["mode_support"] if m["class_id"] in (1,2)]
    for mode in support[:6]:mode["KD_active"] = False
    assert r.adjudicate(prepared, rows, pairs)["G1_G7"]["G1"]["pass_"]
    support[6]["KD_active"] = False
    assert not r.adjudicate(prepared, rows, pairs)["G1_G7"]["G1"]["pass_"]


@pytest.mark.parametrize("dot,passed", [(-1e-10,True),(np.nextafter(-1e-10,-np.inf),False),(None,False)])
def test_G2_dot_and_undefined_boundary(dot, passed):
    prepared, rows, pairs=evidence();rows[0]["projections"]["P4"]["raw_guard_dots"]=[dot]
    assert r.adjudicate(prepared,rows,pairs)["G1_G7"]["G2"]["pass_"] is passed


@pytest.mark.parametrize("better,passed", [(29,True),(28,False)])
def test_G3_fixed_fraction_boundary(better, passed):
    prepared, rows, pairs=evidence()
    for row in rows[better:]:row["evaluated"]["P0"]["panels"]["previous"]["after"]["CE"]=0.
    assert r.adjudicate(prepared,rows,pairs)["G1_G7"]["G3"]["pass_"] is passed


@pytest.mark.parametrize("value,passed", [(1e-4,True),(np.nextafter(1e-4,np.inf),False)])
def test_G4_CE_boundary(value, passed):
    prepared, rows, pairs=evidence()
    for row in rows:row["evaluated"]["P4"]["panels"]["current"]["after"]["CE"]=value
    assert r.adjudicate(prepared,rows,pairs)["G1_G7"]["G4"]["pass_"] is passed


@pytest.mark.parametrize("better,passed", [(29,True),(28,False)])
def test_G5_worst_mode_fraction_boundary(better, passed):
    prepared, rows, pairs=evidence()
    for row in rows[better:]:row["evaluated"]["P2"]["panels"]["previous"]["after"]["mode_CE"]=[0.]*6
    # Keep four positive unit medians while changing the exact pair fraction.
    if better == 29:
        row=rows[33];row["evaluated"]["P2"]["panels"]["previous"]["after"]["mode_CE"]=[2e-4]*6
        rows[0]["evaluated"]["P2"]["panels"]["previous"]["after"]["mode_CE"]=[0.]*6
    value=r.adjudicate(prepared,rows,pairs)["G1_G7"]["G5"]
    assert value["better_count"] == better
    assert value["pass_"] is passed


@pytest.mark.parametrize("value,passed", [(.5,True),(np.nextafter(.5,0),False)])
def test_G6_median_boundary(value, passed):
    prepared, rows, pairs=evidence()
    for row in rows:row["projections"]["P4"]["norm_ratio"]=value
    assert r.adjudicate(prepared,rows,pairs)["G1_G7"]["G6"]["pass_"] is passed


def test_zero_direction_cannot_pass_retention():
    prepared, rows, pairs=evidence();rows[0]["projections"]["P4"]["zero_direction"]=True
    assert not r.adjudicate(prepared,rows,pairs)["G1_G7"]["G6"]["pass_"]


def test_G7_mutation_fails_and_controls_cannot_rescue():
    prepared, rows, pairs=evidence();rows[0]["isolation"]["banks_unchanged"]=False
    verdict=r.adjudicate(prepared,rows,pairs)
    assert verdict["status"] == "FAIL_PMGC_FEASIBILITY" and "G7" in verdict["failed_gates"]
    assert verdict["prototype_derived_new_method_line"] == "ENDED" and not verdict["additional_variants_authorized"]


def test_all_failed_gates_retained():
    prepared, rows, pairs=evidence();prepared[0]["mode_support"][2]["center_active"]=False
    rows[0]["isolation"]["banks_unchanged"]=False;rows[0]["projections"]["P4"]["zero_direction"]=True
    assert set(r.adjudicate(prepared,rows,pairs)["failed_gates"]) == {"G1","G6","G7"}


@pytest.mark.parametrize("change", ["missing","duplicate","reorder"])
def test_exact_48_denominator_fail_closed(change):
    prepared, rows, pairs=evidence()
    if change=="missing":rows.pop()
    elif change=="duplicate":rows[-1]=copy.deepcopy(rows[0])
    else:rows[0],rows[1]=rows[1],rows[0]
    with pytest.raises(Blocked):r.adjudicate(prepared,rows,pairs)


@pytest.mark.parametrize("limit", [.05,.75,-1e-10,.60,1e-4,5e-5,.002,.55,.003,.50,.20,.40])
def test_numeric_gate_thresholds_have_no_hidden_epsilon(limit):
    assert r.at_least(limit,limit) and r.at_most(limit,limit)
    assert not r.at_least(np.nextafter(limit,-np.inf),limit)
    assert not r.at_most(np.nextafter(limit,np.inf),limit)


def test_nonfinite_report_fail_closed():
    prepared, rows, pairs=evidence();rows[0]["evaluated"]["P4"]["panels"]["previous"]["after"]["CE"]=float("nan")
    with pytest.raises(Blocked):r.adjudicate(prepared,rows,pairs)
