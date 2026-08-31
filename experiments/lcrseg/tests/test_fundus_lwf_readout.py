import copy
import json
from pathlib import Path

import numpy as np
import pytest

from lcrseg.engine.evaluator import _case_metrics, _site_summary
from scripts.evaluate_fundus_lwf_v1 import adjudicate, audit_training


REG = json.loads((Path(__file__).resolve().parents[1] / "docs/fundus_lwf_v1/registration.json").read_text())


def cells():
    rows = []
    reference = [[0.80], [0.72, 0.82], [0.70, 0.74, 0.84]]
    for seed in REG["seeds"]:
        for arm in REG["arms"]:
            for stage, values in enumerate(reference):
                for index, value in enumerate(values):
                    if arm == "uniform_kd" and index < stage:
                        value += 0.03
                    rows.append(dict(seed=seed, arm=arm, stage=stage, site=REG["domain_order"][index],
                                     dice_class_1=value, dice_class_2=value, mean_foreground_dice=value))
    return rows


def test_hand_calculated_pairing_and_signs():
    result = adjudicate(cells(), REG)
    assert result["status"] == "PASS_BASELINE_FEASIBILITY"
    for row in result["paired"]:
        assert row["F"] == pytest.approx(0.02)
        assert row["I"] == pytest.approx(0)
        assert row["BWT"] == pytest.approx(0.03)
    zero = cells()
    for row in zero:
        if row["arm"] == "uniform_kd" and REG["domain_order"].index(row["site"]) < row["stage"]:
            for key in ("dice_class_1", "dice_class_2", "mean_foreground_dice"):
                row[key] -= 0.03
    failed = adjudicate(zero, REG)
    assert failed["status"] == "FAIL_BASELINE_FEASIBILITY"
    assert set(failed["failed_gates"]) == {"mean_final_dice_improvement", "positive_final_dice_seeds", "mean_bwt_improvement"}


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unexpected", "nan", "range", "mean"])
def test_bad_required_evidence_never_becomes_scientific_success(defect):
    rows = cells()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "unexpected":
        rows[0]["stage"] = 99
    else:
        rows[0]["dice_class_1"] = {"nan": float("nan"), "range": 1.1, "mean": 0.2}[defect]
    with pytest.raises(RuntimeError):
        adjudicate(rows, REG)


def test_case_mean_and_empty_class_convention():
    empty = np.zeros((1, 2, 2), dtype=np.uint8)
    foreground = np.ones_like(empty)
    first = _case_metrics(empty, empty, num_classes=3)
    second = _case_metrics(empty, foreground, num_classes=3)
    summary = _site_summary("synthetic", [first, second])
    assert first["dice_class_1"] == first["dice_class_2"] == 1
    assert second["dice_class_1"] == 0 and second["dice_class_2"] == 1
    assert summary["mean_foreground_dice"] == 0.75


def test_incomplete_training_stops_before_checkpoint_or_test_access(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.evaluate_fundus_lwf_v1.load_checkpoint", lambda *a, **k: pytest.fail("checkpoint accessed before admission"))
    monkeypatch.setattr("scripts.evaluate_fundus_lwf_v1.load_training_records", lambda *a, **k: pytest.fail("test role accessed before admission"))
    with pytest.raises(FileNotFoundError):
        audit_training(tmp_path, REG)
