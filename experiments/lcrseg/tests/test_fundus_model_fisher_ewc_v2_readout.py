import copy
import json
from pathlib import Path

import pytest

from scripts.evaluate_fundus_model_fisher_ewc_v2 import adjudicate


REG = json.loads(
    (Path(__file__).resolve().parents[1] / "docs/fundus_model_fisher_ewc_v2/registration.json").read_text()
)


def cells():
    rows = []
    reference = [[0.80], [0.72, 0.82], [0.70, 0.74, 0.84]]
    for seed in REG["seeds"]:
        for arm in REG["arms"]:
            for stage, values in enumerate(reference):
                for index, value in enumerate(values):
                    if arm == "model_fisher_ewc_v2" and index < stage:
                        value += 0.03
                    rows.append(
                        dict(
                            seed=seed,
                            arm=arm,
                            stage=stage,
                            site=REG["domain_order"][index],
                            dice_class_1=value,
                            dice_class_2=value,
                            mean_foreground_dice=value,
                        )
                    )
    return rows


def test_registered_pairing_and_signs():
    result = adjudicate(cells(), REG)
    assert result["status"] == "PASS_EWC_FEASIBILITY"
    assert all(row["F"] == pytest.approx(0.02) and row["I"] == pytest.approx(0) and row["BWT"] == pytest.approx(0.03)
               for row in result["paired"])


def test_incomplete_matrix_cannot_pass():
    rows = cells()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(RuntimeError):
        adjudicate(rows, REG)
