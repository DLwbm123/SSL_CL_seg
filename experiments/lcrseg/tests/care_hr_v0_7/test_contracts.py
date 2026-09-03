import inspect

import pytest

from care_hr_v0_7 import contracts


def test_candidate_grid_is_fixed_eight():
    grid = contracts.candidate_grid()
    assert len(grid) == 8
    assert {row["blend_lambda"] for row in grid} == {0.5, 0.75}
    assert {row["epsilon_gain"] for row in grid} == {0.0, 0.0025}
    assert {row["delta_harm"] for row in grid} == {0.005, 0.01}
    assert {row["rho"] for row in grid} == {0.8}


def test_lambda_one_is_not_primary_candidate():
    assert 1.0 not in {row["blend_lambda"] for row in contracts.candidate_grid()}


def test_review_contract_has_no_authority():
    value = contracts.review_contract()
    assert value["status"] == contracts.REVIEW_STATUS
    assert value["draft_state"] == "DRAFT_NOT_REGISTERED"
    assert value["training_authority"] == "NO_TRAINING_AUTHORITY"
    assert value["evaluation_authority"] == "NO_EVALUATION_AUTHORITY"


def test_external_review_authorization_always_blocks():
    with pytest.raises(contracts.ReviewBlocked, match=contracts.REVIEW_STATUS):
        contracts.require_external_review_authorization(True)


def test_feature_schema_is_exact():
    assert len(contracts.FEATURE_NAMES) == 20
    assert contracts.FEATURE_NAMES[0] == "class_is_cup"
    assert contracts.FEATURE_NAMES[-1] == "ridge_top1_top2_margin"


def test_review_contract_is_pure_function():
    assert inspect.signature(contracts.review_contract).parameters == {}
    assert contracts.review_contract() == contracts.review_contract()
