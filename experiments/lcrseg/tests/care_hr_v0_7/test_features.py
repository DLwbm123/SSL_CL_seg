import inspect

import numpy as np
import pytest

from care_hr_v0_7.contracts import FEATURE_NAMES, Proposal
from care_hr_v0_7.features import build_features


def inputs():
    current = np.zeros((3, 10, 10), dtype=np.float64); current[0] = 1
    historical = current.copy()
    historical[0, :2, :4] = 0; historical[1, :2, :4] = 1
    mask = np.zeros((10, 10), dtype=bool); mask[:2, :4] = True
    proposal = Proposal("p", 1, "add", 8, 0.5, 1.5, mask)
    return current, historical, (proposal,)


def test_feature_shape_and_order_are_exact():
    matrix = build_features(*inputs(), 0.9, 0.8, 1.2, 0.3)
    assert matrix.shape == (1, 20) and len(FEATURE_NAMES) == 20
    assert matrix[0, 0] == 0 and matrix[0, 1] == 1


def test_feature_api_rejects_forbidden_semantic_fields_by_signature():
    names = {name.lower() for name in inspect.signature(build_features).parameters}
    assert not names & {"domain", "domain_index", "site", "vendor", "label", "gt", "dice", "utility", "patient_outcome"}


def test_all_features_are_finite():
    assert np.all(np.isfinite(build_features(*inputs(), 0.9, 0.8, 1.2, 0.3)))


def test_target_probability_delta_has_expected_sign():
    matrix = build_features(*inputs(), 0.9, 0.8, 1.2, 0.3)
    assert matrix[0, 8] == 1.0 and matrix[0, 9] == 1.0


def test_nonfinite_external_feature_hard_fails():
    with pytest.raises(ValueError, match="nonfinite"):
        build_features(*inputs(), np.nan, 0.8, 1.2, 0.3)
