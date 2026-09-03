import pytest

from care_hr_v0_7.metrics import aligned_global, proposal_precision


def test_zero_acceptance_precision_is_null_with_counts():
    value = proposal_precision([False, False], [True, False])
    assert value == {"proposal_precision": None, "proposal_precision_numerator": 0,
                     "proposal_precision_denominator": 0}


def test_proposal_precision_counts_are_explicit():
    value = proposal_precision([True, True, False], [True, False, True])
    assert value["proposal_precision"] == 0.5
    assert value["proposal_precision_numerator"] == 1 and value["proposal_precision_denominator"] == 2


def test_global_alignment_is_order_invariant_and_strict():
    assert [row["value"] for row in aligned_global([{"row_index": 1, "value": "b"},
                                                     {"row_index": 0, "value": "a"}], 2)] == ["a", "b"]
    with pytest.raises(ValueError):
        aligned_global([{"row_index": 2}], 1)
