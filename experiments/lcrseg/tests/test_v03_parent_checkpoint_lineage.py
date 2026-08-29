import pytest

from lcrseg.methods.lcrseg_v0_3 import validate_parent_checkpoint_lineage


def _payload() -> dict:
    return {
        "site_id": "REFUGE",
        "site_index": 0,
        "global_step": 8000,
        "config_resolved": {
            "method": {
                "assimilation_mode": "progressive_admission",
                "consolidation_mode": "uniform_relation",
            }
        },
    }


def test_v03_parent_checkpoint_lineage() -> None:
    result = validate_parent_checkpoint_lineage(
        _payload(), checkpoint_sha256="expected", expected_sha256="expected"
    )
    assert result["completed_parent_steps"] == 8000
    with pytest.raises(ValueError):
        validate_parent_checkpoint_lineage(
            _payload(), checkpoint_sha256="wrong", expected_sha256="expected"
        )

