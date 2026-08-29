import pytest

from lcrseg.methods.lcrseg_v0_2a import resolve_v02a_method_config


def test_assimilation_mode_is_explicit_enum() -> None:
    resolved = resolve_v02a_method_config({"progressive_admission": True, "compatibility_calibration": False, "compatibility_rejection": False, "variant_id": "R1"})
    assert resolved["assimilation_mode"] == "progressive_admission"
    assert resolved["consolidation_mode"] == "uniform_relation"
    assert "progressive_admission" not in resolved
    with pytest.raises(ValueError, match="conflicts"):
        resolve_v02a_method_config({"variant_id": "R1", "assimilation_mode": "legacy_continuous_v01", "progressive_admission": True})
