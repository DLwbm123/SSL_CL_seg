import pytest

from lcrseg.methods.lcrseg_v0_3 import resolve_v03_method_config


def test_v03_admission_schedule_frozen() -> None:
    resolved = resolve_v03_method_config(
        {"variant_id": "R1", "assimilation_mode": "progressive_admission", "consolidation_mode": "uniform_relation"}
    )
    assert resolved["progressive_schedule"]["start_fraction"] == 0.40
    assert resolved["progressive_schedule"]["end_fraction"] == 0.80
    altered = dict(resolved)
    altered["progressive_schedule"] = {**resolved["progressive_schedule"], "end_fraction": 0.90}
    with pytest.raises(ValueError):
        resolve_v03_method_config(altered)

