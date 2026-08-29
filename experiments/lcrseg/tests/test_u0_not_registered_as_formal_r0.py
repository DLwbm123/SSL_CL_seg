import json
from pathlib import Path

import pytest

from lcrseg.methods.lcrseg_v0_2a import resolve_v02a_method_config


def test_u0_not_registered_as_formal_r0() -> None:
    with pytest.raises(ValueError):
        resolve_v02a_method_config({"variant_id": "R0", "assimilation_mode": "unit_all", "consolidation_mode": "uniform_relation"})
    amendment = json.loads((Path(__file__).parents[1] / "reports/experiment_status/PROTOCOL_AMENDMENT_V0_2A.json").read_text())
    assert amendment["u0_registered_as_formal_r0"] is False
