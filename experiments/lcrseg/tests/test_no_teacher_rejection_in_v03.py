import json
from pathlib import Path

from lcrseg.methods.lcrseg_v0_3 import resolve_v03_method_config


ROOT = Path(__file__).resolve().parents[1]


def test_no_teacher_rejection_in_v03() -> None:
    for variant in ("r0", "r1", "p0"):
        method = json.loads((ROOT / f"configs/experiments/lcrseg_v0_3_{variant}.yaml").read_text())["method"]
        resolved = resolve_v03_method_config(method)
        assert resolved["consolidation_mode"] in {"uniform_relation", "none"}
        assert resolved["teacher_rejection_enabled"] is False
        assert resolved["multi_agent"] is False
        assert resolved["ric"] is False

