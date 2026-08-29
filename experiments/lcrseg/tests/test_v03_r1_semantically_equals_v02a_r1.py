import json
from pathlib import Path

from lcrseg.methods.lcrseg_v0_2a import resolve_v02a_method_config
from lcrseg.methods.lcrseg_v0_3 import resolve_v03_method_config, semantic_method_view


ROOT = Path(__file__).resolve().parents[1]


def test_v03_r1_semantically_equals_v02a_r1() -> None:
    old = json.loads((ROOT / "configs/experiments/lcrseg_v0_2a_r1.yaml").read_text())["method"]
    new = json.loads((ROOT / "configs/experiments/lcrseg_v0_3_r1.yaml").read_text())["method"]
    assert semantic_method_view(resolve_v03_method_config(new)) == semantic_method_view(
        resolve_v02a_method_config(old)
    )

