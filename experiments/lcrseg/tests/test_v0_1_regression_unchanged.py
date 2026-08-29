from __future__ import annotations

from lcrseg.methods import resolve_method_config


def test_v0_1_defaults_remain_frozen_after_v0_2_registration() -> None:
    resolved = resolve_method_config("lcrseg_v0_1")
    assert resolved["use_learnability"] is True
    assert resolved["use_compatibility"] is True
    assert resolved["anchor_k"] == 1
    assert resolved["lambda_assim"] == 1.0
    assert resolved["lambda_relation"] == 1.0
