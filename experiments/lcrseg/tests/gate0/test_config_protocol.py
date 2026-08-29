from __future__ import annotations

import copy
from pathlib import Path

import pytest

from di_dmpa_jascl.config import ALL_NEW_MODULE_SWITCHES, load_yaml, validate_gate0_config


ROOT = Path(__file__).resolve().parents[2]


def _payloads():
    config = load_yaml(ROOT / "configs/gate0_repaired/fundus.yaml")
    protocol = load_yaml(ROOT / "docs/di_dmpa_jascl/DOMAIN_PROTOCOL.yaml")
    return config, protocol


def test_gate0_has_no_registered_method_or_extension() -> None:
    config, protocol = _payloads()
    resolved = validate_gate0_config(config, protocol)
    assert resolved["method_registered"] is False
    assert resolved["model"]["implementation"] == "lcrseg_unet2d_jascl_3x3_stochastic_head"
    assert resolved["model"]["base_channels"] == 16
    assert resolved["model"]["normalization"] == "groupnorm"
    assert resolved["model"]["reference_root"] == "third_party/JASCL_REFERENCE"
    assert all(resolved["method"][name] is False for name in ALL_NEW_MODULE_SWITCHES)
    assert resolved["method"]["use_constant_patch_classifier_regularization"] is False


@pytest.mark.parametrize("field", ["domain_order", "test_gt_policy"])
def test_protocol_boundary_rejects_mutation(field: str) -> None:
    config, protocol = _payloads()
    mutated = copy.deepcopy(config)
    mutated["data"][field] = ["Drishti_GS"] if field == "domain_order" else "training_allowed"
    with pytest.raises(ValueError):
        validate_gate0_config(mutated, protocol)


def test_three_benchmarks_are_independent_and_fixed_class() -> None:
    _, protocol = _payloads()
    assert protocol["benchmarks_are_independent"] is True
    assert {name: spec["class_count"] for name, spec in protocol["benchmarks"].items()} == {
        "fundus": 3,
        "prostate": 2,
        "mnms": 4,
    }
