from __future__ import annotations

import copy
from pathlib import Path

import pytest

from di_dmpa_jascl.config import ALL_NEW_MODULE_SWITCHES, load_yaml, validate_gate0_config


ROOT = Path(__file__).resolve().parents[2]


def _payloads():
    config = load_yaml(ROOT / "configs/gate0_repaired_v2/fundus_pas_probmse.yaml")
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


def test_v1_status_and_matrices_are_preserved_byte_for_byte():
    import subprocess
    frozen = "46e892960240543c946c570a9378d409b226384b"
    docs = ROOT / "docs/di_dmpa_jascl"
    def original(relative):
        return subprocess.check_output(["git", "-C", str(ROOT), "show",
                                        f"{frozen}:experiments/lcrseg/docs/di_dmpa_jascl/{relative}"])
    assert (docs / "GATE0_STATUS_V1_ARCHIVED.json").read_bytes() == original("GATE0_STATUS.json")
    for path in (docs / "gate0_results").rglob("*"):
        if path.is_file():
            relative = path.relative_to(docs)
            expected = original(relative.as_posix())
            assert path.read_bytes() == expected
            assert (docs / "gate0_results_v1_zero_u_grad" / relative).read_bytes() == expected
