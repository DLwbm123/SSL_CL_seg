from __future__ import annotations

import numpy as np
import pytest
import torch

from lcrseg.memory import (
    MonotonicReliabilityCalibrator,
    SitePrototypeBuilder,
    SitePrototypeMemory,
    estimate_transport,
)


def test_aspr_pava_calibrator_is_monotonic_and_serializable() -> None:
    score = np.linspace(0.0, 1.0, 6000, dtype=np.float32)
    predicted = np.where(np.arange(6000) % 2, 1, 2)
    correctness = (score > 0.35).astype(np.float32)
    calibrator = MonotonicReliabilityCalibrator().fit(score, correctness, predicted, np.ones(6000, dtype=bool))
    probe = torch.linspace(0, 1, 200)
    first = calibrator.predict(probe, torch.ones(200, dtype=torch.long))
    second = MonotonicReliabilityCalibrator.from_state_dict(calibrator.state_dict()).predict(
        probe, torch.ones(200, dtype=torch.long)
    )
    assert torch.all(first[1:] >= first[:-1])
    assert torch.equal(first, second)


def test_aspr_memory_is_append_only_background_free_and_checkpointable() -> None:
    memory = SitePrototypeMemory(4)
    records = {
        class_id: {
            "prototype": torch.nn.functional.one_hot(torch.tensor(class_id), 4).float(),
            "dispersion": 0.1,
            "labeled_case_count": 2,
            "unlabeled_case_count": 1,
            "labeled_pixel_weight": 64.0,
            "unlabeled_pixel_weight": 32.0,
        }
        for class_id in (1, 2)
    }
    memory.append_site(
        "REFUGE",
        records,
        source_checkpoint_sha256="a" * 64,
        class_semantics_sha256="b" * 64,
        manifest_sha256="c" * 64,
        split_sha256="d" * 64,
    )
    with pytest.raises(RuntimeError):
        memory.append_site(
            "REFUGE",
            records,
            source_checkpoint_sha256="a" * 64,
            class_semantics_sha256="b" * 64,
            manifest_sha256="c" * 64,
            split_sha256="d" * 64,
        )
    restored = SitePrototypeMemory(4)
    restored.load_state_dict(memory.state_dict(), strict=True)
    restored.validate()
    assert restored.historical_sites() == ("REFUGE",)
    assert list(restored.parameters()) == []


def test_aspr_builder_requires_weighted_support_and_excludes_background() -> None:
    builder = SitePrototypeBuilder(4)
    feature = torch.nn.functional.normalize(torch.randn(4, 8, 8), dim=0)
    label = torch.ones((8, 8), dtype=torch.long)
    assert builder.add_labeled("case", feature, label)
    with pytest.raises(ValueError):
        SitePrototypeBuilder(4, foreground_ids=(0, 1))


def test_aspr_transport_uses_frozen_shrinkage_formula() -> None:
    old = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.1], [1.0, -0.1]]), dim=1)
    current = torch.nn.functional.normalize(torch.tensor([[0.9, 0.2], [0.9, 0.3], [0.9, 0.1]]), dim=1)
    estimate = estimate_transport(old, current)
    assert estimate.valid
    assert 0.0 <= estimate.shrinkage <= 1.0
    expected = estimate.signal / (estimate.signal + estimate.variance / estimate.case_count + 1.0e-8)
    assert estimate.shrinkage == pytest.approx(expected)
    assert torch.allclose(estimate.delta, estimate.mean_displacement * estimate.shrinkage)


def test_aspr_memory_package_has_no_diagnostic_label_import() -> None:
    import lcrseg.memory.reliability_calibrator as calibrator_module
    import lcrseg.memory.site_prototype_builder as builder_module

    for module in (calibrator_module, builder_module):
        source = open(module.__file__, encoding="utf-8").read()
        assert "diagnostic_records" not in source
        assert "hidden_label" not in source
