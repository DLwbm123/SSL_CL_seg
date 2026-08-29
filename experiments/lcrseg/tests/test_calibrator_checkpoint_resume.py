from __future__ import annotations

import torch

from lcrseg.methods.components.compatibility_calibrator import LabeledOnlyCompatibilityCalibrator


def test_calibrator_state_round_trips_exactly_for_resume() -> None:
    score = torch.linspace(0.0, 1.0, 1200).reshape(1, 1, 1, 1200)
    predicted = torch.zeros((1, 1, 1200), dtype=torch.long)
    correct = score[:, 0].gt(0.45)
    valid = torch.ones_like(score, dtype=torch.bool)
    first = LabeledOnlyCompatibilityCalibrator(num_classes=1, bins=10, min_pixels=500)
    first.fit(score, predicted, correct, valid, epoch=9)
    expected, available = first.calibrate(score, predicted)
    assert available
    resumed = LabeledOnlyCompatibilityCalibrator(num_classes=1, bins=10, min_pixels=500)
    resumed.load_state_dict(first.state_dict())
    actual, resumed_available = resumed.calibrate(score, predicted)
    assert resumed_available
    assert torch.equal(expected, actual)
    assert resumed.last_update_epoch == 9
