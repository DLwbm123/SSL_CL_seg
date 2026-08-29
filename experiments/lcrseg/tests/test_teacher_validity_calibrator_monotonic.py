import torch

from lcrseg.methods.components.teacher_validity import TeacherValidityCalibrator


def test_teacher_validity_calibrator_monotonic() -> None:
    score = torch.linspace(0, 1, 4000).reshape(1, 1, 40, 100)
    predicted = torch.zeros((1, 40, 100), dtype=torch.long)
    correct = torch.arange(4000).reshape(1, 40, 100).remainder(7).ne(0)
    calibrator = TeacherValidityCalibrator(num_classes=1, minimum_pixels_per_class=32)
    calibrator.fit(score, predicted, correct, torch.ones_like(score, dtype=torch.bool), site_id="B")
    calibrated, available = calibrator.calibrate(score, predicted)
    assert available
    assert bool(calibrated.reshape(-1)[1:].ge(calibrated.reshape(-1)[:-1]).all())
