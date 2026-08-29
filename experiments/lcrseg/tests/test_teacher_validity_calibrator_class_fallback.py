import torch

from lcrseg.methods.components.teacher_validity import TeacherValidityCalibrator


def test_teacher_validity_calibrator_class_fallback() -> None:
    score = torch.linspace(0, 1, 3100).reshape(1, 1, 31, 100)
    predicted = torch.zeros((1, 31, 100), dtype=torch.long)
    predicted[:, -1] = 1
    correct = score[:, 0].gt(0.4)
    calibrator = TeacherValidityCalibrator(num_classes=2, minimum_pixels_per_class=2048)
    rows = calibrator.fit(score, predicted, correct, torch.ones_like(score, dtype=torch.bool), site_id="B")
    assert 0 in calibrator.class_mappings and 1 not in calibrator.class_mappings
    assert any(row["class_id"] == 1 and row["fallback_scope"] == "global_fallback" for row in rows)
