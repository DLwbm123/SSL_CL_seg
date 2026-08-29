import inspect
import torch

from lcrseg.methods.components.teacher_validity import compute_teacher_validity
from tests.v0_2_test_utils import relation


def test_teacher_validity_old_model_only() -> None:
    assert all("current" not in name for name in inspect.signature(compute_teacher_validity).parameters)
    result = compute_teacher_validity(torch.randn(1, 3, 8, 8, requires_grad=True), relation(torch.rand(1, 3, 2, 2)), margin_temperature=0.05, spatial_floor=0.25)
    assert not result.raw_score.requires_grad
