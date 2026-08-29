import torch

from lcrseg.methods.components.teacher_validity import compute_teacher_validity
from lcrseg.models import UNet2D
from tests.v0_2_test_utils import relation


def test_teacher_validity_independent_of_current_model() -> None:
    old_logits = torch.randn(1, 3, 8, 8)
    old_relation = relation(torch.rand(1, 3, 2, 2))
    current = UNet2D(3, 3)
    first = compute_teacher_validity(old_logits, old_relation, margin_temperature=0.05, spatial_floor=0.25).raw_score
    with torch.no_grad():
        for parameter in current.parameters():
            parameter.add_(torch.randn_like(parameter))
    second = compute_teacher_validity(old_logits, old_relation, margin_temperature=0.05, spatial_floor=0.25).raw_score
    assert torch.equal(first, second)
