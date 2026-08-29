import math
import torch

from lcrseg.methods.components.progressive_admission import classwise_progressive_admission
from tests.v0_2_test_utils import pseudo


def test_progressive_admission_classwise_fraction() -> None:
    labels = torch.tensor([[[0] * 40 + [1] * 60]])
    output = classwise_progressive_admission(pseudo(labels), torch.linspace(0, 1, 100).reshape(1, 1, 1, 100), torch.ones((1, 1, 1, 100), dtype=torch.bool), num_classes=2, site_step=0, total_site_steps=2, minimum_pixels_for_class_quantile=32)
    assert output.selected_counts == (math.ceil(0.4 * 40), math.ceil(0.4 * 60))
