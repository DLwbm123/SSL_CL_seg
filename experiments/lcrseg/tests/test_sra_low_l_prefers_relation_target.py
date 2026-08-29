import torch

from lcrseg.methods.components.soft_reliability_allocation import soft_reliability_allocation
from tests.v04a_test_utils import pseudo


def test_sra_low_l_prefers_relation_target() -> None:
    scores = torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10)
    labels = torch.ones((1, 10, 10), dtype=torch.long)
    output = soft_reliability_allocation(pseudo(labels), scores, torch.ones_like(scores, dtype=torch.bool), num_classes=3, site_step=0, total_site_steps=10)
    assert float(output.alpha.flatten()[0]) < 0.01
    assert float(1.0 - output.alpha.flatten()[0]) > 0.99
