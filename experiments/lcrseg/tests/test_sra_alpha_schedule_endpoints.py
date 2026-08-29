import torch

from lcrseg.methods.components.soft_reliability_allocation import soft_reliability_allocation
from tests.v04a_test_utils import pseudo


def test_sra_alpha_schedule_endpoints() -> None:
    scores = torch.linspace(0.0, 1.0, 100).reshape(1, 1, 10, 10)
    labels = torch.ones((1, 10, 10), dtype=torch.long)
    valid = torch.ones_like(scores, dtype=torch.bool)
    first = soft_reliability_allocation(pseudo(labels), scores, valid, num_classes=3, site_step=0, total_site_steps=11)
    last = soft_reliability_allocation(pseudo(labels), scores, valid, num_classes=3, site_step=10, total_site_steps=11)
    assert first.target_hard_fraction == 0.40 and first.quantile_boundary == 0.60
    assert last.target_hard_fraction == 0.80 and abs(last.quantile_boundary - 0.20) < 1.0e-12
