import torch

from lcrseg.methods.components.soft_reliability_allocation import soft_reliability_allocation
from tests.v04a_test_utils import pseudo


def test_sra_alpha_monotonic_in_learnability() -> None:
    scores = torch.linspace(0.0, 1.0, 100).reshape(1, 1, 10, 10)
    labels = torch.ones((1, 10, 10), dtype=torch.long)
    output = soft_reliability_allocation(pseudo(labels), scores, torch.ones_like(scores, dtype=torch.bool), num_classes=3, site_step=0, total_site_steps=10)
    alpha = output.alpha.flatten()
    assert torch.all(alpha[1:] >= alpha[:-1])
    assert not alpha.requires_grad
