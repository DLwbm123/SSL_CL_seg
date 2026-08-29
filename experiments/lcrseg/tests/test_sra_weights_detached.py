import torch

from lcrseg.methods.components.soft_reliability_allocation import anchor_update_weights, soft_reliability_allocation
from tests.v04a_test_utils import pseudo, relation


def test_sra_weights_detached() -> None:
    scores = torch.linspace(0.0, 1.0, 100, requires_grad=True).reshape(1, 1, 10, 10)
    labels = torch.ones((1, 10, 10), dtype=torch.long)
    allocation = soft_reliability_allocation(pseudo(labels), scores, torch.ones_like(scores, dtype=torch.bool), num_classes=3, site_step=0, total_site_steps=10)
    logits = torch.zeros((1, 3, 10, 10)); logits[:, 1] = 1.0
    weights = anchor_update_weights(pseudo(labels), relation(logits), allocation)
    assert not allocation.alpha.requires_grad
    assert not weights.requires_grad
