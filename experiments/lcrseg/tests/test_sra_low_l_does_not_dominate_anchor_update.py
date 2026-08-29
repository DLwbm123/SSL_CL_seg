import torch

from lcrseg.methods.components.soft_reliability_allocation import anchor_update_weights, soft_reliability_allocation
from tests.v04a_test_utils import pseudo, relation


def test_sra_low_l_does_not_dominate_anchor_update() -> None:
    labels = torch.ones((1, 10, 10), dtype=torch.long)
    valid = torch.ones((1, 1, 10, 10), dtype=torch.bool)
    allocation = soft_reliability_allocation(pseudo(labels), torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10), valid, num_classes=3, site_step=0, total_site_steps=2)
    logits = torch.zeros((1, 3, 10, 10)); logits[:, 1] = 2.0
    weights = anchor_update_weights(pseudo(labels), relation(logits), allocation).flatten()
    assert float(weights[:10].mean()) < float(weights[-10:].mean()) * 0.05
