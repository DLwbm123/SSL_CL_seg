import torch

from lcrseg.methods.components.soft_reliability_allocation import anchor_update_weights, soft_reliability_allocation
from tests.v04a_test_utils import pseudo, relation


def test_sra_anchor_update_weighted_by_alpha() -> None:
    labels = torch.ones((1, 10, 10), dtype=torch.long)
    valid = torch.ones((1, 1, 10, 10), dtype=torch.bool)
    allocation = soft_reliability_allocation(pseudo(labels), torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10), valid, num_classes=3, site_step=0, total_site_steps=2)
    logits = torch.zeros((1, 3, 10, 10)); logits[:, 1] = 2.0
    weights = anchor_update_weights(pseudo(labels), relation(logits), allocation)
    assert torch.equal(weights, allocation.alpha)
    disagree = logits.clone(); disagree[:, 1] = 0.0; disagree[:, 2] = 2.0
    assert torch.count_nonzero(anchor_update_weights(pseudo(labels), relation(disagree), allocation)) == 0
