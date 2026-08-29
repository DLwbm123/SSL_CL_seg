import torch

from lcrseg.methods.components.soft_reliability_allocation import soft_reliability_allocation, soft_reliability_assimilation_loss
from tests.v04a_test_utils import pseudo, relation


def test_sra_loss_safe_empty_mask() -> None:
    labels = torch.zeros((1, 4, 4), dtype=torch.long)
    valid = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
    allocation = soft_reliability_allocation(pseudo(labels, valid), torch.zeros_like(valid, dtype=torch.float32), valid, num_classes=3, site_step=0, total_site_steps=2)
    segmentation = torch.randn((1, 3, 16, 16), requires_grad=True)
    relation_logits = torch.randn((1, 3, 4, 4), requires_grad=True)
    output = soft_reliability_assimilation_loss(segmentation, pseudo(labels, valid), relation(relation_logits.detach()), relation(relation_logits), allocation, torch.ones((1, 1, 16, 16), dtype=torch.bool))
    output.loss.backward()
    assert output.valid_count == 0 and float(output.loss) == 0.0
    assert segmentation.grad is not None and relation_logits.grad is not None
