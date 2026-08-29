import torch

from lcrseg.methods.components.soft_reliability_allocation import soft_reliability_allocation, soft_reliability_assimilation_loss
from tests.v04a_test_utils import pseudo, relation


def test_sra_current_relation_target_stopgrad() -> None:
    labels = torch.ones((1, 4, 4), dtype=torch.long)
    valid = torch.ones((1, 1, 4, 4), dtype=torch.bool)
    allocation = soft_reliability_allocation(pseudo(labels), torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4), valid, num_classes=3, site_step=0, total_site_steps=2, minimum_pixels_for_class_cdf=1)
    segmentation = torch.randn((1, 3, 16, 16), requires_grad=True)
    weak_logits = torch.randn((1, 3, 4, 4), requires_grad=True)
    strong_logits = torch.randn((1, 3, 4, 4), requires_grad=True)
    output = soft_reliability_assimilation_loss(segmentation, pseudo(labels), relation(weak_logits), relation(strong_logits), allocation, torch.ones((1, 1, 16, 16), dtype=torch.bool))
    output.loss.backward()
    assert weak_logits.grad is None
    assert strong_logits.grad is not None and segmentation.grad is not None
