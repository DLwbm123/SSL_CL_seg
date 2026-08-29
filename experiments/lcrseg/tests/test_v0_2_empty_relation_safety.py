from __future__ import annotations

import torch

from lcrseg.methods.components.rejection_only_routing import rejection_only_relation_loss, rejection_only_weights
from tests.v0_2_test_utils import relation


def test_empty_relation_valid_set_is_differentiable_zero() -> None:
    probability = torch.tensor([[[[0.2, 0.2]], [[0.8, 0.8]]]])
    current_logits = probability.log().detach().clone().requires_grad_(True)
    current = relation(current_logits.softmax(dim=1))
    old = relation(probability)
    valid = torch.zeros((1, 1, 1, 2), dtype=torch.bool)
    routing = rejection_only_weights(
        torch.zeros_like(valid, dtype=torch.float32),
        old.predicted_class,
        valid,
        num_classes=2,
        calibrator_available=False,
    )
    loss = rejection_only_relation_loss(current, old, routing, torch.zeros((1, 1, 4, 8), dtype=torch.bool), distill_temperature=0.5)
    assert float(loss) == 0.0
    loss.backward()
    assert current_logits.grad is not None
