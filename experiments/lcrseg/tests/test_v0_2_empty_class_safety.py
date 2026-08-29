from __future__ import annotations

import torch

from lcrseg.methods.components.progressive_admission import admission_assimilation_loss, classwise_progressive_admission
from tests.v0_2_test_utils import pseudo


def test_empty_pseudo_classes_and_empty_selection_are_safe() -> None:
    labels = torch.full((1, 2, 2), -100, dtype=torch.long)
    valid = torch.zeros((1, 1, 2, 2), dtype=torch.bool)
    output = classwise_progressive_admission(
        pseudo(labels, valid=valid), torch.zeros((1, 1, 2, 2)), valid,
        num_classes=3, site_step=0, total_site_steps=10,
    )
    assert output.candidate_counts == (0, 0, 0)
    assert output.selected_counts == (0, 0, 0)
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    loss = admission_assimilation_loss(logits, pseudo(labels, valid=valid), output, valid)
    assert float(loss) == 0.0
    loss.backward()
    assert logits.grad is not None
