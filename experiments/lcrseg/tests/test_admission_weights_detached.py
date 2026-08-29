from __future__ import annotations

import torch

from lcrseg.methods.components.progressive_admission import admission_assimilation_loss, classwise_progressive_admission
from tests.v0_2_test_utils import pseudo


def test_admission_mask_is_detached_and_selected_loss_has_unit_weights() -> None:
    labels = torch.tensor([[[0, 1], [1, 0]]])
    valid = torch.ones((1, 1, 2, 2), dtype=torch.bool)
    score = torch.tensor([[[[0.1, 0.9], [0.8, 0.2]]]], requires_grad=True)
    admission = classwise_progressive_admission(
        pseudo(labels, valid=valid), score, valid, num_classes=2,
        site_step=0, total_site_steps=2, pi_start=0.5, pi_end=0.5,
    )
    assert not admission.mask.requires_grad
    logits = torch.randn(1, 2, 2, 2, requires_grad=True)
    loss = admission_assimilation_loss(logits, pseudo(labels, valid=valid), admission, valid)
    loss.backward()
    assert logits.grad is not None
    assert score.grad is None
