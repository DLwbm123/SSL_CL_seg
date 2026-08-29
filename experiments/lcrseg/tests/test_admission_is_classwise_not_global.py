from __future__ import annotations

import torch

from lcrseg.methods.components.progressive_admission import classwise_progressive_admission
from tests.v0_2_test_utils import pseudo


def test_admission_selects_each_pseudo_class_instead_of_global_topk() -> None:
    labels = torch.tensor([[[0, 0, 0, 0, 1, 1, 1, 1]]])
    valid = torch.ones((1, 1, 1, 8), dtype=torch.bool)
    # A global top-50% would choose only class 1.  Class-wise top-50% must
    # retain two pixels from each class.
    score = torch.tensor([[[[0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6]]]])
    output = classwise_progressive_admission(
        pseudo(labels, valid=valid), score, valid, num_classes=2,
        site_step=0, total_site_steps=2, pi_start=0.5, pi_end=0.5,
    )
    selected_labels = labels[output.mask[:, 0]]
    assert int(selected_labels.eq(0).sum()) == 2
    assert int(selected_labels.eq(1).sum()) == 2
