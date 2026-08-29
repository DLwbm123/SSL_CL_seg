from __future__ import annotations

import torch

from lcrseg.methods.components.progressive_admission import classwise_progressive_admission
from tests.v0_2_test_utils import pseudo


def test_classwise_admission_hits_the_registered_fraction_with_rounding() -> None:
    labels = torch.tensor([[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]])
    valid = torch.ones((1, 1, 1, 10), dtype=torch.bool)
    output = classwise_progressive_admission(
        pseudo(labels, valid=valid),
        torch.linspace(0.0, 1.0, 10).reshape(1, 1, 1, 10),
        valid,
        num_classes=2,
        site_step=0,
        total_site_steps=10,
        pi_start=0.6,
        pi_end=0.6,
    )
    assert output.candidate_counts == (5, 5)
    assert output.selected_counts == (3, 3)
    assert output.selected_fraction_by_class == (0.6, 0.6)
