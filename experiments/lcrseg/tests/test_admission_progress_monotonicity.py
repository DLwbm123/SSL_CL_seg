from __future__ import annotations

import torch

from lcrseg.methods.components.progressive_admission import classwise_progressive_admission
from tests.v0_2_test_utils import pseudo


def test_admission_expands_monotonically_from_40_to_80_percent() -> None:
    labels = torch.zeros((1, 1, 10), dtype=torch.long)
    valid = torch.ones((1, 1, 1, 10), dtype=torch.bool)
    score = torch.arange(10, dtype=torch.float32).reshape(1, 1, 1, 10)
    early = classwise_progressive_admission(pseudo(labels, valid=valid), score, valid, num_classes=1, site_step=0, total_site_steps=10)
    late = classwise_progressive_admission(pseudo(labels, valid=valid), score, valid, num_classes=1, site_step=9, total_site_steps=10)
    assert early.target_fraction == 0.4
    assert late.target_fraction == 0.8
    assert early.selected_counts == (4,)
    assert late.selected_counts == (8,)
    assert bool((early.mask & ~late.mask).any()) is False
