from __future__ import annotations

import torch

from lcrseg.methods.components.rejection_only_routing import rejection_only_weights


def test_consolidation_weights_never_drop_below_registered_floor_or_exceed_one() -> None:
    score = torch.tensor([[[[0.1, 0.2, 0.9, 0.95, 0.1]]]])
    old_class = torch.zeros((1, 1, 5), dtype=torch.long)
    valid = torch.ones_like(score, dtype=torch.bool)
    output = rejection_only_weights(
        score, old_class, valid, num_classes=1, calibrator_available=True,
        probability_threshold=0.7, max_reject_fraction_per_class=0.4,
        rejected_weight_floor=0.5,
    )
    assert float(output.weights.min()) == 0.5
    assert float(output.weights.max()) == 1.0
    assert not bool(output.weights.eq(0).any())
