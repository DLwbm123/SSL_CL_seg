from __future__ import annotations

import torch

from lcrseg.methods.components.rejection_only_routing import rejection_only_weights


def test_rejection_is_capped_per_old_predicted_class() -> None:
    score = torch.zeros((1, 1, 1, 20))
    old_class = torch.tensor([[[0] * 10 + [1] * 10]])
    valid = torch.ones_like(score, dtype=torch.bool)
    output = rejection_only_weights(
        score, old_class, valid, num_classes=2, calibrator_available=True,
        probability_threshold=0.7, max_reject_fraction_per_class=0.2,
        rejected_weight_floor=0.5,
    )
    assert output.candidate_counts == (10, 10)
    assert output.rejected_counts == (2, 2)
    assert output.rejected_fraction_by_class == (0.2, 0.2)
