from __future__ import annotations

import torch

from lcrseg.methods.components.rejection_only_routing import rejection_only_weights


def test_unavailable_calibrator_is_exact_uniform_relation_kd() -> None:
    score = torch.tensor([[[[0.01, 0.99, 0.02, 0.98]]]])
    old_class = torch.tensor([[[0, 0, 1, 1]]])
    valid = torch.ones_like(score, dtype=torch.bool)
    output = rejection_only_weights(score, old_class, valid, num_classes=2, calibrator_available=False)
    assert not output.calibrator_available
    assert not bool(output.rejection_mask.any())
    assert bool(output.weights.eq(1.0).all())
    assert output.candidate_counts == (2, 2)
