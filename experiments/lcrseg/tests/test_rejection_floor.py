import torch

from lcrseg.methods.components.rejection_only_routing import rejection_only_weights


def test_rejection_floor() -> None:
    score = torch.linspace(0, 1, 20).reshape(1, 1, 1, 20)
    output = rejection_only_weights(score, torch.zeros((1, 1, 20), dtype=torch.long), torch.ones_like(score, dtype=torch.bool), num_classes=1, calibrator_available=True, rejected_weight_floor=0.5)
    assert set(output.weights.unique().tolist()).issubset({0.5, 1.0})
    assert bool(output.weights[output.rejection_mask].eq(0.5).all())
