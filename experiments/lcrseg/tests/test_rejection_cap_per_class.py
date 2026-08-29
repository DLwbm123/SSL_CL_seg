import torch

from lcrseg.methods.components.rejection_only_routing import rejection_only_weights


def test_rejection_cap_per_class() -> None:
    score = torch.linspace(0, 1, 100).reshape(1, 1, 1, 100)
    predicted = torch.tensor([[[0] * 50 + [1] * 50]])
    output = rejection_only_weights(score, predicted, torch.ones_like(score, dtype=torch.bool), num_classes=2, calibrator_available=True)
    assert output.rejected_counts == (10, 10)
