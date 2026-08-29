import torch

from lcrseg.methods.lcrseg_v0_2a import _effective_sample_size


def test_relation_ess_not_collapsed() -> None:
    weights = torch.ones((1, 1, 1, 100))
    weights[..., :20] = 0.5
    ess = _effective_sample_size(weights, torch.ones_like(weights, dtype=torch.bool))
    assert ess >= 80.0
