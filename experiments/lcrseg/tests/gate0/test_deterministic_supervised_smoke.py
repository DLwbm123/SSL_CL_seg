from __future__ import annotations

import copy

import torch
from torch.nn import functional as F

from .test_model_checkpoint import TinySegNet


def _one_step(model, optimizer, image, label):
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(image, stochastic_classifier=False)
    loss = F.cross_entropy(logits, label)
    loss.backward()
    optimizer.step()
    return loss.detach(), copy.deepcopy(model.state_dict())


def test_identical_supervised_steps_are_deterministic_smoke() -> None:
    torch.manual_seed(13)
    baseline = TinySegNet()
    off_switch = TinySegNet()
    off_switch.load_state_dict(baseline.state_dict(), strict=True)
    baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=1.0e-3, weight_decay=4.0e-5)
    off_optimizer = torch.optim.Adam(off_switch.parameters(), lr=1.0e-3, weight_decay=4.0e-5)
    image = torch.randn(2, 3, 8, 8)
    label = torch.randint(0, 3, (2, 8, 8))
    torch.manual_seed(99)
    baseline_loss, baseline_state = _one_step(baseline, baseline_optimizer, image, label)
    torch.manual_seed(99)
    off_loss, off_state = _one_step(off_switch, off_optimizer, image, label)
    assert torch.equal(baseline_loss, off_loss)
    assert baseline_state.keys() == off_state.keys()
    assert all(torch.equal(baseline_state[key], off_state[key]) for key in baseline_state)
