from __future__ import annotations

import copy

import torch
from torch.nn import functional as F

from .test_model_checkpoint import TinySegNet


def test_unlabeled_backward_is_followed_by_optimizer_step() -> None:
    torch.manual_seed(21)
    student = TinySegNet()
    optimizer = torch.optim.Adam(student.parameters(), lr=1.0e-3, weight_decay=4.0e-5)
    before = copy.deepcopy(student.state_dict())
    image = torch.randn(2, 3, 8, 8)
    label = torch.randint(0, 3, (2, 8, 8))
    logits, _ = student(image)
    supervised_loss = F.cross_entropy(logits, label)
    upstream_pseudo_consistency = torch.tensor(0.5)
    total_loss = supervised_loss + 0.5 * upstream_pseudo_consistency
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    assert any(not torch.equal(before[key], student.state_dict()[key]) for key in before)
