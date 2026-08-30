from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from di_dmpa_jascl.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
from di_dmpa_jascl.modeling import RepairedMeanTeacher, assert_complete_classifier_load, update_gas_from_supervised_gradient


class TinyClassifier(nn.Module):
    def __init__(self, channels: int = 4, classes: int = 3) -> None:
        super().__init__()
        self.mu = nn.Conv2d(channels, classes, 3, padding=1, bias=False)
        self.grad_update = nn.Parameter(torch.zeros_like(self.mu.weight))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.conv2d(features, self.mu.weight, padding=1)


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_logit = TinyClassifier()


class TinySegNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 3, padding=1)
        self.decoder = TinyDecoder()

    def forward(self, image: torch.Tensor, *, stochastic_classifier: bool):
        features = torch.tanh(self.encoder(image))
        if stochastic_classifier:
            features = features + torch.randn_like(features) * 0.02
        return self.decoder.conv_logit(features), features


def _wrapper_optimizer_scheduler():
    torch.manual_seed(7)
    wrapper = RepairedMeanTeacher(TinySegNet(), TinySegNet())
    optimizer = torch.optim.Adam(wrapper.student.parameters(), lr=1.0e-3, weight_decay=4.0e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - min(epoch, 100) / 100) ** 0.9)
    wrapper.assert_optimizer_excludes_teacher(optimizer)
    return wrapper, optimizer, scheduler


def test_teacher_is_frozen_excluded_and_no_grad() -> None:
    wrapper, optimizer, _ = _wrapper_optimizer_scheduler()
    assert all(not parameter.requires_grad for parameter in wrapper.teacher.parameters())
    wrapper.assert_optimizer_excludes_teacher(optimizer)
    with torch.no_grad():
        teacher_logits, _ = wrapper.teacher(torch.randn(2, 3, 8, 8), stochastic_classifier=True)
    assert teacher_logits.requires_grad is False


def test_full_classifier_is_required() -> None:
    wrapper, _, _ = _wrapper_optimizer_scheduler()
    state = copy.deepcopy(wrapper.student.state_dict())
    state.pop("decoder.conv_logit.mu.weight")
    with pytest.raises(RuntimeError):
        assert_complete_classifier_load(state, wrapper.student)


def test_checkpoint_roundtrip_restores_full_state_and_rng(tmp_path: Path) -> None:
    wrapper, optimizer, scheduler = _wrapper_optimizer_scheduler()
    image = torch.randn(2, 3, 8, 8)
    label = torch.randint(0, 3, (2, 8, 8))
    logits, _ = wrapper.student(image, stochastic_classifier=True)
    loss = F.cross_entropy(logits, label)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    update_gas_from_supervised_gradient(wrapper.student)
    optimizer.step()
    scheduler.step()
    wrapper.update_teacher(0.99)
    payload = build_checkpoint(
        wrapper=wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        stage_state={"stage_index": 1, "epoch": 4, "global_step": 17},
        sampler_state={"stage_index": 1, "epoch": 4, "phase": "unlabeled", "next_batch": 3},
        prototypes=torch.randn(3, 4),
        config_hash="fixed",
        evaluation_matrices={"mean_iou": {}},
        best_metric=0.25,
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, payload)
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(1)

    restored, restored_optimizer, restored_scheduler = _wrapper_optimizer_scheduler()
    loaded = load_checkpoint(
        path,
        wrapper=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_config_hash="fixed",
    )
    assert loaded["stage_state"]["global_step"] == 17
    assert loaded["sampler_state"]["next_batch"] == 3
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
    for key, value in wrapper.student.state_dict().items():
        assert torch.equal(value, restored.student.state_dict()[key])
    for key, value in wrapper.teacher.state_dict().items():
        assert torch.equal(value, restored.teacher.state_dict()[key])
