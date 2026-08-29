"""Projection head for the dense semantic relation field."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for candidate in range(min(8, channels), 0, -1):
        if channels % candidate == 0:
            return candidate
    return 1


class ProjectionHead(nn.Module):
    def __init__(self, in_channels: int, relation_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, relation_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(relation_dim), relation_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(relation_dim, relation_dim, kernel_size=1, bias=True),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(feature), p=2, dim=1, eps=1e-8)
