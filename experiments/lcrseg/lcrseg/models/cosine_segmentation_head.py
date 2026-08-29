"""Bias-free cosine pixel classifier used by the registered SR-GAS variants."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CosineSegmentationHead(nn.Module):
    """A 1x1 classifier whose forward normalizes features and weights in FP32."""

    def __init__(self, in_channels: int, num_classes: int, *, temperature: float = 10.0, eps: float = 1.0e-8) -> None:
        super().__init__()
        if in_channels < 1 or num_classes < 1:
            raise ValueError("cosine classifier dimensions must be positive")
        if temperature <= 0 or eps <= 0:
            raise ValueError("cosine temperature and epsilon must be positive")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.empty(num_classes, in_channels, 1, 1))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    @classmethod
    def from_conv2d(
        cls,
        source: nn.Conv2d,
        *,
        temperature: float = 10.0,
        eps: float = 1.0e-8,
    ) -> "CosineSegmentationHead":
        if source.kernel_size != (1, 1) or source.groups != 1:
            raise ValueError("SR-GAS requires the existing final 1x1 classifier")
        head = cls(source.in_channels, source.out_channels, temperature=temperature, eps=eps)
        with torch.no_grad():
            head.weight.copy_(source.weight.detach())
        return head

    def forward(self, features: torch.Tensor, *, weight_override: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.in_channels:
            raise ValueError("cosine classifier features must be [B,D,H,W]")
        weight = self.weight if weight_override is None else weight_override
        if weight.shape != self.weight.shape:
            raise ValueError(f"weight override shape mismatch: {tuple(weight.shape)} vs {tuple(self.weight.shape)}")
        feature_dtype = features.dtype
        normalized_features = F.normalize(features.float(), p=2.0, dim=1, eps=self.eps)
        normalized_weight = F.normalize(weight.float().flatten(1), p=2.0, dim=1, eps=self.eps).view_as(weight)
        logits = F.conv2d(normalized_features, normalized_weight, bias=None)
        return (self.temperature * logits).to(dtype=feature_dtype)


__all__ = ["CosineSegmentationHead"]
