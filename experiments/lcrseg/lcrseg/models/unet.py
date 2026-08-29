"""Small, shared 2D U-Net used by every baseline and LCR-Seg V0.1."""
from __future__ import annotations

import torch
from torch import nn

from ..contracts import SegModelOutput
from .projection_head import ProjectionHead, _groups


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.merge = ConvNormAct(out_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        if value.shape[-2:] != skip.shape[-2:]:
            raise ValueError(f"U-Net geometry mismatch: up={value.shape[-2:]}, skip={skip.shape[-2:]}")
        return self.merge(torch.cat((value, skip), dim=1))


class UNet2D(nn.Module):
    """Channels 16/32/64/128 with relation feature at one-quarter resolution."""

    def __init__(self, in_channels: int, num_classes: int, *, base_channels: int = 16, relation_dim: int = 128) -> None:
        super().__init__()
        if base_channels != 16:
            raise ValueError("V0.1 fixes base_channels=16 for fair baseline comparisons")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.relation_dim = int(relation_dim)
        self.enc1 = ConvNormAct(in_channels, 16)
        self.enc2 = ConvNormAct(16, 32)
        self.enc3 = ConvNormAct(32, 64)
        self.bottleneck = ConvNormAct(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dec3 = UpBlock(128, 64, 64)
        self.dec2 = UpBlock(64, 32, 32)
        self.dec1 = UpBlock(32, 16, 16)
        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.projection_head = ProjectionHead(64, relation_dim=relation_dim)

    def forward(self, image: torch.Tensor) -> SegModelOutput:
        if image.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(image.shape)}")
        if image.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {image.shape[1]}")
        height, width = image.shape[-2:]
        if height % 8 or width % 8:
            raise ValueError(f"input H/W must be divisible by 8, got {(height, width)}")
        enc1 = self.enc1(image)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        dec3 = self.dec3(bottleneck, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)
        return SegModelOutput(
            logits=self.segmentation_head(dec1),
            relation_features=self.projection_head(dec3),
            decoder_features={"dec3": dec3, "dec1": dec1},
        )
