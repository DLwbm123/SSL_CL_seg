"""Two query segmentation adapters; no network or weight download on import."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Module):
    # Source: experiments/lcrseg/di_dmpa_jascl/modeling.py at f6b9697.
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, output_channels), nn.ReLU(inplace=False),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, output_channels), nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, output_channels, 2, stride=2)
        self.merge = ConvNormAct(output_channels + skip_channels, output_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError("U-Net skip geometry mismatch")
        return self.merge(torch.cat((x, skip), dim=1))


class UNetBody(nn.Module):
    """The existing 16/32/64/128 GroupNorm body, without its old classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = ConvNormAct(3, 16)
        self.enc2 = ConvNormAct(16, 32)
        self.enc3 = ConvNormAct(32, 64)
        self.bottleneck = ConvNormAct(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = UpBlock(128, 64, 64)
        self.dec2 = UpBlock(64, 32, 32)
        self.dec1 = UpBlock(32, 16, 16)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3 or any(s % 8 for s in image.shape[-2:]):
            raise ValueError("expected RGB BCHW with height/width divisible by 8")
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        z = self.bottleneck(self.pool(e3))
        return self.dec1(self.dec2(self.dec3(z, e3), e2), e1)


class UNetDense(nn.Module):
    """Deterministic valid 3x3 readout and the legacy align_corners geometry."""

    def __init__(self) -> None:
        super().__init__()
        self.body = UNetBody()
        self.head = nn.Conv2d(16, 3, 3, padding=0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(self.head(self.body(image)), size=image.shape[-2:], mode="bilinear", align_corners=True)


def semantic_probabilities(class_logits: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
    """C=3 includes background; the fourth class is no-object and is discarded."""
    if class_logits.ndim != 3 or class_logits.shape[-1] != 4 or mask_logits.shape[:2] != class_logits.shape[:2]:
        raise ValueError("expected class [B,K,4] and mask [B,K,H,W]")
    scores = torch.einsum("bkc,bkhw->bchw", class_logits.softmax(-1)[..., :3], mask_logits.sigmoid())
    denominator = scores.sum(1, keepdim=True)
    return torch.where(denominator > 0, scores / denominator.clamp_min(1e-12), torch.full_like(scores, 1 / 3))


class QueryHead(nn.Module):
    def __init__(self, dim: int, queries: int = 12) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 4)

    def forward(self, mask_features: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, dim = mask_features.shape[:2]
        q = self.queries.unsqueeze(0).expand(batch, -1, -1)
        tokens = context.flatten(2).transpose(1, 2)
        q = self.norm1(q + self.attention(q, tokens, tokens, need_weights=False)[0])
        q = self.norm2(q + self.ffn(q))
        masks = torch.einsum("bkd,bdhw->bkhw", q, mask_features) / math.sqrt(dim)
        return self.classifier(q), masks, q


class UNetQuery(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = UNetBody()
        self.embedding = nn.Conv2d(16, 128, 1)
        self.head = QueryHead(128)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.embedding(self.body(image))
        masks_at_quarter = F.avg_pool2d(feature, 4)
        context = F.avg_pool2d(masks_at_quarter, 2)
        classes, masks, queries = self.head(masks_at_quarter, context)
        masks = F.interpolate(masks, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return dict(class_logits=classes, mask_logits=masks, queries=queries,
                    pixels=masks_at_quarter, semantic=semantic_probabilities(classes, masks))


class DINOv2Adapter(nn.Module):
    """Last-block query injection into pinned, non-register DINOv2 ViT-S/14."""

    def __init__(self, backbone: nn.Module, *, query: bool) -> None:
        super().__init__()
        if (backbone.embed_dim, backbone.patch_size, backbone.num_register_tokens, len(backbone.blocks)) != (384, 14, 0, 12):
            raise ValueError("expected official DINOv2 ViT-S/14 without registers")
        self.backbone = backbone
        self.query = query
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.queries = nn.Parameter(torch.randn(12, 384) * 0.02) if query else None
        self.upsample = nn.Sequential(nn.ConvTranspose2d(384, 192, 2, stride=2), nn.GELU(),
                                      nn.ConvTranspose2d(192, 128, 2, stride=2))
        self.classifier = nn.Linear(384, 4) if query else nn.Conv2d(128, 3, 1)
        self.mask_projection = nn.Linear(384, 128) if query else None

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        if image.ndim != 4 or image.shape[1:] != (3, 384, 384):
            raise ValueError("DINOv2 adapter requires RGB 384x384")
        padded = F.pad((image - self.mean) / self.std, (0, 8, 0, 8), mode="reflect")
        x = self.backbone.prepare_tokens_with_masks(padded)
        for block in self.backbone.blocks[:-1]:
            x = block(x)
        if self.query:
            x = torch.cat((self.queries.unsqueeze(0).expand(x.shape[0], -1, -1), x), dim=1)
        x = self.backbone.norm(self.backbone.blocks[-1](x))
        if self.query:
            queries, patch_tokens = x[:, :12], x[:, 13:]
        else:
            patch_tokens = x[:, 1:]
        if patch_tokens.shape[1] != 28 * 28:
            raise ValueError("wrong padded patch geometry")
        pixel_tokens = patch_tokens.transpose(1, 2).reshape(x.shape[0], 384, 28, 28)
        pixels = self.upsample(pixel_tokens)
        if not self.query:
            return F.interpolate(self.classifier(pixels), size=(392, 392), mode="bilinear", align_corners=False)[..., :384, :384]
        classes = self.classifier(queries)
        mask_features = self.mask_projection(queries)
        masks = torch.einsum("bkd,bdhw->bkhw", mask_features, pixels) / math.sqrt(128)
        masks = F.interpolate(masks, size=(392, 392), mode="bilinear", align_corners=False)[..., :384, :384]
        return dict(class_logits=classes, mask_logits=masks, queries=queries,
                    pixels=pixel_tokens, semantic=semantic_probabilities(classes, masks))


def load_pinned_dinov2(source_root: Path, weight: Path, *, commit: str, weight_sha256: str) -> nn.Module:
    """Construct from local official source and a single local file. Never downloads."""
    source_root, weight = source_root.resolve(strict=True), weight.resolve(strict=True)
    source_manifest = json.loads((source_root / "DINOV2_SOURCE_SHA256.json").read_text())
    if source_manifest["commit"] != commit:
        raise ValueError("DINOv2 source commit mismatch")
    for relative, expected in source_manifest["files"].items():
        if hashlib.sha256((source_root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("DINOv2 source file differs: " + relative)
    digest = hashlib.sha256()
    with weight.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != weight_sha256:
        raise ValueError("DINOv2 weight SHA256 mismatch")
    if "dinov2" in sys.modules and not Path(sys.modules["dinov2"].__file__).resolve().is_relative_to(source_root):
        raise RuntimeError("a different DINOv2 source is already imported")
    sys.path.insert(0, str(source_root))
    try:
        from dinov2.hub.backbones import dinov2_vits14
        backbone = dinov2_vits14(pretrained=False, block_chunks=0)
    finally:
        sys.path.pop(0)
    backbone.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
    return backbone
