"""Shared-geometry weak/strong transforms and strict batch constructors."""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from ..contracts import LabeledBatch, UnlabeledBatch


def _flip(image: torch.Tensor, *, hflip: bool, vflip: bool) -> torch.Tensor:
    dimensions: list[int] = []
    if hflip:
        dimensions.append(-1)
    if vflip:
        dimensions.append(-2)
    return torch.flip(image, dimensions) if dimensions else image


class LabeledTransform:
    def __init__(self, *, flip_probability: float = 0.5) -> None:
        self.flip_probability = float(flip_probability)

    def __call__(self, *, image: torch.Tensor, label: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hflip = bool(torch.rand(()) < self.flip_probability)
        vflip = bool(torch.rand(()) < self.flip_probability)
        return {
            "image": _flip(image, hflip=hflip, vflip=vflip),
            "label": _flip(label, hflip=hflip, vflip=vflip),
            "valid_mask": _flip(valid_mask, hflip=hflip, vflip=vflip),
        }


class WeakStrongTransform:
    """Apply identical geometry and strong-view-only appearance/cutout changes."""

    def __init__(
        self,
        *,
        flip_probability: float = 0.5,
        strong_noise_std: float = 0.03,
        brightness_delta: float = 0.10,
        contrast_delta: float = 0.10,
        cutout_probability: float = 0.5,
        cutout_fraction: float = 0.20,
    ) -> None:
        self.flip_probability = float(flip_probability)
        self.strong_noise_std = float(strong_noise_std)
        self.brightness_delta = float(brightness_delta)
        self.contrast_delta = float(contrast_delta)
        self.cutout_probability = float(cutout_probability)
        self.cutout_fraction = float(cutout_fraction)

    def __call__(self, *, image: torch.Tensor) -> dict[str, Any]:
        hflip = bool(torch.rand(()) < self.flip_probability)
        vflip = bool(torch.rand(()) < self.flip_probability)
        weak_image = _flip(image, hflip=hflip, vflip=vflip)
        strong_image = weak_image.clone()
        brightness = 1.0 + (float(torch.rand(())) * 2.0 - 1.0) * self.brightness_delta
        contrast = 1.0 + (float(torch.rand(())) * 2.0 - 1.0) * self.contrast_delta
        center = strong_image.mean(dim=(-2, -1), keepdim=True)
        strong_image = (strong_image - center) * contrast + center
        strong_image = strong_image * brightness
        if self.strong_noise_std > 0:
            strong_image = strong_image + torch.randn_like(strong_image) * self.strong_noise_std
        valid_mask = torch.ones((1, *strong_image.shape[-2:]), dtype=torch.bool)
        cutout = False
        cutout_box: tuple[int, int, int, int] | None = None
        if self.cutout_probability > 0 and bool(torch.rand(()) < self.cutout_probability):
            height, width = strong_image.shape[-2:]
            cut_height = max(1, int(round(height * self.cutout_fraction)))
            cut_width = max(1, int(round(width * self.cutout_fraction)))
            top = int(torch.randint(0, height - cut_height + 1, ()).item())
            left = int(torch.randint(0, width - cut_width + 1, ()).item())
            strong_image[:, top : top + cut_height, left : left + cut_width] = 0.0
            valid_mask[:, top : top + cut_height, left : left + cut_width] = False
            cutout = True
            cutout_box = (top, left, cut_height, cut_width)
        return {
            "weak_image": weak_image,
            "strong_image": strong_image,
            "strong_valid_mask": valid_mask,
            "geometry_record": {"hflip": hflip, "vflip": vflip, "cutout": cutout, "cutout_box": cutout_box},
        }


def collate_labeled(samples: list[dict[str, Any]]) -> LabeledBatch:
    if not samples:
        raise ValueError("cannot collate an empty labeled batch")
    required = {"image", "label", "valid_mask", "case_id", "patient_id", "site", "slice_index"}
    for sample in samples:
        missing = required.difference(sample)
        if missing:
            raise ValueError(f"labeled sample misses keys: {sorted(missing)}")
    return LabeledBatch(
        image=torch.stack([sample["image"] for sample in samples]),
        label=torch.stack([sample["label"] for sample in samples]).long(),
        valid_mask=torch.stack([sample["valid_mask"] for sample in samples]).bool(),
        case_id=[str(sample["case_id"]) for sample in samples],
        patient_id=[str(sample["patient_id"]) for sample in samples],
        site=[str(sample["site"]) for sample in samples],
        slice_index=[sample["slice_index"] for sample in samples],
    )


def collate_unlabeled(samples: list[dict[str, Any]]) -> UnlabeledBatch:
    if not samples:
        raise ValueError("cannot collate an empty unlabeled batch")
    forbidden = {"label", "hidden_label", "label_h5_relpath", "diagnostic_path"}
    required = {"weak_image", "strong_image", "strong_valid_mask", "case_id", "patient_id", "site", "slice_index", "geometry_record"}
    for sample in samples:
        leaked = forbidden.intersection(sample)
        if leaked:
            raise RuntimeError(f"hidden label leakage in unlabeled batch: {sorted(leaked)}")
        missing = required.difference(sample)
        if missing:
            raise ValueError(f"unlabeled sample misses keys: {sorted(missing)}")
    return UnlabeledBatch(
        weak_image=torch.stack([sample["weak_image"] for sample in samples]),
        strong_image=torch.stack([sample["strong_image"] for sample in samples]),
        strong_valid_mask=torch.stack([sample["strong_valid_mask"] for sample in samples]).bool(),
        case_id=[str(sample["case_id"]) for sample in samples],
        patient_id=[str(sample["patient_id"]) for sample in samples],
        site=[str(sample["site"]) for sample in samples],
        slice_index=[sample["slice_index"] for sample in samples],
        geometry_record=[dict(sample["geometry_record"]) for sample in samples],
    )


def downsample_valid_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask.float(), size=size, mode="nearest").bool()
