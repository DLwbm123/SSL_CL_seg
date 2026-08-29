"""Deterministic, same-geometry CRISP style probes using frozen augmentations only.

This module is audit-only until the CRISP feasibility compiler emits the exact
``CRISP_FEASIBILITY_SUPPORTED`` status.  It deliberately accepts images and
identity metadata only: labels, pseudo-labels, models, and optimizers are not
part of the contract.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from ..data.transforms import _flip


# These values are the existing WeakStrongTransform/continual-runner defaults
# and the frozen values in every formal Fundus R0 configuration.  They are not
# constructor arguments so a feasibility run cannot silently tune them.
_FLIP_PROBABILITY = 0.5
_STRONG_NOISE_STD = 0.03
_BRIGHTNESS_DELTA = 0.10
_CONTRAST_DELTA = 0.10


def crisp_style_probe_contract() -> dict[str, Any]:
    """Return the canonical, serialisable style-probe contract."""

    contract: dict[str, Any] = {
        "protocol_id": "crispseg_v0_1",
        "purpose": "audit_only_channel_role_estimation",
        "seed_components": ["protocol_seed", "site_id", "case_id"],
        "geometry": {
            "operators": ["horizontal_flip", "vertical_flip"],
            "flip_probability": _FLIP_PROBABILITY,
            "paired_views_share_exact_geometry": True,
        },
        "appearance": {
            "operators_in_order": ["contrast", "brightness", "gaussian_noise"],
            "contrast_delta": _CONTRAST_DELTA,
            "brightness_delta": _BRIGHTNESS_DELTA,
            "strong_noise_std": _STRONG_NOISE_STD,
            "source": "existing WeakStrongTransform strong appearance path",
        },
        "datasets": {
            "fundus": "three-channel intensity path; shared scalar contrast/brightness and elementwise Gaussian noise",
            "prostate": "single-channel MRI intensity path; scalar contrast/brightness and elementwise Gaussian noise",
        },
        "cutout": False,
        "new_augmentation": False,
        "uses_global_rng": False,
        "hidden_gt_usage": "none",
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**contract, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def _case_seed(*, protocol_seed: int, site_id: str, case_id: str) -> int:
    if not site_id or not case_id:
        raise ValueError("site_id and case_id must be non-empty")
    identity = f"crispseg_v0_1\0{int(protocol_seed)}\0{site_id}\0{case_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & ((1 << 63) - 1)


class FrozenStyleProbeTransform:
    """Produce clean/style views without changing geometry or global RNG state."""

    def __init__(self, *, protocol_seed: int) -> None:
        self.protocol_seed = int(protocol_seed)

    def __call__(
        self,
        *,
        image: torch.Tensor,
        dataset: str,
        site_id: str,
        case_id: str,
    ) -> dict[str, Any]:
        if image.ndim != 3 or not image.is_floating_point():
            raise ValueError(f"style-probe image must be floating [C,H,W], got {tuple(image.shape)} {image.dtype}")
        if not bool(torch.isfinite(image).all()):
            raise ValueError("style-probe image contains non-finite values")
        expected_channels = {"fundus": 3, "prostate": 1}
        if dataset not in expected_channels:
            raise ValueError(f"unsupported frozen style-probe dataset: {dataset!r}")
        if image.shape[0] != expected_channels[dataset]:
            raise ValueError(f"{dataset} style-probe image requires {expected_channels[dataset]} channels")

        case_seed = _case_seed(protocol_seed=self.protocol_seed, site_id=site_id, case_id=case_id)
        generator = torch.Generator(device=image.device)
        generator.manual_seed(case_seed)
        hflip = bool(torch.rand((), generator=generator, device=image.device) < _FLIP_PROBABILITY)
        vflip = bool(torch.rand((), generator=generator, device=image.device) < _FLIP_PROBABILITY)

        # View A is the existing weak-appearance path.  View B begins from the
        # exact same tensor and applies only the existing strong appearance ops.
        clean_image = _flip(image, hflip=hflip, vflip=vflip).clone()
        style_image = clean_image.clone()
        brightness = 1.0 + (float(torch.rand((), generator=generator, device=image.device)) * 2.0 - 1.0) * _BRIGHTNESS_DELTA
        contrast = 1.0 + (float(torch.rand((), generator=generator, device=image.device)) * 2.0 - 1.0) * _CONTRAST_DELTA
        center = style_image.mean(dim=(-2, -1), keepdim=True)
        style_image = (style_image - center) * contrast + center
        style_image = style_image * brightness
        noise = torch.randn(
            style_image.shape,
            dtype=style_image.dtype,
            device=style_image.device,
            generator=generator,
        )
        style_image = style_image + noise * _STRONG_NOISE_STD
        valid_mask = torch.ones((1, *style_image.shape[-2:]), dtype=torch.bool, device=style_image.device)
        return {
            "clean_image": clean_image,
            "style_image": style_image,
            "style_valid_mask": valid_mask,
            "geometry_record": {
                "hflip": hflip,
                "vflip": vflip,
                "cutout": False,
                "cutout_box": None,
            },
            "style_record": {
                "dataset": dataset,
                "site_id": site_id,
                "case_id": case_id,
                "case_seed": case_seed,
                "brightness": brightness,
                "contrast": contrast,
                "strong_noise_std": _STRONG_NOISE_STD,
                "contract_sha256": crisp_style_probe_contract()["contract_sha256"],
            },
        }


__all__ = ["FrozenStyleProbeTransform", "crisp_style_probe_contract"]
