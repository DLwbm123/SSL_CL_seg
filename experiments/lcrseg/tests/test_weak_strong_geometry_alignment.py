from __future__ import annotations

import torch

from lcrseg.data.transforms import WeakStrongTransform


def test_weak_strong_geometry_alignment_and_cutout_mask() -> None:
    image = torch.ones((1, 20, 20))
    image[:, 3, 5] = 7.0
    transform = WeakStrongTransform(
        flip_probability=1.0,
        strong_noise_std=0.0,
        brightness_delta=0.0,
        contrast_delta=0.0,
        cutout_probability=1.0,
        cutout_fraction=0.25,
    )
    torch.manual_seed(123)
    result = transform(image=image)
    assert result["geometry_record"]["hflip"] is True
    assert result["geometry_record"]["vflip"] is True
    valid = result["strong_valid_mask"]
    assert valid.dtype is torch.bool
    assert not bool(valid.all())
    assert torch.equal(result["weak_image"] * valid, result["strong_image"] * valid)
    assert torch.equal(result["strong_image"][~valid.expand_as(result["strong_image"])], torch.zeros_like(result["strong_image"][~valid.expand_as(result["strong_image"])]))
