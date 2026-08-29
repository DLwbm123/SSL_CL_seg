from __future__ import annotations

import pytest
import torch

from lcrseg.models import UNet2D


@pytest.mark.parametrize(
    ("channels", "classes", "shape"),
    ((1, 2, (2, 1, 256, 256)), (1, 4, (2, 1, 384, 384)), (3, 3, (2, 3, 384, 384))),
)
def test_model_output_contract(channels: int, classes: int, shape: tuple[int, ...]) -> None:
    output = UNet2D(channels, classes)(torch.randn(shape))
    assert output.logits.shape == (shape[0], classes, shape[-2], shape[-1])
    assert output.relation_features.shape == (shape[0], 128, shape[-2] // 4, shape[-1] // 4)
    assert output.decoder_features is not None
    assert set(output.decoder_features) == {"dec3", "dec1"}
    assert output.decoder_features["dec3"].shape == (shape[0], 64, shape[-2] // 4, shape[-1] // 4)
    assert output.decoder_features["dec1"].shape == (shape[0], 16, shape[-2], shape[-1])
    norms = output.relation_features.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
