from __future__ import annotations

from pathlib import Path

import torch

from di_dmpa_jascl.modeling import LCRSegUNet2DJASCL, build_lcrseg_unet_jascl_model
from di_dmpa_jascl.provenance import assert_upstream_unchanged


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = ROOT / "third_party" / "JASCL_REFERENCE"
UPSTREAM_PATH = "Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT"


def test_lcrseg_unet_body_and_official_classifier_contract() -> None:
    assert_upstream_unchanged(REFERENCE_ROOT, UPSTREAM_PATH)
    model = build_lcrseg_unet_jascl_model(
        REFERENCE_ROOT,
        upstream_path=UPSTREAM_PATH,
        input_channels=3,
        num_classes=3,
    )
    assert isinstance(model, LCRSegUNet2DJASCL)
    assert model.enc1.block[0].in_channels == 3
    assert model.enc1.block[0].out_channels == 16
    assert model.bottleneck.block[0].out_channels == 128
    classifier = model.decoder.conv_logit
    assert classifier.mu.in_channels == 16
    assert classifier.mu.out_channels == 3
    assert classifier.mu.kernel_size == (3, 3)
    assert classifier.mu.padding == (1, 1)
    assert tuple(classifier.grad_update.shape[-2:]) == (3, 3)
    logits, features = model(torch.randn(1, 3, 32, 32), stochastic_classifier=False)
    assert logits.shape == (1, 3, 32, 32)
    assert features.shape == (1, 16, 32, 32)


def test_unet_supports_single_channel_medical_benchmarks() -> None:
    model = build_lcrseg_unet_jascl_model(
        REFERENCE_ROOT,
        upstream_path=UPSTREAM_PATH,
        input_channels=1,
        num_classes=4,
    )
    logits, features = model(torch.randn(1, 1, 32, 32), stochastic_classifier=False)
    assert logits.shape == (1, 4, 32, 32)
    assert features.shape == (1, 16, 32, 32)
