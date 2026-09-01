from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from di_dmpa_jascl.metrics import ConfusionMetrics
from di_dmpa_jascl.modeling import assert_complete_classifier_load, build_lcrseg_unet_jascl_model
from pres_jascl_v0_1.core import pixel_confusion, segmentation_metrics
from .test_official_model_contract import REFERENCE_ROOT, UPSTREAM_PATH


def test_pres_checkpoint_classifier_contract_is_complete_and_read_only():
    model = build_lcrseg_unet_jascl_model(
        REFERENCE_ROOT, upstream_path=UPSTREAM_PATH, input_channels=3, num_classes=3,
    ).eval().requires_grad_(False)
    state = copy.deepcopy(model.state_dict())
    assert_complete_classifier_load(state, model)
    state.pop("decoder.conv_logit.mu.weight")
    with pytest.raises(RuntimeError, match="classifier state is incomplete"):
        assert_complete_classifier_load(state, model)


def test_pres_segmentation_arithmetic_matches_frozen_evaluator():
    prediction = np.array([[0, 1, 2], [1, 2, 2]])
    target = np.array([[0, 1, 1], [1, 2, 255]])
    frozen = ConfusionMetrics(3, 255)
    frozen.update(torch.from_numpy(prediction), torch.from_numpy(target))
    observed = segmentation_metrics(pixel_confusion(prediction, target))
    expected = frozen.summary()
    assert observed["confusion_matrix"] == expected["confusion_matrix"]
    for key in ("mean_iou", "mean_dice", "mean_foreground_dice"):
        assert observed[key] == pytest.approx(expected[key], abs=1e-15)
