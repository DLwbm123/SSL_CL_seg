from __future__ import annotations

import pytest
import torch

from lcrseg.contracts import UnlabeledBatch
from lcrseg.methods.lcrseg_v0_2 import LCRSegV02Method
from lcrseg.models import UNet2D


class _UnlabeledOnlyBatcher:
    steps_per_epoch = 1

    @staticmethod
    def batch_at(index: int) -> UnlabeledBatch:
        image = torch.zeros((1, 3, 8, 8))
        return UnlabeledBatch(
            weak_image=image,
            strong_image=image,
            strong_valid_mask=torch.ones((1, 1, 8, 8), dtype=torch.bool),
            case_id=["u"],
            patient_id=["u"],
            site=["A"],
            slice_index=[None],
            geometry_record=[{}],
        )


def test_calibrator_rejects_any_non_labeled_batch_source() -> None:
    method = LCRSegV02Method(UNet2D(3, 3))
    # The type gate occurs before a batch is read, so no train-unlabeled data
    # can be accidentally consumed as calibration data.
    method.old_model = method.model
    method.old_anchor_bank = method.current_anchor_bank
    with pytest.raises(TypeError, match="LabeledBatch"):
        method.on_epoch_end(epoch=9, calibration_batcher=_UnlabeledOnlyBatcher(), device="cpu")
