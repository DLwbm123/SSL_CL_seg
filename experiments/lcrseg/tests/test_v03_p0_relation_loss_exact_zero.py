import torch

from lcrseg.methods.lcrseg_v0_3 import LCRSegV03Method
from lcrseg.models import UNet2D
from tests.v0_2a_test_utils import batches, previous_checkpoint, routing_config


def test_v03_p0_relation_loss_exact_zero(tmp_path) -> None:
    method = LCRSegV03Method(
        UNet2D(3, 3),
        config=routing_config(
            protocol_id="lcrseg_v0_3",
            variant_id="P0",
            assimilation_mode="progressive_admission",
            consolidation_mode="none",
            lambda_relation=0.0,
        ),
    )
    method.begin_site("B", previous_checkpoint(tmp_path), 4)
    labeled, unlabeled = batches()
    result = method.training_step(labeled, unlabeled, global_step=1, site_step=0)
    assert torch.equal(result.losses["loss_relation"], result.losses["loss_relation"].new_zeros(()))
    assert result.scalars["lambda_relation_effective"] == 0.0
    assert result.scalars["relation_loss_numerator"] == 0.0
    assert result.scalars["relation_denominator"] == 0.0

