import torch

from lcrseg.methods.lcrseg_v0_3 import LCRSegV03Method
from lcrseg.models import UNet2D
from tests.v0_2a_test_utils import batches, previous_checkpoint, routing_config


def test_v03_p0_old_relation_no_backward(tmp_path) -> None:
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
    result = method.training_step(*batches(), global_step=1, site_step=0)
    gradients = torch.autograd.grad(
        result.losses["loss_relation"],
        tuple(method.model.parameters()),
        allow_unused=True,
        retain_graph=True,
    )
    assert all(gradient is None or torch.count_nonzero(gradient) == 0 for gradient in gradients)
    assert method.old_model is not None
    assert all(parameter.grad is None for parameter in method.old_model.parameters())

