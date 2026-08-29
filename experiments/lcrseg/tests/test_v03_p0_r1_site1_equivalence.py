import copy

import torch

from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from lcrseg.engine.trainer import TrainerState
from lcrseg.methods.lcrseg_v0_2a import LCRSegV02AMethod
from lcrseg.methods.lcrseg_v0_3 import LCRSegV03Method
from lcrseg.models import UNet2D
from tests.v0_2a_test_utils import batches, routing_config, trainer


def test_v03_p0_r1_site1_equivalence() -> None:
    torch.manual_seed(20260827)
    r1 = LCRSegV02AMethod(
        UNet2D(3, 3),
        config=routing_config(
            variant_id="R1",
            assimilation_mode="progressive_admission",
            consolidation_mode="uniform_relation",
        ),
    )
    p0 = LCRSegV03Method(
        UNet2D(3, 3),
        config=routing_config(
            protocol_id="lcrseg_v0_3",
            variant_id="P0",
            assimilation_mode="progressive_admission",
            consolidation_mode="none",
            lambda_relation=0.0,
        ),
    )
    p0.model.load_state_dict(copy.deepcopy(r1.model.state_dict()), strict=True)
    r1.begin_site("REFUGE", None, 4)
    p0.begin_site("REFUGE", None, 4)
    first_trainer, second_trainer = trainer(r1, 4), trainer(p0, 4)
    labeled, unlabeled = batches()
    for step in range(2):
        rng = capture_rng_state()
        first = first_trainer.train_step(
            labeled, unlabeled, state=TrainerState(global_step=step, site_step=step, epoch=0)
        )
        restore_rng_state(rng)
        second = second_trainer.train_step(
            labeled, unlabeled, state=TrainerState(global_step=step, site_step=step, epoch=0)
        )
        assert torch.equal(first.total_loss, second.total_loss)
        for key, value in r1.model.state_dict().items():
            assert torch.equal(value, p0.model.state_dict()[key])
        assert torch.equal(r1.current_anchor_bank.anchors, p0.current_anchor_bank.anchors)

