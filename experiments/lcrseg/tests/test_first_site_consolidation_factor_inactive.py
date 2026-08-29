import copy

import torch

from lcrseg.engine.trainer import TrainerState
from lcrseg.methods.lcrseg_v0_2a import LCRSegV02AMethod
from lcrseg.models import UNet2D
from tests.v0_2a_test_utils import batches, routing_config, trainer


def test_first_site_consolidation_factor_inactive() -> None:
    torch.manual_seed(17)
    r1 = LCRSegV02AMethod(
        UNet2D(3, 3),
        config=routing_config(
            variant_id="R1",
            assimilation_mode="progressive_admission",
            consolidation_mode="uniform_relation",
        ),
    )
    r3 = LCRSegV02AMethod(
        UNet2D(3, 3),
        config=routing_config(
            variant_id="R3",
            assimilation_mode="progressive_admission",
            consolidation_mode="calibrated_teacher_rejection",
        ),
    )
    r3.model.load_state_dict(copy.deepcopy(r1.model.state_dict()), strict=True)
    r1.begin_site("REFUGE", None, 4)
    r3.begin_site("REFUGE", None, 4)
    labeled, unlabeled = batches()
    trainer_r1, trainer_r3 = trainer(r1, 4), trainer(r3, 4)
    for step in range(2):
        first = trainer_r1.train_step(
            labeled,
            unlabeled,
            state=TrainerState(global_step=step, site_step=step, epoch=0),
        )
        second = trainer_r3.train_step(
            labeled,
            unlabeled,
            state=TrainerState(global_step=step, site_step=step, epoch=0),
        )
        assert torch.equal(first.total_loss, second.total_loss)
        for key, value in r1.model.state_dict().items():
            assert torch.equal(value, r3.model.state_dict()[key])
        assert torch.equal(r1.current_anchor_bank.anchors, r3.current_anchor_bank.anchors)
