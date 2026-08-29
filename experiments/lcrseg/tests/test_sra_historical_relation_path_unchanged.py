import copy

from lcrseg.methods.lcrseg_v0_2a import LCRSegV02AMethod
from lcrseg.methods.lcrseg_v0_4a import LCRSegV04AMethod
from lcrseg.models import UNet2D
from tests.v0_2a_test_utils import batches, previous_checkpoint, routing_config


def test_sra_historical_relation_path_unchanged(tmp_path) -> None:
    first_model = UNet2D(3, 3)
    second_model = copy.deepcopy(first_model)
    r1 = LCRSegV02AMethod(first_model, config=routing_config(variant_id="R1", assimilation_mode="progressive_admission", consolidation_mode="uniform_relation"))
    sra = LCRSegV04AMethod(second_model, config=routing_config(protocol_id="lcrseg_v0_4a"))
    parent = previous_checkpoint(tmp_path)
    r1.begin_site("B", parent, 4)
    sra.begin_site("B", parent, 4)
    labeled, unlabeled = batches()
    r1_result = r1.training_step(labeled, unlabeled, global_step=1, site_step=0)
    sra_result = sra.training_step(labeled, unlabeled, global_step=1, site_step=0)
    assert float(r1_result.losses["loss_relation"]) == float(sra_result.losses["loss_relation"])
    assert sra_result.scalars["sra_historical_relation_exact_path"] == "frozen_uniform_relation_v0_2a"
