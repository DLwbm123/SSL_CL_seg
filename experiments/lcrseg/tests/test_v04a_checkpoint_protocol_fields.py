from lcrseg.methods.lcrseg_v0_4a import FROZEN_RANK_SCHEDULE, FROZEN_SOFT_ALLOCATION, LCRSegV04AMethod
from lcrseg.models import UNet2D


def test_v04a_checkpoint_protocol_fields() -> None:
    method = LCRSegV04AMethod(UNet2D(3, 3))
    semantics = method.method_state_dict()["method_statistics"]["protocol_semantics"]
    assert semantics["protocol_id"] == "lcrseg_v0_4a"
    assert semantics["assimilation_mode"] == "soft_reliability_allocation"
    assert semantics["consolidation_mode"] == "uniform_relation"
    assert semantics["rank_schedule"] == FROZEN_RANK_SCHEDULE
    assert semantics["soft_allocation"] == FROZEN_SOFT_ALLOCATION
    assert semantics["single_anchor"] is True
    assert semantics["multi_agent"] is False and semantics["ric"] is False
    assert semantics["teacher_rejection"] is False
