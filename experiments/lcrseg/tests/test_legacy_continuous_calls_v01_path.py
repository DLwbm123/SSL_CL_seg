from lcrseg.methods.components.routing import assimilation_loss as frozen_v01_assimilation_loss
from lcrseg.methods.lcrseg_v0_2a import assimilation_loss as amended_assimilation_loss


def test_legacy_continuous_calls_v01_path() -> None:
    assert amended_assimilation_loss is frozen_v01_assimilation_loss
