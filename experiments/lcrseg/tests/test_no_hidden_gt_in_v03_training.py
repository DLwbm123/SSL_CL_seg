import inspect

from lcrseg.methods.lcrseg_v0_2a import LCRSegV02AMethod
from lcrseg.methods.lcrseg_v0_3 import LCRSegV03Method


def test_no_hidden_gt_in_v03_training() -> None:
    assert LCRSegV03Method.training_step is LCRSegV02AMethod.training_step
    source = inspect.getsource(LCRSegV03Method) + inspect.getsource(LCRSegV02AMethod.training_step)
    assert "hidden_label" not in source
    assert "diagnostics" not in source

