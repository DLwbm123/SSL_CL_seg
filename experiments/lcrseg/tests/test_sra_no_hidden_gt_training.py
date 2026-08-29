import inspect

from lcrseg.contracts import UnlabeledBatch
from lcrseg.methods.lcrseg_v0_4a import LCRSegV04AMethod


def test_sra_no_hidden_gt_training() -> None:
    assert "label" not in UnlabeledBatch.__dataclass_fields__
    source = inspect.getsource(LCRSegV04AMethod.training_step)
    assert "hidden_gt" not in source
    assert LCRSegV04AMethod._new_v04a_statistics()["hidden_gt_training_usage"] == 0
