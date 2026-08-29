import inspect

from lcrseg.methods.components.progressive_admission import classwise_progressive_admission


def test_progressive_admission_no_hidden_gt() -> None:
    parameters = set(inspect.signature(classwise_progressive_admission).parameters)
    assert not {"gt", "label", "hidden_gt", "hidden_label"}.intersection(parameters)
