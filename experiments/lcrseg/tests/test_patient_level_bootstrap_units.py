from lcrseg.analysis.v0_3 import paired_patient_bootstrap


def test_patient_level_bootstrap_units() -> None:
    first = {"patient_a": 0.8, "patient_b": 0.6, "patient_c": 0.7}
    second = {"patient_a": 0.7, "patient_b": 0.5, "patient_c": 0.6}
    result = paired_patient_bootstrap(first, second, samples=1000, seed=7)
    assert result["sampling_unit"] == "patient"
    assert result["paired"] is True
    assert result["patients"] == 3
    assert abs(result["mean_difference"] - 0.1) < 1.0e-12

