import numpy as np

from lcrseg.analysis.v0_4 import signed_distance_and_component_size


def test_v04_boundary_distance_and_component_size_are_processed_pixel_units() -> None:
    label = np.zeros((9, 9), dtype=np.int64)
    label[2:7, 2:7] = 1
    distance, component = signed_distance_and_component_size(label, 1)

    assert distance.shape == label.shape
    assert distance.dtype == np.float32
    assert distance[4, 4] == 3.0
    assert distance[2, 2] == 1.0
    assert distance[1, 2] == -1.0
    assert distance[0, 0] < -2.0
    assert component[4, 4] == 25
    assert component[0, 0] == 0
    np.testing.assert_array_equal(np.abs(distance) <= 3.0, np.abs(distance) <= 3)


def test_v04_missing_class_has_zero_components_and_negative_distance() -> None:
    label = np.zeros((5, 5), dtype=np.int64)
    distance, component = signed_distance_and_component_size(label, 2)

    assert np.all(distance < 0)
    assert np.all(component == 0)

