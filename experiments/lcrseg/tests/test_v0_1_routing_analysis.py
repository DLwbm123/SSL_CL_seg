from __future__ import annotations

import numpy as np

from lcrseg.analysis.v0_1_routing import _binary_boundary, _component_areas, _small_component_grid, effective_sample_size


def test_effective_sample_size_matches_uniform_and_sparse_limits() -> None:
    assert np.isclose(effective_sample_size(np.asarray([1.0, 1.0, 1.0])), 3.0)
    assert np.isclose(effective_sample_size(np.asarray([1.0, 0.0, 0.0])), 1.0)
    assert effective_sample_size(np.asarray([], dtype=np.float64)) == 0.0


def test_spatial_helpers_use_predicted_components_and_two_sided_boundaries() -> None:
    label = np.asarray([[0, 0, 1], [0, 0, 1], [2, 2, 1]], dtype=np.int64)
    boundary = _binary_boundary(label)
    assert boundary.dtype == bool
    assert bool(boundary[0, 1])
    assert bool(boundary[1, 0])
    prediction = np.asarray([[0, 0, 1], [0, 2, 1], [0, 0, 1]], dtype=np.int64)
    areas = _component_areas(prediction, 3)
    assert areas[2].tolist() == [1]
    small = _small_component_grid(prediction, {0: 0.0, 1: 0.0, 2: 2.0}, prediction.shape)
    assert bool(small[1, 1])
    assert int(small.sum()) == 1
