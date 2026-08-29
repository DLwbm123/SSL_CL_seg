import numpy as np

from lcrseg.analysis.v0_4 import spherical_kmeans


def test_v04_spherical_kmeans_is_deterministic_and_normalized() -> None:
    rng = np.random.default_rng(7)
    first = rng.normal(loc=(1.0, 0.0, 0.0), scale=0.05, size=(32, 3))
    second = rng.normal(loc=(0.0, 1.0, 0.0), scale=0.05, size=(32, 3))
    features = np.concatenate((first, second), axis=0)

    labels_a, centers_a, objective_a = spherical_kmeans(features, k=2, seed=19, restarts=5)
    labels_b, centers_b, objective_b = spherical_kmeans(features, k=2, seed=19, restarts=5)

    np.testing.assert_array_equal(labels_a, labels_b)
    np.testing.assert_allclose(centers_a, centers_b, atol=0.0, rtol=0.0)
    assert objective_a == objective_b
    np.testing.assert_allclose(np.linalg.norm(centers_a, axis=1), 1.0, atol=1.0e-12)
    assert set(np.bincount(labels_a).tolist()) == {32}

