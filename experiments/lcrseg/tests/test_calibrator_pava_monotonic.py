from __future__ import annotations

import numpy as np

from lcrseg.methods.components.compatibility_calibrator import fit_pava_mapping


def test_pava_mapping_is_monotonic_after_laplace_smoothing() -> None:
    score = np.linspace(0.0, 1.0, 1000)
    # Intentionally non-monotonic empirical accuracy.
    correct = np.array([(index % 5) in (0, 1, 4) for index in range(score.size)])
    mapping = fit_pava_mapping(score, correct, bins=10, scope="global", class_id=None)
    assert all(second + 1.0e-12 >= first for first, second in zip(mapping.probabilities, mapping.probabilities[1:]))
    assert all(0.0 <= value <= 1.0 for value in mapping.probabilities)
