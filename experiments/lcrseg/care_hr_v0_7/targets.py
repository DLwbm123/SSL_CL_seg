"""Training/evaluator-only target and guarded oracle helpers."""
from __future__ import annotations

import numpy as np

from .contracts import require_external_review_authorization


def _dice(hard, truth, selected):
    left = np.isin(hard, selected)
    right = np.isin(truth, selected)
    denominator = int(left.sum() + right.sum())
    return 1.0 if denominator == 0 else 2.0 * int(np.count_nonzero(left & right)) / denominator


def proposal_targets(current_probability, revised_probability, ground_truth):
    current = np.argmax(np.asarray(current_probability), axis=0)
    revised = np.argmax(np.asarray(revised_probability), axis=0)
    truth = np.asarray(ground_truth)
    if current.shape != revised.shape or current.shape != truth.shape:
        raise ValueError("target shapes differ")
    gain_fg = _dice(revised, truth, (1, 2)) - _dice(current, truth, (1, 2))
    gain_rim = _dice(revised, truth, (1,)) - _dice(current, truth, (1,))
    gain_cup = _dice(revised, truth, (2,)) - _dice(current, truth, (2,))
    return {
        "gain_fg": gain_fg,
        "gain_rim": gain_rim,
        "gain_cup": gain_cup,
        "harm": max(0.0, -gain_fg, -gain_rim, -gain_cup),
    }


def c9_regional_oracle(current_probability, historical_probability, regions,
                       ground_truth, evaluator_authorized=False):
    require_external_review_authorization(evaluator_authorized)
