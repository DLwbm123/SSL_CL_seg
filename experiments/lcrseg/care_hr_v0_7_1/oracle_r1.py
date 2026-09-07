"""Evaluator-only exhaustive combination oracle, never a deployable router."""
import numpy as np

from .actions import LAMBDAS
from .scoring_r1 import confusion, from_confusion, gains, labels


def tie_key(row):
    return (-row["macro_fg_dice"], row["mask_union_pixels"], len(row["indices"]),
            0.0 if row["blend_lambda"] is None else row["blend_lambda"], tuple(row["proposal_ids"]))


def score_actions(prepared, truth):
    current, truth = labels(prepared["current_hard"], truth)
    masks, blended = prepared["masks"], prepared["blended_hard"]
    base = confusion(current, truth)
    before = from_confusion(base, current.size)
    increments = np.zeros((2, len(masks), 3, 3), dtype=np.int64)
    hard_changes = np.zeros((2, len(masks)), dtype=np.int64)
    prob_changes = np.zeros_like(hard_changes)
    valid_changes = np.zeros_like(hard_changes)
    for li in range(2):
        for i, mask in enumerate(masks):
            valid = mask & (truth != 255)
            old = np.bincount(3 * truth[valid].astype(np.int64) + current[valid], minlength=9).reshape(3, 3)
            new = np.bincount(3 * truth[valid].astype(np.int64) + blended[li][valid], minlength=9).reshape(3, 3)
            increments[li, i] = new - old
            changed = mask & (current != blended[li])
            hard_changes[li, i] = np.count_nonzero(changed)
            valid_changes[li, i] = np.count_nonzero(changed & (truth != 255))
            prob_changes[li, i] = np.count_nonzero(mask & prepared["probability_changed"][li])
    rows = []
    for action in prepared["actions"]:
        indices = action["indices"]
        li = 0 if action["blend_lambda"] is None else LAMBDAS.index(action["blend_lambda"])
        matrix = base + increments[li, indices].sum(axis=0, dtype=np.int64) if indices else base
        score = from_confusion(matrix, current.size)
        delta = gains(before, score)
        rows.append({**action, **score, **delta,
                     "hard_label_changed_pixels": int(hard_changes[li, indices].sum()) if indices else 0,
                     "probability_changed_pixels": int(prob_changes[li, indices].sum()) if indices else 0,
                     "GT_valid_hard_changed_pixels": int(valid_changes[li, indices].sum()) if indices else 0,
                     "O_SAFE0": action["O_CAP"] and delta["gain_rim"] >= 0 and delta["gain_cup"] >= 0})
    return rows


def select_oracles(rows):
    selected = {}
    for space in ("O_CAP", "O_SAFE0", "O_NO_AREA", "O_FREE_SUBSET"):
        eligible = [r for r in rows if space == "O_FREE_SUBSET" or r[space]]
        if not eligible or not any(not r["indices"] for r in eligible):
            raise ValueError("oracle space must contain unique no-op")
        for coefficient, label in ((0.5, "lambda050"), (0.75, "lambda075"), (None, "envelope")):
            candidates = eligible if coefficient is None else [r for r in eligible if r["blend_lambda"] in (None, coefficient)]
            selected[space + "_" + label] = min(candidates, key=tie_key)
    return selected


def pareto_actions(rows):
    """Exact nondominated (macro gain, harm); all equal points are retained."""
    ordered = sorted(rows, key=lambda r: (-r["gain_macro_fg"], r["harm"], r["action_id"]))
    output, best_harm, best_gain = [], float("inf"), None
    for row in ordered:
        g, h = row["gain_macro_fg"], row["harm"]
        if h < best_harm or (h == best_harm and g == best_gain):
            output.append(row)
            best_harm, best_gain = h, g
    return output
