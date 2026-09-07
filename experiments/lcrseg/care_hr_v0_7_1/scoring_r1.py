"""Evaluator-only SCORING_CONTRACT_R1. Confusion rows=truth, columns=prediction."""
import numpy as np


def labels(prediction, truth):
    p, t = np.asarray(prediction), np.asarray(truth)
    if p.ndim != 2 or p.size == 0 or p.shape != t.shape:
        raise ValueError("nonempty same-shape two-dimensional prediction/GT required")
    if any(not np.issubdtype(x.dtype, np.integer) for x in (p, t)):
        raise ValueError("integer hard labels required")
    if not np.isin(p, (0, 1, 2)).all() or not np.isin(t, (0, 1, 2, 255)).all():
        raise ValueError("illegal prediction/GT class")
    return p, t


def confusion(prediction, truth):
    p, t = labels(prediction, truth)
    valid = t != 255
    return np.bincount(3 * t[valid].astype(np.int64) + p[valid], minlength=9).reshape(3, 3)


def from_confusion(matrix, image_pixels):
    m = np.asarray(matrix)
    if m.shape != (3, 3) or m.dtype != np.int64 or np.any(m < 0):
        raise ValueError("nonnegative int64 3x3 confusion required")
    n = int(m.sum())
    if not isinstance(image_pixels, (int, np.integer)) or image_pixels <= 0 or n > image_pixels:
        raise ValueError("invalid image/valid pixel accounting")
    dice, iou = [], []
    for c in range(3):
        tp, ps, ts = int(m[c, c]), int(m[:, c].sum()), int(m[c, :].sum())
        denom, union = ps + ts, ps + ts - tp
        dice.append(1.0 if denom == 0 else 2.0 * tp / denom)
        iou.append(1.0 if union == 0 else tp / union)
    union_tp = int(m[1:, 1:].sum())
    union_den = int(m[:, 1:].sum()) + int(m[1:, :].sum())
    macro = float((dice[1] + dice[2]) / 2.0)
    return {"macro_fg_dice": macro, "foreground_dice": macro,
            "background_dice": dice[0], "rim_dice": dice[1], "cup_dice": dice[2],
            "background_iou": iou[0], "rim_iou": iou[1], "cup_iou": iou[2],
            "mean_iou": float(np.mean(iou)),
            "union_fg_dice": 1.0 if union_den == 0 else 2.0 * union_tp / union_den,
            "n_valid_pixels": n, "n_ignored_pixels": int(image_pixels) - n,
            "has_evaluable_gt": n > 0, "all_pixels_ignored": n == 0,
            "metric_from_empty_support_convention": n == 0}


def case_metrics(prediction, truth):
    return from_confusion(confusion(prediction, truth), np.asarray(prediction).size)


def gains(before, after):
    if before["n_valid_pixels"] != after["n_valid_pixels"]:
        raise ValueError("before/after scoring supports differ")
    macro = after["macro_fg_dice"] - before["macro_fg_dice"]
    rim = after["rim_dice"] - before["rim_dice"]
    cup = after["cup_dice"] - before["cup_dice"]
    return {"gain_macro_fg": macro, "gain_rim": rim, "gain_cup": cup,
            "gain_union_fg": after["union_fg_dice"] - before["union_fg_dice"],
            "harm": max(0.0, -macro, -rim, -cup),
            "has_evaluable_gt": before["has_evaluable_gt"]}


def action_target(current_probability, revised_probability, truth, action_id, blend_lambda):
    from .actions import probability_pair
    current, revised = probability_pair(current_probability, revised_probability)
    if not isinstance(action_id, str) or not action_id or blend_lambda not in (None, 0.5, 0.75):
        raise ValueError("explicit action ID and registered common lambda required")
    return {"action_id": action_id, "blend_lambda": blend_lambda,
            **gains(case_metrics(current.argmax(axis=0), truth), case_metrics(revised.argmax(axis=0), truth))}
