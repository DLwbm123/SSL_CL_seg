"""Prediction-only action materialization. No GT, valid-mask or domain API."""
import numpy as np

from .actions import (LAMBDAS, apply_action, enumerate_actions, probability_pair,
                      proposals_for_route, validate_proposals)


def prepare_case(current_probability, historical_probability, route, proposals=None):
    current, history = probability_pair(current_probability, historical_probability)
    if isinstance(route, bool) or not isinstance(route, (int, np.integer)) or route not in (0, 1, 2):
        raise ValueError("invalid frozen route")
    if proposals is None:
        proposals = proposals_for_route(current, history, route)
    elif route == 2 and len(proposals):
        raise ValueError("current route permits no proposals")
    proposals = validate_proposals(proposals, current.shape[1:])
    hard = current.argmax(axis=0).astype(np.uint8)
    cap = set(enumerate_actions(proposals, hard, "O_CAP"))
    no_area = set(enumerate_actions(proposals, hard, "O_NO_AREA"))
    free = list(enumerate_actions(proposals, hard, "O_FREE_SUBSET"))
    masks = np.asarray([p.mask for p in proposals], dtype=bool).reshape(len(proposals), *hard.shape)
    union = masks.any(axis=0) if len(proposals) else np.zeros(hard.shape, dtype=bool)
    blended, probability_changed = [], []
    word = np.uint32 if current.dtype == np.float32 else np.uint64
    for coefficient in LAMBDAS:
        out = apply_action(current, history, proposals, (tuple(range(len(proposals))), coefficient)) if proposals else current.copy()
        if out[:, ~union].tobytes() != current[:, ~union].tobytes():
            raise AssertionError("outside-region probability bytes changed")
        blended.append(out.argmax(axis=0).astype(np.uint8))
        probability_changed.append(np.any(out.view(word) != current.view(word), axis=0))
    actions = []
    for index, (indices, coefficient) in enumerate(free):
        counts = [sum(proposals[i].target_class == c for i in indices) for c in (1, 2)]
        area = sum(proposals[i].area for i in indices)
        actions.append({"action_id": f"a{index:04d}", "indices": list(indices),
                        "proposal_ids": [proposals[i].proposal_id for i in indices],
                        "blend_lambda": coefficient, "mask_union_pixels": area,
                        "O_CAP": (indices, coefficient) in cap, "O_NO_AREA": (indices, coefficient) in no_area,
                        "quantity_rejected": len(indices) > 4 or max(counts) > 3,
                        "foreground_area_rejected": 100 * area > 15 * int(np.count_nonzero(hard)),
                        "image_area_rejected": 100 * area > 2 * hard.size})
    return {"current_hard": hard, "blended_hard": np.asarray(blended), "masks": masks,
            "probability_changed": np.asarray(probability_changed), "actions": actions,
            "proposals": [{"proposal_id": p.proposal_id, "target_class": p.target_class,
                            "direction": p.direction, "area": p.area,
                            "centroid_row": p.centroid_row, "centroid_col": p.centroid_col} for p in proposals],
            "route": int(route), "current_foreground_pixels": int(np.count_nonzero(hard)),
            "image_pixels": int(hard.size), "source_probability_dtype": str(current.dtype)}
