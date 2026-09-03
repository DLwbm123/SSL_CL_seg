"""Deterministic proposal construction from current and historical hard masks."""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

from .contracts import Proposal


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    for row, col in zip(*np.nonzero(mask)):
        if seen[row, col]:
            continue
        todo = deque([(int(row), int(col))])
        seen[row, col] = True
        pixels = []
        while todo:
            r, c = todo.popleft()
            pixels.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    todo.append((rr, cc))
        yield pixels


def _sort_key(row):
    return (-row["area"], row["centroid_row"], row["centroid_col"],
            row["target_class"], 0 if row["direction"] == "add" else 1)


def generate_proposals(current_hard, historical_hard, minimum_pixels=8, maximum_proposals=12):
    current = np.asarray(current_hard)
    historical = np.asarray(historical_hard)
    if current.ndim != 2 or current.shape != historical.shape:
        raise ValueError("hard masks must be same-shape 2D arrays")
    rows = []
    for target_class in (1, 2):
        masks = (
            ("add", (historical == target_class) & (current != target_class)),
            ("remove", (current == target_class) & (historical != target_class)),
        )
        for direction, candidate_mask in masks:
            for pixels in _components(candidate_mask):
                if len(pixels) < minimum_pixels:
                    continue
                mask = np.zeros(current.shape, dtype=bool)
                rr, cc = zip(*pixels)
                mask[rr, cc] = True
                rows.append({
                    "target_class": target_class,
                    "direction": direction,
                    "area": len(pixels),
                    "centroid_row": float(np.mean(rr)),
                    "centroid_col": float(np.mean(cc)),
                    "mask": mask,
                })
    rows.sort(key=_sort_key)
    claimed = np.zeros(current.shape, dtype=bool)
    output = []
    for row in rows:
        mask = row["mask"] & ~claimed
        if int(mask.sum()) < minimum_pixels:
            continue
        claimed |= mask
        digest = hashlib.sha256(
            bytes([row["target_class"], row["direction"] == "remove"]) + mask.tobytes()
        ).hexdigest()[:16]
        rr, cc = np.nonzero(mask)
        output.append(Proposal(
            proposal_id=f"p{len(output):02d}_{digest}",
            target_class=row["target_class"],
            direction=row["direction"],
            area=int(mask.sum()),
            centroid_row=float(np.mean(rr)),
            centroid_col=float(np.mean(cc)),
            mask=mask,
        ))
        if len(output) == maximum_proposals:
            break
    return tuple(output)
