"""GT-free frozen action enumeration; no risk fitting or learned acceptance."""
from itertools import combinations

import numpy as np

from care_hr_v0_7.policy import apply_anchored_revision
from care_hr_v0_7.proposals import generate_proposals


LAMBDAS = (0.5, 0.75)
SPACES = ("O_CAP", "O_NO_AREA", "O_FREE_SUBSET")


def probability_pair(current, historical):
    arrays = tuple(np.asarray(x) for x in (current, historical))
    for value in arrays:
        if value.ndim != 3 or value.shape[0] != 3 or min(value.shape) < 1 or value.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError("expected nonempty 3xHxW float32/float64 probabilities")
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError("probability outside finite [0,1]")
        if not np.allclose(value.sum(axis=0), 1, atol=1e-7):
            raise ValueError("invalid probability sum")
    if arrays[0].shape != arrays[1].shape or arrays[0].dtype != arrays[1].dtype:
        raise ValueError("probability shapes/dtypes differ")
    return arrays


def validate_proposals(proposals, shape):
    proposals = tuple(proposals)
    if len(proposals) > 12 or len({p.proposal_id for p in proposals}) != len(proposals):
        raise ValueError("duplicate IDs or more than 12 proposals")
    union = np.zeros(shape, dtype=bool)
    for p in proposals:
        mask = np.asarray(p.mask)
        if mask.shape != tuple(shape) or mask.dtype != bool:
            raise ValueError("proposal masks must be same-shape boolean arrays")
        if p.target_class not in (1, 2) or p.direction not in ("add", "remove"):
            raise ValueError("invalid proposal class/direction")
        if isinstance(p.area, bool) or not isinstance(p.area, (int, np.integer)) or p.area != int(mask.sum()) or p.area < 8:
            raise ValueError("proposal area mismatch or below minimum")
        if np.any(union & mask):
            raise ValueError("overlapping or duplicate proposal masks")
        rr, cc = np.nonzero(mask)
        if p.centroid_row != float(rr.mean()) or p.centroid_col != float(cc.mean()):
            raise ValueError("proposal centroid mismatch")
        union |= mask
    return proposals


def proposals_for_route(current_probability, historical_probability, route):
    current, historical = probability_pair(current_probability, historical_probability)
    if isinstance(route, bool) or not isinstance(route, (int, np.integer)) or route not in (0, 1, 2):
        raise ValueError("route must be frozen expert index 0,1,2")
    if route == 2:
        return ()
    proposals = generate_proposals(current.argmax(axis=0), historical.argmax(axis=0))
    return validate_proposals(proposals, current.shape[1:])


def enumerate_actions(proposals, current_hard, space="O_CAP", blend_lambda=None):
    """Yield (indices, common lambda); the unique no-op is ((), None)."""
    current = np.asarray(current_hard)
    if current.ndim != 2 or current.size == 0 or not np.issubdtype(current.dtype, np.integer) or not np.isin(current, (0, 1, 2)).all():
        raise ValueError("current hard mask must contain only classes 0,1,2")
    proposals = validate_proposals(proposals, current.shape)
    if space not in SPACES or (blend_lambda is not None and blend_lambda not in LAMBDAS):
        raise ValueError("unregistered action space or lambda")
    lambdas = LAMBDAS if blend_lambda is None else (blend_lambda,)
    foreground = int(np.count_nonzero(current))
    yield (), None
    if foreground == 0:
        return
    maximum = len(proposals) if space == "O_FREE_SUBSET" else min(4, len(proposals))
    for count in range(1, maximum + 1):
        for indices in combinations(range(len(proposals)), count):
            chosen = [proposals[i] for i in indices]
            if space != "O_FREE_SUBSET" and any(sum(p.target_class == c for p in chosen) > 3 for c in (1, 2)):
                continue
            area = sum(p.area for p in chosen)  # validated non-overlap: exact union area
            if space == "O_CAP" and (100 * area > 15 * foreground or 100 * area > 2 * current.size):
                continue
            for coefficient in lambdas:
                yield indices, coefficient


def apply_action(current, historical, proposals, action):
    current, historical = probability_pair(current, historical)
    proposals = validate_proposals(proposals, current.shape[1:])
    indices, coefficient = action
    if not isinstance(indices, tuple) or any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
        raise ValueError("action indices must be integer tuples")
    if tuple(sorted(set(indices))) != indices or any(i < 0 or i >= len(proposals) for i in indices):
        raise ValueError("invalid or duplicate action indices")
    if not indices:
        if coefficient is not None:
            raise ValueError("no-op must have null lambda")
        return current.copy()
    if coefficient not in LAMBDAS:
        raise ValueError("unregistered common lambda")
    return apply_anchored_revision(current, historical, [proposals[i] for i in indices], coefficient)


def validate_vote_counts(accepted_votes, finite_predictions, proposal_count):
    votes, finite = (np.asarray(x) for x in (accepted_votes, finite_predictions))
    if votes.shape != (proposal_count,) or finite.shape != votes.shape:
        raise ValueError("vote lengths differ from proposal count")
    if any(not np.issubdtype(x.dtype, np.integer) for x in (votes, finite)):
        raise ValueError("counts must be integers")
    if np.any(votes < 0) or np.any(votes > finite) or np.any(finite > 200) or np.any(finite < 0):
        raise ValueError("require 0 <= votes <= finite predictions <= 200")
    return votes, finite
