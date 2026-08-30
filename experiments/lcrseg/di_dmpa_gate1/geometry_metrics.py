"""Registered equal-case spherical geometry, not library-default quantiles."""
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from .binding import NumericalError, require


def finite(x, name):
    if not np.isfinite(x).all():
        raise NumericalError(f"nonfinite {name}")


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    finite(x, "features/centers")
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if (norms <= 1e-12).any():
        raise NumericalError("feature/center norm <= 1e-12")
    return x / norms


def weighted_ecdf(values, weights, q, uid_rank=None):
    values, weights = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    require(values.ndim == weights.ndim == 1 and len(values) == len(weights) > 0, "empty/mismatched ECDF")
    finite(values, "ECDF values"); finite(weights, "ECDF weights")
    require((weights >= 0).all() and weights.sum() > 0 and 0 <= q <= 1, "invalid ECDF weights/quantile")
    rank = np.arange(len(values)) if uid_rank is None else uid_rank
    order = np.lexsort((rank, values))
    index = min(int(np.searchsorted(np.cumsum(weights[order]) / weights.sum(), q, side="left")), len(values)-1)
    return float(values[order[index]])


def boundary_band(labels, class_id):
    mask = np.asarray(labels) == class_id
    structure = np.ones((7, 7), dtype=bool)
    band = binary_dilation(mask, structure=structure, border_value=0) & ~binary_erosion(mask, structure=structure, border_value=0)
    return band & mask & (np.asarray(labels) != 255)


def nearest(x, centers, active):
    require(np.any(active), "all prototype slots inactive")
    similarity = np.asarray(x, dtype=np.float64) @ centers.T
    finite(similarity, "cosine matrix")
    similarity[:, ~np.asarray(active)] = -np.inf
    assignments = np.argmax(similarity, axis=1)
    best = np.clip(similarity[np.arange(len(x)), assignments], -1, 1)
    return assignments, 1 - best


def geometry(x, weights, centers, active, uid_rank=None):
    assignments, cosine = nearest(x, centers, active)
    distances = np.sqrt(np.maximum(0, 2*cosine))
    weights = np.asarray(weights, dtype=np.float64)
    mass = weights.sum()
    require(mass > 0, "empty class mass")
    occupancy = np.bincount(assignments, weights=weights, minlength=len(centers))/mass
    selected = centers[active]
    separation = angular = None
    if len(selected) > 1:
        cos = np.clip(selected @ selected.T, -1, 1)
        pairs = cos[np.triu_indices(len(selected), 1)]
        separation = float(np.sqrt(np.maximum(0, 2-2*pairs)).min())
        angular = float(np.arccos(pairs).min())
    return dict(Q_K=float(np.dot(weights, cosine)/mass), cosine_distance_p95=weighted_ecdf(cosine,weights,.95,uid_rank),
                R95=weighted_ecdf(distances,weights,.95,uid_rank), occupancy=occupancy.tolist(),
                active_assignment_slots=(occupancy>0).tolist(), inactive_count=int((~active).sum()),
                minimum_inter_prototype_euclidean=separation, minimum_inter_prototype_angular=angular)
