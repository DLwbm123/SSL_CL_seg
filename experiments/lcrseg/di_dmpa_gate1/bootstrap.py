"""Frozen case multiplicities and maximum-cosine Hungarian stability."""
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from .binding import H, require
from .geometry_metrics import finite


def multiplicity_weights(case_ids, draws):
    counts=Counter(draws)
    # A registered case with zero pixels of this class contributes zero mass;
    # it stays in the common case draw and is never replaced or rerolled.
    sizes=Counter(case_ids)
    weights=np.asarray([counts[c]/sizes[c] for c in case_ids],dtype=np.float64)
    require(weights.sum()>0,"empty bootstrap mass")
    return weights/weights.sum()


def registered_draws(prereg, seed, stage):
    plan=next(p for p in prereg["shared_sampling"]["plans"] if p["seed"]==seed and p["stage_index"]==stage)
    require(len(plan["bootstrap"])==5,"five bootstraps required")
    for draw in plan["bootstrap"]:
        require(H(draw["case_ids_with_replacement"])==draw["case_draw_sha256"],"bootstrap draw changed")
    return plan["bootstrap"]


def matched_cosines(original, original_active, bootstrap, bootstrap_active):
    K=len(original)
    require(len(bootstrap)==K,"bootstrap silently changed K")
    scores=np.zeros(K,dtype=np.float64)
    oi=np.flatnonzero(original_active); bi=np.flatnonzero(bootstrap_active)
    if len(oi) and len(bi):
        cosine=np.clip(original[oi]@bootstrap[bi].T,-1,1)
        finite(cosine,"bootstrap cosine")
        rows,cols=linear_sum_assignment(-cosine)
        scores[oi[rows]]=cosine[rows,cols]
    return scores
