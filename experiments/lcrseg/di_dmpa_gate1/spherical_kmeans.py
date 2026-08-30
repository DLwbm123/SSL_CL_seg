"""CPU float64 weighted spherical K-means with registered restarts/ties."""
from __future__ import annotations

import numpy as np

from .binding import S, require
from .geometry_metrics import finite, nearest, normalize


def clustering_seed(seed, stage, class_id, K, replicate, restart):
    return S(["kmeans-v1",seed,stage,class_id,K,replicate,restart])


def weighted_initialization(x, weights, K, rng):
    centers = np.zeros((K,x.shape[1]),dtype=np.float64)
    active = np.zeros(K,dtype=bool)
    mass = weights/weights.sum()
    first = int(rng.choice(len(x),p=mass))
    centers[0],active[0]=x[first],True
    for slot in range(1,K):
        _, distance = nearest(x,centers,active)
        # Exact duplicate vectors cannot seed two distinct modes.
        duplicate = np.any(np.all(x[:,None,:] == centers[None,active,:],axis=2),axis=1)
        probabilities = mass*distance
        probabilities[duplicate]=0
        if probabilities.sum() <= 0:
            break
        index=int(rng.choice(len(x),p=probabilities/probabilities.sum()))
        centers[slot],active[slot]=x[index],True
    return centers,active


def _iterate(x,weights,centers,active,uid_rank,max_iterations,tolerance):
    converged=False
    for iteration in range(1,max_iterations+1):
        assignments,_=nearest(x,centers,active)
        new=np.zeros_like(centers)
        new_active=np.zeros_like(active)
        for slot in range(len(centers)):
            selected=(assignments==slot)&(weights>0)
            if selected.any():
                new[slot]=normalize(np.sum(x[selected]*weights[selected,None],axis=0))
                new_active[slot]=True
        for slot in np.flatnonzero(~new_active):
            _,distance=nearest(x,new,new_active)
            residual=weights*distance
            order=np.lexsort((uid_rank,-residual))
            for index in order:
                if weights[index]>0 and not np.any(np.all(new[new_active]==x[index],axis=1)):
                    new[slot],new_active[slot]=x[index],True
                    break
        common=active&new_active
        movement=float(np.arccos(np.clip(np.sum(centers[common]*new[common],axis=1),-1,1)).max()) if common.any() else np.inf
        converged=bool(np.array_equal(active,new_active) and movement<=tolerance)
        centers,active=new,new_active
        if converged:
            break
    _,distance=nearest(x,centers,active)
    return centers,active,float(np.dot(weights,distance)/weights.sum()),iteration,converged


def fit(x, weights, K, *, seed, stage, class_id, replicate=-1, uid_rank=None):
    x=normalize(x)
    weights=np.asarray(weights,dtype=np.float64)
    finite(weights,"clustering weights")
    require(K in (1,2,3,5) and len(weights)==len(x) and (weights>=0).all() and weights.sum()>0,"invalid clustering input")
    uid_rank=np.arange(len(x)) if uid_rank is None else np.asarray(uid_rank)
    records=[]; best=None
    for restart in range(5):
        rng_seed=clustering_seed(seed,stage,class_id,K,replicate,restart)
        if K==1:
            centers=normalize(np.sum(x*weights[:,None],axis=0))[None]
            active=np.ones(1,dtype=bool)
            _,distance=nearest(x,centers,active)
            q=float(np.dot(weights,distance)/weights.sum()); iterations=1; converged=True
        else:
            centers,active=weighted_initialization(x,weights,K,np.random.Generator(np.random.PCG64(rng_seed)))
            centers,active,q,iterations,converged=_iterate(x,weights,centers,active,uid_rank,100,1e-6)
        finite(centers,"cluster centers")
        finite(q,"restart quantization")
        records.append(dict(restart=restart,seed=rng_seed,Q_K=q,iterations=iterations,converged=converged))
        if best is None or q<best[0]:
            best=(q,centers.copy(),active.copy(),restart)
    _,centers,active,restart=best
    return dict(centers=centers,active=active,K=K,selected_restart=restart,restarts=records,
                center_norms=np.linalg.norm(centers,axis=1).tolist(),finite=True,
                iterations=records[restart]["iterations"],converged=records[restart]["converged"])
