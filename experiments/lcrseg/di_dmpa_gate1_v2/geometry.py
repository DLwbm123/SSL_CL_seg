"""Null-aware spherical fit/metrics: every null UID retains fixed distance2."""
from collections import Counter
import numpy as np

from di_dmpa_gate1.geometry_metrics import geometry as v1_geometry, nearest, weighted_ecdf
from di_dmpa_gate1.spherical_kmeans import fit as v1_fit, clustering_seed
from di_dmpa_gate1.bootstrap import matched_cosines
from di_dmpa_gate1.binding import NumericalError
from .binding import InvalidCenter, NonfiniteFeature, require


def validate_directions(directions, active_mask):
    x=np.asarray(directions); active=np.asarray(active_mask)
    require(x.ndim==2 and x.shape[1]==16 and active.shape==(len(x),) and active.dtype==np.bool_, 'wrong direction/mask schema')
    if not np.isfinite(x).all():raise NonfiniteFeature('nonfinite cached direction')
    require(np.all(x[~active]==0),'null placeholder is not zero')
    require(np.all(np.abs(np.linalg.norm(x[active],axis=1)-1)<=1e-12),'active direction is not unit norm')
    return x.astype(np.float64,copy=False),active


def validate_centers(centers, center_active):
    p=np.asarray(centers,dtype=np.float64); mask=np.asarray(center_active,dtype=bool)
    require(p.ndim==2 and p.shape[1]==16 and mask.shape==(len(p),),'wrong center schema')
    if not np.isfinite(p).all():raise InvalidCenter('nonfinite center')
    norms=np.linalg.norm(p[mask],axis=1)
    if (norms<=1e-12).any() or (np.abs(norms-1)>1e-12).any():raise InvalidCenter('active center is zero or not unit norm')
    require(np.all(p[~mask]==0),'inactive center placeholder must be zero')
    return p,mask


def fit(directions, active_mask, weights, K, *, seed, stage, class_id, replicate=-1, uid_rank=None):
    x,active=validate_directions(directions,active_mask)
    w=np.asarray(weights,dtype=np.float64)
    require(w.shape==(len(x),) and np.isfinite(w).all() and (w>=0).all(),'invalid fit weights')
    rank=np.arange(len(x)) if uid_rank is None else np.asarray(uid_rank)
    mass=float(w[active].sum())
    if mass==0:
        result=dict(centers=np.zeros((K,16)),active=np.zeros(K,dtype=bool),K=K,selected_restart=0,
            restarts=[dict(restart=r,seed=clustering_seed(seed,stage,class_id,K,replicate,r),Q_K=None,
                Q_null_worst_case=2.,iterations=0,converged=True,directional_support='NONE') for r in range(5)],
            center_norms=[0.]*K,finite=True,iterations=0,converged=True)
    else:
        try:result=v1_fit(x[active],w[active],K,seed=seed,stage=stage,class_id=class_id,replicate=replicate,uid_rank=rank[active])
        except NumericalError as error:raise InvalidCenter(str(error)) from error
    validate_centers(result['centers'],result['active'])
    return dict(result,directional_support='PRESENT' if mass>0 else 'NONE',original_active_weight_sum=mass,
                original_registered_count=len(x),original_null_count=int((~active).sum()))


def metrics(directions, active_mask, weights, centers, center_active, uid_rank=None):
    x,active=validate_directions(directions,active_mask); p,pa=validate_centers(centers,center_active)
    w=np.asarray(weights,dtype=np.float64)
    require(w.shape==(len(x),) and np.isfinite(w).all() and (w>=0).all() and w.sum()>0,'invalid metric weights')
    mass=float(w.sum()); am=float(w[active].sum()/mass); nm=float(w[~active].sum()/mass)
    cosine=np.full(len(x),2.,dtype=np.float64); euclidean=np.full(len(x),2.,dtype=np.float64)
    conditional=None
    if am>0 and pa.any():
        _,distance=nearest(x[active],p,pa)
        cosine[active]=distance; euclidean[active]=np.sqrt(np.maximum(0,2*distance))
        conditional=v1_geometry(x[active],w[active],p,pa,None if uid_rank is None else np.asarray(uid_rank)[active])
    # All null/no-direction observations retain distance2 in the full ECDF.
    wc_q=float(np.dot(w,cosine)/mass)
    result=dict(metric_schema='NULL_AWARE_SPHERE_V2',admission_radius_field='R95_null_worst_case',
        registered_count=len(x),active_count=int(active.sum()),null_count=int((~active).sum()),
        full_uid_count_used=len(x),null_rows_retained=int((~active).sum()),null_mass=nm,active_direction_mass=am,
        directional_support='PRESENT' if conditional is not None else 'NONE',
        Q_directional_conditional=None if conditional is None else conditional['Q_K'],
        cosine_distance_p95_directional=None if conditional is None else conditional['cosine_distance_p95'],
        R95_directional=None if conditional is None else conditional['R95'],
        Q_null_worst_case=wc_q,cosine_distance_p95_null_worst_case=weighted_ecdf(cosine,w,.95,uid_rank),
        R95_null_worst_case=weighted_ecdf(euclidean,w,.95,uid_rank),
        occupancy=[0.]*len(p) if conditional is None else conditional['occupancy'],
        active_assignment_slots=[False]*len(p) if conditional is None else conditional['active_assignment_slots'],
        inactive_count=int((~pa).sum()),minimum_inter_prototype_euclidean=None,
        minimum_inter_prototype_angular=None)
    if conditional is not None:
        for name in ('minimum_inter_prototype_euclidean','minimum_inter_prototype_angular'):result[name]=conditional[name]
    return result


def bootstrap_weights(case_ids, draws):
    counts=Counter(draws); sizes=Counter(case_ids)
    weights=np.asarray([counts[c]/sizes[c] for c in case_ids],dtype=np.float64)
    return weights/weights.sum() if weights.sum()>0 else weights


def verify_null_identity(reference, candidate):
    require(reference['registered_count']==candidate['registered_count'] and reference['null_count']==candidate['null_count'],
            'K-dependent UID/null count')
    require(reference['null_mass']==candidate['null_mass'],'K-dependent null mass')
    if reference['Q_directional_conditional'] is None or candidate['Q_directional_conditional'] is None:
        require(reference['Q_null_worst_case']==candidate['Q_null_worst_case']==2,'no-support K-dependent gain')
        return dict(status='PASS_NO_DIRECTIONAL_SUPPORT_CONSTANT_TWO',absolute_error=0.)
    left=reference['Q_null_worst_case']-candidate['Q_null_worst_case']
    right=reference['active_direction_mass']*(reference['Q_directional_conditional']-candidate['Q_directional_conditional'])
    require(np.isclose(left,right,atol=1e-12,rtol=1e-10),'null-term K-independence identity failed')
    return dict(status='PASS',absolute_error=float(abs(left-right)))
