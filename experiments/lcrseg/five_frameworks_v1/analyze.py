"""Paired development-patient inference; no invented outcomes or efficacy gate."""
import numpy as np


def sequence_metrics(matrix):
    r=np.asarray(matrix,float)
    if r.shape!=(3,3):raise ValueError('expected R_stage,domain with 3 stages/domains')
    old=(r[2,0]+r[2,1])/2
    forget=((r[0,0]-r[2,0])+(r[1,1]-r[2,1]))/2
    return {'Final':float(r[2].mean()),'Old':float(old),'Incoming':float(r[2,2]),
            'N_incremental':float((r[1,1]+r[2,2])/2),'Forget':float(forget),'BWT':float(-forget)}


def select_candidate(rows):
    if not rows:raise ValueError('unresolved selection: no sealed trajectory scores')
    if any(not r.get('sealed_all_orders') for r in rows):raise ValueError('selection requires complete sealed trajectories')
    top=max(r['Final'] for r in rows)
    tied=[r for r in rows if top-r['Final']<=1e-4]
    # Costs can be used only with the same declared measurement scope/unit.
    units={r.get('compute_unit') for r in tied}
    comparable=len(units)==1 and None not in units
    return sorted(tied,key=lambda r:(-r['Old'],r['compute'] if comparable else 0,r['candidate_id']))[0]


def paired_patient_bootstrap(differences,repetitions=2000,seed=161):
    """domain -> array [seed, order, stage_or_metric, SAME ordered patients].

    Caller aligns patient identity once per physical domain. One resampling per
    domain/repetition is shared by every seed/order/stage. This function receives
    paired differences so method labels never get independent patient weights.
    """
    rng=np.random.default_rng(seed);samples=[]
    values={d:np.asarray(v,float) for d,v in differences.items()}
    if not values or any(v.ndim!=4 or v.shape[-1]==0 for v in values.values()):raise ValueError('empty or malformed paired domains')
    for _ in range(repetitions):
        domains=[]
        for v in values.values():
            indices=rng.integers(v.shape[-1],size=v.shape[-1]);domains.append(v[...,indices].mean(-1))
        samples.append(np.stack(domains).mean(0))
    a=np.asarray(samples)
    return {'estimate':np.stack([v.mean(-1) for v in values.values()]).mean(0),
            'patient_CI':np.quantile(a,[.025,.975],axis=0),'replicates':a,
            'equal_seed_order_estimate':np.stack([v.mean(-1) for v in values.values()]).mean((0,1,2)),
            'equal_seed_order_patient_CI':np.quantile(a.mean((1,2)),[.025,.975],axis=0),
            'training_seed_SD':np.stack([v.mean(-1) for v in values.values()]).mean((0,2)).std(0,ddof=1) if next(iter(values.values())).shape[0]>1 else None,
            'population':'development patients; training seeds and orders are not new patients'}
