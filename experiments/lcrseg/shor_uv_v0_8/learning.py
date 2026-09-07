"""Patient grouped nested utility fitting. Only explicit split arrays reach fit."""
import hashlib
import numpy as np
from .features import predict, choose

LAMBDAS = (0.1, 1.0, 10.0, 100.0)
SALT = 'shor-uv-v0.8-inner'


def patient_weights(patients):
    _, inv, count = np.unique(patients, return_inverse=True, return_counts=True)
    return 1.0/count[inv]


def inner_folds(patients):
    unique = sorted(set(patients), key=lambda p: (hashlib.sha256((SALT+'|'+p).encode()).hexdigest(), p))
    mapping = {p:i % 4 for i,p in enumerate(unique)}
    return np.array([mapping[p] for p in patients], dtype=int)


def fit(x, y, weights, regularization):
    x, y, w = (np.asarray(a, dtype=np.float64) for a in (x,y,weights))
    if (x.ndim != 2 or x.shape[1] != 18 or y.shape != (len(x),3) or w.shape != (len(x),)
            or not len(x) or not all(np.isfinite(a).all() for a in (x,y,w))
            or (w <= 0).any() or regularization not in LAMBDAS):
        raise ValueError('invalid fit split')
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    constant = np.all(x == x[0], axis=0)
    mean[constant] = x[0,constant]
    scale[(scale == 0) | constant] = 1
    z = (x-mean)/scale
    intercept = np.average(y, axis=0, weights=w)
    coef = np.linalg.solve(np.einsum('ni,n,nj->ij', z, w, z, optimize=False) + regularization*np.eye(18),
                           np.einsum('ni,n,nj->ij', z, w, y-intercept, optimize=False))
    if not np.isfinite(coef).all(): raise ValueError('nonfinite Ridge solution')
    return dict(mean=mean.tolist(), scale=scale.tolist(), coef=coef.tolist(),
                intercept=intercept.tolist(), regularization=regularization)


def candidate_grid(kind):
    if kind == 'confidence':
        return [dict(id='confidence_%d'%i, kind=kind, threshold=t, epsilon=0, harm_limit=None)
                for i,t in enumerate((None, -0.1, 0.0, 0.1))]
    if kind == 'gain':
        return [dict(id='gain_%d'%i, kind=kind, epsilon=e, harm_limit=None)
                for i,e in enumerate((0.,.005))]
    return [dict(id='cf_%d'%i, kind='cf', epsilon=e, harm_limit=h)
            for i,(e,h) in enumerate((e,h) for e in (0.,.005) for h in (.01,.025,.05,.10))]


def selection_metrics(delta, seeds, domains, evaluable):
    """Inner evaluator alone gets domains; no domain enters the fit API."""
    groups = {(s,d): np.flatnonzero((seeds==s)&(domains==d)) for s in range(3) for d in range(3)}
    if any(not len(ix) or not evaluable[ix].any() for ix in groups.values()):
        return dict(safe=False, support=False)
    v = {k:delta[ix].mean(0) for k,ix in groups.items()}
    macro = {k:float(x[:2].mean()) for k,x in v.items()}
    current = np.mean([v[s,2][:2] for s in range(3)], axis=0)
    result = dict(support=True, overall_gain=float(np.mean(list(macro.values()))),
        historical_gain=float(np.mean([macro[s,d] for s in range(3) for d in (0,1)])),
        current_drop=max(0.,-float(current.mean())),
        current_class_drop=max(0.,-float(current.min())),
        maximum_seed_domain_drop=max(0.,-min(macro.values())),
        current_large_harm_rows=int(np.sum(delta[domains==2,:2].mean(1)<-.10)))
    result['safe'] = (result['current_drop']<=.01 and result['current_class_drop']<=.015
        and result['maximum_seed_domain_drop']<=.02 and result['current_large_harm_rows']==0)
    return result


def select_threshold(pred, x, targets, routes, seeds, domains, evaluable, kind):
    candidates=[]
    for c in candidate_grid(kind):
        delta = np.zeros((len(routes),3))
        accepted=[]
        for i,h in enumerate(routes):
            ok=False
            if h<2:
                if kind=='confidence': ok=c['threshold'] is None or x[i,h,9]-x[i,h,8]>=c['threshold']
                else:
                    p=pred[i,h]
                    ok=float(p[:2].mean())>c['epsilon'] and (kind=='gain' or p[2]<=c['harm_limit'])
                if ok: delta[i]=targets[i,h]
            accepted.append(bool(ok))
        m=selection_metrics(delta,seeds,domains,evaluable)
        candidates.append(dict(candidate=c,metrics=m,accepted=int(sum(accepted))))
    valid=[r for r in candidates if r['metrics']['safe']]
    def rank(r):
        m,c=r['metrics'],r['candidate']
        return (-m['historical_gain'],-m['overall_gain'],m['current_drop'],
                c['harm_limit'] if c['harm_limit'] is not None else float('inf'),
                -c['epsilon'],c['id'])
    chosen=min(valid,key=rank)['candidate'] if valid else None
    return chosen,candidates


def train_nested(x, targets, patients, routes, seeds, domains, evaluable, routed_only=False, ledger_sink=None):
    x,targets=np.asarray(x),np.asarray(targets)
    patients=np.asarray(patients); routes=np.asarray(routes); seeds=np.asarray(seeds); domains=np.asarray(domains)
    evaluable=np.asarray(evaluable,dtype=bool)
    folds=inner_folds(patients)
    pair_patients=np.repeat(patients,2)
    eligible=np.repeat(evaluable,2)
    if routed_only: eligible &= np.tile(np.arange(2),len(x))==np.repeat(routes,2)
    xp,yp=x.reshape(-1,18),targets.reshape(-1,3)
    pf=np.repeat(folds,2); ledger=[]; oofs={}; scores={}
    def do_fit(mask, lam, phase, inner):
        ids=pair_patients[mask]; unique=sorted(set(ids))
        record=dict(phase=phase,inner_fold=inner,regularization=lam,patients=len(unique),
                    pairs=int(mask.sum()),patient_ids=unique,outputs=['d_rim','d_cup','H'],
                    status='NO_FIT_EMPTY_SPLIT',actual_design_solves=0,actual_scalar_heads=0)
        ledger.append(record)
        if not mask.any():
            if ledger_sink: ledger_sink('no_fit',record)
            return None
        if ledger_sink: ledger_sink('fit_attempt',record)
        w=patient_weights(ids)
        try:
            model=fit(xp[mask],yp[mask],w,lam)
        except Exception:
            record['status']='FIT_FAILED'
            if ledger_sink: ledger_sink('fit_failed',record)
            raise
        record.update(status='FIT_COMPLETED',actual_design_solves=1,actual_scalar_heads=3,
                      weight_sum=float(w.sum()),zero_variance_features=int(np.sum(np.all(xp[mask]==xp[mask][0],axis=0))))
        if ledger_sink: ledger_sink('fit_completed',record)
        return model
    for lam in LAMBDAS:
        oof=np.full((len(x)*2,3),np.nan)
        for inner in range(4):
            mask=eligible & (pf!=inner)
            model=do_fit(mask,lam,'inner',inner)
            if model is not None: oof[pf==inner]=predict(model,xp[pf==inner])
        good=eligible & np.isfinite(oof).all(1)
        if np.array_equal(good,eligible) and good.any() and np.isfinite(oof).all():
            w=patient_weights(pair_patients[good])
            scores[lam]=float(np.average(((oof[good]-yp[good])**2).mean(1),weights=w))
            oofs[lam]=oof.reshape(-1,2,3)
    if not scores:
        return dict(model=None,selected_lambda=None,candidates={},choices={'cf':None,'gain':None},
                    fits=ledger,status='CURRENT_FALLBACK_NO_SOLVABLE_INNER_FIT',lambda_mse={})
    lam=min(scores,key=lambda v:(scores[v],-v))
    model=do_fit(eligible,lam,'outer_refit',None)
    choices={}; candidates={}
    for kind in (('cf',) if routed_only else ('cf','gain')):
        choices[kind],candidates[kind]=select_threshold(oofs[lam],x,targets,routes,seeds,domains,evaluable,kind)
    return dict(model=model,selected_lambda=lam,lambda_mse=scores,choices=choices,candidates=candidates,
                fits=ledger,status='FIT_COMPLETED',inner_folds=folds.tolist())
