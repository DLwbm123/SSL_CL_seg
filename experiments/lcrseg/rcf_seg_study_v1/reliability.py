"""RCF equations 9--21; independent of labels, anatomy, and expert identity."""
import numpy as np


def posterior(x, state):
    z=(np.asarray(x,dtype=np.float64)-state['center'])/state['scale']
    logs=[]
    for w,m,c in zip(state['weights'],state['means'],state['covariances']):
        d=z-m
        sign,det=np.linalg.slogdet(c)
        if sign<=0: raise FloatingPointError('nonpositive covariance')
        logs.append(np.log(w)-.5*(det+np.einsum('ni,ij,nj->n',d,np.linalg.inv(c),d)+2*np.log(2*np.pi)))
    logs=np.stack(logs,1)
    maximum=logs.max(1,keepdims=True)
    probabilities=np.exp(logs-maximum)
    normalizer=probabilities.sum(1,keepdims=True)
    return probabilities/normalizer, float((maximum+np.log(normalizer)).mean())


def fit_gmm(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=2 or len(x)<4 or not np.isfinite(x).all():
        raise ValueError('finite two-dimensional unlabeled evidence required')
    center=np.median(x,axis=0)
    scale=np.maximum(1.4826*np.median(abs(x-center),axis=0),.05)
    z=(x-center)/scale
    if np.max(np.var(z,axis=0))<1e-10: return None
    best=None
    for q in (.10,.25,.35):
        score=z.sum(1); low=score<=np.quantile(score,q); high=score>=np.quantile(score,1-q)
        state=dict(center=center,scale=scale,means=np.stack([z[low].mean(0),z[high].mean(0)]),
                   weights=np.array([.5,.5]),covariances=np.stack([np.eye(2),np.eye(2)]))
        previous=-np.inf
        for _ in range(25):
            resp,ll=posterior(x,state)
            mass=resp.sum(0)
            if mass.min()<2: break
            means=resp.T@z/mass[:,None];cov=[]
            for j in range(2):
                d=z-means[j];c=(d.T*resp[:,j])@d/mass[j]
                cov.append(.8*c+.2*np.diag(np.diag(c))+np.eye(2)*1e-4)
            state.update(weights=mass/len(x),means=means,covariances=np.stack(cov))
            if abs(ll-previous)<1e-4: break
            previous=ll
        if mass.min()<2: continue
        _,ll=posterior(x,state)
        if np.isfinite(ll) and (best is None or ll>best['log_likelihood']):
            state['clean']=int(state['means'].sum(1).argmax());state['log_likelihood']=ll;best=state
    return best


def threshold(scores, eta, previous=None):
    scores=np.asarray(scores,dtype=np.float64)
    if not len(scores) or not np.isfinite(scores).all() or not 0<eta<=1:
        raise ValueError('invalid posterior population or reliability target')
    ordered=np.sort(scores)[::-1]
    valid=np.flatnonzero(np.cumsum(ordered)/np.arange(1,len(ordered)+1)>=eta)
    # No admissible prefix means reject all. Finite sentinel keeps checkpoints JSON-safe.
    candidate=float(ordered[valid[-1]]) if len(valid) else 1.000001
    return candidate if previous is None or candidate>=previous else .7*previous+.3*candidate
