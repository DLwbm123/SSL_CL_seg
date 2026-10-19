"""Finite fixed-coordinate Gaussian sites; no analytic segmentation head."""
import torch
from torch.nn import functional as F


def graph_sites(features, pairs, enabled):
    n=len(features)
    lap=features.new_zeros(n,n,dtype=torch.float64)
    edges=[]
    if enabled:
        z=F.normalize(features.double(),dim=1)
        similarity=z@z.T
        similarity.fill_diagonal_(-torch.inf)
        value,index=similarity.topk(3,dim=1)
        weights=value.softmax(1)
        for i in range(n):
            for j,w in zip(index[i].tolist(),weights[i]):
                lap[i,i]+=w; lap[j,j]+=w; lap[i,j]-=w; lap[j,i]-=w
                edges.append((i,j))
        for i,j in pairs.tolist():
            lap[i,i]+=1; lap[j,j]+=1; lap[i,j]-=1; lap[j,i]-=1
    return torch.kron(lap.contiguous(),torch.eye(3,device=lap.device,dtype=lap.dtype)),edges


def label_sites(f, labels):
    n=len(f); precision=f.new_zeros(3*n,3*n,dtype=torch.float64)
    natural=f.new_zeros(3*n,dtype=torch.float64)
    for i in (labels>=0).nonzero().flatten().tolist():
        point=f[i].double()
        p=torch.cat((point.new_zeros(1),point)).softmax(0)[1:]
        target=F.one_hot(labels[i],4)[1:].to(p)
        hessian=torch.diag(p)-p[:,None]*p[None,:]
        gradient=p-target
        precision[3*i:3*i+3,3*i:3*i+3]=hessian
        natural[3*i:3*i+3]=hessian@point-gradient
    return precision,natural


def rebuild(base_precision,base_natural,label_precision,label_natural,relation):
    precision=base_precision+label_precision+relation
    factor=torch.linalg.cholesky(precision)
    mean=torch.cholesky_solve((base_natural+label_natural)[:,None],factor)[:,0]
    return precision,mean


def self_check():
    torch.manual_seed(163)
    f=torch.randn(8,3,dtype=torch.double)
    labels=torch.tensor([0,1,2,3,-1,-1,-1,-1])
    base=torch.eye(24,dtype=torch.double)
    tp,hp=label_sites(f,labels)
    relation,_=graph_sites(torch.randn(8,4),torch.tensor([[4,5]]),True)
    p,m=rebuild(base,f.flatten(),tp,hp,relation)
    p2,m2=rebuild(base,f.flatten(),tp,hp,relation)
    assert torch.equal(p,p2) and torch.equal(m,m2)
    # Rebuild from retained PSD sites, rather than accumulating exposures.
    zero=torch.zeros_like(relation)
    removed,m0=rebuild(base,f.flatten(),tp,hp,zero)
    assert torch.allclose(removed,base+tp)
    revised=labels.clone(); revised[4]=2
    tr,hr=label_sites(f,revised)
    _,mr=rebuild(base,f.flatten(),tr,hr,relation)
    assert (mr-m).norm()>0
    # Exact Gaussian-site objective gradient agrees with posterior matching.
    x=f.flatten().clone().requires_grad_()
    objective=.5*(x-m)@p@(x-m)
    g=torch.autograd.grad(objective,x)[0]
    expected=base@(x-f.flatten())+tp@x-hp+relation@x
    assert torch.allclose(g,expected,atol=1e-12)
    assert torch.isfinite(m0).all()
    return dict(status='PASS',optimizer_calls=0,patient_forwards=0,
                replacement_idempotent=True,remove_rebuild_exact=True,
                legal_label_revises_state=True,quadratic_gradient_equivalent=True)
