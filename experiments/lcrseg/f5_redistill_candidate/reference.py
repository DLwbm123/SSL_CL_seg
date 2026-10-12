"""Literal fixed D_(2/3)(entry || student), for bounded mechanism checks."""
import torch
from ..five_frameworks_v1.kernels import _finite,_prob,_mask


def power_reference(logits,target,valid):
    if logits.shape!=target.shape:raise ValueError('reference grids must match')
    _finite(logits,'student logits');_prob(target,'teacher probabilities');valid=_mask(valid,logits)
    p=target.detach().double()
    if not torch.allclose(p.sum(1),torch.ones_like(p[:,0]),atol=1e-5,rtol=1e-5):
        raise ValueError('teacher probabilities must sum to one')
    logq=torch.log_softmax(logits.double(),dim=1)
    # p=0 contributes exactly0; finite positive q follows from finite log-softmax logits.
    exponent=torch.where(p>0,(5/3)*p.log()-(2/3)*logq,torch.full_like(p,-torch.inf))
    per_pixel=.9*(exponent.exp().sum(1)-1)
    _finite(per_pixel,'power divergence')
    labels=p.argmax(1);terms=[]
    for c in range(logits.shape[1]):
        selected=valid & (labels==c)
        if selected.any():terms.append(per_pixel[selected].mean())
    return torch.stack(terms).mean() if terms else logits.sum()*0
