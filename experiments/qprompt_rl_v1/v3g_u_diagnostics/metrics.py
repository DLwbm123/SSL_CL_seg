"""Matched training-probe summaries; no optimizer operations."""
import math
import torch


def gradient_metrics(left,right):
    a=sum(float(g.detach().double().square().sum()) for g in left if g is not None)
    b=sum(float(g.detach().double().square().sum()) for g in right if g is not None)
    dot=sum(float((g.detach().double()*h.detach().double()).sum()) for g,h in zip(left,right) if g is not None and h is not None)
    assert all(math.isfinite(v) for v in (a,b,dot))
    cosine=max(-1.,min(1.,dot/math.sqrt(a*b))) if a>0 and b>0 else None
    return dict(gradient_cosine=cosine,negative_cosine=None if cosine is None else float(cosine<0),
                norm_L=math.sqrt(a),norm_U=math.sqrt(b),weighted_U_over_L=.5*math.sqrt(b/a) if a>0 else None)


def kl(q,p):
    return (q*(q.clamp_min(1e-8).log()-p.clamp_min(1e-8).log())).sum(1)


def check():
    x=torch.tensor([1.,2.],requires_grad=True)
    a=torch.autograd.grad(x.square().sum(),x,retain_graph=True)
    b=torch.autograd.grad(-x.square().sum(),x)
    assert gradient_metrics(a,b)['gradient_cosine']==-1
    assert gradient_metrics(a,a)['gradient_cosine']==1
    assert gradient_metrics(a,[torch.zeros_like(x)])['gradient_cosine'] is None
    assert gradient_metrics(a,b)['weighted_U_over_L']==.5
    q=torch.tensor([[[[.2]],[[.3]],[[.5]]]])
    assert float(kl(q,q))==0 and float(kl(q,q.flip(1)))>0
    print('PASS positive, opposing, zero gradients and directed KL')


if __name__=='__main__':check()
