"""GYRO supplement Algorithm1 and Eq9; no anatomy labels or logit binding."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def frame(seed, k=10, d=128):
    if not 2 <= k <= d:
        raise ValueError('Gaussian thin-QR construction requires 2<=K<=D')
    g=torch.Generator().manual_seed(seed)
    q,r=torch.linalg.qr(torch.randn(d,k,generator=g,dtype=torch.float64),mode='reduced')
    q=q*torch.where(r.diag()<0,-1.,1.)
    return (math.sqrt(k/(k-1))*(torch.eye(k,dtype=torch.float64)-torch.ones(k,k,dtype=torch.float64)/k)@q.T).float()


def spectral(z, frames, ids, beta):
    if z.ndim!=3 or z.shape[1:]!=(2,128): raise ValueError('expected P,V=2,D=128')
    c=F.normalize(z,dim=-1,eps=1e-12).mean(1)
    if beta:
        e=frames[ids].detach()
        j=torch.einsum('pd,pkd->pk',c.detach(),e).argmax(1)
        a=e[torch.arange(len(c),device=c.device),j]
        y=(1-beta)*c+beta*a
    else:
        y=c
    value=-torch.linalg.svdvals(y.float()).sum()/math.sqrt(len(c)*min(len(c),128))
    return value


class Projector(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(128,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Linear(512,128))


def self_check():
    e=frame(163); e2=frame(164)
    gram=e@e.T
    assert torch.allclose(gram,torch.eye(10)*(10/9)-torch.ones(10,10)/9,atol=2e-6)
    assert torch.allclose(e.sum(0),torch.zeros(128),atol=1e-6)
    assert torch.equal(e,frame(163)) and not torch.equal(e,e2)
    z=torch.randn(24,2,128,generator=torch.Generator().manual_seed(90),requires_grad=True)
    ids=torch.tensor([1]*16+[0]*8); fs=torch.stack([e,e2])
    loss=spectral(z,fs,ids,1/3); g=torch.autograd.grad(loss,z)[0]
    assert torch.isfinite(g).all() and g.norm()>0
    perm=torch.arange(23,-1,-1)
    assert torch.allclose(loss,spectral(z[perm],fs,ids[perm],1/3),atol=1e-6)
    c=F.normalize(z,dim=-1).mean(1)
    assert torch.allclose(spectral(z,fs,ids,0),-torch.linalg.svdvals(c).sum()/24)
    return dict(status='PASS',checks=['simplex_geometry','frame_determinism','finite_shared_gradient','row_permutation','beta0_MMCR'],optimizer_updates=0)
