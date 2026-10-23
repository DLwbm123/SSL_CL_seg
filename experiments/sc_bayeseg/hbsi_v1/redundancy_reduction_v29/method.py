"""Spatial cross-view redundancy reduction on the shared recognition backbone."""
import math
import torch
from ..spatial_correspondence_v28.method import feature
from ..semantic_alignment_v23.method import Objective as BaseObjective


def redundancy(z,w):
    assert z.shape==w.shape and z.ndim==3 and z.shape[1:]==(64,16)
    z=z-z.mean(1,keepdim=True);w=w-w.mean(1,keepdim=True)
    vz=z.square().mean(1,keepdim=True);vw=w.square().mean(1,keepdim=True)
    zn=z/torch.sqrt(vz+1e-8);wn=w/torch.sqrt(vw+1e-8)
    c=zn.transpose(1,2)@wn/64
    diagonal=c.diagonal(dim1=-2,dim2=-1)
    mask=~torch.eye(16,device=z.device,dtype=torch.bool)
    invariance=(1-diagonal).square().mean()
    off=c[:,mask].square().sum(1).mean()/16
    loss=invariance+.005*off
    return loss,dict(invariance=invariance.detach(),redundancy=off.detach(),correlation_diagonal=diagonal.detach().mean(),correlation_offdiag_square=c[:,mask].detach().square().mean(),feature_spatial_sd=(vz.detach().sqrt().mean()+vw.detach().sqrt().mean())/2)


def auxiliary(s,x):
    assert x.shape[-2:]==(128,128)
    flip=s.steps%2==0
    view=torch.flip(.95*x+.025,[-1]) if flip else 1.05*x-.025
    f=feature(s.phi,x);g=feature(s.phi,view)
    if flip:g=torch.flip(g,[-1])
    z=f[:,:,8::16,8::16].flatten(2).transpose(1,2)
    w=g[:,:,8::16,8::16].flatten(2).transpose(1,2)
    return redundancy(z,w)


def check_fixture():
    n=torch.arange(64,dtype=torch.float64)[:,None];k=torch.arange(1,17,dtype=torch.float64)[None]
    z=torch.sin(2*math.pi*n*k/64)[None]
    aligned,_=redundancy(z,z)
    duplicated,_=redundancy(z[:,:,:1].expand_as(z),z[:,:,:1].expand_as(z))
    wrong,_=redundancy(z,torch.roll(z,1,-1))
    flat=torch.ones_like(z);collapsed,_=redundancy(flat,flat)
    assert aligned<1e-12 and duplicated>aligned and wrong>aligned and abs(float(collapsed)-1)<1e-12
    p=z.clone().requires_grad_();rng=torch.Generator().manual_seed(29)
    noisy=z+.1*torch.randn(z.shape,generator=rng,dtype=z.dtype)
    loss,_=redundancy(p,noisy);loss.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0
    return dict(orthogonal_identity_near_zero=True,duplicate_channels_penalized=True,channel_mismatch_penalized=True,collapsed_constant_loss_one=True,finite_nondegenerate_gradient=True,fixture_reverse_calls=1)


class Objective:
    def __init__(self,state,refiner,use_u):
        self.s=state;self.counter=refiner.counter;self.use_u=use_u
        self.base=BaseObjective(state,refiner,False)

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        s=self.s;assert U is not None and 'observations' not in U
        if self.use_u:s.check_batch(U,'U')
        else:assert U['role']=='appearance_donor_L' and U['stage']==s.active['id']
        base,record=self.base.estimate(L,None,counts,generator,sampling_weights)
        aux,stats=auxiliary(s,U['x']);loss=base+aux
        if not torch.isfinite(loss):raise FloatingPointError('redundancy reduction objective')
        self.counter.add('redundancy_backbone_forwards',2*len(U['x']))
        return loss,dict(record,scaled=loss.detach(),base_objective=base.detach(),auxiliary_risk=aux.detach(),auxiliary_pool_U=loss.new_tensor(float(self.use_u)),**stats)
