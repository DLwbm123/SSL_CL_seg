"""Image-only spatial correspondence on the shared recognition backbone."""
import math
import torch
from torch.nn import functional as F
from ..semantic_alignment_v23.method import Objective as BaseObjective


def feature(phi,x):
    # Explicit coordinates would solve location matching without image content.
    if phi.config.coordinate_channels:
        x=torch.cat((x,x.new_zeros((len(x),2,*x.shape[-2:]))),1)
    h=torch.tanh(phi.enc(x))
    low=torch.tanh(phi.down(F.avg_pool2d(h,2)))
    return torch.tanh(phi.up(torch.cat((h,F.interpolate(low,x.shape[-2:],mode='bilinear',align_corners=False)),1)))


def contrast(z,w):
    assert z.shape==w.shape and z.ndim==3 and z.shape[1]==64
    z=F.normalize(z,dim=-1,eps=1e-8);w=F.normalize(w,dim=-1,eps=1e-8)
    similarities=z@w.transpose(-1,-2);logits=similarities/.1
    target=torch.arange(64,device=z.device).expand(len(z),-1).reshape(-1)
    ce=(F.cross_entropy(logits.reshape(-1,64),target)+F.cross_entropy(logits.transpose(-1,-2).reshape(-1,64),target))/2
    diagonal=similarities.diagonal(dim1=-2,dim2=-1)
    return ce/math.log(64),dict(correspondence_positive_cos=diagonal.detach().mean(),correspondence_negative_cos=(similarities.detach().sum()-diagonal.detach().sum())/(len(z)*64*63),correspondence_accuracy=(logits.detach().argmax(-1)==target.reshape(len(z),64)).double().mean())


def auxiliary(s,x):
    assert x.shape[-2:]==(128,128)
    # Matched to the already frozen v25 view strengths, with no search.
    flip=s.steps%2==0
    view=torch.flip(.95*x+.025,[-1]) if flip else 1.05*x-.025
    f=feature(s.phi,x);g=feature(s.phi,view)
    if flip:g=torch.flip(g,[-1])
    # 64 well separated interior centers; positives use exact inverse alignment.
    z=f[:,:,8::16,8::16].flatten(2).transpose(1,2)
    w=g[:,:,8::16,8::16].flatten(2).transpose(1,2)
    return contrast(z,w)


def check_fixture():
    z=torch.eye(64,dtype=torch.float64)[None].requires_grad_()
    aligned,_=contrast(z,z);wrong,_=contrast(z,torch.roll(z,1,1))
    assert aligned<wrong
    flat=torch.ones_like(z);chance,_=contrast(flat,flat)
    assert abs(float(chance)-1)<1e-12
    aligned.backward();assert z.grad is not None and torch.isfinite(z.grad).all()
    return dict(identity_better_than_wrong=True,collapsed_features_chance_one=True,finite_fixture_gradient=True,fixture_reverse_calls=1)


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
        if not torch.isfinite(loss):raise FloatingPointError('spatial correspondence objective')
        self.counter.add('correspondence_backbone_forwards',2*len(U['x']))
        return loss,dict(record,scaled=loss.detach(),base_objective=base.detach(),correspondence=aux.detach(),auxiliary_pool_U=loss.new_tensor(float(self.use_u)),**stats)
