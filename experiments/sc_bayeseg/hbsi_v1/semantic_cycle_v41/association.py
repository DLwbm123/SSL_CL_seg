"""Labeled-pixel round trips through image-only bridge features."""
import math
import torch
from ..spatial_correspondence_v28.method import feature


def support_indices(y):
    # Keep empty slices. Selection uses only the already-observed current L mask.
    labels=y.flatten().long();classes=labels.unique(sorted=True)
    chosen=[]
    for c in classes:
        members=torch.nonzero(labels==c).flatten()
        n=64//len(classes)
        chosen.append(members[torch.arange(n,device=y.device)*len(members)//n])
    index=torch.cat(chosen)
    assert len(index)==64
    return index,labels[index]


def association(a,b,labels):
    assert a.ndim==b.ndim==2 and a.shape[1]==b.shape[1]
    logits=a@b.T
    ab=torch.log_softmax(logits,1);ba=torch.log_softmax(logits.T,1)
    cycle=torch.logsumexp(ab[:,:,None]+ba[None,:,:],dim=1)
    same=labels[:,None]==labels[None,:]
    target=same.to(a.dtype)/same.sum(1,keepdim=True)
    walker=-(target*cycle).sum(1).mean()
    logvisit=torch.logsumexp(ab,dim=0)-math.log(len(a))
    visit=-logvisit.mean()
    value=walker+visit
    return value,dict(cycle_walker=walker.detach(),cycle_visit=visit.detach(),cycle_correct_mass=(cycle.detach().exp()*same).sum(1).mean(),cycle_visit_entropy=-(logvisit.detach().exp()*logvisit.detach()).sum(),cycle_single_class=value.new_tensor(float(labels.unique().numel()==1)),cycle_support_fg_fraction=(labels==1).to(a.dtype).mean(),cycle_feature_norm_A=a.detach().norm(dim=1).mean(),cycle_feature_norm_B=b.detach().norm(dim=1).mean())


def auxiliary(s,L,x):
    assert L['x'].shape==x.shape==(1,1,128,128)
    obs=L['observations'];assert obs['obs_mask'].all() and (obs['y_obs'].sum(1)==1).all()
    truth=obs['y_obs'].argmax(1)
    index,labels=support_indices(truth)
    a=feature(s.phi,L['x'])[0].flatten(1).T[index]
    b=feature(s.phi,x)[0,:,8::16,8::16].flatten(1).T
    assert a.shape==b.shape==(64,16)
    return association(a,b,labels)


def check_fixture():
    a=torch.tensor([[2.,0.],[2.,0.],[0.,2.],[0.,2.]],dtype=torch.float64,requires_grad=True)
    b=a.detach().clone().requires_grad_();y=torch.tensor([0,0,1,1])
    good,stats=association(a,b,y);flat,_=association(torch.zeros_like(a),torch.zeros_like(b),y)
    assert good<flat and stats['cycle_correct_mass']>.9
    perm=torch.tensor([2,0,3,1]);other,_=association(a[perm],b.flip(0),y[perm]);assert torch.allclose(good,other,atol=1e-12)
    assert abs(float(flat)-2*math.log(4))<1e-12
    empty,_=association(a,b,torch.zeros_like(y));assert torch.isfinite(empty)
    zero_idx,zero_label=support_indices(torch.zeros(128,128,dtype=torch.long));assert len(zero_idx.unique())==64 and zero_label.sum()==0
    mixed=torch.zeros(128,128,dtype=torch.long);mixed[1,1]=1
    idx,lab=support_indices(mixed);assert (lab==1).sum()==32 and (idx[lab==1]==129).all()
    ga,gb=torch.autograd.grad(good,(a,b));assert torch.isfinite(ga).all() and torch.isfinite(gb).all() and ga.norm()>0 and gb.norm()>0
    h=1e-5;ap=a.detach().clone();am=ap.clone();ap[0,0]+=h;am[0,0]-=h
    fd=(association(ap,b.detach(),y)[0]-association(am,b.detach(),y)[0])/(2*h)
    assert torch.allclose(fd,ga[0,0],atol=1e-8,rtol=1e-5)
    return dict(class_preserving_better_than_collapse=True,permutation_invariant=True,uniform_loss_exact=True,empty_support_retained=True,small_class_repeats_explicit=True,both_feature_gradients=True,finite_difference=True,fixture_reverse_calls=1)
