"""Stateless current-label prototype reassignment, preserving target probability values."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def assignments(labeled_features,labels,unlabeled_features,fallback):
    assert labeled_features.ndim==unlabeled_features.ndim==4 and labeled_features.shape[1]==unlabeled_features.shape[1]
    assert labels.shape==labeled_features[:,0].shape and fallback.shape==unlabeled_features[:,0].shape
    assert torch.isfinite(labeled_features).all() and torch.isfinite(unlabeled_features).all()
    assert ((labels==255)|((labels>=0)&(labels<3))).all()
    lf=F.normalize(labeled_features.detach().float(),dim=1,eps=1e-8);uf=F.normalize(unlabeled_features.detach().float(),dim=1,eps=1e-8)
    support=[];centers=[]
    for cls in range(3):
        mask=labels==cls;n=int(mask.sum());support.append(n)
        center=(lf*mask[:,None]).sum((0,2,3))/max(n,1);centers.append(center)
    centers=torch.stack(centers);present=(torch.tensor(support,device=centers.device)>0)&(centers.norm(dim=1)>1e-8)
    if not present.any():return fallback.detach().clone(),support,False
    centers=F.normalize(centers,dim=1,eps=1e-8)
    score=torch.einsum('bchw,kc->bkhw',uf,centers).masked_fill(~present[None,:,None,None],-torch.inf)
    chosen=score.argmax(1)
    zero=unlabeled_features.detach().float().norm(dim=1)<=1e-8
    chosen=torch.where(zero,fallback,chosen)
    return chosen,support,True


@torch.no_grad()
def permute(target,chosen,mask):
    target=target.detach();old=target.argmax(1);chosen=chosen.detach().long();mask=mask.to(target.device).bool()
    assert target.shape[1]==3 and chosen.shape==old.shape==mask.shape and ((chosen>=0)&(chosen<3)).all()
    assert torch.isfinite(target).all() and (target>=0).all()
    out=target.clone();old_value=target.gather(1,old[:,None]);new_value=target.gather(1,chosen[:,None])
    out.scatter_(1,old[:,None],new_value);out.scatter_(1,chosen[:,None],old_value)
    return torch.where(mask[:,None],out,target)


def selfcheck():
    labels=torch.tensor([[[0,1],[2,255]]]);lf=F.one_hot(labels.clamp_max(2),3).permute(0,3,1,2).float().requires_grad_()
    uf=lf.detach().clone().requires_grad_();fallback=torch.zeros_like(labels)
    chosen,support,present=assignments(lf,labels,uf,fallback)
    assert present and support==[1,1,1] and torch.equal(chosen[labels!=255],labels[labels!=255]) and not chosen.requires_grad
    missing=torch.full_like(labels,255);assert torch.equal(assignments(lf,missing,uf,fallback)[0],fallback)
    assert torch.equal(assignments(lf,labels,uf*0,fallback)[0],fallback)
    absent=labels.clone();absent[absent==2]=255;assert not (assignments(lf,absent,uf,fallback)[0]==2).any()
    target=torch.tensor([.8,.15,.05]).reshape(1,3,1,1).expand(1,3,2,2).clone().requires_grad_();mask=labels!=255
    result=permute(target,chosen,mask)
    assert torch.equal(result.sort(1).values,target.detach().sort(1).values) and torch.equal(result[:,:,1,1],target.detach()[:,:,1,1])
    assert torch.equal(result.argmax(1)[mask],chosen[mask]) and not result.requires_grad
    assert torch.equal(permute(target,target.argmax(1),mask),target.detach()) and torch.equal(permute(target,chosen,mask&False),target.detach())
    rotated=permute(target,(chosen+1)%3,mask);assert torch.equal(rotated.sort(1).values,target.detach().sort(1).values)
    logits=torch.randn(1,3,2,2,requires_grad=True);loss=-(result*logits.log_softmax(1)).sum()
    grads=torch.autograd.grad(loss,(logits,target,lf,uf),allow_unused=True);assert torch.isfinite(grads[0]).all() and grads[1:]==(None,None,None)
    return dict(status='PASS',model_forwards=0,optimizer_calls=0,query_calls=0,checks=['known semantic class assignment','missing-class mask','empty/zero-feature fallback','exact target probability multiset preservation','unselected identity','identity repair','rotated-class control','detached teachers/features/labels; finite student gradient'])
