"""Training-only two-level prototypes; label ancestry is supplied by the task."""
import math
import torch
from torch.nn import functional as F


def validate_mapping(parent_map):
    if not isinstance(parent_map,(tuple,list)) or len(parent_map)<2 or any(type(p) is not int or p<0 for p in parent_map):
        raise ValueError('each leaf needs one nonnegative integer parent')
    if sorted(set(parent_map))!=list(range(max(parent_map)+1)):
        raise ValueError('parent IDs must be contiguous from zero')


def prototypes(features,labels,classes):
    means=[];supported=[];counts=[]
    for c in range(classes):
        mask=labels==c;count=mask.sum(1);present=count>0;counts.append(int(count.sum()))
        mean=((features*mask[:,None]).sum(2)/count.clamp_min(1)[:,None])[present].mean(0) if present.any() else features.sum((0,2))*0
        means.append(F.normalize(mean,dim=0,eps=1e-12));supported.append(bool(mean.detach().norm()>1e-12))
    return torch.stack(means),torch.tensor(supported,device=features.device),counts


def contrast(features,labels,anchors,supported,temperature):
    ids=supported.nonzero().flatten()
    if len(ids)<2:return features.sum()*0
    mapping=torch.full((len(anchors)+1,),255,dtype=torch.long,device=features.device)
    mapping[ids]=torch.arange(len(ids),device=features.device)
    target=mapping[torch.where(labels==255,len(anchors),labels)]
    scores=torch.einsum('bdn,cd->bcn',features,anchors[ids].detach())/temperature
    losses=F.cross_entropy(scores,target,reduction='none',ignore_index=255)
    return torch.stack([losses[labels==c].mean() for c in ids]).mean()


def hierarchy_terms(features,labels,parent_map,temperature=.1,margin=.25):
    validate_mapping(parent_map)
    if features.ndim not in (4,5) or min(features.shape)<1 or features.shape[1]<2 or not features.is_floating_point():
        raise ValueError('expected nonempty 2D/3D batch features with channel dimension >=2')
    if labels.shape!=features.shape[:1]+features.shape[2:] or labels.dtype!=torch.long or labels.device!=features.device:
        raise ValueError('integer labels and features must share their spatial grid/device')
    if not math.isfinite(temperature) or temperature<=0 or not math.isfinite(margin) or not 0<=margin<=2:
        raise ValueError('finite positive temperature and margin in [0,2] required')
    if not torch.isfinite(features).all():raise FloatingPointError('nonfinite hierarchy features')
    if ((labels!=255)&((labels<0)|(labels>=len(parent_map)))).any():raise ValueError('label outside ancestry map')
    flat=features.flatten(2);norm=torch.linalg.vector_norm(flat,dim=1,keepdim=True)
    if not torch.isfinite(norm).all():raise FloatingPointError('overflow hierarchy feature norm')
    z=flat/norm.clamp_min(1e-12);leaf=labels.flatten(1)
    mapping=torch.tensor(tuple(parent_map)+(255,),device=features.device)
    parent=mapping[torch.where(leaf==255,len(parent_map),leaf)]
    lp,ls,lc=prototypes(z,leaf,len(parent_map));hp,hs,hc=prototypes(z,parent,max(parent_map)+1)
    coarse=contrast(z,parent,hp,hs,temperature);parts=[];alignment=[];edges=0
    for p in range(len(hp)):
        children=torch.tensor([c for c,v in enumerate(parent_map) if v==p],device=features.device)
        children=children[ls[children]]
        if len(children)<2:continue
        member=parent==p
        selected=torch.where(member,leaf,torch.full_like(leaf,255))
        child_support=torch.zeros_like(ls);child_support[children]=True
        parts.append(contrast(z,selected,lp,child_support,temperature))
        if hs[p]:
            # The parent is a detached target; only current labeled child means move.
            alignment.append((torch.linalg.vector_norm(lp[children]-hp[p].detach(),dim=1)-margin).clamp_min(0).mean())
            edges+=len(children)
    zero=z.sum()*0
    terms=dict(coarse=coarse,parts=torch.stack(parts).mean() if parts else zero,
               alignment=torch.stack(alignment).mean() if alignment else zero)
    stats=dict(leaf_pixels=lc,parent_pixels=hc,coarse_classes=int(hs.sum()),fine_groups=len(parts),
               aligned_edges=edges,valid_pixels=int((leaf!=255).sum()),
               zero_leaf_prototypes=sum(n>0 and not bool(s) for n,s in zip(lc,ls)),
               zero_parent_prototypes=sum(n>0 and not bool(s) for n,s in zip(hc,hs)))
    return terms,stats
