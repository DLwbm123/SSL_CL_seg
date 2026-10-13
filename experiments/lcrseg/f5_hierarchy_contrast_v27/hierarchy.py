"""V21 coarse/sibling contrasts, with no live child-parent distance objective."""
import math
import torch
from ..f5_hierarchy_proto_v21.hierarchy import validate_mapping,prototypes,contrast

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
    coarse=contrast(z,parent,hp,hs,temperature);parts=[]
    for p in range(len(hp)):
        children=torch.tensor([c for c,v in enumerate(parent_map) if v==p],device=features.device)
        children=children[ls[children]]
        if len(children)<2:continue
        member=parent==p
        selected=torch.where(member,leaf,torch.full_like(leaf,255))
        child_support=torch.zeros_like(ls);child_support[children]=True
        parts.append(contrast(z,selected,lp,child_support,temperature))
    zero=z.sum()*0
    terms=dict(coarse=coarse,parts=torch.stack(parts).mean() if parts else zero,
               alignment=zero)
    stats=dict(leaf_pixels=lc,parent_pixels=hc,coarse_classes=int(hs.sum()),fine_groups=len(parts),
               aligned_edges=0,alignment_removed=True,valid_pixels=int((leaf!=255).sum()),
               zero_leaf_prototypes=sum(n>0 and not bool(s) for n,s in zip(lc,ls)),
               zero_parent_prototypes=sum(n>0 and not bool(s) for n,s in zip(hc,hs)))
    return terms,stats
