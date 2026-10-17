import torch
from torch.nn import functional as F
from ..di_dmpa_jascl.data import stable_seed
from ..siemens_ssl_reproduction_v1.methods import pixel_ce, strong_view

ARMS = ('dense_fixmatch', 'label_relation')
STEPS = 10000
GRID = 16
SUPPORT_PER_CLASS = 32
TEMPERATURE = .1
WEIGHT = .1


def parts(model, images):
    h = model.parent.features(images)
    logits = model.parent.native_readout(h, images.shape[-2:])
    # Match centers of the native valid 3x3 readout before output interpolation.
    aligned = F.interpolate(h[...,1:-1,1:-1], size=images.shape[-2:], mode='bilinear', align_corners=True)
    return logits, aligned


def grid_indices(shape, device):
    y = torch.linspace(0,shape[-2]-1,GRID,device=device).round().long()
    x = torch.linspace(0,shape[-1]-1,GRID,device=device).round().long()
    return (y[:,None]*shape[-1]+x[None,:]).flatten()


def nodes(field):
    indices=grid_indices(field.shape,field.device)
    return field.flatten(2)[:,:,indices].transpose(1,2)


def support_indices(labels, step):
    if labels.dtype!=torch.long or labels.ndim!=3 or not ((labels>=0)&(labels<4)).all():
        raise ValueError('support requires current training L labels')
    result=[]
    for patient,y in enumerate(labels.detach().cpu()):
        chosen=[]
        for c in range(4):
            idx=torch.where(y.flatten()==c)[0]
            g=torch.Generator().manual_seed(stable_seed(163,'relation_support',step,patient,c))
            chosen.append(idx[torch.randperm(len(idx),generator=g)[:SUPPORT_PER_CLASS]])
        result.append(torch.cat(chosen).to(labels.device))
    return result


def supports(features, labels, indices, patients=None):
    patients=range(len(indices)) if patients is None else patients
    values=[]; classes=[]
    for i in patients:
        values.append(features[i].flatten(1)[:,indices[i]].T)
        classes.append(labels[i].flatten()[indices[i]])
    return torch.cat(values),torch.cat(classes)


def readout(query, support, labels, base=None):
    if query.shape[-1]!=support.shape[-1] or labels.shape!=(len(support),) or len(support)==0:
        raise ValueError('invalid relation supports')
    if labels.dtype!=torch.long or not ((labels>=0)&(labels<4)).all():
        raise ValueError('invalid support classes')
    weights=(F.normalize(query,dim=-1)@F.normalize(support,dim=-1).T/TEMPERATURE).softmax(-1)
    raw=weights@F.one_hot(labels,4).to(weights)
    present=torch.bincount(labels,minlength=4)>0
    if base is None: return raw,present
    if base.shape!=raw.shape: raise ValueError('base/query grids differ')
    # Missing support is not negative evidence: keep its original probability mass.
    mass=base[...,present].sum(-1,keepdim=True)
    return torch.where(present,raw*mass,base),present


@torch.no_grad()
def target(query_features, support_features, labels, base, indices):
    sf,sy=supports(support_features,labels,indices)
    distribution,present=readout(nodes(query_features),sf,sy,base)
    confidence,pred=distribution.max(-1)
    keep=(base.max(-1).values<.95)&(confidence>=.95)&present[pred]
    if int(present.sum())<2:keep.zero_()
    return distribution.detach(),keep.detach(),present


def labeled_readout(features, labels, indices):
    if len(indices)!=2:raise ValueError('two distinct L patients required')
    query=nodes(features); truth=labels.flatten(1)[:,grid_indices(labels.shape,labels.device)]
    loss=features.sum()*0; active=0
    for i in range(2):
        sf,sy=supports(features,labels,indices,[1-i])
        r,present=readout(query[i],sf,sy)
        if int(present.sum())<2:continue
        for c in range(4):
            mask=truth[i]==c
            if bool(present[c]) and bool(mask.any()):
                loss=loss-r[mask,c].clamp_min(1e-8).log().mean();active+=1
    return loss/max(1,active),active


def transfer(strong_logits, strong_features, support_features, labels, indices, q, keep):
    sf,sy=supports(support_features.detach(),labels,indices)
    strong=nodes(strong_logits).log_softmax(-1)
    student,_=readout(nodes(strong_features),sf,sy,strong.exp())
    q=q.detach();keep=keep.detach()
    ce=-(q*strong).sum(-1)
    kl=(torch.xlogy(q,q)-q*student.clamp_min(1e-8).log()).sum(-1)
    return ((ce+kl)*keep).mean()


def losses(arm,model,ema,provider,step):
    if arm not in ARMS:raise PermissionError('unregistered source arm')
    x,y,ids=provider.labeled(step)
    if len(set(ids))!=2:raise ValueError('L support/query must be distinct patients')
    u=torch.cat([provider.unlabeled(step*7+i)[0] for i in range(7)])
    with torch.no_grad():weak=model(u);confidence,pseudo=weak.softmax(1).max(1);base_keep=confidence>=.95
    if arm=='label_relation':l_logits,l_features=parts(model,x)
    else:l_logits=model(x)
    supervised=pixel_ce(l_logits,y).mean()
    strong_images=strong_view(u,step)
    if arm=='label_relation':s_logits,s_features=parts(model,strong_images)
    else:s_logits=model(strong_images)
    unsupervised=(pixel_ce(s_logits,pseudo)*base_keep).mean()
    read=supervised*0;extra=supervised*0;active=0
    accepted=[0]*4;query_count=0
    if arm=='label_relation':
        indices=support_indices(y,step)
        read,active=labeled_readout(l_features,y,indices)
        with torch.no_grad():
            _,e_l=parts(ema,x);_,e_u=parts(ema,u)
            q,keep,_=target(e_u,e_l,y,nodes(weak.softmax(1)),indices)
        extra=transfer(s_logits,s_features,l_features,y,indices,q,keep)
        accepted=[int((keep&(q.argmax(-1)==c)).sum()) for c in range(4)]
        query_count=keep.numel()
    total=supervised+unsupervised+WEIGHT*read+WEIGHT*extra
    return total,dict(supervised=float(supervised.detach()),unsupervised=float(unsupervised.detach()),
        relation_labeled=float(read.detach()),relation_transfer=float(extra.detach()),
        supervised_relation_groups=active,accepted_class_pixels=[int((base_keep&(pseudo==c)).sum()) for c in range(4)],
        U_pixels=base_keep.numel(),extra_accepted_class_nodes=accepted,relation_queries=query_count)
