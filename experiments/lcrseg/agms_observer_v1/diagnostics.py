"""Detached aggregate observers. No label input on U; no loss/gate feedback."""
import math
import torch
import torch.nn.functional as F
from ..agms_cl_v0_1.core import selector, brier


@torch.no_grad()
def selection_counts(qs,geometry,fine,risk):
    masks={};stats={}
    for name,arm in [('main','A1'),('equal','A4'),('risk','A5')]:
        masks[name],stats[name]=selector(qs,geometry,fine,arm,risk)
    g=geometry.bool();candidate=masks['main']
    out=dict(steps=1,geometry=int(g.sum()),pas_rejected=int((g&~fine).sum()),candidate=int(candidate.sum()),
             invalid_geometry=int((~g).sum()),fine=int((g&fine).sum()),
             rejected_main_disc=int((g&~fine&~candidate).sum()))
    for name in masks:
        out[name+'_selected']=int(masks[name].sum())
        # Disjoint sequential rejection partition of main-disc eligible candidates.
        low=candidate&(stats[name]['mean']<.9) if name!='main' else candidate&False
        high=candidate&~low&(stats[name]['variance']>.01) if name!='main' else candidate&False
        out[name+'_reject_mean']=int(low.sum());out[name+'_reject_variance']=int(high.sum())
        assert out[name+'_selected']+int(low.sum())+int(high.sum())==out['candidate']
    for a,b in [('main','equal'),('main','risk'),('equal','risk')]:
        key=a+'__'+b
        out[key+'_xor']=int((masks[a]^masks[b]).sum())
        out[key+'_intersection']=int((masks[a]&masks[b]).sum())
        out[key+'_union']=int((masks[a]|masks[b]).sum())
    return out


def selection_ratios(counts):
    """Ratios of cumulative pixel counts; no mean-of-step ratios."""
    out=dict(counts)
    for key,value in counts.items():
        if key.endswith(('_selected','_xor','_reject_mean','_reject_variance')):
            for denom in ('geometry','candidate'):
                out[key+'_per_'+denom]=value/counts[denom] if counts[denom] else None
        if key.endswith('_intersection'):
            union=counts[key.replace('_intersection','_union')]
            out[key.replace('_intersection','_jaccard')]=value/union if union else None
    return out


def kl_parts(logp,q,mask):
    """Numeric shadow only: per-pixel categorical KL = parent + qD conditional.

    q is detached. Uses logsumexp for extreme student probabilities. Same
    geometry/mask denominator as the original masked_kl; never a second loss.
    """
    logp=logp.log_softmax(1);q=q.detach();mask=mask.detach().bool();tiny=torch.finfo(q.dtype).tiny
    qd=q[:,1:3].sum(1);lpD=torch.logsumexp(logp[:,1:3],dim=1)
    parent=q[:,0]*(q[:,0].clamp_min(tiny).log()-logp[:,0])+qd*(qd.clamp_min(tiny).log()-lpD)
    conditional=(q[:,1:3]*(q[:,1:3].clamp_min(tiny).log()-qd.clamp_min(tiny).log()[:,None]-(logp[:,1:3]-lpD[:,None]))).sum(1)
    denom=mask.sum().clamp_min(1)
    reduce=lambda x:(x*mask).sum()/denom
    return reduce(parent),reduce(conditional)


@torch.no_grad()
def labeled_diagnostics(qs,labels):
    qs=[q.detach() for q in qs];g=labels!=255;disc=g&((labels==1)|(labels==2));cup=labels==2
    near=lambda m:F.max_pool2d(m.float()[:,None],3,1,1)[:,0].bool()
    # One-pixel GT bands, excluding pixels neighboring unknown labels.
    safe=g&~near(~g)
    inner=safe&(near(cup)&near(labels==1))&disc
    outer=safe&near(labels==0)&near(disc)
    regions={'inner':inner,'outer':outer&~inner,'interior':safe&~inner&~outer}
    cond=torch.stack([q[:,2]/q[:,1:3].sum(1).clamp_min(torch.finfo(q.dtype).tiny) for q in qs])
    disagreement=cond.var(0,unbiased=False)
    entropy=-(qs[0]*qs[0].clamp_min(torch.finfo(qs[0].dtype).tiny).log()).sum(1)
    main_error=(cond[0]>=.5)!=cup
    scales=[]
    for i,q in enumerate(qs):
        parent_brier=brier([q],labels)
        classes=[]
        for cls in (1,2):
            m=disc&(labels==cls);n=int(m.sum())
            classes.append(dict(label=cls,count=n,conditional_errors=int((((cond[i]>=.5)!=cup)&m).sum()),
              conditional_brier_sum=float(((cond[i]-cup.float()).square()*m).sum())))
        present=[c for c in classes if c['count']]
        scales.append(dict(scale=i,parent_brier=None if parent_brier is None else float(parent_brier[0]),
          conditional_classes=classes,
          conditional_balanced_error=sum(c['conditional_errors']/c['count'] for c in present)/len(present) if present else None,
          conditional_balanced_brier=sum(c['conditional_brier_sum']/c['count'] for c in present)/len(present) if present else None))
    ranking=[]
    for region,m0 in regions.items():
        for cls in (1,2):
            m=m0&disc&(labels==cls);n=int(m.sum());k=math.ceil(n*.25)
            item=dict(region=region,label=cls,eligible=n,selected=k,coverage=k/n if n else None,total_main_errors=int(main_error[m].sum()))
            for name,score in [('conditional_disagreement',disagreement),('main_entropy',entropy)]:
                idx=torch.argsort(score[m],descending=True,stable=True)[:k]
                item[name+'_selected_main_errors']=int(main_error[m][idx].sum())
            ranking.append(item)
    return dict(scales=scales,ranking=ranking,boundary_width_pixels=1,inner_outer_overlap='inner takes precedence',
                interpretation='current training L; matched class/region/coverage; descriptive, not independent calibration or pixel significance')
