"""Exact detached capacity assignment; the R1 GRQA objective is otherwise unchanged."""
import itertools
import math
import torch
from torch.nn import functional as F
from qprompt.losses import grqa_loss as original_grqa
from r1_12h.core import image_prototypes, image_alignment_loss, supervised_query_loss, DINOv2Adapter


def exact_assignment(scores, supported, bounds=None):
    """DP over counts; maximize FP64 score, break exact ties by assignment tuple."""
    if scores.ndim != 3 or scores.shape[-1] != 3: raise ValueError('expected B,K,3')
    classes = supported.nonzero().flatten().tolist(); k = scores.shape[1]
    if not classes: return None
    if bounds is None:
        if k != 12: raise ValueError('production K must equal 12')
        bounds = {3:(3,6),2:(4,8),1:(12,12)}[len(classes)]
    lo,hi = bounds
    if not len(classes)*lo <= k <= len(classes)*hi: raise ValueError('infeasible capacity')
    values = scores.detach().double().cpu().tolist(); output=[]
    for batch in values:
        states = {(0,)*len(classes):(0.,())}
        for row in batch:
            new={}
            for counts,(value,path) in states.items():
                for j,c in enumerate(classes):
                    if counts[j] >= hi: continue
                    if not math.isfinite(row[c]): raise FloatingPointError('nonfinite supported similarity')
                    nc=list(counts);nc[j]+=1;nc=tuple(nc)
                    candidate=(value+row[c],path+(c,));old=new.get(nc)
                    if old is None or candidate[0]>old[0] or (candidate[0]==old[0] and candidate[1]<old[1]):new[nc]=candidate
            states=new
        feasible=[v for counts,v in states.items() if min(counts)>=lo]
        if not feasible: raise ValueError('no feasible assignment')
        output.append(min(feasible,key=lambda v:(-v[0],v[1]))[1])
    return torch.tensor(output,device=scores.device,dtype=torch.long)


def weight(arm, local_step):
    if arm=='A0':return 0.
    return 5.*min(1.,local_step/300) if arm in ('A2','A4') else 5.


def routed_grqa(queries, reference_queries, bank, cc=False, diagnostics=False):
    if queries.shape != reference_queries.shape or queries.shape[-1] != bank.vectors.shape[-1]:raise ValueError('geometry mismatch')
    if not bank.supported.any():
        z=queries.sum()*0
        return dict(total=z,group=z,paper_selected_k3_surrogate=z,categorical_kl=z.detach()),dict(status='NO_SUPPORTED_CLASS',routing_mode='cc' if cc else 'top1',supported_classes=[])
    anchors=bank.vectors.detach()
    current=F.normalize(queries,dim=-1)@anchors.T
    reference=F.normalize(reference_queries.detach(),dim=-1)@anchors.T
    current=current.masked_fill(~bank.supported[None,None],-torch.inf)
    reference=reference.masked_fill(~bank.supported[None,None],-torch.inf)
    selected=exact_assignment(current,bank.supported) if cc else current.detach().argmax(-1)
    reward=current.detach().gather(-1,selected[...,None]).squeeze(-1);advantage=torch.zeros_like(reward)
    sizes=[]
    for b in range(queries.shape[0]):
        sizes.append([int((selected[b]==c).sum()) for c in range(3)])
        for c in range(3):
            group=selected[b]==c
            if group.sum()>1:
                v=reward[b,group];std=v.std(unbiased=False)
                if std>0:advantage[b,group]=(v-v.mean())/(std+1e-6)
    lc=current.log_softmax(-1);lr=reference.log_softmax(-1).detach()
    lcs=lc.gather(-1,selected[...,None]).squeeze(-1);lrs=lr.gather(-1,selected[...,None]).squeeze(-1)
    ratio=(lcs-lrs).exp()
    group=-torch.minimum(ratio*advantage,ratio.clamp(.9,1.1)*advantage).mean()
    delta=lrs-lcs;k3=(torch.expm1(delta)-delta).mean()
    kl=(lc[...,bank.supported].exp()*(lc[...,bank.supported]-lr[...,bank.supported])).sum(-1).mean()
    total=group+.001*k3
    if not (torch.isfinite(total) and torch.isfinite(kl)):raise FloatingPointError('nonfinite GRQA')
    terms=dict(total=total,group=group,paper_selected_k3_surrogate=k3,categorical_kl=kl.detach())
    info={}
    if diagnostics:
        regret=(current.detach().max(-1).values-reward).flatten();nonempty=[n for row in sizes for n in row if n]
        info=dict(routing_mode='cc' if cc else 'top1',supported_classes=bank.supported.nonzero().flatten().tolist(),
          status='DEGENERATE_SINGLE_SUPPORTED_CLASS' if int(bank.supported.sum())==1 else 'OK',
          query_utilization=[sum(row[c] for row in sizes) for c in range(3)],group_sizes=sizes,
          singleton_fraction=sum(n==1 for n in nonempty)/len(nonempty),min_group_size=min(nonempty),max_group_size=max(nonempty),
          reward_mean=float(reward.mean()),reward_std=float(reward.std(unbiased=False)),advantage_std=float(advantage.std(unbiased=False)),
          ratio_min=float(ratio.detach().min()),ratio_mean=float(ratio.detach().mean()),ratio_max=float(ratio.detach().max()),
          clip_fraction=float(((ratio.detach()<.9)|(ratio.detach()>1.1)).float().mean()),
          assignment_total_similarity=float(reward.sum()),assignment_regret_mean=float(regret.mean()),
          assignment_regret_p50=float(regret.quantile(.5)),assignment_regret_p95=float(regret.quantile(.95)),assignment_regret_max=float(regret.max()),
          selected_k3=float(k3.detach()),categorical_kl=float(kl.detach()))
    return terms,info


def components(model, image, label, bank, reference, arm, step, diagnostic=False, both=False):
    out=model(image)
    with torch.autocast(image.device.type,enabled=False):
        seg=supervised_query_loss(out['class_logits'].float(),out['mask_logits'].float(),label)['total']
        if arm=='QUERY_PREFIX':return seg,dict(Lseg=seg),None,None,{}
        current,support=image_prototypes(out['pixels'].float(),label,vit_pad=isinstance(model,DINOv2Adapter))
        img=image_alignment_loss(current,support,bank)
    with torch.no_grad():ref=reference(image)['queries'].float() if reference is not None else out['queries'].detach().float()
    terms=dict(Lseg=seg,Limg=img);info={};total=seg+10*img
    with torch.autocast(image.device.type,enabled=False):
        if arm!='A0' or diagnostic or both:
            gr,info=routed_grqa(out['queries'].float(),ref,bank,cc=arm in ('A3','A4'),diagnostics=diagnostic)
            terms.update(group=gr['group'],k3=gr['paper_selected_k3_surrogate'],categorical_kl=gr['categorical_kl'])
            terms['GRQA']=gr['total'];terms['regularizer']=weight(arm,step-1999)*gr['total']
            total=total+terms['regularizer']
        if both:
            cc,ci=routed_grqa(out['queries'].float(),ref,bank,cc=True,diagnostics=True)
            terms['GRQA_CC']=cc['total'];info={'original':info,'cc':ci}
    return total,terms,current,support,info


def gradient_interaction(terms, model, names):
    params=tuple(p for p in model.parameters() if p.requires_grad)
    grads={name:torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True) for name,loss in names.items()}
    norm={name:math.sqrt(sum(float(g.detach().float().square().sum()) for g in gs if g is not None)) for name,gs in grads.items()}
    result={'norm_'+k:v for k,v in norm.items()}
    for a,b in itertools.combinations(names,2):
        key='cos_'+a+'_'+b
        if not norm[a] or not norm[b]:result[key]=None;result[key+'_reason']='ZERO_GRADIENT_NORM'
        else:
            dot=sum(float((x.detach().float()*y.detach().float()).sum()) for x,y in zip(grads[a],grads[b]) if x is not None and y is not None)
            result[key]=dot/(norm[a]*norm[b])
    return result
