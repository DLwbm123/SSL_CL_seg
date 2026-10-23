"""Consistency-specific probes; no donor parameter/Adam/RNG writes."""
import copy
from pathlib import Path
import torch
from torch.nn import functional as F
from ..semantic_u_v20.diagnostic import parameters, groups, metrics
from ..semantic_u_v20.method import rng, components, posterior_probability, strong_view, consistency
from ..stream_state import StreamState, digest
from ..paired_semantic_v19.run import data


class Budget:
    def __init__(self):
        self.reverse = 0
        self.candidate_steps = 0

    def grad(self, loss, params, **kwargs):
        self.reverse += 1
        if self.reverse>128:
            raise RuntimeError('128 reverse limit')
        result=torch.autograd.grad(loss,params,allow_unused=True,**kwargs)
        assert all(v is None or torch.isfinite(v).all() for v in result)
        return result

    def step(self, optimizer):
        self.candidate_steps += 1
        if self.candidate_steps>24:
            raise RuntimeError('24 candidate Adam step limit')
        optimizer.step()


def batch(d,role,seed):
    b=d.batch(role,2,torch.Generator().manual_seed(seed));b['x']=b['x'].cuda()
    if 'observations' in b:
        b['observations']={k:v.cuda() for k,v in b['observations'].items()}
    return b


@torch.no_grad()
def samples(s,x,tag):
    values=[]
    for outer in range(s.config.global_mc):
        gen=rng(s,tag+outer);theta=s.q.rsample(gen);a=s.current.rsample(theta,gen)
        context=s.phi.context(x,theta,a);offsets=s.phi.initial_offsets(len(x),x)
        for local in range(s.config.local_mc):
            noise=s.phi.noise(len(x),x,rng(s,tag+10+outer*3+local))
            z,_=s.phi.rsample_and_log_prob(context,noise,offsets,s.model,theta)
            values.append(s.model.class_probabilities(z,theta))
    return torch.stack(values)


def rate(n,d):
    return float(n/d) if int(d)>0 else None


def quality(draws,other,truth=None):
    mean=draws.mean(0);sd=draws.std(0,unbiased=False);label=mean.argmax(1)
    confidence=mean.max(1).values
    score=confidence-sd.gather(1,label[:,None]).squeeze(1)
    accepted=score>=.9
    other_mean=other.mean(0);other_label=other_mean.argmax(1)
    other_score=other_mean.max(1).values-other.std(0,unbiased=False).gather(1,other_label[:,None]).squeeze(1)
    other_accepted=other_score>=.9
    result=dict(pixels=label.numel(),entropy=float(-(mean*mean.clamp_min(1e-12).log()).sum(1).mean()),probability_variance=float(draws.var(0,unbiased=False).mean()),sample_class_disagreement=float((draws.argmax(2)!=label[None]).double().mean()),mask_agreement=float((accepted==other_accepted).double().mean()),classes={})
    for k,name in [(0,'background'),(1,'foreground')]:
        region=label==k;mask=region & accepted
        other_mask=(other_label==k)&other_accepted
        union=(mask|other_mask).sum()
        result['classes'][name]=dict(predicted_pixels=int(region.sum()),accepted_pixels=int(mask.sum()),coverage=rate(mask.sum(),region.sum()),accepted_class_jaccard=rate((mask&other_mask).sum(),union))
    if truth is not None:
        gt=truth[:,1].bool();pred=label==1;tp=(gt&pred).sum()
        wrong=label!=gt.long();high=confidence>=.9
        dilation=F.max_pool2d(gt[:,None].double(),7,1,3)[:,0]>0
        erosion=1-F.max_pool2d((~gt)[:,None].double(),7,1,3)[:,0]
        boundary=dilation & (erosion<1)
        result['label_audit']=dict(precision=rate(tp,pred.sum()),recall=rate(tp,gt.sum()),gt_foreground_pixels=int(gt.sum()),pred_foreground_pixels=int(pred.sum()),high_confidence_pixels=int(high.sum()),high_confidence_errors=int((high&wrong).sum()),high_confidence_error_rate=rate((high&wrong).sum(),high.sum()),boundary_pixels=int(boundary.sum()),boundary_errors=int((boundary&wrong).sum()),boundary_error_rate=rate((boundary&wrong).sum(),boundary.sum()),interior_error_rate=rate(((~boundary)&wrong).sum(),(~boundary).sum()),accepted_foreground_precision=rate((accepted & pred & gt).sum(),(accepted&pred).sum()))
    return result


def run(p,root,save_progress):
    donor=Path(p['v19_root']);p1=Path(p['p1_root']);p2=Path(p['p2_root'])
    paths=[('L400',donor/'results/s20261023_L_A_attempt001/ACTIVE_STEP_000400.pt'),('A800',p1/'results/P1_A_attempt001/ACTIVE_STEP_000800.pt'),('C800',p1/'results/P1_C_attempt001/ACTIVE_STEP_000800.pt'),('A1600',p2/'results/s20261023_A_attempt001/ACTIVE_STEP_001600.pt'),('C1600',p2/'results/s20261023_C_attempt001/ACTIVE_STEP_001600.pt')]
    budget=Budget();rows=[]
    for name,path in paths:
        s=StreamState.load(path).to_device('cuda');original=digest(s.payload());active=copy.deepcopy(s.active)
        d=data(p,s.config,'RUNMC','LU');s.active['evidence']=copy.deepcopy(d.evidence);probe_payload=digest(s.payload())
        for index in range(2):
            L,U,probe=batch(d,'L',9410+index),batch(d,'U',9420+index),batch(d,'L',9430+index)
            # Predict using x alone, before passing original L truth to the audit.
            ldraw=samples(s,L['x'],70);lother=samples(s,L['x'],170)
            udraw=samples(s,U['x'],70);uother=samples(s,U['x'],170)
            with torch.no_grad():torch.testing.assert_close(udraw.mean(0),posterior_probability(s,U['x']))
            lquality=quality(ldraw,lother,L['observations']['y_obs']);uquality=quality(udraw,uother)
            terms=components(s,L,None,d.counts);base=terms['task']+terms['image_L']+terms['kl']
            student=posterior_probability(s,strong_view(s,U['x']))
            sem,details=consistency(udraw.mean(0),student)
            ps=parameters(s)
            tg=budget.grad(terms['task'],ps,retain_graph=True);ag=budget.grad(base,ps);sg=budget.grad(sem,ps)
            tt,ss,aa=groups(s,tg),groups(s,sg),groups(s,ag);gr={}
            for key in tt:
                t,u,a=tt[key],ss[key],aa[key];nt,nu,na=t.norm(),u.norm(),a.norm()
                gr[key]=dict(task_norm=float(nt),semantic_norm=float(nu),A_norm=float(na),semantic_task_ratio=float(nu/nt) if nt>1e-12 else None,semantic_task_cosine=float(torch.dot(t,u)/(nt*nu)) if nt>1e-12 and nu>1e-12 else None,semantic_A_ratio=float(nu/na) if na>1e-12 else None)
            scalars=dict(task=float(terms['task'].detach()),A=float(base.detach()),semantic=float(sem.detach()),**{k:float(v) for k,v in details.items()})
            del terms,base,sem,student
            candidate={'unchanged':metrics(s,probe)}
            for use_sem in (False,True):
                clone=copy.deepcopy(s);clone.optimizer.zero_grad(set_to_none=True)
                for j,param in enumerate(parameters(clone)):
                    g=ag[j]
                    if use_sem and sg[j] is not None:g=sg[j] if g is None else g+sg[j]
                    param.grad=None if g is None else g.detach().clone()
                budget.step(clone.optimizer);candidate['A_sem' if use_sem else 'A']=metrics(clone,probe)
                del clone
            assert digest(s.payload())==probe_payload and all(v.grad is None for v in ps)
            rows.append(dict(state=name,batch_index=index,L_teacher=lquality,U_teacher=uquality,terms=scalars,gradient_groups=gr,adam_candidates=candidate))
            save_progress(dict(completed_probes=len(rows),reverse_calls=budget.reverse,candidate_steps=budget.candidate_steps))
        s.active=active;assert digest(s.payload())==original
        del s,d
    return dict(status='COMPLETE',rows=rows,reverse_calls=budget.reverse,reverse_limit=128,candidate_steps_discarded=budget.candidate_steps,candidate_step_limit=24,source_updates=0,payload_unchanged=True,training_only=True,validation_scoring_calls=0,teacher_labels_used_for_prediction=False)
