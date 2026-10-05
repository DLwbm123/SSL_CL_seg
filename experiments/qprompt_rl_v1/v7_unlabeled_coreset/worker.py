"""Select U images while retaining the established native student update path."""
import copy
import fcntl
import math
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.qprompt_rl_v1.v5_grpo_group_control import worker as w
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e, runner as io
from experiments.lcrseg.five_frameworks_v1.native_data import NativeCurrentDomain
from experiments.lcrseg.five_frameworks_v1.recipes import generator
from . import controller as q, protocol as p

ROOT,C=io.ROOT,io.C
COST=w.COST


class SubsetView:
    def __init__(self,original,indices):
        self.original,self.indices=original,list(indices)
        self.rows=[original.rows[i] for i in indices];self.role=original.role
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):return self.original[self.indices[i]]


class Provider(e.Provider):
    def __init__(self,*args,development=False,**kwargs):
        super().__init__(*args,development=False,**kwargs)
        self.development=True
        patients=sorted(set(self._patients.values()))
        order=np.random.RandomState(e.stable('V7/roles',p.ROLE_SEED,self.domain)).permutation(len(patients))
        held={patients[i] for i in order[:p.REWARD_LABELS[self.domain]]}
        assert len(patients)==p.LABELS[self.domain]
        self.roles=dict(fit=sorted(set(patients)-held),online=sorted(held),audit=[])
        original=self._l
        for role,ids in self.roles.items():
            ds=copy.copy(original);ds.rows=[r for r in original.rows if self._patients[r['case_id']] in ids];ds.checked=set()
            if role=='fit':self._l=ds
            else:self.feedback[role]=ds
        assert len(self._u)==p.UNLABELED[self.domain] and len(self._l)>=2
        self.selected=list(range(len(self._u)));self.exposure=[0]*len(self._u)
    def semantic_metadata(self):
        return dict(super().semantic_metadata(),role_seed=p.ROLE_SEED,roles=self.roles)
    def unlabeled(self,step):
        assert len(self.selected)>=2
        self.u_reads+=1;view=SubsetView(self._u,self.selected)
        epoch,index=divmod(step,self.steps_per_epoch);cycle,offset=divmod(2*index,len(view))
        order=torch.randperm(len(view),generator=generator(self.seed,self.order,self.stage,epoch,f'unlabeled/order/{cycle}')).tolist()
        for i in (order[offset],order[(offset+1)%len(view)]):self.exposure[self.selected[i]]+=1
        return self._batch(view,step,'unlabeled')


def snapshot(t):
    return dict(state=w.snapshot(t),selected=list(t.provider.selected),exposure=list(t.provider.exposure))


def restore(t,s):
    w.restore(t,s['state']);t.provider.selected=list(s['selected']);t.provider.exposure=list(s['exposure'])


@contextmanager
def preserve(t):
    extra=w.extra(t);selected=list(t.provider.selected);exposure=list(t.provider.exposure)
    try:
        with t.readonly():yield
    finally:
        w.restore_extra(t,extra);t.provider.selected=selected;t.provider.exposure=exposure


def create(ledger):
    torch.manual_seed(168);torch.cuda.manual_seed_all(168);np.random.seed(168);e.random.seed(168)
    e.Provider=Provider
    t=e.create(C,168,C['domain'],True,ledger)
    e.write(ROOT/'ROLES.private.json',t.provider.roles)
    return t


def image(t,i):
    item=t.provider._u[i];assert 'label' not in item,'U label access'
    COST['U_image_accesses']+=1
    return item['image'][None].to(t.provider.device)


def anchors(t):
    ids=np.random.RandomState(e.stable('V7/anchors',C['domain'])).permutation(len(t.provider._u))[:8]
    with preserve(t),torch.no_grad():
        result=[(int(i),t.clean(image(t,int(i))).detach().cpu()) for i in ids]
    COST['source_anchor_forward_images']+=len(result)
    return result


def kl(current,source):
    source=source.to(current);mask=source.max(1).values>.7
    return ((source*(source.clamp_min(1e-8).log()-current.clamp_min(1e-8).log())).sum(1)*mask).mean()


def objective_terms(t,anchor):
    ds=t.provider.feedback['online']
    for i in range(len(ds)):
        b=ds[i];COST['reward_label_sample_accesses']+=1
        yield e.quality(t.clean(b['image'][None].to(t.provider.device)),b['label'][None].to(t.provider.device))/len(ds)
    for i,source in anchor:
        COST['retention_forward_images']+=1
        yield -.1*kl(t.clean(image(t,i)),source)/len(anchor)


def objective(t,anchor):
    with preserve(t),torch.no_grad(),w.measured('reward'):
        value=sum(float(v) for v in objective_terms(t,anchor))
    assert math.isfinite(value)
    return value


def features(t):
    rows,emb=[],[]
    with preserve(t),torch.no_grad(),w.measured('features'):
        t.model.eval();t.ema.eval()
        for i in range(len(t.provider._u)):
            x=image(t,i);h=t.model.parent.features(x,'eval')
            pred=t.model.parent.native_readout(h,x.shape[-2:],'eval').softmax(1)
            teacher=t.ema(x,mode='teacher').softmax(1)
            conf,cls=teacher.max(1);fg=cls>0;mix=(teacher+pred)/2
            js=((teacher*(teacher.clamp_min(1e-8).log()-mix.clamp_min(1e-8).log())).sum(1)+(pred*(pred.clamp_min(1e-8).log()-mix.clamp_min(1e-8).log())).sum(1))/2/math.log(2)
            loss=e.u_loss(pred,teacher,torch.ones_like(cls,dtype=torch.bool),2)[0]
            rows.append([float(conf[fg].mean() if fg.any() else conf.mean()),float((conf>.7).float().mean()),float(-(teacher*teacher.clamp_min(1e-8).log()).sum(1).mean()/math.log(3)),float((cls==1).float().mean()),float((cls==2).float().mean()),float(teacher[:,1:].sum(1).mean()),float(js.mean()),float((cls!=pred.argmax(1)).float().mean()),math.log1p(float(loss)),t.step/p.DOMAINS[C['domain']],t.optimizer.param_groups[0]['lr']/t.scheduler.base_lrs[0],t.provider.exposure[i]/max(1,max(t.provider.exposure))])
            emb.append(F.normalize(h.mean((-2,-1)),dim=1).cpu()[0])
    COST['feature_image_pairs']+=len(rows)
    x=torch.tensor(rows);embedding=torch.stack(emb)
    assert x.shape==(len(rows),q.FEATURES) and torch.isfinite(x).all() and torch.isfinite(embedding).all()
    return x,embedding


def flatgrad(loss,params):
    COST['VJP']+=1
    gradients=torch.autograd.grad(loss,params,allow_unused=True)
    return torch.cat([(torch.zeros_like(v) if g is None else g).reshape(-1) for v,g in zip(params,gradients)]).detach()


def retrieve(t,anchor,k):
    """Eq. 6/7-style first-order selection; clean proxy and native parameter-group LR."""
    params=[v for v in t.model.parameters() if v.requires_grad]
    saved=[v.detach().clone() for v in params];base=torch.cat([v.reshape(-1) for v in saved])
    rates={id(v):group['lr'] for group in t.optimizer.param_groups for v in group['params']}
    lr=torch.cat([torch.full_like(v,rates[id(v)]).reshape(-1) for v in params])
    with preserve(t),w.measured('retrieve'):
        try:
            t.model.eval();t.ema.eval();ug=[]
            for i in range(len(t.provider._u)):
                x=image(t,i)
                with torch.no_grad():teacher=t.ema(x,mode='teacher').softmax(1)
                pred=t.clean(x);ug.append(flatgrad(e.u_loss(pred,teacher,torch.ones_like(pred[:,0],dtype=torch.bool),2)[0],params))
            ug=torch.stack(ug);lg=torch.zeros_like(base);ds=t.provider._l
            for i in range(len(ds)):
                b=ds[i];COST['retrieve_fit_label_sample_accesses']+=1
                loss=-e.quality(t.clean(b['image'][None].to(t.provider.device)),b['label'][None].to(t.provider.device))
                lg+=flatgrad(loss,params)/len(ds)
            chosen=[];acc=torch.zeros_like(base)
            for _ in range(k):
                virtual=base-lr*(lg+.5*acc/k);offset=0
                with torch.no_grad():
                    for v in params:
                        v.copy_(virtual[offset:offset+v.numel()].reshape_as(v));offset+=v.numel()
                outer=torch.zeros_like(base)
                for term in objective_terms(t,anchor):outer+=flatgrad(-term,params)
                gain=ug@(lr*outer);gain[chosen]=-torch.inf
                idx=int(gain.argmax());chosen.append(idx);acc+=ug[idx]
            assert len(set(chosen))==k
            return chosen
        finally:
            with torch.no_grad():
                for v,s in zip(params,saved):v.copy_(s)


def choose(t,method,controller,block,policy=None,anchor=None):
    n=len(t.provider._u);k=p.k(C['domain'])
    if method=='NO_U':return []
    if method=='ALL_U':return list(range(n))
    if method=='RANDOM':return np.random.RandomState(e.stable('V7/random',controller,C['domain'],block)).permutation(n)[:k].tolist()
    if method=='RETRIEVE_FO':return retrieve(t,anchor,k)
    x,emb=features(t)
    if method=='CONFIDENCE':return sorted(range(n),key=lambda i:(-float(x[i,0]),i))[:k]
    if method=='GRPO':
        with torch.no_grad():return q.sequence(policy,x,emb,k,greedy=True)[0]
    assert method=='KCENTER'
    chosen=[int(((emb-emb.mean(0))**2).sum(1).argmax())]
    while len(chosen)<k:
        distance=1-(emb@emb[chosen].T).max(1).values;distance[chosen]=-torch.inf;chosen.append(int(distance.argmax()))
    return chosen


def advance(t,ledger,category,key,selected,steps):
    t.provider.selected=sorted(selected)
    w.advance(t,ledger,category,key,2 if selected else 0,steps)


def save(t,path,**extra):
    e.atomic_save(dict(state=snapshot(t),commit=C['commit'],**extra),path)


def qualification(ledger):
    t=create(ledger);entry=snapshot(t);anchor=anchors(t)
    x,emb=features(t);objective(t,anchor);selected=retrieve(t,anchor,p.k(C['domain']))
    assert e.same(entry,snapshot(t)),'diagnostic or virtual selection mutated native state'
    assert len(set(selected))==p.k(C['domain'])
    # Full view equals original loader exactly, including augmentation streams.
    with preserve(t):
        actual=t.provider.unlabeled(0);expected=NativeCurrentDomain.unlabeled(t.provider,0)
        assert e.same(actual,expected),'ALL_U is not native batch-equivalent'
    assert e.same(entry,snapshot(t))
    for method,indices in [('ALL_U',list(range(len(t.provider._u)))),('SUBSET',selected),('NO_U',[])]:
        restore(t,entry);advance(t,ledger,'smoke',method+'/first',indices,2);expected=snapshot(t)
        restore(t,entry);save(t,ROOT/'resume.private.pt')
        loaded=torch.load(ROOT/'resume.private.pt',map_location='cpu',weights_only=False);restore(t,loaded['state'])
        advance(t,ledger,'smoke',method+'/resume',indices,2)
        assert e.same(expected,snapshot(t)),method+' continuation differs'
        assert all(math.isfinite(float(t.last[k])) for k in ('labeled_loss',))
    e.write(ROOT/'QUALIFICATION.json',dict(status='PASS',native_calls=12,all_U_batch_parity=True,exact_checkpoint_continuation=True,feature_reward_virtual_state_isolation=True,unlabeled_label_absent=True,fit=len(t.provider._l),reward=len(t.provider.feedback['online']),U=len(t.provider._u),selected=p.k(C['domain'])))


def learn(ledger):
    t=create(ledger);anchor=anchors(t);student_rng=e.rng_state()
    torch.manual_seed(C['controller']);model=q.Policy();opt=torch.optim.Adam(model.parameters(),lr=.001);e.rng_restore(student_rng)
    diagnostics=[];k=p.k(C['domain'])
    for block in range(p.DOMAINS[C['domain']]//p.BLOCK):
        entry=snapshot(t);x,emb=features(t);before=objective(t,anchor);samples=[];rewards=[];branch0=None
        behavior=e.cpu(model.state_dict())
        for branch in range(p.GROUP):
            g=torch.Generator().manual_seed(e.stable('V7/group',C['controller'],C['domain'],block,branch))
            with torch.no_grad():actions,old,_=q.sequence(model,x,emb,k,generator=g)
            restore(t,entry);advance(t,ledger,'development',f'{block}/{branch}',actions,p.BLOCK)
            rewards.append(objective(t,anchor)-before);samples.append((actions,old))
            if branch==0:branch0=snapshot(t)
        assert e.same(behavior,e.cpu(model.state_dict()))
        e.atomic_save(dict(block=block,entry=entry,branch0=branch0,x=x,embeddings=emb,samples=samples,rewards=rewards,policy=behavior,optimizer=e.cpu(opt.state_dict()),anchors=anchor),ROOT/'group_latest.private.pt')
        restore(t,branch0);student=snapshot(t)
        diag=q.update(model,opt,x,emb,samples,rewards,k,e.stable('V7/update',C['controller'],C['domain'],block),lambda optimizer,key:ledger.step(optimizer,'actor',f'{block}/{key}'))
        assert e.same(student,snapshot(t)),'policy update mutated student'
        diagnostics.append(dict(block=block,unique_subsets=len({tuple(sorted(s[0])) for s in samples}),**diag))
        e.write(ROOT/'GROUP_DIAGNOSTICS.json',diagnostics)
        save(t,ROOT/'student_latest.private.pt',block=block,policy=e.cpu(model.state_dict()),optimizer=e.cpu(opt.state_dict()),anchors=anchor)
        w.v4.status('LEARNING',block=block+1,blocks=p.DOMAINS[C['domain']]//p.BLOCK,physical_calls=dict(ledger.count))
    e.atomic_save(dict(state=e.cpu(model.state_dict()),controller=C['controller'],domain=C['domain'],commit=C['commit']),ROOT/'FROZEN_POLICY.private.pt')


def endpoint(ledger):
    t=create(ledger);save(t,ROOT/'ENTRY.private.pt');io.export(t,ROOT/'entry.pt')
    m,c=C['method'],C.get('controller');model=None;anchor=anchors(t) if m=='RETRIEVE_FO' else None
    if m=='GRPO':
        prior=e.rng_state();model=q.Policy()
        raw=torch.load(Path(C['campaign'])/'jobs'/f'learn_{C["domain"]}_{c}'/'FROZEN_POLICY.private.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(raw['state']);model.eval();e.rng_restore(prior);policy_before=e.cpu(model.state_dict())
    started=time.time();history=[]
    for block in range(p.DOMAINS[C['domain']]//p.BLOCK):
        selected=choose(t,m,c,block,model,anchor);previous=set(history[-1]) if history else set()
        e.append(ROOT/'SELECTIONS.private.jsonl',dict(block=block,indices=selected))
        current=set(selected);union=current|previous
        e.append(ROOT/'SELECTION_SUMMARY.jsonl',dict(block=block,size=len(selected),previous_jaccard=len(current&previous)/len(union) if union else None))
        history.append(selected);advance(t,ledger,'endpoint',f'{m}/{block}',selected,p.BLOCK)
        save(t,ROOT/'student_latest.private.pt',method=m,controller=c)
        w.v4.status('ENDPOINT_TRAINING',method=m,controller=c,step=t.step,physical_calls=dict(ledger.count))
    if model is not None:assert e.same(policy_before,e.cpu(model.state_dict()))
    path=ROOT/'endpoint.pt';io.export(t,path)
    e.write(ROOT/'ENDPOINT_REGISTRY.json',[dict(seed=168,domain=C['domain'],method=m,controller=c,step=t.step,path=str(path),training_wall_seconds=time.time()-started,selected_union=len(set(i for h in history for i in h)),U_size=len(t.provider._u),actual_U_image_accesses=sum(t.provider.exposure))])
    e.write(ROOT/'FREEZE.json',dict(status='FROZEN',time=time.time(),controller_updates=0,reward_label_sample_accesses=COST['reward_label_sample_accesses'],commit=C['commit']))
    if m!='RETRIEVE_FO':assert COST['reward_label_sample_accesses']==0


def main():
    torch.set_num_threads(2);torch.cuda.set_device(0)
    lock=(ROOT/'EXECUTOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'PHYSICAL_LEDGER.jsonl').exists(),'no automatic retry'
    ledger=w.v4.Ledger(ROOT);ledger.caps=C['caps'];ledger.seen=set()
    w.v4.status('STARTING',job=C['job'])
    try:
        dict(qualification=qualification,learn=learn,endpoint=endpoint,evaluate=w.evaluate)[C['job']](ledger)
        w.audit(ledger)
        e.write(ROOT/'FINAL.json',dict(status='COMPLETE',time=time.time(),physical_calls=dict(ledger.count),peak_cuda_allocated=torch.cuda.max_memory_allocated(),commit=C['commit']))
        w.v4.status('COMPLETE',job=C['job'])
    except BaseException as exc:
        w.v4.status('FAILED',error=repr(exc),traceback=traceback.format_exc(),physical_calls=dict(ledger.count));raise
    finally:e.write(ROOT/'MEASURED_COSTS.json',dict(COST))


if __name__=='__main__':
    with w.NativeOperations(ROOT/'operations'):main()
