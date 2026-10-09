"""Test early source-memory protection against a matched selected-EMA prefix."""
import fcntl
import importlib.util
import json
import os
import random
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch

METHODS=('EARLY_MEMORY','EARLY_SELECTED_EMA')
CAPS=dict(qualification=108,prior=19200)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def configure(cfg):
    global H,N,c,b,S,A
    H=load('historical',str(Path(cfg['v124'])/'run.py'));H.configure(cfg)
    N=H.N;c=H.c;b=H.b;S=H.S;A=load('analysis_new',cfg['readout_entry']);N.STAGE='V127';c.CAPS.update(CAPS)


def target_for(mode,step,target,q):
    assert mode in METHODS
    return q.detach() if mode=='EARLY_SELECTED_EMA' and step<100 else target.detach()


def selfcheck():
    H.selfcheck();A.selfcheck()
    p=torch.tensor([.2,.3,.5]).reshape(1,3,1,1).requires_grad_();q=torch.tensor([.8,.1,.1]).reshape_as(p).requires_grad_();target=torch.tensor([.2,.7,.1]).reshape_as(p).requires_grad_()
    valid=torch.ones_like(p[:,0],dtype=torch.bool)
    for mode in METHODS:
        for step in (0,99,100,299):
            out=target_for(mode,step,target,q);assert not out.requires_grad
            assert torch.equal(out,q if mode=='EARLY_SELECTED_EMA' and step<100 else target)
            loss=S.selected_loss(p,q,out,valid,valid);g=torch.autograd.grad(loss,(p,q,target),allow_unused=True)
            assert torch.isfinite(g[0]).all() and g[1:]==(None,None)
    return dict(status='PASS',checks=['inherited native selection and detach','early-only target routing','synthetic absolute and matched-control gates'],model_forwards=0,image_reads=0,updates=0)


def worker(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    torch.set_num_threads(2);torch.cuda.set_device(0)
    original=N.E.install_actions(c);N.E.action_check(c,original);transfer=c.transfer_loss
    roles=c.split_roles(cfg['data']);assert roles==N.D.read(Path(cfg['action_root'])/'ROLES.private.json')
    ctx=H.A.contexts()[cfg['context_index']];stream=cfg['stream'];r=H.fold_roles(roles,ctx['fold'],ctx['labeled_images']);condition=(ctx['condition'],ctx['factor'])
    ledger=b.JobLedger(root,cfg['caps']);counts=Counter();allowed=set(r['A_fit']);u_allowed=True
    sample=c.e.primitive.CurrentData.__getitem__
    def guarded(ds,i):
        assert ds.role in ('train_labeled','train_unlabeled')
        assert ds.rows[i]['case_id'] in (allowed if ds.role=='train_labeled' else set(r['U_adapt']) if u_allowed else set())
        counts['image_attempts_'+ds.role]+=1;out=sample(ds,i);counts['image_success_'+ds.role]+=1;return out
    c.e.primitive.CurrentData.__getitem__=guarded
    def seed():
        random.seed(123000+stream);np.random.seed(123000+stream);torch.manual_seed(123000+stream);torch.cuda.manual_seed_all(123000+stream)
    seed();t=b.make(cfg,r,ledger,(ctx['source_step'],ctx['labeled_images'],condition,'heldout'),300);t.provider.seed=10168+10000*stream;seed()
    initial=c.snapshot(t);initial_memory=c.e.cpu(t.memory.state_dict());profile={'mode':'WARM'}
    for name,model in [('student',t.model),('ema',t.ema),('memory',t.memory)]:
        def hook(module,args,name=name):counts[name+'_image_forwards']+=len(args[0])
        model.register_forward_pre_hook(hook)
    clean=t.clean
    def counted_clean(x):
        counts['clean_student_forward_attempts']+=len(x);out=clean(x);counts['clean_student_forward_success']+=len(x);return out
    t.clean=counted_clean
    step=t.optimizer.step
    def rate_step(*args,**kwargs):
        rates=[g['lr'] for g in t.optimizer.param_groups];scale=1. if t.step<100 else .25
        counts['rate_records']+=1;c.e.append(root/'RATE_LEDGER.jsonl',dict(ordinal=counts['rate_records'],key=t.key,category=t.category,step=t.step,scale=scale,base=rates,effective=[a*scale for a in rates]))
        try:
            for g,lr in zip(t.optimizer.param_groups,rates):g['lr']=lr*scale
            return step(*args,**kwargs)
        finally:
            for g,lr in zip(t.optimizer.param_groups,rates):g['lr']=lr
    rate_step._wrapped_by_lr_sched=True;t.optimizer.step=rate_step
    def selected(p,q,valid,action,qm=None,qflip=None):
        if profile['mode']=='WARM':assert action==0;return transfer(p,q,valid,action,qm,qflip)
        assert action==11
        _,stats,target=transfer(p,q,valid,action,qm,qflip)
        features,eligible,classes=S.features(p,q,qm,qflip,target,valid,t.step);mask,selection=S.choose(features,eligible,classes,'COVERAGE',None,None)
        out=target_for(profile['mode'],t.step,target,q);loss=S.selected_loss(p,q,out,valid,mask)
        counts['selector_calls']+=1;c.e.append(root/'SELECTION_LEDGER.jsonl',dict(ordinal=counts['selector_calls'],key=t.key,step=t.step,mode=profile['mode'],target='EMA' if profile['mode']=='EARLY_SELECTED_EMA' and t.step<100 else 'MIXED_MEMORY',image_budgets=[(v['eligible'],v['selected']) for v in selection]))
        return loss,stats,out
    c.transfer_loss=selected
    def freeze_check():
        assert c.e.same(initial_memory,t.memory.state_dict()) and all(p.grad is None for model in (t.memory,t.ema) for p in model.parameters())
    campaign=Path(cfg['campaign']);job=cfg['job']
    if job=='qualification':
        t.action=0;t.category='qualification';t.key='HISTORICAL_WARM';profile['mode']='WARM'
        for _ in range(100):t.update()
        historical=torch.load(Path(cfg['v123'])/'jobs/entry0_3/ENTRY100.private.pt',map_location='cpu',weights_only=False)
        assert c.e.same(historical,c.snapshot(t)), 'historical warm100 exact state mismatch'
        for mode in METHODS:
            expected=None
            for repeat in range(2):
                c.restore(t,initial);t.action=11;t.category='qualification';t.key=mode;profile['mode']=mode
                for _ in range(2):t.update()
                freeze_check();now=c.snapshot(t)
                if repeat==0:
                    c.e.atomic_save(now,root/(mode+'_CHECK.private.pt'));expected=torch.load(root/(mode+'_CHECK.private.pt'),map_location='cpu',weights_only=False)
                else:assert c.e.same(expected,now)
        assert dict(ledger.count)==dict(qualification=108)
        N.D.write(root/'QUALIFICATION.json',dict(status='PASS',historical_entry100_exact=True,both_new_checkpoint_replays_exact=True,frozen_memory=True,teacher_gradients_none=True,updates=108))
    elif job=='train':
        assert N.D.read(campaign/'jobs/qualification/QUALIFICATION.json')['status']=='PASS'
        for mode in METHODS:
            c.restore(t,initial);t.action=11;t.category='prior';t.key=f"ctx{ctx['context']}/stream{stream}/{mode}";profile['mode']=mode
            for _ in range(300):
                t.update()
                if t.step in (100,150,300):c.e.atomic_save(c.snapshot(t),root/f'{mode}_{t.step}.private.pt')
                if t.step%50==0:N.B.write(root/'STATUS.json',dict(status='RUNNING',mode=mode,step=t.step,physical=dict(ledger.count)))
            freeze_check();N.D.write(root/(mode+'_TRAINING.json'),dict(status='SEALED',updates=300,time=time.time()))
        assert dict(ledger.count)==dict(prior=600)
    elif job=='evaluate':
        assert N.D.read(campaign/'ENDPOINT_LOCK.json')['snapshots']==192
        u_allowed=False;qr=dict(r,Q_train_new=r['A_hold']);rows=[]
        for mode in METHODS:
            for at in (100,150,300):
                state=torch.load(campaign/f"jobs/train{ctx['context']}_{stream}/{mode}_{at}.private.pt",map_location='cpu',weights_only=False);c.restore(t,state);out={}
                for role,cond in [('Q_train_new',condition),('Q_train_old',('identity',1.))]:
                    allowed=set(qr[role]);counts['query_attempts']+=1;c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='attempt',ordinal=counts['query_attempts'],method=mode,step=at,role=role))
                    out[role]=scores(t,qr,role,cond);counts['query_success']+=1;c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='success',ordinal=counts['query_success']))
                assert c.e.same(state,c.snapshot(t))
                new=out['Q_train_new'];old=out['Q_train_old'];rows.append(dict(ctx,stream=stream,method=mode,step=at,new=new['macro'],old=old['macro'],new_rim=new['rim'],new_cup=new['cup'],old_rim=old['rim'],old_cup=old['cup']))
        assert not ledger.count and counts['query_attempts']==counts['query_success']==12 and counts['clean_student_forward_success']==48
        N.D.write(root/'RESULTS.json',rows)
    else:raise ValueError(job)
    N.D.write(root/'COUNTS.json',dict(counts));N.D.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count),time=time.time()))


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert N.D.read(Path(cfg['v126'])/'COMPLETION_AUDIT.json')['status']=='PASS'
    assert N.D.read(Path(cfg['v126'])/'FINAL_PUBLICATION.json')['anonymous_http']=='200'
    (root/'jobs').mkdir();(root/'PHYSICAL_LEDGER.jsonl').touch()
    qual=[dict(id='qualification',job='qualification',context_index=0,stream=3,caps=dict(qualification=108))]
    jobs=lambda kind,caps:[dict(id=f'{kind}{i}_{s}',job=kind,context_index=i,stream=s,caps=caps) for i in range(16) for s in (3,4)]
    train=jobs('train',dict(prior=600));evaluate=jobs('evaluate',{});account=Accounting(root,caps=CAPS,jobs=qual+train+evaluate)
    N.schedule(root,cfg,qual,account);assert N.D.read(root/'jobs/qualification/QUALIFICATION.json')['status']=='PASS'
    N.schedule(root,cfg,train,account);assert dict(account.count)==CAPS and not account.failure
    assert all((root/'jobs'/j['id']/f'{m}_{t}.private.pt').exists() for j in train for m in METHODS for t in (100,150,300))
    N.D.write(root/'ENDPOINT_LOCK.json',dict(status='SEALED_BEFORE_PERFORMANCE_READOUT',snapshots=192,trajectories=64,time=time.time()))
    N.schedule(root,cfg,evaluate,account)
    rows=sum((N.D.read(root/'jobs'/j['id']/'RESULTS.json') for j in evaluate),[])
    N.D.write(root/'NEW_RESULTS.json',rows)
    for name,value in A.analyze(rows,N.D.read(Path(cfg['v126'])/'RESULTS.json')).items():N.D.write(root/(name+'.json'),value)
    N.D.table(root/'RESULTS.csv',N.D.read(root/'RESULTS.json'))
    counts=Counter()
    for j in qual+train+evaluate:counts.update(N.D.read(root/'jobs'/j['id']/'COUNTS.json'))
    assert counts['query_success']==counts['query_attempts']==384 and counts['clean_student_forward_attempts']==counts['clean_student_forward_success']==1536
    N.B.write(root/'COSTS.json',dict(native_updates=sum(account.count.values()),physical=dict(account.count),success=dict(account.success),failures=dict(account.failure),counts=dict(counts),actor_updates=0,linear_solves=0,new_annotations=0,Q_dev=0,performance_query_calls=384,performance_query_images=1536))
    N.D.write(root/'FINAL.json',dict(status='COMPLETE',decision=N.D.read(root/'DECISION.json')['status'],time=time.time()));N.B.write(root/'STATUS.json',N.D.read(root/'FINAL.json'))


if __name__=='__main__':
    root=Path(os.environ['EXEC_RUN']);cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());configure(cfg)
    if os.environ.get('EXEC_SELFCHECK')=='1':print(json.dumps(selfcheck()))
    else:
        N.D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':worker(root,cfg)
            else:coordinator(root,cfg)
        except BaseException as exc:N.D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
