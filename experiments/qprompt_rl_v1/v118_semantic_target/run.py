"""Train-only closed-loop semantic target reassignment with matched controls."""
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

METHODS=('BASE','SEMANTIC','ROTATED')
STREAMS=(3,4)
SNAPSHOTS=(150,200,250,300)
CAPS=dict(qualification=12,prior=9600)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def configure(cfg,root):
    global Q,N,c,b,S,A,P
    Q=load('quarter',cfg['quarter_entry']);Q.configure(cfg,root);N=Q.N;c=N.c;b=N.b
    N.STAGE='V118';N.CAPS=CAPS;c.CAPS.update(CAPS)
    S=load('selector',cfg['selector_entry'])
    A=load('analysis118',cfg['analysis_entry']);P=load('target118',cfg['target_entry'])


def worker(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    torch.set_num_threads(2);torch.cuda.set_device(0);random.seed(168);np.random.seed(168);torch.manual_seed(168)
    original=N.E.install_actions(c);N.E.action_check(c,original)
    roles=c.split_roles(cfg['data']);assert roles==N.D.read(Path(cfg['action_root'])/'ROLES.private.json')
    counts=Counter();ledger=b.JobLedger(root,cfg['caps']);campaign=Path(cfg['campaign']);profile={};active={};region_ordinal=0
    allowed=set(roles['A_fit']);sample=c.e.primitive.CurrentData.__getitem__
    def guarded(ds,i):
        assert ds.role in ('train_labeled','train_unlabeled') and ds.rows[i]['case_id'] in (allowed if ds.role=='train_labeled' else set(roles['U_adapt']))
        counts['item_attempts_'+ds.role]+=1;value=sample(ds,i);counts['item_success_'+ds.role]+=1;return value
    c.e.primitive.CurrentData.__getitem__=guarded
    original_loss=c.Trainer.losses;transfer=c.transfer_loss
    def losses(t):active['trainer']=t;return original_loss(t)
    def selected(p,q,valid,action,qm=None,qflip=None):
        nonlocal region_ordinal
        assert action==11 and qm is not None and qflip is not None
        _,stats,target=transfer(p,q,valid,action,qm,qflip);t=active['trainer']
        x,eligible,classes=S.features(p,q,qm,qflip,target,valid,t.step)
        mask,rows=S.choose(x,eligible,classes,'COVERAGE',None,None)
        if profile['mode']!='ORIGINAL':
            counts['prototype_attempts']+=1
            rec=dict(ordinal=counts['prototype_attempts'],category=t.category,key=t.key,step=t.step,mode=profile['mode'])
            c.e.append(root/'PROTOTYPE_LEDGER.jsonl',dict(event='attempt',**rec))
            try:
                with torch.no_grad():
                    hu=t.ema.parent.feature_to_output(t._captured_feature,q.shape[-2:])
                    labeled,labels,_=t.provider.labeled(t.cursor)
                    assert len(labeled)==2
                    before=counts['ema_image_forwards']
                    t.ema(labeled,mode='teacher')
                    assert counts['ema_image_forwards']-before==2
                    counts['prototype_image_forwards']+=2
                    hl=t.ema.parent.feature_to_output(t._captured_feature,labels.shape[-2:])
                    chosen,support,present=P.assignments(hl,labels,hu,target.argmax(1))
                    if profile['mode']=='ROTATED' and present:chosen=torch.where(hu.float().norm(dim=1)>1e-8,(chosen+1)%3,chosen)
                    updated=target if profile['mode']=='BASE' else P.permute(target,chosen,mask)
                    assert not updated.requires_grad and torch.equal(updated.max(1).values,target.max(1).values)
                    assert torch.allclose(updated.sum(1),target.sum(1),atol=1e-7,rtol=1e-7)
                    changed=(updated.argmax(1)!=target.argmax(1))&mask.to(updated.device)
                    detail=dict(support=support,present=present,changed=int(changed.sum()),selected=int(mask.sum()),target_class_counts=torch.bincount(updated.argmax(1)[mask.to(updated.device)],minlength=3).tolist())
                c.e.append(root/'PROTOTYPE_LEDGER.jsonl',dict(event='success',**rec,**detail));counts['prototype_success']+=1;counts['prototype_changed_pixels']+=detail['changed']
                target=updated
            except BaseException:c.e.append(root/'PROTOTYPE_LEDGER.jsonl',dict(event='failure',**rec));raise
        loss=S.selected_loss(p,q,target,valid,mask)
        region_ordinal+=1;record=dict(ordinal=region_ordinal,category=t.category,key=t.key,step=t.step,mode=profile['mode'],images=len(rows),eligible=sum(r['eligible'] for r in rows),selected=sum(r['selected'] for r in rows),tiles=sum(r['tiles'] for r in rows),classes=[sum(r['classes'][i] for r in rows) for i in range(3)],image_budgets=[(r['eligible'],r['selected']) for r in rows])
        c.e.append(root/'SELECTION_LEDGER.jsonl',record);counts['selector_invocations']+=1;counts['selection_decisions']+=record['tiles'];counts['eligible_pixels']+=record['eligible'];counts['selected_pixels']+=record['selected']
        stats.update(region_selected=record['selected']);return loss,stats,target
    c.Trainer.losses=losses;c.transfer_loss=selected
    def create(ctx):
        t=b.make(cfg,roles,ledger,ctx,300);t.action=11
        for name,model in [('student',t.model),('ema',t.ema),('memory',t.memory)]:
            def hook(module,args,name=name):counts[name+'_image_forwards']+=len(args[0])
            model.register_forward_pre_hook(hook)
        def capture(module,args):t._captured_feature=args[0].detach()
        t.ema.parent.native.decoder.conv_logit.register_forward_pre_hook(capture)
        return t
    def setprofile(mode):profile.clear();profile.update(mode=mode)
    def query(t,role,condition):
        nonlocal allowed
        assert role in ('Q_train_new','Q_train_old')
        allowed=set(roles[role]);counts['query_images']+=len(allowed);counts['query_'+role]+=len(allowed)
        record=dict(ordinal=counts['query_images']//4,role=role,step=t.step);c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='attempt',**record))
        try:value=scores(t,roles,role,condition)['macro']
        except BaseException:c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='failure',**record));raise
        else:c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='success',**record));return value
        finally:allowed=set(roles['A_fit'])
    def state(path):return torch.load(path,map_location='cpu',weights_only=False)
    def entry_state(i):return state(Path(cfg['v116'])/f'jobs/learn601/GROUP_{i:02d}_ENTRY.private.pt')['student']
    if cfg['job']=='qualification':
        t=create(b.contexts()[0]);t.provider.seed=10168;entry=entry_state(0);gold=None
        for mode in ('ORIGINAL','BASE'):
            c.restore(t,entry);t.category='qualification';t.key=mode;t.action=11;setprofile(mode)
            for _ in range(2):t.update()
            now=c.snapshot(t)
            if gold is None:gold=now
            else:
                for key in ('student','ema','memory','optimizer','scheduler','scaler','rng','grads','step','cursor'):assert c.e.same(gold[key],now[key]),key
        for i,mode in ((0,'SEMANTIC'),(7,'ROTATED')):
            t=create(b.contexts()[i]);t.provider.seed=10168;entry=entry_state(i);expected=None
            for replay in range(2):
                c.restore(t,entry);t.category='qualification';t.key=mode;t.action=11;setprofile(mode)
                for _ in range(2):t.update()
                now=c.snapshot(t);assert c.e.same(entry['memory'],now['memory']) and all(p.grad is None for p in t.ema.parameters()) and all(p.grad is None for p in t.memory.parameters())
                if replay==0:
                    c.e.atomic_save(now,root/(mode+'_QUALIFIED.private.pt'));expected=state(root/(mode+'_QUALIFIED.private.pt'))
                else:assert c.e.same(expected,now)
        assert dict(ledger.count)==dict(qualification=12) and counts['prototype_success']==10 and counts['selector_invocations']==12
        N.D.write(root/'QUALIFICATION.json',dict(status='PASS',native_updates=12,prototype_forwards=10,BASE_matches_original_model_optimizer_RNG=True,exact_checkpoint_replays=2,frozen_memory=True,teacher_gradients_none=True,query_calls=0))
    elif cfg['job'] in ('train','evaluate'):
        i,stream=cfg['context_index'],cfg['stream'];ctx=b.contexts()[i];t=create(ctx);t.provider.seed=10168
        c.restore(t,entry_state(i));assert t.step==100;t.provider.seed=10168+10000*stream
        random.seed(118000+stream);np.random.seed(118000+stream);torch.manual_seed(118000+stream);torch.cuda.manual_seed_all(118000+stream)
        paired=c.snapshot(t)
        if cfg['job']=='train':
            for mode in METHODS:
                c.restore(t,paired);t.action=11;t.category='prior';t.key=f'ctx{i}/stream{stream}/{mode}';setprofile(mode)
                for _ in range(200):
                    t.update()
                    if t.step in SNAPSHOTS:
                        c.e.atomic_save(c.snapshot(t),root/f'{mode}_{t.step}.private.pt')
                        N.B.write(root/'STATUS.json',dict(status='RUNNING',method=mode,step=t.step,physical=dict(ledger.count)))
                N.D.write(root/(mode+'_TRAINING.json'),dict(status='SEALED',context=i,stream=stream,method=mode,updates=200,snapshots=list(SNAPSHOTS),time=time.time()))
            assert dict(ledger.count)==dict(prior=600) and counts['prototype_success']==600 and counts['selector_invocations']==600
        else:
            assert N.D.read(campaign/'ENDPOINT_LOCK.json')['snapshots']==192
            previous=N.D.read(Path(cfg['v117'])/'RESULTS.json');ref=next(r for r in previous if r['seed']==601 and r['group']==i and r['method']=='COVERAGE')
            assert ref['entry_step']==100;rows=[]
            src=campaign/f'jobs/train{i}_{stream}'
            for mode in METHODS:
                for step in SNAPSHOTS:
                    c.restore(t,state(src/f'{mode}_{step}.private.pt'))
                    new=query(t,'Q_train_new',ctx[2]);old=query(t,'Q_train_old',('identity',1.))
                    rows.append(dict(context=i,stream=stream,method=mode,step=step,new=new,old=old,new_entry=ref['new_entry'],old_entry=ref['old_entry'],new_gain=new-ref['new_entry'],old_gain=old-ref['old_entry']))
            assert counts['query_images']==96 and not ledger.count;N.D.write(root/'RESULTS.json',rows)
    else:raise ValueError(cfg['job'])
    N.D.write(root/'COUNTS.json',dict(counts));N.D.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count),time=time.time()))


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=Path(cfg['v117']);assert N.D.read(previous/'COMPLETION_AUDIT.json')['status']=='PASS'
    assert N.D.read(previous/'FINAL_PUBLICATION.json')['anonymous_http']=='200' and N.D.read(previous/'POSTHOC_PUBLICATION.json')['anonymous_http']=='200'
    (root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    qual=[dict(id='qualification',job='qualification',caps=dict(qualification=12))]
    train=[dict(id=f'train{i}_{s}',job='train',context_index=i,stream=s,caps=dict(prior=600)) for i in range(8) for s in STREAMS]
    evaluate=[dict(id=f'evaluate{i}_{s}',job='evaluate',context_index=i,stream=s,caps={}) for i in range(8) for s in STREAMS]
    account=Accounting(root,caps=CAPS,jobs=qual+train+evaluate)
    N.schedule(root,cfg,qual,account);N.schedule(root,cfg,train,account)
    assert dict(account.count)==dict(account.success)==CAPS and not account.failure
    assert sum(len(list((root/'jobs'/j['id']).glob('*_TRAINING.json'))) for j in train)==48
    N.D.write(root/'ENDPOINT_LOCK.json',dict(status='SEALED_BEFORE_QUERY',trajectories=48,snapshots=192,time=time.time()))
    N.schedule(root,cfg,evaluate,account)
    rows=sum((N.D.read(root/'jobs'/j['id']/'RESULTS.json') for j in evaluate),[]);assert len(rows)==192
    counts=Counter()
    for j in qual+train+evaluate:counts.update(N.D.read(root/'jobs'/j['id']/'COUNTS.json'))
    assert counts['query_images']==1536 and counts['selector_invocations']==9612 and counts['prototype_success']==counts['prototype_attempts']==9610 and counts['prototype_image_forwards']==19220
    N.D.write(root/'RESULTS.json',rows);N.D.table(root/'RESULTS.csv',rows)
    N.B.write(root/'COSTS.json',dict(native_updates=9612,actor_optimizer_updates=0,linear_solves=0,physical=dict(account.count),success=dict(account.success),failures=dict(account.failure),counts=dict(counts),new_annotation_cases=0,query_calls=384,query_images=1536,Q_dev_images=0,additional_EMA_forward_calls=9610,additional_EMA_image_forwards=19220))
    for name,result in A.analyze(rows).items():N.D.write(root/(name+'.json'),result)
    N.D.write(root/'FINAL.json',dict(status='COMPLETE',decision='PENDING_FULL_AUDIT_AND_PUBLICATION',time=time.time()));N.B.write(root/'STATUS.json',N.D.read(root/'FINAL.json'))


if __name__=='__main__':
    root=Path(os.environ['EXEC_RUN']);cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());configure(cfg,root)
    if os.environ.get('EXEC_SELFCHECK')=='1':
        A.selfcheck();Q.selfcheck();result=P.selfcheck();print(json.dumps(result))
    else:
        N.D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':worker(root,cfg)
            else:coordinator(root,cfg)
        except BaseException as exc:N.D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
