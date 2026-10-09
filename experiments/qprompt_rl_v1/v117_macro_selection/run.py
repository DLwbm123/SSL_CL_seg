"""Train-only paired macro-selector action-space and learnability diagnostic."""
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

METHODS=('RANDOM','CONFIDENCE','COVERAGE','OFF')
CAPS=dict(qualification=8,prior=12800)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def configure(cfg,root):
    global Q,N,c,b,S
    Q=load('quarter',cfg['quarter_entry']);Q.configure(cfg,root);N=Q.N;c=N.c;b=N.b
    N.STAGE='V117';N.CAPS=CAPS;c.CAPS.update(CAPS)
    S=load('selector',cfg['selector_entry'])
    A=load('analysis117',cfg['analysis_entry']);globals()['A']=A


def worker(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    torch.set_num_threads(2);torch.cuda.set_device(0);random.seed(168);np.random.seed(168);torch.manual_seed(168)
    original=N.E.install_actions(c);N.E.action_check(c,original);N.EXTRA_ACTION_INSTALLER(c)
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
        if 'feature' not in profile:
            values=x[x[:,:,11]>0].double();profile['feature']=torch.cat((values.mean(0),values.std(0,unbiased=False))).tolist() if len(values) else [0.]*26
        mask,rows=S.choose(x,eligible,classes,profile['mode'],profile.get('actor'),profile['rng'],profile.get('trace'))
        loss=S.selected_loss(p,q,target,valid,mask)
        region_ordinal+=1;record=dict(ordinal=region_ordinal,category=t.category,key=t.key,step=t.step,mode=profile['name'],images=len(rows),eligible=sum(r['eligible'] for r in rows),selected=sum(r['selected'] for r in rows),tiles=sum(r['tiles'] for r in rows),classes=[sum(r['classes'][i] for r in rows) for i in range(3)],image_budgets=[(r['eligible'],r['selected']) for r in rows])
        c.e.append(root/'SELECTION_LEDGER.jsonl',record);counts['selector_invocations']+=1;counts['selection_decisions']+=record['tiles'];counts['eligible_pixels']+=record['eligible'];counts['selected_pixels']+=record['selected']
        stats.update(region_selected=record['selected']);return loss,stats,target
    c.Trainer.losses=losses;c.transfer_loss=selected
    def create(ctx):
        t=b.make(cfg,roles,ledger,ctx,300);t.action=11
        for name,model in [('student',t.model),('ema',t.ema),('memory',t.memory)]:
            def hook(module,args,name=name):counts[name+'_image_forwards']+=len(args[0])
            model.register_forward_pre_hook(hook)
        return t
    def setprofile(mode,actor,rng,trace=None,name=None):profile.clear();profile.update(mode=mode,actor=actor,rng=rng,trace=trace,name=name or mode)
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
    def entry_state(seed,group):
        return state(Path(cfg['v116'])/f'jobs/learn{seed}/GROUP_{group:02d}_ENTRY.private.pt')['student']
    if cfg['job']=='qualification':
        for i,mode in ((0,'RANDOM'),(7,'OFF')):
            t=create(b.contexts()[i]);t.provider.seed=10168;entry=entry_state(601,i)
            first=None;first_feature=None
            for replay in range(2):
                c.restore(t,entry);t.category='qualification';t.key=str(i);t.action=12 if mode=='OFF' else 11
                setprofile(mode,None,torch.Generator().manual_seed(11700+i))
                for _ in range(2):t.update()
                now=c.snapshot(t);assert c.e.same(entry['memory'],now['memory'])
                if replay==0:first=now;first_feature=profile.get('feature')
                else:assert c.e.same(first,now) and first_feature==profile.get('feature')
                if mode=='OFF':assert not t.last['active_U'] and 'feature' not in profile
        assert dict(ledger.count)==dict(qualification=8) and counts['selector_invocations']==4
        N.D.write(root/'QUALIFICATION.json',dict(status='PASS',native_updates=8,exact_full_state_replays=2,frozen_memory=True,OFF_has_no_U=True,query_calls=0))
    elif cfg['job']=='collect':
        seed=cfg['controller'];rows=[];features=[]
        for group in cfg['groups']:
            i=group%8;ctx=b.contexts()[i];t=create(ctx);t.provider.seed=10168;entry=entry_state(seed,group)
            c.restore(t,entry);assert t.step==100+50*(group//8)
            new_entry=query(t,'Q_train_new',ctx[2]);old_entry=query(t,'Q_train_old',('identity',1.))
            first_feature=None
            for mode in METHODS:
                c.restore(t,entry);t.category='prior';t.key=f'g{group}/{mode}';t.action=12 if mode=='OFF' else 11
                setprofile(mode,None,torch.Generator().manual_seed(117000+1000*seed+group))
                for k in range(50):
                    t.update()
                    if k==12:mid=query(t,'Q_train_new',ctx[2])
                new=query(t,'Q_train_new',ctx[2]);old=query(t,'Q_train_old',('identity',1.))
                gain=.25*(mid-new_entry)+.75*(new-new_entry);reward=gain+old-old_entry
                row=dict(seed=seed,group=group,context=i,cycle=group//8,entry_step=entry['step'],method=mode,mid_new=mid,new=new,old=old,new_entry=new_entry,old_entry=old_entry,gain=gain,final_new_gain=new-new_entry,old_gain=old-old_entry,reward=reward)
                rows.append(row);c.e.append(root/'ROWS.jsonl',row)
                if mode!='OFF':
                    if first_feature is None:first_feature=profile['feature']
                    else:assert first_feature==profile['feature'], 'paired branch entry prediction features differ'
                else:assert 'feature' not in profile and not t.last['active_U']
            features.append(dict(seed=seed,group=group,x=first_feature))
            N.B.write(root/'STATUS.json',dict(status='RUNNING',groups=len(features),total=len(cfg['groups']),physical=dict(ledger.count)))
            del t;torch.cuda.empty_cache()
        assert dict(ledger.count)==dict(prior=1600) and counts['query_images']==448 and counts['selector_invocations']==1200
        N.D.write(root/'FEATURES.private.json',features);N.D.write(root/'RESULTS.json',rows)
    else:raise ValueError(cfg['job'])
    N.D.write(root/'COUNTS.json',dict(counts));N.D.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count),time=time.time()))


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=Path(cfg['v116']);assert N.D.read(previous/'COMPLETION_AUDIT.json')['status']=='PASS'
    assert N.D.read(previous/'FINAL_PUBLICATION.json')['anonymous_http']=='200'
    assert N.D.read(previous/'posthoc_policy_diagnosis/PUBLICATION.json')['anonymous_http']=='200'
    (root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    qual=[dict(id='qualification',job='qualification',caps=dict(qualification=8))]
    collect=[dict(id=f'collect{s}_{part}',job='collect',controller=s,groups=list(range(part*8,(part+1)*8)),caps=dict(prior=1600)) for s in (601,602) for part in range(4)]
    account=Accounting(root,caps=CAPS,jobs=qual+collect);N.schedule(root,cfg,qual,account);N.schedule(root,cfg,collect,account)
    assert dict(account.count)==dict(account.success)==CAPS and not account.failure
    rows=sum((N.D.read(root/'jobs'/j['id']/'RESULTS.json') for j in collect),[])
    features=sum((N.D.read(root/'jobs'/j['id']/'FEATURES.private.json') for j in collect),[])
    counts=Counter()
    for j in qual+collect:counts.update(N.D.read(root/'jobs'/j['id']/'COUNTS.json'))
    assert counts['query_images']==3584 and counts['selector_invocations']==9604
    N.D.write(root/'RESULTS.json',rows);N.D.table(root/'RESULTS.csv',rows);N.D.write(root/'FEATURES.private.json',features)
    N.B.write(root/'COSTS.json',dict(native_updates=12808,actor_optimizer_updates=0,linear_solves=0,physical=dict(account.count),success=dict(account.success),failures=dict(account.failure),counts=dict(counts),new_annotation_cases=0,query_calls=896,query_images=3584,Q_dev_images=0))
    N.D.write(root/'DATASET_LOCK.json',dict(status='SEALED',states=64,actions=4,time=time.time()))
    for name,result in A.analyze(rows,features).items():N.D.write(root/(name+'.json'),result)
    N.D.write(root/'FINAL.json',dict(status='COMPLETE',decision='PENDING_FULL_AUDIT_AND_PUBLICATION',time=time.time()));N.B.write(root/'STATUS.json',N.D.read(root/'FINAL.json'))


if __name__=='__main__':
    root=Path(os.environ['EXEC_RUN']);cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());configure(cfg,root)
    if os.environ.get('EXEC_SELFCHECK')=='1':
        A.selfcheck();Q.selfcheck();original=N.E.install_actions(c);N.E.action_check(c,original)
        forwards=Counter();forward=S.Actor.forward
        def counted(actor,x):forwards['calls']+=1;return forward(actor,x)
        S.Actor.forward=counted;result=S.selfcheck(c.transfer_loss)
        result.update(model_forwards=forwards['calls'],CPU_actor_forward_calls=forwards['calls'],segmentation_forwards=0)
        result['checks']+=['LOCO heldout exclusion','training-only scale and tie handling','positive and negative diagnostic gates','quarter rate restoration']
        print(json.dumps(result))
    else:
        N.D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':worker(root,cfg)
            else:coordinator(root,cfg)
        except BaseException as exc:N.D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
