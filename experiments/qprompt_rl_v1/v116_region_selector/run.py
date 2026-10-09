"""Native-engine regional selection pilot with paired on-policy branch feedback."""
import copy
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

METHODS=('RL_601','RL_602','CE_601','CE_602','RANDOM','CONFIDENCE','COVERAGE')
CAPS=dict(qualification=8,prior=12800,development=16800,actor_qualification=2,selector_RL=64,selector_CE=64)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def configure(cfg,root):
    global Q,N,c,b,S
    Q=load('quarter',cfg['quarter_entry']);Q.configure(cfg,root);N=Q.N;c=N.c;b=N.b
    N.STAGE='V116';N.CAPS=CAPS;c.CAPS.update(CAPS)
    S=load('selector',cfg['selector_entry'])


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
        assert role in (('Q_dev_new','Q_dev_old') if cfg['job']=='evaluate' else ('Q_train_new','Q_train_old'))
        allowed=set(roles[role]);counts['query_images']+=len(allowed);counts['query_'+role]+=len(allowed)
        record=dict(ordinal=counts['query_images']//4,role=role,step=t.step);c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='attempt',**record))
        try:value=scores(t,roles,role,condition)['macro']
        except BaseException:c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='failure',**record));raise
        else:c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='success',**record));return value
        finally:allowed=set(roles['A_fit'])
    def state(path):return torch.load(path,map_location='cpu',weights_only=False)
    def actors(seed):return S.Actor(seed),S.Actor(seed)
    if cfg['job']=='qualification':
        rl,ce=actors(601);traces=[]
        for i,mode in ((0,'RL'),(7,'COVERAGE')):
            t=create(b.contexts()[i]);t.provider.seed=10168
            entry=state(Path(cfg['v92'])/f'jobs/collect{i}_1/ENTRY100_STREAM.private.pt');c.restore(t,entry);t.category='qualification';t.key=str(i);t.action=11
            generator=torch.Generator().manual_seed(11600+i);initial=generator.get_state();trace=[];setprofile(mode,rl,generator,trace)
            for _ in range(2):t.update()
            expected=c.snapshot(t);assert c.e.same(entry['memory'],expected['memory']);traces.append(S.pack(trace));c.restore(t,entry);t.action=11;generator.set_state(initial);again=[];setprofile(mode,rl,generator,again)
            for _ in range(2):t.update()
            assert c.e.same(expected,c.snapshot(t)) and c.e.same(traces[-1],S.pack(again))
        before=c.snapshot(t);rlopt=torch.optim.Adam(rl.parameters(),lr=.001);ceopt=torch.optim.Adam(ce.parameters(),lr=.001)
        details=S.update(rl,ce,rlopt,ceopt,traces,[.01,-.01],lambda label,fn:ledger.call('actor_qualification',label,fn))
        assert c.e.same(before,c.snapshot(t)) and all(p.grad is None for p in t.memory.parameters())
        assert dict(ledger.count)==dict(qualification=8,actor_qualification=2) and counts['selector_invocations']==8
        N.D.write(root/'QUALIFICATION.json',dict(status='PASS',native_updates=8,actor_updates=2,full_state_replays=2,selection_trace_replay_exact=True,actor_update_student_isolation=True,frozen_memory=True,query_calls=0,actor_check=details))
    elif cfg['job']=='learn':
        seed=cfg['controller'];rl,ce=actors(seed);rlopt=torch.optim.Adam(rl.parameters(),lr=.001);ceopt=torch.optim.Adam(ce.parameters(),lr=.001);generator=torch.Generator().manual_seed(116000+seed)
        for group in range(32):
            i=group%8;cycle=group//8;ctx=b.contexts()[i];t=create(ctx);t.provider.seed=10168
            entry=state(Path(cfg['v92'])/f'jobs/collect{i}_1/ENTRY100_STREAM.private.pt') if cycle==0 else state(root/f'STATE_{i}.private.pt')
            c.restore(t,entry);assert t.step==100+cycle*50;t.category='prior';t.action=11
            c.e.atomic_save(dict(student=entry,rl=rl.state_dict(),ce=ce.state_dict(),rl_optimizer=rlopt.state_dict(),ce_optimizer=ceopt.state_dict(),selector_rng=generator.get_state()),root/f'GROUP_{group:02d}_ENTRY.private.pt')
            old_entry=query(t,'Q_train_old',('identity',1.));new_entry=query(t,'Q_train_new',ctx[2]);outputs=[];branches=[];retained=None
            for branch in range(4):
                c.restore(t,entry);t.category='prior';t.action=11;t.key=f'g{group}/branch{branch}';trace=[];setprofile('RL',rl,generator,trace,name=f'RL_{seed}')
                for k in range(50):
                    t.update()
                    if k==12:mid=query(t,'Q_train_new',ctx[2])
                new=query(t,'Q_train_new',ctx[2]);old=query(t,'Q_train_old',('identity',1.));reward=.25*(mid-new_entry)+.75*(new-new_entry)+old-old_entry
                outputs.append(dict(branch=branch,mid_new=mid,new=new,old=old,new_entry=new_entry,old_entry=old_entry,reward=reward,decisions=len(trace)))
                branches.append(S.pack(trace))
                if branch==0:retained=c.snapshot(t)
            c.restore(t,retained);before=c.snapshot(t)
            details=S.update(rl,ce,rlopt,ceopt,branches,[r['reward'] for r in outputs],lambda label,fn:ledger.call(label,f'g{group}',fn))
            assert c.e.same(before,c.snapshot(t))
            c.e.atomic_save(retained,root/f'STATE_{i}.private.pt');c.e.atomic_save(branches,root/f'GROUP_{group:02d}_TRAJECTORIES.private.pt')
            c.e.append(root/'GROUPS.jsonl',dict(group=group,context=i,cycle=cycle,seed=seed,entry_step=entry['step'],retained_branch=0,branches=outputs,actor=details))
            c.e.atomic_save(dict(rl=rl.state_dict(),ce=ce.state_dict(),rl_optimizer=rlopt.state_dict(),ce_optimizer=ceopt.state_dict(),selector_rng=generator.get_state(),groups=group+1),root/'ACTORS.private.pt')
            N.B.write(root/'STATUS.json',dict(status='RUNNING',groups=group+1,total=32,physical=dict(ledger.count)));del t;torch.cuda.empty_cache()
        assert dict(ledger.count)==dict(prior=6400,selector_RL=32,selector_CE=32) and counts['query_images']==1792
        N.D.write(root/'ACTOR_LOCK.json',dict(status='SEALED',seed=seed,groups=32,actors=2,time=time.time()))
    elif cfg['job'] in ('train','evaluate'):
        i,stream=cfg['context_index'],cfg['stream'];m,n,kind,factor=N.D.read(Path(cfg['code'])/'experiments/qprompt_rl_v1/v8_grpo_transfer_prior/ACTION_AUDIT_CONFIG.json')['dev_contexts'][i];ctx=(m,n,(kind,factor),f'dev{i}')
        fitted={}
        if cfg['job']=='train':
            for seed in (601,602):
                assert N.D.read(campaign/f'jobs/learn{seed}/ACTOR_LOCK.json')['status']=='SEALED'
                saved=state(campaign/f'jobs/learn{seed}/ACTORS.private.pt')
                for arm in ('RL','CE'):
                    actor=S.Actor(seed);actor.load_state_dict(saved[arm.lower()]);fitted[f'{arm}_{seed}']=actor
        t=create(ctx);c.restore(t,state(Path(cfg['original'])/f'jobs/development/dev{i}/ENTRY.private.pt'));assert t.step==100
        t.provider.seed=168+10000*stream;random.seed(860100+stream);np.random.seed(860100+stream);torch.manual_seed(860100+stream);torch.cuda.manual_seed_all(860100+stream);paired=c.snapshot(t)
        if cfg['job']=='train':
            c.e.atomic_save(paired,root/'ENTRY.private.pt')
            for method in METHODS:
                c.restore(t,paired);t.category='development';t.key=f'dev{i}/stream{stream}/{method}';t.action=11
                setprofile(method.split('_')[0],fitted.get(method),torch.Generator().manual_seed(862600+1000*stream+i),name=method)
                for _ in range(200):
                    t.update()
                    if t.step==150:c.e.atomic_save(c.snapshot(t),root/(method+'_MID.private.pt'))
                    if t.step%50==0:N.B.write(root/'STATUS.json',dict(status='RUNNING',method=method,step=t.step,physical=dict(ledger.count)))
                c.e.atomic_save(c.snapshot(t),root/(method+'_FINAL.private.pt'));N.D.write(root/(method+'_TRAINING.json'),dict(status='SEALED',context=ctx[3],stream=stream,method=method,updates=200))
            assert dict(ledger.count)==dict(development=1400)
        else:
            assert N.D.read(campaign/'ENDPOINT_LOCK.json')['trajectories']==84
            references=[r for r in N.D.read(Path(cfg['v113'])/'DEVELOPMENT_RESULTS.json')['rows'] if r['context']==ctx[3]];assert len({r['new_entry'] for r in references})==len({r['old_memory_reference'] for r in references})==1
            entrynew,reference=references[0]['new_entry'],references[0]['old_memory_reference'];rows=[];src=campaign/f'jobs/train{i}_{stream}'
            for method in METHODS:
                c.restore(t,state(src/(method+'_MID.private.pt')));mid=query(t,'Q_dev_new',ctx[2]);c.restore(t,state(src/(method+'_FINAL.private.pt')));new=query(t,'Q_dev_new',ctx[2]);old=query(t,'Q_dev_old',('identity',1.));value=N.B.reward(mid,new,entrynew,old,reference)
                rows.append(dict(context=ctx[3],stream=stream,method=method,mid_new=mid,new=new,old=old,utility=value['reward'],gain=value['gain'],forget_penalty=value['forget_penalty'],new_entry=entrynew,old_memory_reference=reference))
            assert counts['query_images']==84 and not ledger.count;N.D.write(root/'RESULTS.json',rows)
    else:raise ValueError(cfg['job'])
    N.D.write(root/'COUNTS.json',dict(counts));N.D.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count),time=time.time()))


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert N.D.read(Path(cfg['v115'])/'COMPLETION_AUDIT.json')['status']=='PASS' and N.D.read(Path(cfg['v115'])/'FINAL_PUBLICATION.json')['anonymous_http']=='200'
    (root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    qual=[dict(id='qualification',job='qualification',caps=dict(qualification=8,actor_qualification=2))]
    learn=[dict(id=f'learn{s}',job='learn',controller=s,caps=dict(prior=6400,selector_RL=32,selector_CE=32)) for s in (601,602)]
    train=[dict(id=f'train{i}_{s}',job='train',context_index=i,stream=s,caps=dict(development=1400)) for i in range(4) for s in (3,4,5)]
    evaluate=[dict(id=f'evaluate{i}_{s}',job='evaluate',context_index=i,stream=s,caps={}) for i in range(4) for s in (3,4,5)]
    account=Accounting(root,caps=CAPS,jobs=qual+learn+train+evaluate)
    N.schedule(root,cfg,qual,account);N.schedule(root,cfg,learn,account)
    N.D.write(root/'ACTOR_LOCK.json',dict(status='SEALED_BEFORE_DEPLOYMENT',time=time.time(),actors=4))
    N.schedule(root,cfg,train,account);assert dict(account.count)==CAPS and not account.failure
    assert sum(len(list((root/'jobs'/j['id']).glob('*_TRAINING.json'))) for j in train)==84
    N.D.write(root/'ENDPOINT_LOCK.json',dict(status='SEALED_BEFORE_QUERY_READOUT',time=time.time(),trajectories=84,snapshots=168))
    N.schedule(root,cfg,evaluate,account);rows=sum((N.D.read(root/'jobs'/j['id']/'RESULTS.json') for j in evaluate),[]);assert len(rows)==84
    counts=Counter()
    for j in qual+learn+train+evaluate:counts.update(N.D.read(root/'jobs'/j['id']/'COUNTS.json'))
    assert counts['query_images']==4592 and counts['selector_invocations']==29608
    N.D.write(root/'DEVELOPMENT_RESULTS.json',dict(rows=rows));N.D.table(root/'DEVELOPMENT_RESULTS.csv',rows)
    N.B.write(root/'COSTS.json',dict(native_updates=29608,actor_optimizer_updates=130,linear_solves=0,physical=dict(account.count),success=dict(account.success),failures=dict(account.failure),counts=dict(counts),new_annotation_cases=0))
    N.D.write(root/'FINAL.json',dict(status='COMPLETE',decision='PENDING_FULL_READOUT',time=time.time()));N.B.write(root/'STATUS.json',N.D.read(root/'FINAL.json'))


if __name__=='__main__':
    root=Path(os.environ['EXEC_RUN']);cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());configure(cfg,root)
    if os.environ.get('EXEC_SELFCHECK')=='1':
        original=N.E.install_actions(c);N.E.action_check(c,original);print(json.dumps(S.selfcheck(c.transfer_loss)))
    else:
        N.D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':worker(root,cfg)
            else:coordinator(root,cfg)
        except BaseException as exc:N.D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
