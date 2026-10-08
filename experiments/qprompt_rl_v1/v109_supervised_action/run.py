"""Append one truly supervised-only action to frozen training-state return tables."""
import fcntl
import json
import os
import random
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch

CAPS=dict(qualification=12,audit=12000)


def install_off(c):
    original=c.Trainer.losses
    def losses(t):
        if t.action!=12:return original(t)
        assert t.model.family=='B0_PARENT_LCTX'
        value=c.e.StageTrainer.losses(t)
        assert value[1] is None and value[2] is None and not t.last['active_U']
        return value
    c.Trainer.losses=losses
    return original


def contexts():return b.contexts()+J.contexts()


def margin_rows(keys,old,new,oldrows):
    assert old.shape==(80,12) and len(keys)==len(new)==80 and len({tuple(k) for k in keys})==80
    lookup={(r['context'],r['stream'],r['step']):r for r in new};assert len(lookup)==80 and set(lookup)=={tuple(k) for k in keys}
    result=[]
    for i,key in enumerate(keys):
        row=lookup[tuple(key)];assert row['action']==12 and all(np.isfinite(row[k]) for k in ('gain','old_entry','old_final','dense_reward'))
        assert abs(row['dense_reward']-(row['gain']+row['old_final']-row['old_entry']))<1e-12
        best=int(old[i].argmax());previous=oldrows[(*key,best)]
        result.append(dict(context=key[0],stream=key[1],step=key[2],off_return=row['dense_reward'],best_old_action=best,best_old_return=float(old[i,best]),off_minus_best=float(row['dense_reward']-old[i,best]),off_minus_mean=float(row['dense_reward']-old[i].mean()),gain_difference=row['gain']-previous['gain'],old_retention_difference=row['old_final']-previous['old_final']))
    return result


def gpu_job(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    torch.set_num_threads(2);torch.cuda.set_device(0);random.seed(168);np.random.seed(168);torch.manual_seed(168)
    original_actions=E.install_actions(c);E.action_check(c,original_actions);original_losses=install_off(c);wrapped=c.Trainer.losses
    roles=c.split_roles(cfg['data']);assert roles==D.read(Path(cfg['action_root'])/'ROLES.private.json')
    assert len(roles['Q_train_new'])==len(roles['Q_train_old'])==4
    allowed=set(roles['A_fit']);counts=Counter();sample=c.e.primitive.CurrentData.__getitem__
    def guarded(ds,i):
        assert ds.role in ('train_labeled','train_unlabeled')
        assert ds.rows[i]['case_id'] in (allowed if ds.role=='train_labeled' else set(roles['U_adapt']))
        counts['item_attempts_'+ds.role]+=1;value=sample(ds,i);counts['item_success_'+ds.role]+=1;return value
    c.e.primitive.CurrentData.__getitem__=guarded;ledger=b.JobLedger(root,cfg['caps'])
    def create(ctx):
        t=b.make(cfg,roles,ledger,ctx,300)
        for name,model in [('student',t.model),('ema',t.ema),('memory',t.memory)]:
            def hook(module,args,name=name):counts[name+'_image_forwards']+=len(args[0])
            model.register_forward_pre_hook(hook)
        return t
    def update(t):
        before=t.provider.u_reads;t.update()
        if t.action==12:
            assert t.provider.u_reads==before and not t.last['active_U'];counts['off_updates']+=1
        else:counts['original_action_updates']+=1
    def query(t,role,condition):
        nonlocal allowed
        assert role in ('Q_train_new','Q_train_old');allowed=set(roles[role]);counts['query_images']+=len(allowed)
        record=dict(role=role,step=t.step,ordinal=counts['query_images']//4);c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='attempt',**record))
        try:value=scores(t,roles,role,condition)['macro']
        except BaseException:c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='failure',**record));raise
        else:c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='success',**record));return value
        finally:allowed=set(roles['A_fit'])
    if cfg['job']=='qualification':
        for i in (0,7):
            t=create(b.contexts()[i]);t.provider.seed=10168
            entry=torch.load(Path(cfg['v92'])/f'jobs/collect{i}_1/ENTRY100_STREAM.private.pt',map_location='cpu',weights_only=False)
            t.category='qualification';t.key=str(i)
            for action,steps,reference in ((12,2,c.e.StageTrainer.losses),(9,1,original_losses)):
                c.restore(t,entry);t.action=action
                for _ in range(steps):update(t)
                expected=c.snapshot(t);c.restore(t,entry);t.action=action;c.Trainer.losses=reference
                try:
                    for _ in range(steps):update(t)
                finally:c.Trainer.losses=wrapped
                assert c.e.same(expected,c.snapshot(t)), 'native branch differs from its reference'
            del t;torch.cuda.empty_cache()
        assert dict(ledger.count)=={'qualification':12} and counts['off_updates']==8 and counts['original_action_updates']==4
        D.write(root/'QUALIFICATION.json',dict(status='PASS',off_supervised_full_snapshot_exact=2,old_action_full_snapshot_exact=2,off_zero_U_reads=True,off_unlabeled_loss_none=True,native_updates=12,query_calls=0))
    elif cfg['job']=='collect':
        i,s=cfg['context_index'],cfg['stream'];ctx=contexts()[i];t=create(ctx);t.provider.seed=168+10000*s
        src=Path(cfg['v101'])/f'jobs/collect{i}_{s}' if i<8 else Path(cfg['v107'])/f'jobs/collect{i-8}_{s}'
        entrysrc=Path(cfg['v92'])/f'jobs/collect{i}_{s}' if i<8 else src
        entry=torch.load(entrysrc/'ENTRY100_STREAM.private.pt',map_location='cpu',weights_only=False)
        retained=torch.load(src/'ON_POLICY_ENTRY200.private.pt',map_location='cpu',weights_only=False)
        oldrows=D.read(src/'TRAINING_TABLE.private.json');behavior=D.read(src/'BEHAVIOR.private.json');assert len(oldrows)==24
        oldentry=oldrows[0]['old_entry'];assert all(r['old_entry']==oldentry for r in oldrows)
        if i<8:
            refs=[json.loads(x) for x in (Path(cfg['original'])/'jobs/audit_1/ACTION_ROWS.jsonl').read_text().splitlines()]
            refs=[r for r in refs if r['context']==ctx[3] and r['entry_step']==100];assert len(refs)==9 and len({r['new_entry'] for r in refs})==1;newentry=refs[0]['new_entry']
        else:newentry=D.read(Path(cfg['v107'])/f'jobs/entry{i-8}/BASELINES.json')['new_entry']
        models=[V.load12(Path(cfg['v99'])/f'jobs/fit/{a}_{seed}.private.pt') for a in ('EXPANDED_CE','EXPANDED_RL') for seed in (601,602)]
        def select(t,g):counts['behavior_extractions']+=1;return B.select(t,models,g)
        c.restore(t,entry);generator=torch.Generator().manual_seed(860901+100*s+i);first,trace=select(t,generator)
        assert trace==behavior['first'] and all(r['state']==trace['state'] for r in oldrows if r['step']==100)
        c.restore(t,retained);counts['behavior_extractions']+=1;state=t.extract().tolist();assert state==behavior['second']['state'] and all(r['state']==state for r in oldrows if r['step']==200)
        chosen=[r for r in oldrows if r['step']==200 and r['reused']];assert len(chosen)==1 and chosen[0]['action']==behavior['second']['action'];basegain=chosen[0]['gain']
        c.restore(t,torch.load(src/f'FIRST_{first}_FINAL.private.pt',map_location='cpu',weights_only=False));baseline=query(t,'Q_train_new',ctx[2])
        t.category='audit';t.key=f'{ctx[3]}/{s}/firstOFF';c.restore(t,entry);t.action=12;mid=None
        for _ in range(200):
            if t.step==200:
                c.e.atomic_save(c.snapshot(t),root/'OFF_ENTRY200.private.pt');second,secondtrace=select(t,generator);assert 0<=second<12;t.action=second
            update(t)
            if t.step==150:mid=query(t,'Q_train_new',ctx[2])
            if t.step%50==0:B.write(root/'STATUS.json',dict(status='RUNNING',branch='firstOFF',step=t.step,physical=dict(ledger.count)))
        new=query(t,'Q_train_new',ctx[2]);old=query(t,'Q_train_old',('identity',1.));gain=.25*(mid-newentry)+.75*(new-newentry)
        rows=[dict(context=ctx[3],stream=s,step=100,action=12,gain=gain,old_entry=oldentry,old_final=old,dense_reward=gain+old-oldentry,continuation_action=second,new_final=new,mid_new=mid)]
        c.e.atomic_save(c.snapshot(t),root/'FIRST_OFF_FINAL.private.pt')
        c.restore(t,retained);t.action=12;t.key=f'{ctx[3]}/{s}/secondOFF'
        for _ in range(100):update(t)
        new=query(t,'Q_train_new',ctx[2]);old=query(t,'Q_train_old',('identity',1.));gain=basegain+.75*(new-baseline)
        rows.append(dict(context=ctx[3],stream=s,step=200,action=12,gain=gain,old_entry=oldentry,old_final=old,dense_reward=gain+old-oldentry,continuation_action=-1,new_final=new,mid_new=None))
        c.e.atomic_save(c.snapshot(t),root/'SECOND_OFF_FINAL.private.pt')
        assert dict(ledger.count)=={'audit':300} and counts['query_images']==24 and counts['behavior_extractions']==3 and counts['off_updates']==200 and counts['original_action_updates']==100
        D.write(root/'RESULTS.json',rows);D.write(root/'REFERENCE_REPLAY.json',dict(first_selection_exact=True,retained_state_exact=True,baseline_final_new=baseline,baseline_gain=basegain,baseline_final_source_action=first))
        D.write(root/'OFF_CONTINUATION.private.json',secondtrace)
    else:raise ValueError('unknown job')
    D.write(root/'COUNTS.json',dict(counts));D.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count),counts=dict(counts),time=time.time()))


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert len(contexts())==20
    qual=[dict(id='qualification',job='qualification',caps=dict(qualification=12))]
    collect=[dict(id=f'collect{i}_{s}',job='collect',context_index=i,stream=s,caps=dict(audit=300)) for i in range(20) for s in (1,2)]
    jobs=qual+collect;(root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False);account=Accounting(root,caps=CAPS,jobs=jobs)
    N.schedule(root,cfg,qual,account);N.schedule(root,cfg,collect,account)
    assert dict(account.count)==dict(account.success)==CAPS and not account.failure
    rows=sum((D.read(root/'jobs'/j['id']/'RESULTS.json') for j in collect),[]);assert len(rows)==80
    keys=D.read(Path(cfg['v107'])/'KEYS.private.json');old=np.load(Path(cfg['v107'])/'DATASET.private.npz')['returns'];oldrows={}
    for i in range(20):
        for s in (1,2):
            src=Path(cfg['v101'])/f'jobs/collect{i}_{s}' if i<8 else Path(cfg['v107'])/f'jobs/collect{i-8}_{s}'
            for r in D.read(src/'TRAINING_TABLE.private.json'):oldrows[(r['context'],r['stream'],r['step'],r['action'])]=r
    margins=margin_rows(keys,old,rows,oldrows);lookup={(r['context'],r['stream'],r['step']):r for r in rows}
    returns=np.column_stack((old,[lookup[tuple(k)]['dense_reward'] for k in keys]));assert returns.shape==(80,13) and np.array_equal(returns[:,:12],old) and np.isfinite(returns).all()
    np.savez(root/'RETURNS13.private.npz',keys=np.array(keys,dtype=str),returns=returns)
    counts=Counter()
    for j in jobs:counts.update(D.read(root/'jobs'/j['id']/'COUNTS.json'))
    assert counts['query_images']==960 and counts['behavior_extractions']==120 and counts['off_updates']==8008 and counts['original_action_updates']==4004
    for name,values in [('NEW_RETURN_TABLE',rows),('ACTION_MARGINS',margins)]:D.write(root/(name+'.json'),values);D.table(root/(name+'.csv'),values)
    decision=dict(status='ACTION_MARGIN_DIAGNOSTIC_COMPLETE_NO_DEPLOYMENT_CLAIM',states=80,new_action=12,mean_off_minus_best=float(np.mean([r['off_minus_best'] for r in margins])),positive_states=sum(r['off_minus_best']>0 for r in margins),negative_states=sum(r['off_minus_best']<0 for r in margins))
    D.write(root/'DECISION.json',decision);D.write(root/'DATASET_LOCK.json',dict(status='SEALED',states=80,actions=13,old960_returns_unchanged=True,new_returns=80,time=time.time()))
    B.write(root/'COSTS.json',dict(native_updates=12012,actor_optimizer_updates=0,linear_solves=0,physical=dict(account.count),success=dict(account.success),failures=dict(account.failure),counts=dict(counts),stable_probes=0,development_query_images=0,new_annotation_cases=0))
    D.write(root/'FINAL.json',dict(status='COMPLETE',decision=decision['status'],physical=dict(account.count),publication='PENDING',time=time.time()));B.write(root/'STATUS.json',D.read(root/'FINAL.json'))


if __name__=='__main__':
    import importlib.util
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN'])
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c,stage_b as b
    B=load('base_loss',cfg['base_entry']);D=load('diagnostic',cfg['diagnostic_entry']);V=load('actor12',cfg['v99_entry']);E=load('expanded12',cfg['expanded_entry']);J=load('coverage_contexts',cfg['coverage_entry'])
    N=load('scheduler',cfg['sequential_entry']);N.B=B;N.D=D
    if os.environ.get('EXEC_SELFCHECK')=='1':
        assert len(contexts())==20 and 40*300==CAPS['audit'];print('PASS import/matrix admission; native qualification required before collection')
    else:
        D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':gpu_job(root,cfg)
            else:coordinator(root,cfg)
        except BaseException as exc:D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
