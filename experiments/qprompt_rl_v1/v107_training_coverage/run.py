"""Training-only context expansion, preserving frozen native collection semantics."""
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

CAPS = dict(qualification=8, entries=1200, audit=84000)


def contexts():
    return [(m,n,(kind,factor),f'coverage_m{m}_n{n}_{kind}') for m in (2000,8000) for n in (2,8) for kind,factor in (('identity',1.),('brightness',1.2),('contrast',.8))]


def assemble(rows):
    keys=sorted({(r['context'],r['stream'],r['step']) for r in rows});groups=[]
    for key in keys:
        group=sorted((r for r in rows if (r['context'],r['stream'],r['step'])==key),key=lambda r:r['action'])
        assert [r['action'] for r in group]==list(range(12)), 'incomplete or duplicate actions'
        assert all(r['state']==group[0]['state'] for r in group)
        groups.append(group)
    original=np.array([g[0]['state'] for g in groups],dtype=np.float64)
    rewards=np.array([[r['dense_reward'] for r in g] for g in groups],dtype=np.float64)
    assert np.isfinite(original).all() and np.isfinite(rewards).all()
    return keys,original,rewards


def selfcheck():
    rows=[dict(context='synthetic',stream=1,step=100,action=a,state=[0.]*24,dense_reward=float(a)) for a in range(12)]
    keys,x,y=assemble(rows);assert x.shape==(1,24) and y[0,11]==11
    for bad in (rows[:-1],rows+[rows[0]],[dict(r,dense_reward=float('nan')) for r in rows]):
        try:assemble(bad)
        except AssertionError:pass
        else:raise AssertionError('invalid table accepted')
    assert len(contexts())==12 and 24*3500==CAPS['audit'] and 12*100==CAPS['entries']


def gpu_job(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    torch.set_num_threads(2);torch.cuda.set_device(0)
    random.seed(168);np.random.seed(168);torch.manual_seed(168)
    original_actions=E.install_actions(c);E.action_check(c,original_actions)
    roles=c.split_roles(cfg['data']);assert roles==D.read(Path(cfg['action_root'])/'ROLES.private.json')
    assert len(roles['Q_train_new'])==len(roles['Q_train_old'])==4
    allowed=set(roles['A_fit']);counts=Counter();sample=c.e.primitive.CurrentData.__getitem__
    def guarded(ds,i):
        assert ds.role in ('train_labeled','train_unlabeled')
        assert ds.rows[i]['case_id'] in (allowed if ds.role=='train_labeled' else set(roles['U_adapt']))
        counts['item_attempts_'+ds.role]+=1;value=sample(ds,i);counts['item_success_'+ds.role]+=1
        return value
    c.e.primitive.CurrentData.__getitem__=guarded
    ledger=b.JobLedger(root,cfg['caps']);campaign=Path(cfg['campaign'])
    def create(ctx):
        t=b.make(cfg,roles,ledger,ctx,300)
        for name,model in [('student',t.model),('ema',t.ema),('memory',t.memory)]:
            def hook(module,args,name=name):counts[name+'_image_forwards']+=len(args[0])
            model.register_forward_pre_hook(hook)
        return t
    def query(t,role,condition):
        nonlocal allowed
        assert role in ('Q_train_new','Q_train_old');allowed=set(roles[role]);counts['query_images']+=len(allowed)
        record=dict(role=role,step=t.step,ordinal=counts['query_images']//4);c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='attempt',**record))
        try:value=scores(t,roles,role,condition)['macro']
        except BaseException:c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='failure',**record));raise
        else:c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='success',**record));return value
        finally:allowed=set(roles['A_fit'])
    def stable(t,key):
        parts=[]
        for j in range(4):
            counts['probe_extractions']+=1;record=dict(key=key,probe=j,ordinal=counts['probe_extractions'])
            c.e.append(root/'EXTRACTION_LEDGER.jsonl',dict(event='attempt',**record))
            try:parts.append(P.probe(t,j,c).numpy().astype(np.float64))
            except BaseException:c.e.append(root/'EXTRACTION_LEDGER.jsonl',dict(event='failure',**record));raise
            c.e.append(root/'EXTRACTION_LEDGER.jsonl',dict(event='success',**record))
        return np.stack(parts)
    models=[V.load12(Path(cfg['v99'])/f'jobs/fit/{arm}_{s}.private.pt') for arm in ('EXPANDED_CE','EXPANDED_RL') for s in (601,602)]
    def select(t,generator):
        counts['behavior_extractions']+=1
        return B.select(t,models,generator)
    if cfg['job']=='qualification':
        keys,_,_,_,_=D.dataset(cfg,'V101');saved=np.load(Path(cfg['v103'])/'FEATURES.private.npz')['probes'].mean(1)
        for i,action in ((0,9),(7,11)):
            ctx=b.contexts()[i];t=create(ctx);t.provider.seed=10168
            entry=torch.load(Path(cfg['v92'])/f'jobs/collect{i}_1/ENTRY100_STREAM.private.pt',map_location='cpu',weights_only=False)
            c.restore(t,entry);before=c.snapshot(t);value=stable(t,str(i))
            assert np.array_equal(value.mean(0),saved[keys.index((ctx[3],1,100))])
            selected,trace=select(t,torch.Generator().manual_seed(860901+100+i));assert c.e.same(before,c.snapshot(t))
            t.category='qualification';t.key=str(i);t.action=action
            for _ in range(2):t.update()
            expected=c.snapshot(t);c.restore(t,entry);again=stable(t,str(i)+'/repeat')
            repeated,trace2=select(t,torch.Generator().manual_seed(860901+100+i))
            assert np.array_equal(value,again) and selected==repeated and trace==trace2 and c.e.same(before,c.snapshot(t))
            t.action=action
            for _ in range(2):t.update()
            assert c.e.same(expected,c.snapshot(t));del t;torch.cuda.empty_cache()
        assert dict(ledger.count)=={'qualification':8} and counts['probe_extractions']==16
        D.write(root/'QUALIFICATION.json',dict(status='PASS',stable_original_exact=2,paired_continuation_exact=2,probe_and_policy_state_RNG_isolation=True))
    elif cfg['job']=='entry':
        ctx=contexts()[cfg['context_index']];t=create(ctx);t.category='entries';t.key=ctx[3];t.action=0
        for _ in range(100):t.update()
        assert t.step==100 and t.options['total_steps']==300
        entry=c.snapshot(t);c.e.atomic_save(entry,root/'ENTRY100.private.pt')
        new=query(t,'Q_train_new',ctx[2]);old=query(t,'Q_train_old',('identity',1.))
        assert c.e.same(entry,c.snapshot(t))
        D.write(root/'BASELINES.json',dict(new_entry=new,old_entry=old))
        assert dict(ledger.count)=={'entries':100} and counts['query_images']==8
    elif cfg['job']=='collect':
        i=cfg['context_index'];stream=cfg['stream'];ctx=contexts()[i];t=create(ctx)
        src=campaign/f'jobs/entry{i}';entry=torch.load(src/'ENTRY100.private.pt',map_location='cpu',weights_only=False);c.restore(t,entry)
        t.provider.seed=168+10000*stream
        random.seed(860100+stream);np.random.seed(860100+stream);torch.manual_seed(860100+stream);torch.cuda.manual_seed_all(860100+stream)
        entry=c.snapshot(t);c.e.atomic_save(entry,root/'ENTRY100_STREAM.private.pt')
        generator=torch.Generator().manual_seed(860901+100*stream+i+8);first,trace=select(t,generator);afterfirst=generator.get_state()
        probes100=stable(t,'100');base=D.read(src/'BASELINES.json');newentry=base['new_entry'];oldentry=base['old_entry'];rows=[];t.category='audit'
        def row(step,action,state,gain,oldfinal,continuation,reused=False):
            return dict(context=ctx[3],stream=stream,step=step,action=action,state=state,gain=gain,old_entry=oldentry,old_final=oldfinal,dense_reward=gain+oldfinal-oldentry,continuation_action=continuation,reused=reused)
        for action in range(12):
            c.restore(t,entry);t.action=action;t.key=f'{ctx[3]}/{stream}/first{action}';paired=torch.Generator();paired.set_state(afterfirst);values={}
            for _ in range(200):
                if t.step==200:
                    prefix=c.snapshot(t);second,secondtrace=select(t,paired);t.action=second
                    if action==first:
                        retained=prefix;selectedsecond=second;retainedtrace=secondtrace;c.e.atomic_save(prefix,root/'ON_POLICY_ENTRY200.private.pt')
                t.update()
                if t.step in (150,300):values[t.step]=query(t,'Q_train_new',ctx[2])
                if t.step==300:oldfinal=query(t,'Q_train_old',('identity',1.))
            gain=.25*(values[150]-newentry)+.75*(values[300]-newentry);rows.append(row(100,action,trace['state'],gain,oldfinal,second))
            c.e.atomic_save(c.snapshot(t),root/f'FIRST_{action}_FINAL.private.pt')
            if action==first:baseline=values[300];basegain=gain;baseold=oldfinal
            B.write(root/'STATUS.json',dict(rows=len(rows),physical=dict(ledger.count)))
        c.restore(t,retained);counts['behavior_extractions']+=1;state=t.extract().tolist();assert state==retainedtrace['state'];probes200=stable(t,'200')
        for action in range(12):
            if action==selectedsecond:gain=basegain;oldfinal=baseold
            else:
                c.restore(t,retained);t.action=action;t.key=f'{ctx[3]}/{stream}/second{action}'
                for _ in range(100):t.update()
                gain=basegain+.75*(query(t,'Q_train_new',ctx[2])-baseline);oldfinal=query(t,'Q_train_old',('identity',1.))
                c.e.atomic_save(c.snapshot(t),root/f'SECOND_{action}_FINAL.private.pt')
            rows.append(row(200,action,state,gain,oldfinal,-1,action==selectedsecond))
            B.write(root/'STATUS.json',dict(rows=len(rows),physical=dict(ledger.count)))
        assert len(rows)==24 and dict(ledger.count)=={'audit':3500} and counts['query_images']==232 and counts['probe_extractions']==8 and counts['behavior_extractions']==14
        assemble(rows)
        D.write(root/'TRAINING_TABLE.private.json',rows);D.write(root/'BEHAVIOR.private.json',dict(first=trace,second=retainedtrace))
        np.savez(root/'PROBES.private.npz',step100=probes100,step200=probes200)
        D.write(root/'COLLECTION_RECEIPT.json',dict(status='PASS',rows=24,first_action=first,second_action=selectedsecond,reused_current_second_returns=1))
    else:raise ValueError('unknown GPU job')
    D.write(root/'COUNTS.json',dict(counts));D.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count),counts=dict(counts),time=time.time()))


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selfcheck();D.write(root/'SELF_CHECK.json',dict(status='PASS',optimizer_updates=0,checks=['valid_assembly','missing_duplicate_nonfinite_rejection','matrix_counts']))
    qual=[dict(id='qualification',job='qualification',caps=dict(qualification=8))]
    entries=[dict(id=f'entry{i}',job='entry',context_index=i,caps=dict(entries=100)) for i in range(12)]
    collect=[dict(id=f'collect{i}_{s}',job='collect',context_index=i,stream=s,caps=dict(audit=3500)) for i in range(12) for s in (1,2)]
    jobs=qual+entries+collect;(root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    account=Accounting(root,caps=CAPS,jobs=jobs)
    for phase in (qual,entries,collect):N.schedule(root,cfg,phase,account)
    assert dict(account.count)==dict(account.success)==CAPS and not account.failure
    rows=sum((D.read(root/'jobs'/j['id']/'TRAINING_TABLE.private.json') for j in collect),[])
    keys,original,rewards=assemble(rows);assert len(keys)==48 and len(rows)==576
    features={}
    for j in collect:
        values=np.load(root/'jobs'/j['id']/'PROBES.private.npz');ctx=contexts()[j['context_index']][3]
        for step in (100,200):features[(ctx,j['stream'],step)]=values[f'step{step}']
    probes=np.stack([features[key] for key in keys]);assert probes.shape==(48,4,24) and np.isfinite(probes).all()
    oldkeys,oldoriginal,oldrewards,_,_=D.dataset(cfg,'V101');saved=np.load(Path(cfg['v103'])/'FEATURES.private.npz')
    assert np.array_equal(saved['original'],oldoriginal) and not set(keys).intersection(oldkeys)
    allkeys=oldkeys+keys;assert len(allkeys)==80
    np.savez(root/'DATASET.private.npz',keys=np.array(allkeys,dtype=str),original=np.concatenate((oldoriginal,original)),probes=np.concatenate((saved['probes'],probes)),returns=np.concatenate((oldrewards,rewards)))
    D.write(root/'KEYS.private.json',allkeys)
    counts=Counter()
    for j in jobs:counts.update(D.read(root/'jobs'/j['id']/'COUNTS.json'))
    assert counts['query_images']==5664 and counts['probe_extractions']==208 and counts['behavior_extractions']==340
    public=[{k:v for k,v in r.items() if k!='state'} for r in rows]
    D.write(root/'NEW_RETURN_TABLE.json',public);D.table(root/'NEW_RETURN_TABLE.csv',public)
    D.write(root/'DATASET_LOCK.json',dict(status='SEALED',old_states=32,new_states=48,total_states=80,actions=12,total_returns=960,old_arrays_preserved=True,no_development_access=True,time=time.time()))
    B.write(root/'COSTS.json',dict(native_updates=85208,actor_optimizer_updates=0,linear_solves=0,physical=dict(account.count),success=dict(account.success),failures=dict(account.failure),counts=dict(counts),new_annotation_cases=0))
    D.write(root/'FINAL.json',dict(status='COMPLETE',decision='TRAINING_COVERAGE_DATASET_ONLY_NO_PERFORMANCE_CLAIM',physical=dict(account.count),publication='PENDING',time=time.time()));B.write(root/'STATUS.json',D.read(root/'FINAL.json'))


if __name__=='__main__':
    import importlib.util
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN'])
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c,stage_b as b
    B=load('base_loss',cfg['base_entry']);D=load('diagnostic',cfg['diagnostic_entry']);V=load('actor12',cfg['v99_entry']);P=load('probe',cfg['probe_entry']);E=load('actions12',cfg['expanded_entry'])
    N=load('scheduler',cfg['v106_entry']);N.D=D;N.B=B
    if os.environ.get('EXEC_SELFCHECK')=='1':selfcheck();print('PASS assembly validation and matrix counts; zero optimizer/query')
    else:
        D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':gpu_job(root,cfg)
            else:coordinator(root,cfg)
        except BaseException as exc:D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
