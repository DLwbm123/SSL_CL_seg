"""Complete action returns at saved OFF-first prefixes before policy fitting."""
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

CAPS=dict(qualification=8,audit=48000)


def assemble(rows):
    keys=sorted({(r['context'],r['stream'],r['step'],r['prefix']) for r in rows});groups=[]
    for key in keys:
        group=sorted((r for r in rows if (r['context'],r['stream'],r['step'],r['prefix'])==key),key=lambda r:r['action'])
        assert [r['action'] for r in group]==list(range(13)) and sum(r['reused'] for r in group)==1
        assert key[2:]==(200,'OFF') and all(abs(r['gain']+r['old_final']-r['old_entry']-r['dense_reward'])<1e-12 for r in group)
        groups.append(group)
    values=np.array([[r['dense_reward'] for r in g] for g in groups],dtype=np.float64);assert np.isfinite(values).all()
    return keys,values


def selfcheck():
    rows=[dict(context='synthetic',stream=1,step=200,prefix='OFF',action=a,reused=a==9,gain=float(a),old_final=0.,old_entry=0.,dense_reward=float(a)) for a in range(13)]
    keys,values=assemble(rows);assert len(keys)==1 and values.shape==(1,13)
    for bad in (rows[:-1],rows+[rows[0]],[dict(r,dense_reward=float('nan')) for r in rows]):
        try:assemble(bad)
        except AssertionError:pass
        else:raise AssertionError('invalid table accepted')
    assert 40*12*100==CAPS['audit'] and 40*12*8==3840


def gpu_job(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    torch.set_num_threads(2);torch.cuda.set_device(0);random.seed(168);np.random.seed(168);torch.manual_seed(168)
    original=E.install_actions(c);E.action_check(c,original);O.install_off(c)
    contexts=b.contexts()+J.contexts();roles=c.split_roles(cfg['data']);assert roles==D.read(Path(cfg['action_root'])/'ROLES.private.json')
    assert len(roles['Q_train_new'])==len(roles['Q_train_old'])==4
    allowed=set(roles['A_fit']);counts=Counter();sample=c.e.primitive.CurrentData.__getitem__
    def guarded(ds,i):
        assert ds.role in ('train_labeled','train_unlabeled') and ds.rows[i]['case_id'] in (allowed if ds.role=='train_labeled' else set(roles['U_adapt']))
        counts['item_attempts_'+ds.role]+=1;value=sample(ds,i);counts['item_success_'+ds.role]+=1;return value
    c.e.primitive.CurrentData.__getitem__=guarded;ledger=b.JobLedger(root,cfg['caps'])
    def create(i,s):
        t=b.make(cfg,roles,ledger,contexts[i],300);t.provider.seed=168+10000*s
        for name,model in [('student',t.model),('ema',t.ema),('memory',t.memory)]:
            def hook(module,args,name=name):counts[name+'_image_forwards']+=len(args[0])
            model.register_forward_pre_hook(hook)
        entry=torch.load(Path(cfg['v109'])/f'jobs/collect{i}_{s}/OFF_ENTRY200.private.pt',map_location='cpu',weights_only=False)
        c.restore(t,entry);assert t.step==200
        return t,entry
    def update(t):
        before=t.provider.u_reads;t.update()
        if t.action==12:assert t.provider.u_reads==before and not t.last['active_U'];counts['off_updates']+=1
        else:counts['original_action_updates']+=1
    def stable(t,key):
        parts=[]
        for j in range(4):
            counts['probe_extractions']+=1;record=dict(key=key,probe=j,ordinal=counts['probe_extractions']);c.e.append(root/'EXTRACTION_LEDGER.jsonl',dict(event='attempt',**record))
            try:parts.append(P.probe(t,j,c).numpy().astype(np.float64))
            except BaseException:c.e.append(root/'EXTRACTION_LEDGER.jsonl',dict(event='failure',**record));raise
            c.e.append(root/'EXTRACTION_LEDGER.jsonl',dict(event='success',**record))
        return np.stack(parts)
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
            t,entry=create(i,1);t.category='qualification';t.key=str(i);before=c.snapshot(t);probe=stable(t,str(i));assert c.e.same(before,c.snapshot(t))
            t.action=12
            for _ in range(2):update(t)
            expected=c.snapshot(t);c.restore(t,entry);again=stable(t,str(i)+'/repeat');assert np.array_equal(probe,again) and c.e.same(before,c.snapshot(t));t.action=12
            for _ in range(2):update(t)
            assert c.e.same(expected,c.snapshot(t));del t;torch.cuda.empty_cache()
        assert dict(ledger.count)=={'qualification':8} and counts['probe_extractions']==16 and counts['off_updates']==8
        D.write(root/'QUALIFICATION.json',dict(status='PASS',paired_OFF_continuation_exact=2,stable_repeat_exact=2,full_snapshot_RNG_isolation=True,off_zero_U_reads=True,native_updates=8,queries=0))
    elif cfg['job']=='collect':
        i,s=cfg['context_index'],cfg['stream'];ctx=contexts[i];t,entry=create(i,s);src=Path(cfg['v109'])/f'jobs/collect{i}_{s}'
        reference=next(r for r in D.read(src/'RESULTS.json') if r['step']==100);trace=D.read(src/'OFF_CONTINUATION.private.json');chosen=reference['continuation_action'];assert 0<=chosen<12 and chosen==trace['action']
        counts['behavior_extractions']+=1;original=t.extract().numpy().astype(np.float64);assert original.tolist()==trace['state'] and c.e.same(entry,c.snapshot(t))
        probes=stable(t,'OFF200');assert c.e.same(entry,c.snapshot(t));t.category='audit';rows=[]
        for action in range(13):
            if action==chosen:gain=reference['gain'];new=reference['new_final'];old=reference['old_final']
            else:
                c.restore(t,entry);t.action=action;t.key=f'{ctx[3]}/{s}/OFF200/{action}'
                for _ in range(100):update(t)
                new=query(t,'Q_train_new',ctx[2]);old=query(t,'Q_train_old',('identity',1.));gain=reference['gain']+.75*(new-reference['new_final'])
                c.e.atomic_save(c.snapshot(t),root/f'ACTION_{action}_FINAL.private.pt')
            rows.append(dict(context=ctx[3],stream=s,step=200,prefix='OFF',action=action,reused=action==chosen,gain=gain,new_final=new,old_entry=reference['old_entry'],old_final=old,dense_reward=gain+old-reference['old_entry']))
            B.write(root/'STATUS.json',dict(status='RUNNING',actions_complete=len(rows),physical=dict(ledger.count)))
        assemble(rows);assert dict(ledger.count)=={'audit':1200} and counts['query_images']==96 and counts['probe_extractions']==4 and counts['off_updates']==100 and counts['original_action_updates']==1100
        D.write(root/'RESULTS.json',rows);np.savez(root/'FEATURES.private.npz',original=original,probes=probes)
        D.write(root/'COLLECTION_RECEIPT.json',dict(status='PASS',original_state_exact=True,reused_action=chosen,reused_return_exact=rows[chosen]['dense_reward']==reference['dense_reward'],rows=13))
        assert rows[chosen]['dense_reward']==reference['dense_reward']
    else:raise ValueError('unknown job')
    D.write(root/'COUNTS.json',dict(counts));D.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count),counts=dict(counts),time=time.time()))


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);selfcheck()
    D.write(root/'SELF_CHECK.json',dict(status='PASS',checks=['assembly','missing_duplicate_nonfinite_rejection','fixed_matrix'],model_calls=0))
    qual=[dict(id='qualification',job='qualification',caps=dict(qualification=8))]
    collect=[dict(id=f'collect{i}_{s}',job='collect',context_index=i,stream=s,caps=dict(audit=1200)) for i in range(20) for s in (1,2)];jobs=qual+collect
    (root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False);account=Accounting(root,caps=CAPS,jobs=jobs)
    N.schedule(root,cfg,qual,account);N.schedule(root,cfg,collect,account);assert dict(account.count)==dict(account.success)==CAPS and not account.failure
    rows=sum((D.read(root/'jobs'/j['id']/'RESULTS.json') for j in collect),[]);keys,returns=assemble(rows);assert len(keys)==40 and len(rows)==520
    contexts=b.contexts()+J.contexts();features={}
    for j in collect:features[(contexts[j['context_index']][3],j['stream'],200,'OFF')]=np.load(root/'jobs'/j['id']/'FEATURES.private.npz')
    original=np.stack([features[k]['original'] for k in keys]);probes=np.stack([features[k]['probes'] for k in keys]);assert original.shape==(40,24) and probes.shape==(40,4,24) and np.isfinite(original).all() and np.isfinite(probes).all()
    oldkeys=D.read(Path(cfg['v107'])/'KEYS.private.json');old=np.load(Path(cfg['v107'])/'DATASET.private.npz');oldreturns=np.load(Path(cfg['v109'])/'RETURNS13.private.npz');assert np.array_equal(oldreturns['returns'][:,:12],old['returns']) and np.array_equal(oldreturns['keys'],np.array(oldkeys,dtype=str))
    allkeys=[list(k)+['ORIGINAL'] for k in oldkeys]+[list(k) for k in keys];assert len(allkeys)==len({tuple(k) for k in allkeys})==120
    allreturns=np.concatenate((oldreturns['returns'],returns));assert allreturns.shape==(120,13) and np.array_equal(allreturns[:80],oldreturns['returns'])
    np.savez(root/'DATASET.private.npz',keys=np.array(allkeys,dtype=str),original=np.concatenate((old['original'],original)),probes=np.concatenate((old['probes'],probes)),returns=allreturns)
    D.write(root/'KEYS.private.json',allkeys);D.write(root/'NEW_RETURN_TABLE.json',rows);D.table(root/'NEW_RETURN_TABLE.csv',rows)
    counts=Counter()
    for j in jobs:counts.update(D.read(root/'jobs'/j['id']/'COUNTS.json'))
    assert counts['query_images']==3840 and counts['probe_extractions']==176 and counts['behavior_extractions']==40 and counts['off_updates']==4008 and counts['original_action_updates']==44000
    D.write(root/'DATASET_LOCK.json',dict(status='SEALED',states=120,actions=13,returns=1560,old80_states_1040_returns_preserved=True,OFF_prefix_states=40,reused_returns=40,time=time.time()))
    B.write(root/'COSTS.json',dict(native_updates=48008,actor_optimizer_updates=0,linear_solves=0,physical=dict(account.count),success=dict(account.success),failures=dict(account.failure),counts=dict(counts),development_query_images=0,new_annotation_cases=0))
    D.write(root/'FINAL.json',dict(status='COMPLETE',decision='OFF_PREFIX_COVERAGE_COMPLETE_NO_DEPLOYMENT_CLAIM',physical=dict(account.count),publication='PENDING',time=time.time()));B.write(root/'STATUS.json',D.read(root/'FINAL.json'))


if __name__=='__main__':
    import importlib.util
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN'])
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c,stage_b as b
    B=load('base_loss',cfg['base_entry']);D=load('diagnostic',cfg['diagnostic_entry']);E=load('actions12',cfg['expanded_entry']);P=load('probe',cfg['probe_entry']);J=load('coverage_contexts',cfg['coverage_entry']);O=load('off_action',cfg['off_entry'])
    N=load('scheduler',cfg['sequential_entry']);N.B=B;N.D=D
    if os.environ.get('EXEC_SELFCHECK')=='1':selfcheck();print('PASS assembly validation/matrix; zero model calls')
    else:
        D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':gpu_job(root,cfg)
            else:coordinator(root,cfg)
        except BaseException as exc:D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
