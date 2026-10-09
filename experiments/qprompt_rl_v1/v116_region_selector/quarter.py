"""Fixed simple policies with a quarter-rate remaining native schedule."""
import fcntl
import importlib.util
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path


def scaled_step(optimizer, original, record):
    def step(*args,**kwargs):
        rates=[g['lr'] for g in optimizer.param_groups]
        try:
            for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate*.25
            record(rates,[g['lr'] for g in optimizer.param_groups])
            return original(*args,**kwargs)
        finally:
            for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate
    step._wrapped_by_lr_sched=True
    return step


def selfcheck():
    class Dummy: param_groups=[dict(lr=.001),dict(lr=.0005)]
    opt=Dummy();seen=[]
    def inspect():seen.append([g['lr'] for g in opt.param_groups]);return 17
    fn=scaled_step(opt,inspect,lambda old,new:None)
    assert fn()==17 and seen==[[.00025,.000125]] and [g['lr'] for g in opt.param_groups]==[.001,.0005]
    opt.param_groups[0]['lr']=.0008
    assert fn()==17 and seen[-1]==[.0002,.000125] and opt.param_groups[0]['lr']==.0008
    def fail():raise ValueError('synthetic')
    try:scaled_step(opt,fail,lambda old,new:None)()
    except ValueError:pass
    else:raise AssertionError('exception swallowed')
    assert [g['lr'] for g in opt.param_groups]==[.0008,.0005]


def configure(cfg,root):
    global N
    spec=importlib.util.spec_from_file_location('base13',cfg['v111_entry']);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);v.configure(cfg);N=v.N
    N.STAGE='V113';N.METHODS=('OFF','GLOBAL','TIME');N.CAPS=dict(qualification=8,development=7200);N.c.CAPS['development']=600
    original=N.b.make;counts=Counter()
    def make(*args,**kwargs):
        t=original(*args,**kwargs)
        def record(base,effective):
            assert 100<=t.step<300 and len(base)==len(effective) and all(x==y*.25 for x,y in zip(effective,base))
            counts['calls']+=1;N.c.e.append(root/'RATE_LEDGER.jsonl',dict(ordinal=counts['calls'],key=t.key,category=t.category,step=t.step,base=base,effective=effective))
        t.optimizer.step=scaled_step(t.optimizer,t.optimizer.step,record);return t
    N.b.make=make


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);selfcheck()
    assert N.D.read(Path(cfg['v111'])/'FINAL.json')['status']=='COMPLETE' and N.D.read(Path(cfg['v112'])/'FINAL.json')['status']=='COMPLETE'
    baseline=N.D.read(Path(cfg['v112'])/'METHOD_SUMMARY.json')
    assert all(r['weighted_new_gain']<=0 and r['final_new_minus_entry']<=0 for r in baseline if r['stage']=='V111')
    (root/'jobs').mkdir(exist_ok=False);(root/'jobs/fit').symlink_to(Path(cfg['v111'])/'jobs/fit',target_is_directory=True)
    (root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    qual=[dict(id='qualification',job='qualification',caps=dict(qualification=8))]
    train=[dict(id=f'train{i}_{s}',job='train',context_index=i,stream=s,caps=dict(development=600)) for i in range(4) for s in (3,4,5)]
    evaluate=[dict(id=f'evaluate{i}_{s}',job='evaluate',context_index=i,stream=s,caps={}) for i in range(4) for s in (3,4,5)]
    account=Accounting(root,caps=N.CAPS,jobs=qual+train+evaluate)
    N.schedule(root,cfg,qual,account);N.schedule(root,cfg,train,account)
    assert dict(account.count)==N.CAPS and not account.failure
    assert sum(len(list((root/'jobs'/j['id']).glob('*_TRAINING.json'))) for j in train)==36
    N.D.write(root/'ENDPOINT_LOCK.json',dict(status='SEALED_BEFORE_QUERY_READOUT',trajectories=36,snapshots=72,time=time.time(),physical=dict(account.count)))
    N.schedule(root,cfg,evaluate,account)
    rows=sum((N.D.read(root/'jobs'/j['id']/'RESULTS.json') for j in evaluate),[])
    assert len(rows)==len({(r['context'],r['stream'],r['method']) for r in rows})==36
    counts=Counter()
    for j in qual+train+evaluate:counts.update(N.D.read(root/'jobs'/j['id']/'COUNTS.json'))
    assert counts['query_images']==432 and counts['probe_extractions']==304
    N.D.write(root/'DEVELOPMENT_RESULTS.json',dict(status=N.STAGE+'_COMPLETE_REQUIRES_MATCHED_ABSOLUTE_ANALYSIS',rows=rows));N.D.table(root/'DEVELOPMENT_RESULTS.csv',rows)
    N.B.write(root/'COSTS.json',dict(native_updates=7208,actor_optimizer_updates=0,linear_solves=0,physical=dict(account.count),success=dict(account.success),failures=dict(account.failure),counts=dict(counts),new_annotation_cases=0))
    N.D.write(root/'FINAL.json',dict(status='COMPLETE',decision='PENDING_MATCHED_ABSOLUTE_ANALYSIS',time=time.time(),publication='PENDING'));N.B.write(root/'STATUS.json',N.D.read(root/'FINAL.json'))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN']);selfcheck()
    if os.environ.get('EXEC_SELFCHECK')=='1':print('PASS effective rate scaling, restoration and exceptions; zero model calls')
    else:
        configure(cfg,root);N.D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':N.gpu_job(root,cfg)
            else:coordinator(root,cfg)
        except BaseException as exc:N.D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
