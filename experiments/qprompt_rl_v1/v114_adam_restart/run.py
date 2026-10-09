"""Restart inherited Adam state once at ENTRY100, keeping the quarter schedule."""
import copy
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path


def reset(optimizer):
    assert optimizer.state
    optimizer.state.clear()
    assert not optimizer.state


def selfcheck():
    class Dummy:
        state={0:dict(step=100,exp_avg=3.,exp_avg_sq=4.)}
        param_groups=[dict(lr=.0005,params=[17])]
    opt=Dummy();groups=copy.deepcopy(opt.param_groups);reset(opt)
    assert not opt.state and opt.param_groups==groups
    try:reset(opt)
    except AssertionError:pass
    else:raise AssertionError('empty inherited state accepted')


def configure(cfg,root):
    global Q,N
    spec=importlib.util.spec_from_file_location('quarter',cfg['quarter_entry']);Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q);Q.configure(cfg,root);N=Q.N;N.STAGE='V114'
    original=N.b.make;ordinal=0
    def make(*args,**kwargs):
        t=original(*args,**kwargs);step=t.optimizer.step
        def restarted(*a,**kw):
            nonlocal ordinal
            if t.step==100:
                states=list(t.optimizer.state.values());assert len(states)==28 and all(int(s['step'])==100 for s in states)
                before=N.c.snapshot(t) if t.category=='qualification' else None
                groups=[{k:v for k,v in g.items() if k!='params'} for g in t.optimizer.param_groups];schedule=copy.deepcopy(t.scheduler.state_dict())
                reset(t.optimizer)
                assert groups==[{k:v for k,v in g.items() if k!='params'} for g in t.optimizer.param_groups] and N.c.e.same(schedule,t.scheduler.state_dict())
                if before is not None:
                    after=N.c.snapshot(t);before['optimizer']['state']={};assert N.c.e.same(before,after)
                ordinal+=1;N.c.e.append(root/'RESET_LEDGER.jsonl',dict(ordinal=ordinal,key=t.key,category=t.category,native_step=100,inherited_states=28,inherited_adam_step=100,after_states=0,scheduler_preserved=True,qualification_fullsnapshot_check=before is not None))
            return step(*a,**kw)
        restarted._wrapped_by_lr_sched=True;t.optimizer.step=restarted;return t
    N.b.make=make


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN']);selfcheck()
    if os.environ.get('EXEC_SELFCHECK')=='1':print('PASS state clear preserves groups and rejects empty state; zero model/optimizer calls')
    else:
        configure(cfg,root);N.D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':N.gpu_job(root,cfg)
            else:
                assert N.D.read(Path(cfg['v113'])/'COMPLETION_AUDIT.json')['status']=='PASS' and N.D.read(Path(cfg['v113'])/'FINAL_PUBLICATION.json')['anonymous_http']=='200'
                Q.coordinator(root,cfg)
        except BaseException as exc:N.D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
