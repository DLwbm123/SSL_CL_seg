"""Experiment-local physical/functional ledger and independent resource admission."""
import fcntl,json,os,time,subprocess
from datetime import datetime
from pathlib import Path
import torch
from ..main_style_ssl_v1.runtime import write,append
from ..five_frameworks_v1.checkpoint import atomic_save


def allocated(root):
    return int(subprocess.check_output(['du','-s','-B1',str(root)],text=True).split()[0])


def guard(cfg,planned=0,new_stage=False):
    elapsed=time.time()-datetime.fromisoformat(cfg['clock_start']).timestamp()
    if elapsed>23.75*3600 or (new_stage and elapsed>23*3600):raise TimeoutError('shared original experiment clock exhausted')
    used=allocated(cfg['root'])+100_000_000
    if used+planned>=47_000_000_000:raise OSError('47GB protection; 2GB atomic-save reserve')
    return dict(allocated_bytes_including_metadata=used,elapsed_seconds=elapsed,hard_limit_bytes=49_000_000_000)


def event(cfg,kind,stage,phase):
    root=Path(cfg['root'])
    with (root/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'budget_counter.json';c=json.loads(path.read_text()) if path.exists() else dict(total=3,optimizer=0,functional=3,pending=0,by_phase={})
        key=stage+':'+phase
        n=c['by_phase'].get(key,0)
        cap=(32 if phase=='engineering' else (500 if stage=='codec' else 512 if stage in ('C0','C1') else 2000))
        if kind=='reserved':
            if c['total']+c['pending']>=11783 or n+c['pending']>=cap:raise PermissionError('frozen actual-call budget')
            c['pending']+=1
        else:
            assert c['pending']>0
            c['pending']-=1;c['total']+=1;c[kind]+=1;c['by_phase'][key]=n+1
        append(root/'budget_events.jsonl',dict(kind=kind,stage=stage,phase=phase,time=time.time(),total=c['total']))
        write(path,c)


def attach(opt,cfg,stage,phase):
    count=0
    def pre(*_):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in opt.param_groups for p in g['params']):raise FloatingPointError('nonfinite gradient')
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],1.,error_if_nonfinite=True)
        guard(cfg);event(cfg,'reserved',stage,phase)
    def post(*_):
        nonlocal count
        count+=1;event(cfg,'optimizer',stage,phase)
    opt.register_step_pre_hook(pre);opt.register_step_post_hook(post)


def save(cfg,state,path):
    guard(cfg,200_000_000)
    finite(state);atomic_save(state,Path(path))


def finite(value):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():raise FloatingPointError('nonfinite persistent tensor')
    elif isinstance(value,dict):
        for x in value.values():finite(x)
    elif isinstance(value,(list,tuple)):
        for x in value:finite(x)


def snapshot(model,opt,data,step):
    return dict(step=step,model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),reads=[data.l_reads,data.u_reads])


def restore(state,model,opt,data):
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);data.l_reads,data.u_reads=state['reads'];opt.zero_grad(set_to_none=True)


def exact(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b
