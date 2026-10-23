"""Original-clock, storage and physical-call accounting; no automatic replay."""
import fcntl,json,os,subprocess,time
from pathlib import Path
import torch
from ..main_style_ssl_v1.runtime import write,append
from ..five_frameworks_v1.checkpoint import atomic_save
CAPS={'source_qualification_samples':240,'source_qualification_reverse':108,'source_formal_samples':12000,'source_formal_reverse':6000,'target_qualification_samples':240,'target_qualification_reverse':108,'target_formal_samples':12000,'target_formal_reverse':6000,'evaluation_samples':5244,'evaluation_reverse':0,'conditional_solves':1000000}
ACTIVE=None

def guard(cfg,planned=0,new=False):
    elapsed=time.time()-cfg['started_epoch']
    if elapsed>=27900 or (new and elapsed>=25200):raise TimeoutError('original 8h clock')
    size=int(subprocess.check_output(['du','-s','-B1',cfg['root']]).split()[0])+2000000
    if size+planned>=3500000000:raise OSError('3.5GB write protection')
    return dict(elapsed_seconds=elapsed,allocated_bytes_including_ops=size)

def record(cfg,key,n,event='call'):
    root=Path(cfg['root'])
    with (root/'ledger.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p=root/'COUNTERS.private.json';d=json.loads(p.read_text()) if p.exists() else dict(calls={},optimizer={},pending=None)
        table=d['optimizer'] if event=='optimizer' else d['calls']
        cap=32 if key.endswith('_qualification') else 2000
        if event!='optimizer':cap=CAPS[key]
        if table.get(key,0)+n>cap:raise PermissionError('frozen cap '+key)
        table[key]=table.get(key,0)+n
        append(root/'EVENTS.private.jsonl',dict(event=event,key=key,n=n,time=time.time()))
        write(p,d)

class Calls:
    def __init__(self,cfg):self.cfg=cfg;self.bucket=cfg['bucket']
    def sample(self,n):
        guard(self.cfg);record(self.cfg,self.bucket+'_samples',n)
    def solves(self,n):record(self.cfg,'conditional_solves',n)

def install(cfg):
    global ACTIVE
    ACTIVE=cfg
    original=torch.autograd.backward
    def backward(*args,**kwargs):
        record(cfg,cfg['bucket']+'_reverse',1)
        return original(*args,**kwargs)
    torch.autograd.backward=backward

def attach(opt,cfg,model):
    names={id(p) for p in model.parameters() if p.requires_grad}
    assert names=={id(p) for g in opt.param_groups for p in g['params']}
    def pre(*_):
        guard(cfg)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        key=cfg['group']+'_'+cfg['phase']
        p=Path(cfg['root'])/'COUNTERS.private.json';d=json.loads(p.read_text()) if p.exists() else {'optimizer':{}}
        if d['optimizer'].get(key,0)>=(32 if cfg['phase']=='qualification' else 2000):raise PermissionError('optimizer cap')
    def post(*_):record(cfg,cfg['group']+'_'+cfg['phase'],1,'optimizer')
    opt.register_step_pre_hook(pre);opt.register_step_post_hook(post)

def save(cfg,state,path):
    guard(cfg,150000000)
    from ..gyro_spatial_memory_v1.runtime import finite
    finite(state);atomic_save(state,path)
