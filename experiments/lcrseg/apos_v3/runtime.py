"""Shared original clock and physical accounting for the independent V3 round."""
import fcntl,json,subprocess,time
from pathlib import Path
import torch
from ..main_style_ssl_v1.runtime import write,append
from ..five_frameworks_v1.checkpoint import atomic_save
CAPS={'source_evaluation_samples':2176,'target_qualification_samples':600,'target_qualification_reverse':150,'target_formal_samples':24000,'target_formal_reverse':6018,'target_evaluation_samples':7224,'component_samples':200000,'pixel_samples':2500000}

def resources(cfg):
    return dict(elapsed_seconds=time.time()-cfg['started_epoch'],allocated_bytes_including_ops=int(subprocess.check_output(['du','-s','-B1',cfg['root']]).split()[0])+2000000)
def guard(cfg,planned=0,new=False):
    r=resources(cfg)
    if r['elapsed_seconds']>=27900 or (new and r['elapsed_seconds']>=25200):raise TimeoutError('original 8h clock')
    if r['allocated_bytes_including_ops']+planned>=3500000000:raise OSError('3.5GB write protection')
    return r

def record(cfg,key,n,optimizer=False):
    root=Path(cfg['root'])
    with (root/'ledger.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);p=root/'COUNTERS.private.json'
        d=json.loads(p.read_text()) if p.exists() else dict(calls={},optimizer={},pending_optimizer=None)
        table=d['optimizer' if optimizer else 'calls'];cap=(32 if key.endswith('_qualification') else 6000 if key=='source_CAL_formal' else 2000) if optimizer else CAPS[key]
        if table.get(key,0)+n>cap:raise PermissionError('cap '+key)
        if optimizer:
            assert d.get('pending_optimizer',{}).get('key')==key
            d['pending_optimizer']=None
        table[key]=table.get(key,0)+n;append(root/'EVENTS.private.jsonl',dict(key=key,n=n,kind='optimizer' if optimizer else 'call',time=time.time()));write(p,d)
def reserve(cfg,key):
    root=Path(cfg['root'])
    with (root/'ledger.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);p=root/'COUNTERS.private.json'
        d=json.loads(p.read_text()) if p.exists() else dict(calls={},optimizer={},pending_optimizer=None)
        assert d.get('pending_optimizer') is None
        cap=32 if key.endswith('_qualification') else 6000 if key=='source_CAL_formal' else 2000
        if d['optimizer'].get(key,0)>=cap:raise PermissionError('optimizer cap')
        d['pending_optimizer']=dict(key=key,time=time.time());append(root/'EVENTS.private.jsonl',dict(key=key,n=1,kind='reserved_optimizer',time=time.time()));write(p,d)
class Calls:
    def __init__(self,cfg):self.cfg=cfg;self.bucket=cfg['bucket']
    def sample(self,n):guard(self.cfg);record(self.cfg,self.bucket+'_samples',n)
    def component(self,n):record(self.cfg,'component_samples',n)
    def pixel(self,n):record(self.cfg,'pixel_samples',n)
def install(cfg):
    original=torch.autograd.backward
    def backward(*a,**k):record(cfg,cfg['bucket']+'_reverse',1);return original(*a,**k)
    torch.autograd.backward=backward

def attach(opt,cfg,model):
    assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for g in opt.param_groups for p in g['params']}
    def pre(*_):
        guard(cfg)
        for prefix in ('adapter_','calibration'):
            pp=[p for n,p in model.named_parameters() if p.requires_grad and (n.startswith('adapter_') if prefix=='adapter_' else not n.startswith('adapter_'))]
            if pp:torch.nn.utils.clip_grad_norm_(pp,1.,error_if_nonfinite=True)
        reserve(cfg,cfg['group']+'_'+cfg['phase'])
    opt.register_step_pre_hook(pre);opt.register_step_post_hook(lambda *_:record(cfg,cfg['group']+'_'+cfg['phase'],1,True))
def save(cfg,state,path):
    from ..gyro_spatial_memory_v1.runtime import finite
    guard(cfg,150000000);finite(state);atomic_save(state,path)
