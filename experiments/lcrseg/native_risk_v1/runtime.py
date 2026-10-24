"""Shared original clock and physical accounting for the independent acquisition/anatomy round."""
import fcntl,json,subprocess,time
from pathlib import Path
import torch
from ..main_style_ssl_v1.runtime import write,append
from ..five_frameworks_v1.checkpoint import atomic_save
CAPS={'source_qualification_samples': 160, 'source_qualification_reverse': 40, 'source_formal_samples': 8000, 'source_formal_reverse': 2024, 'source_diagnostic_samples': 192, 'source_evaluation_samples': 544, 'target_qualification_samples': 480, 'target_qualification_reverse': 120, 'target_formal_samples': 24000, 'target_formal_reverse': 6072, 'target_evaluation_samples': 3612, 'source_risk_calibration_samples': 460, 'target_risk_calibration_samples': 1270, 'adapter_samples': 39000, 'transform_samples': 18000, 'spline_samples': 36000, 'bias_samples': 18000, 'anatomy_samples': 72000, 'equivariance_samples': 18000, 'risk_class_samples': 120000, 'dual_computations': 8128, 'dual_active_updates': 4064}

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
        table=d['optimizer' if optimizer else 'calls'];cap=(32 if key.endswith('_qualification') else 2000) if optimizer else CAPS[key]
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
        cap=32 if key.endswith('_qualification') else 2000
        if d['optimizer'].get(key,0)>=cap:raise PermissionError('optimizer cap')
        d['pending_optimizer']=dict(key=key,time=time.time());append(root/'EVENTS.private.jsonl',dict(key=key,n=1,kind='reserved_optimizer',time=time.time()));write(p,d)
class Calls:
    def __init__(self,cfg):self.cfg=cfg;self.bucket=cfg['bucket']
    def sample(self,n):guard(self.cfg);record(self.cfg,self.bucket+'_samples',n)
    def extra(self,key,n):record(self.cfg,key,n)

def install(cfg):
    original=torch.autograd.backward
    def backward(*a,**k):record(cfg,cfg['bucket']+'_reverse',1);return original(*a,**k)
    torch.autograd.backward=backward

def attach(opt,cfg,model):
    assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for g in opt.param_groups for p in g['params']}
    def pre(*_):
        guard(cfg)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        reserve(cfg,cfg['group']+'_'+cfg['phase'])
    opt.register_step_pre_hook(pre);opt.register_step_post_hook(lambda *_:record(cfg,cfg['group']+'_'+cfg['phase'],1,True))
def save(cfg,state,path):
    from ..gyro_spatial_memory_v1.runtime import finite
    guard(cfg,150000000);finite(state);atomic_save(state,path)

class SyntheticCalls:
    CAPS={'native_samples':128,'reverse':64,'adapter_samples':128,'anatomy_samples':256,'transform_samples':128,'spline_samples':512,'bias_samples':256,'equivariance_samples':128,'risk_class_samples':384,'dual_computations':64,'dual_active_updates':64}
    def __init__(self,cfg):self.cfg=cfg;self.count={}
    def sample(self,n):self.extra('native_samples',n)
    def extra(self,key,n):
        assert self.count.get(key,0)+n<=self.CAPS[key],key
        self.count[key]=self.count.get(key,0)+n
        write(Path(self.cfg['root'])/'SYNTHETIC_CALLS.json',self.count)
