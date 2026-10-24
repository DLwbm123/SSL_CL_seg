"""Fixed original native core and one prehead residual adapter."""
import copy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .runtime import Calls,attach
from ..space_lora_v1.core import source_model
class Model(nn.Module):
    def __init__(self,parent,stage,arm,groups):
        super().__init__();self.parent=parent;self.adapter=nn.Sequential(nn.Conv2d(16,4,1),nn.ReLU(),nn.Conv2d(4,16,3,padding=1));nn.init.zeros_(self.adapter[2].weight);nn.init.zeros_(self.adapter[2].bias)
        self.stage=stage;self.arm=arm;self.calls=None;self.register_buffer('local_updates',torch.zeros((),dtype=torch.int64));self.register_buffer('dual',torch.zeros(groups));self.last=None
    def train(self,mode=True):super().train(mode);self.parent.eval();return self
    def configure(self):self.train();self.requires_grad_(False);self.adapter.requires_grad_(True);self.parent.eval()
    def forward(self,x):
        if self.calls:self.calls.sample(len(x));self.calls.extra('adapter_samples',len(x))
        h=self.parent.features(x);h=h+self.adapter(h);return F.log_softmax(self.parent.native_readout(h,x.shape[-2:]),1)
def make(cfg):
    from .risk import load_cache
    torch.manual_seed(20271801);torch.cuda.manual_seed_all(20271801);src=source_model(cfg);cache=load_cache(cfg) if cfg.get('cache') and not cfg.get('synthetic') else None
    m=Model(copy.deepcopy(src.parent).cpu(),cfg['stage'],cfg['arm'],len(cache['groups']) if cache else cfg.get('synthetic_groups',0)).cuda();del src
    kind=cfg.get('load',('source' if cfg['stage']=='source_eval' or (cfg['stage']=='target' and cfg['arm']!='B') else 'initial'))
    if not cfg.get('synthetic') and kind!='initial':
        path=Path(cfg['root'])/('source/F/final.pt' if kind=='source' else 'target/'+cfg['arm']+'/final.pt');state=torch.load(path,map_location='cpu',weights_only=False);assert state['step']==state['physical_updates']==2000
        weights={k:v for k,v in state['ema'].items() if k not in ('dual','local_updates')};missing,unexpected=m.load_state_dict(weights,strict=False);assert set(missing)=={'dual','local_updates'} and not unexpected
    m.configure();m.calls=Calls(cfg);object.__setattr__(m,'cfg',cfg);return m

def setup(cfg):
    m=make(cfg);c=m.calls;m.calls=None;ema=copy.deepcopy(m).eval().requires_grad_(False);m.calls=c;ema.calls=Calls(cfg)
    groups=[dict(params=[p for p in m.adapter.parameters() if (p.ndim>1)==matrix],weight_decay=.0005 if matrix else 0.) for matrix in (True,False)];opt=torch.optim.SGD(groups,lr=.001,momentum=.9,nesterov=True);attach(opt,cfg,m);return None,m,ema,opt

def frozen(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items() if k.startswith('parent.')}
def check_frozen(m):
    for k,v in getattr(m,'frozen_reference',{}).items():assert torch.equal(m.state_dict()[k].detach().cpu(),v),k
    assert all(p.grad is None for p in m.parent.parameters())
@torch.no_grad()
def ema_update(ema,m):
    from .runtime import record
    record(m.cfg,'dual_computations',1)
    active=m.stage=='source' or m.arm=='C'
    if active:
        record(m.cfg,'dual_active_updates',1);m.dual.add_(m.last_dual_update,alpha=.01).clamp_(0,10)
    assert torch.isfinite(m.dual).all();m.local_updates.add_(1)
    params=dict(m.named_parameters())
    for n,v in ema.state_dict().items():
        if n in params and params[n].requires_grad:v.mul_(.999).add_(m.state_dict()[n],alpha=.001)
        else:v.copy_(m.state_dict()[n])
    check_frozen(m);check_frozen(ema)
    if m.cfg['stage']=='source' and m.cfg['phase']=='formal' and int(m.local_updates) in (1500,1750,2000):
        from .run import diagnostic
        diagnostic(m,ema,m.current_data,int(m.local_updates))
