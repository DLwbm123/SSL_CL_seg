"""One original source parent, frozen calibration and trainable observation adapter."""
import copy
from pathlib import Path
import torch
from torch.nn import functional as F
from ..apos_v2.model import Model as Previous
from ..space_lora_v1.core import source_model
from .math import solve,area
from .runtime import Calls,attach

class Model(Previous):
    def configure(self):
        self.train();self.requires_grad_(False)
        if self.stage=='target':self.adapter_A.requires_grad_(True);self.adapter_B.requires_grad_(True)
        self.parent.eval();self.aggregate={}
    def forward(self,x):
        if self.calls:self.calls.sample(len(x)*2)
        with torch.no_grad():
            h0=self.parent.features(x);q0=F.log_softmax(self.parent.native_readout(h0,x.shape[-2:]),1)
            features=area(h0,32);s2=1e-6+(1-1e-6)*self.scale_head(features).sigmoid();pi=F.softmax(self.mode_head(features.mean((2,3))),1);w=s2/(s2+self.reference_mse[None,:,None,None])
        hs=self.parent.features(x);h=hs+self.adapter_B(torch.tanh(self.adapter_A(hs)));q=F.log_softmax(self.parent.native_readout(h,x.shape[-2:]),1)
        if self.arm=='N':lp=q;stats={}
        else:lp,stats=self.infer(q,q0,w,pi,self.arm)
        if self.stage=='source_eval':
            err=float((lp.exp()-q0.exp()).abs().max());assert err<=5e-7 and torch.equal(lp.argmax(1),q0.argmax(1)),('source function mismatch',self.arm,err)
            stats.update(source_probability_max_error=err,source_argmax_exact=True)
        self.last=dict(logq=q,logq0=q0,w=w,pi=pi,stats=stats)
        for k,v in stats.items():
            if isinstance(v,(float,int)):
                a=self.aggregate.setdefault(k,dict(sum=0.,max=-float('inf'),count=0));a['sum']+=v;a['max']=max(a['max'],v);a['count']+=1
            elif isinstance(v,list) and k=='class_flip_counts':
                a=self.aggregate.setdefault(k,dict(sum=[0]*4,count=0));a['sum']=[x+y for x,y in zip(a['sum'],v)];a['count']+=1
        return lp
    def infer(self,q,q0,w,pi,arm):
        def capture(meta,tensors):
            from .runtime import guard
            from ..main_style_ssl_v1.runtime import write
            from ..five_frameworks_v1.checkpoint import atomic_save
            root=Path(self.cfg['root']);guard(self.cfg,100000000)
            if not (root/'SOLVER_FAILURE.private.json').exists():
                write(root/'SOLVER_FAILURE.private.json',meta);atomic_save({k:v.detach().cpu() for k,v in tensors.items()},root/'solver_failure.private.pt')
        return solve(q,q0,w,pi,{'mu':self.prior_mu,'factor':self.prior_factor,'sigma2':self.prior_sigma2},arm,self.calls,capture)

def make(cfg,prior_override=None):
    torch.manual_seed(20271611);torch.cuda.manual_seed_all(20271611);original=source_model(cfg)
    prior=prior_override if prior_override is not None else torch.load(Path(cfg['reuse_root'])/'prior.private.pt',map_location='cpu',weights_only=False)
    m=Model(copy.deepcopy(original.parent).cpu(),prior,cfg['arm'],cfg['stage']).cuda();del original
    if prior_override is None:
        state=torch.load(Path(cfg['reuse_root'])/'source/CAL/final.pt',map_location='cpu',weights_only=False);assert state['step']==state['physical_updates']==6000;m.load_state_dict(state['ema'])
    m.local_updates.zero_();m.configure();m.calls=Calls(cfg);object.__setattr__(m,'cfg',cfg);return m

def setup(cfg):
    m=make(cfg);calls=m.calls;m.calls=None;ema=copy.deepcopy(m).eval().requires_grad_(False);m.calls=calls;ema.calls=Calls(cfg)
    groups=[dict(params=[p for p in m.parameters() if p.requires_grad and (p.ndim>1)==matrix],weight_decay=.0005 if matrix else 0.) for matrix in (True,False)];groups=[g for g in groups if g['params']]
    opt=torch.optim.SGD(groups,lr=.001,momentum=.9,nesterov=True);attach(opt,cfg,m);return None,m,ema,opt

def frozen(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items() if not k.startswith(('adapter_','local_updates'))}
def check_frozen(m):
    for k,v in getattr(m,'frozen_reference',{}).items():assert torch.equal(m.state_dict()[k].detach().cpu(),v),k
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('adapter_'))
@torch.no_grad()
def ema_update(ema,m):
    m.local_updates.add_(1);params=dict(m.named_parameters())
    for n,v in ema.state_dict().items():
        if n in params and params[n].requires_grad:v.mul_(.999).add_(m.state_dict()[n],alpha=.001)
        else:v.copy_(m.state_dict()[n])
    check_frozen(m);check_frozen(ema)
