"""Original native readout, a zero-entry adapter, and detached calibration."""
import copy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .math import solve,area
from .runtime import Calls,attach
from ..space_lora_v1.core import source_model

class Model(nn.Module):
    def __init__(self,parent,prior,arm,stage):
        super().__init__();self.parent=parent;self.arm=arm;self.stage=stage;self.calls=None
        self.adapter_A=nn.Conv2d(16,8,1,bias=False);self.adapter_B=nn.Conv2d(8,16,1,bias=False);nn.init.zeros_(self.adapter_B.weight)
        self.scale_head=nn.Conv2d(16,3,1);nn.init.zeros_(self.scale_head.weight);nn.init.constant_(self.scale_head.bias,-4)
        self.mode_head=nn.Linear(16,8);nn.init.zeros_(self.mode_head.weight)
        with torch.no_grad():self.mode_head.bias.copy_(prior['mode_probs'].log())
        for k,v in prior.items():self.register_buffer('prior_'+k,v)
        self.register_buffer('reference_mse',torch.full((3,),1e-3))
        self.register_buffer('source_error_sum',torch.zeros(3,dtype=torch.float64));self.register_buffer('source_error_cells',torch.zeros((),dtype=torch.int64));self.register_buffer('local_updates',torch.zeros((),dtype=torch.int64))
        self.last=None
    def configure(self):
        self.train();self.requires_grad_(False);self.scale_head.requires_grad_(True);self.mode_head.requires_grad_(True)
        if self.stage=='target':self.adapter_A.requires_grad_(True);self.adapter_B.requires_grad_(True)
        self.parent.eval()
    def train(self,mode=True):super().train(mode);self.parent.eval();return self
    def forward(self,x):
        if self.calls:self.calls.sample(len(x))
        h0=self.parent.features(x);h=h0+self.adapter_B(torch.tanh(self.adapter_A(h0)))
        logits=self.parent.native_readout(h,x.shape[-2:]);logq=F.log_softmax(logits,1)
        features=area(h.detach(),32);s2=1e-6+(1-1e-6)*self.scale_head(features).sigmoid();mode=self.mode_head(features.mean((2,3)));pi=F.softmax(mode,1)
        w=(s2/(s2+self.reference_mse[None,:,None,None])).detach()
        if self.arm in ('N','CAL'):lp=logq;stats={}
        else:lp,stats=self.infer(logq,w,pi.detach(),self.arm)
        self.last=dict(logq=logq,logits=logits,s2=s2,mode=mode,w=w,pi=pi,stats=stats)
        return lp
    def infer(self,logq,w,pi,arm):return solve(logq,w,pi,{'mu':self.prior_mu,'factor':self.prior_factor,'sigma2':self.prior_sigma2},arm,self.calls)


def make(cfg,prior_override=None):
    torch.manual_seed(20271611);torch.cuda.manual_seed_all(20271611);original=source_model(cfg)
    prior=prior_override if prior_override is not None else torch.load(Path(cfg['root'])/'prior.private.pt',map_location='cpu',weights_only=False)
    model=Model(copy.deepcopy(original.parent).cpu(),prior,cfg['arm'],cfg['stage']).cuda();del original
    if prior_override is None and cfg['stage'] in ('target','source_eval'):
        state=torch.load(Path(cfg['root'])/'source/CAL/final.pt',map_location='cpu',weights_only=False);assert state['step']==state['physical_updates']==6000
        model.load_state_dict(state['ema'])
        if cfg['stage']=='target':model.local_updates.zero_()
    model.configure();model.calls=Calls(cfg);object.__setattr__(model,'cfg',cfg)
    return model

def setup(cfg):
    model=make(cfg);calls=model.calls;model.calls=None;ema=copy.deepcopy(model).eval().requires_grad_(False);model.calls=calls;ema.calls=Calls(cfg)
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and (n.startswith('adapter_')==adapter) and (p.ndim>1)==matrix],weight_decay=.0005 if matrix else 0.) for adapter in (True,False) for matrix in (True,False)]
    groups=[g for g in groups if g['params']];opt=torch.optim.SGD(groups,lr=.001,momentum=.9,nesterov=True);attach(opt,cfg,model)
    return None,model,ema,opt

def frozen(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(('parent.','prior_')) or (model.stage=='source' and k.startswith('adapter_')) or (model.stage=='target' and k.startswith(('reference_mse','source_error_')))}
def check_frozen(model):
    for k,v in getattr(model,'frozen_reference',{}).items():assert torch.equal(model.state_dict()[k].detach().cpu(),v),k
    assert all(p.grad is None for p in model.parent.parameters())

@torch.no_grad()
def ema_update(ema,model):
    model.local_updates.add_(1)
    if model.stage=='source':
        error=model.last['error2'].double();model.source_error_sum.add_(error.sum((0,2,3)));model.source_error_cells.add_(error.shape[0]*error.shape[2]*error.shape[3]);model.reference_mse.copy_((model.source_error_sum/model.source_error_cells).float().clamp_min(1e-6))
    params=dict(model.named_parameters())
    for n,v in ema.state_dict().items():
        if n in params and params[n].requires_grad:v.mul_(.999).add_(model.state_dict()[n],alpha=.001)
        else:v.copy_(model.state_dict()[n])
    check_frozen(model);check_frozen(ema)
    cfg=model.cfg
    if cfg['stage']=='source' and cfg['phase']=='formal' and int(model.local_updates)%500==0:
        from .run import diagnostic
        diagnostic(model,ema,model.current_data,int(model.local_updates))
