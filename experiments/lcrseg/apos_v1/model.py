"""Shared observation architecture, structural inference in the prediction path."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .math import posterior,lowrank_nll,LOG2PI
from .runtime import Calls,attach
from ..space_lora_v1.core import source_model
HEAD='native.decoder.conv_logit'

class Model(nn.Module):
    def __init__(self,parent,prior,arm):
        super().__init__();self.parent=parent;self.arm=arm;self.calls=None
        self.context=nn.Sequential(nn.Linear(4,16),nn.Tanh(),nn.Linear(16,32))
        self.observation=nn.Conv2d(16,6,1);self.presence=nn.Linear(16,8);self.detail=nn.Conv2d(16,3,1)
        nn.init.zeros_(self.context[-1].weight);nn.init.zeros_(self.context[-1].bias)
        nn.init.normal_(self.observation.weight,std=.01);nn.init.zeros_(self.observation.bias)
        with torch.no_grad():self.observation.bias[3:]=-2.
        nn.init.zeros_(self.detail.weight);nn.init.zeros_(self.detail.bias)
        for k in ('mu','factor','tau2','active'):self.register_buffer('prior_'+k,prior[k])
        g=torch.Generator().manual_seed(20271699)
        self.register_buffer('quadrature_diagonal',torch.randn(3072,generator=g))
        self.register_buffer('quadrature_rank',torch.randn(8,generator=g))
        self.last=None
    def configure(self,stage):
        self.train();self.requires_grad_(True);self.stage=stage
        self.parent.configure_stage_training()
        self.parent.native.decoder.conv_logit.requires_grad_(False)
        if stage=='target':self.parent.eval().requires_grad_(False);self.detail.requires_grad_(False)
    def train(self,mode=True):
        super().train(mode)
        if getattr(self,'stage',None)=='target':self.parent.eval()
        return self
    def forward(self,x):
        if self.calls:self.calls.sample(len(x))
        h=self.parent.features(x)
        stats=torch.stack([x.mean((1,2,3)),x.std((1,2,3),unbiased=False),x.amin((1,2,3)),x.amax((1,2,3))],-1)
        gb=self.context(stats).tanh();h=h*(1+.1*gb[:,:16,None,None])+.1*gb[:,16:,None,None]
        obs=self.observation(F.interpolate(h,size=(32,32),mode='bilinear',align_corners=False))
        t=obs[:,:3].tanh().flatten(1);w=(1+99*obs[:,3:].sigmoid()).flatten(1)
        rawpi=self.presence(h.mean((2,3)));logpi=rawpi.masked_fill(~self.prior_active[None],-1e9)
        detail=.1*self.detail(h).tanh();detail=F.interpolate(detail,size=x.shape[-2:],mode='bilinear',align_corners=False)
        if self.arm=='D':
            samples=t[:,None,None,:];weights=torch.ones(len(x),1,device=x.device,dtype=t.dtype)
            moments=None
        else:
            if self.calls:self.calls.solves(len(x)*8*(4 if self.arm=='T' else 1))
            m,v,f,lp=posterior(t,w,F.log_softmax(logpi,dim=-1),self.prior_mu,self.prior_factor,self.prior_tau2,self.arm=='T')
            noise=v.sqrt()*self.quadrature_diagonal.double()+ (f*self.quadrature_rank.double()).sum(-1)
            samples=torch.stack([m+noise,m-noise],dim=2).to(t);weights=lp.exp().to(t);moments=(m,v,f,lp)
        # Compute probabilities component by component to bound high-resolution memory.
        probability=None
        for k in range(samples.shape[1]):
            value=None
            for q in range(samples.shape[2]):
                d=F.interpolate(samples[:,k,q].reshape(-1,3,32,32),size=x.shape[-2:],mode='bilinear',align_corners=False)+detail
                logits=torch.cat([torch.zeros_like(d[:,:1]),-d/.1],1)
                part=logits.softmax(1)/samples.shape[2];value=part if value is None else value+part
            weighted=value*weights[:,k,None,None,None]
            probability=weighted if probability is None else probability+weighted
        self.last=(t,w,rawpi,moments)
        return probability.clamp_min(1e-12).log()
    def posterior_nll(self,truth):
        t,w,rawpi,moments=self.last
        if moments is None:return .5*(w*(t-truth).square()-w.log()+LOG2PI).mean()
        m,v,f,lp=moments
        nll=lowrank_nll(truth.double()[:,None],m,v,f)
        return -torch.logsumexp(lp-nll,dim=-1).mean()/3072


def make(cfg):
    torch.manual_seed(20271601);torch.cuda.manual_seed_all(20271601)
    original=source_model(cfg)
    prior=torch.load(cfg['root']+'/prior.private.pt',map_location='cpu',weights_only=False)
    model=Model(copy.deepcopy(original.parent).cpu(),prior,cfg['arm']).cuda()
    del original
    if cfg['stage']=='target':
        saved=torch.load(cfg['root']+'/source/'+cfg['arm']+'/final.pt',map_location='cpu',weights_only=False)
        assert saved['step']==saved['physical_updates']==2000
        model.load_state_dict(saved['ema'])
    model.configure(cfg['stage']);model.calls=Calls(cfg)
    object.__setattr__(model,'cfg',cfg)
    return model


def setup(cfg):
    model=make(cfg);calls=model.calls;model.calls=None
    ema=copy.deepcopy(model).eval().requires_grad_(False);model.calls=calls;ema.calls=Calls(cfg)
    source=None
    groups=[dict(params=[p for p in model.parameters() if p.requires_grad and (p.ndim>1)==matrix],weight_decay=.0005 if matrix else 0.) for matrix in (True,False)]
    opt=torch.optim.SGD(groups,lr=.001,momentum=.9,nesterov=True);attach(opt,cfg,model)
    return source,model,ema,opt

@torch.no_grad()
def ema_update(ema,model):
    pars=dict(model.named_parameters())
    for n,v in ema.state_dict().items():
        s=model.state_dict()[n]
        if n in pars and pars[n].requires_grad:v.mul_(.999).add_(s,alpha=.001)
        else:v.copy_(s)
    assert_frozen(model);assert_frozen(ema)


def frozen(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(('prior_','quadrature_','parent.native.decoder.conv_logit.')) or (getattr(model,'stage','source')=='target' and k.startswith(('parent.','detail.')))}

def assert_frozen(model):
    reference=getattr(model,'frozen_reference',None)
    if reference is not None:
        for k,v in reference.items():
            assert torch.equal(model.state_dict()[k].detach().cpu(),v),k
    for n,p in model.named_parameters():
        if not p.requires_grad:assert p.grad is None,n
