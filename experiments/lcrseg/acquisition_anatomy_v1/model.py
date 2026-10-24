"""Native segmentation with a learned acquisition inverse and training-only anatomy."""
import copy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .appearance import Canonicalizer
from .runtime import Calls,attach
from ..space_lora_v1.core import source_model

class Model(nn.Module):
    def __init__(self,parent,stage,arm):
        super().__init__();self.parent=parent;self.canonicalizer=Canonicalizer();self.anatomy=nn.ModuleList([nn.Conv2d(16,4,1) for _ in (32,64)])
        for head in self.anatomy:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.stage=stage;self.arm=arm;self.calls=None;self.register_buffer('local_updates',torch.zeros((),dtype=torch.int64));self.last=None
    def train(self,mode=True):super().train(mode);self.parent.eval();return self
    def configure(self):
        self.train();self.requires_grad_(False);self.canonicalizer.requires_grad_(True)
        if self.stage=='source':
            for name,p in self.parent.named_parameters():
                if name.startswith('native.') and not name.startswith('native.decoder.conv_logit.'):p.requires_grad_(True)
            if self.arm=='F':self.anatomy.requires_grad_(True)
        self.parent.eval()
    def forward(self,x):
        if self.calls:self.calls.sample(len(x))
        corrected=self.canonicalizer(x,self.calls);h=self.parent.features(corrected);lp=F.log_softmax(self.parent.native_readout(h,x.shape[-2:]),1);aa=[]
        if self.cfg.get('task')!='evaluate':
            aligned=F.interpolate(F.avg_pool2d(h,3,1),size=x.shape[-2:],mode='bilinear',align_corners=True)
            for r,head in zip((32,64),self.anatomy):aa.append(F.log_softmax(head(F.avg_pool2d(aligned,384//r)),1))
            if self.calls:self.calls.extra('anatomy_samples',len(x)*2)
        self.last=dict(corrected=corrected,anatomy=aa)
        return lp

def make(cfg):
    torch.manual_seed(20271701);torch.cuda.manual_seed_all(20271701);src=source_model(cfg);m=Model(copy.deepcopy(src.parent).cpu(),cfg['stage'],cfg['arm']).cuda();del src
    if cfg['stage']!='source' and not cfg.get('synthetic'):
        source='B' if cfg['arm']=='B' else 'F';state=torch.load(Path(cfg['root'])/'source'/source/'final.pt',map_location='cpu',weights_only=False);assert state['step']==state['physical_updates']==2000;m.load_state_dict(state['ema'])
    m.local_updates.zero_();m.configure();m.calls=Calls(cfg);object.__setattr__(m,'cfg',cfg);return m

def setup(cfg):
    m=make(cfg);c=m.calls;m.calls=None;ema=copy.deepcopy(m).eval().requires_grad_(False);m.calls=c;ema.calls=Calls(cfg)
    groups=[dict(params=[p for p in m.parameters() if p.requires_grad and (p.ndim>1)==matrix],weight_decay=.0005 if matrix else 0.) for matrix in (True,False)];groups=[g for g in groups if g['params']];opt=torch.optim.SGD(groups,lr=.001,momentum=.9,nesterov=True);attach(opt,cfg,m);return None,m,ema,opt

def frozen(m):
    trainable={n for n,p in m.named_parameters() if p.requires_grad}
    return {k:v.detach().cpu().clone() for k,v in m.state_dict().items() if k not in trainable and k!='local_updates'}
def check_frozen(m):
    for k,v in getattr(m,'frozen_reference',{}).items():assert torch.equal(m.state_dict()[k].detach().cpu(),v),k
    for name,p in m.named_parameters():
        if not name.startswith('canonicalizer.') and (m.stage=='target' or name.startswith('parent.native.decoder.conv_logit.')):assert p.grad is None,name

@torch.no_grad()
def ema_update(ema,m):
    m.local_updates.add_(1);params=dict(m.named_parameters())
    for n,v in ema.state_dict().items():
        if n in params and params[n].requires_grad:v.mul_(.999).add_(m.state_dict()[n],alpha=.001)
        else:v.copy_(m.state_dict()[n])
    check_frozen(m);check_frozen(ema)
    if m.cfg['stage']=='source' and m.cfg['phase']=='formal' and int(m.local_updates) in (1500,1750,2000):
        from .run import diagnostic
        diagnostic(m,ema,m.current_data,int(m.local_updates))
