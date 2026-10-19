"""One spectral operation, full native segmentation training; no KD or readout binding."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .module import Projector,frame,spectral
from .data import historical
from ..space_lora_v1.core import source_model
from ..siemens_ssl_reproduction_v1.methods import optimizer,pixel_ce,strong_view
from ..semantic_condition_v1.core import budget_hooks

ARMS=('M_U','G_U','M_noNewU','G_noNewU','dense_fixmatch')
STEPS=2000


class NativeGyro(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.parent=copy.deepcopy(source.parent)
        self.parent.configure_stage_training()
        self.projector=Projector()
        self.register_buffer('frames',torch.stack([frame(90163),frame(90164)]))

    def forward(self,x):
        return self.parent.native_readout(self.parent.features(x),x.shape[-2:])

    def project(self,x):
        m=self.parent.native
        h=m.enc1(x); h=m.enc2(m.pool(h)); h=m.enc3(m.pool(h)); h=m.bottleneck(m.pool(h))
        return self.projector(h.mean((-2,-1)))

    def teacher(self):
        return copy.deepcopy(self).eval().requires_grad_(False)


def setup(cfg):
    torch.manual_seed(163);torch.cuda.manual_seed_all(163)
    source=source_model(cfg)
    model=NativeGyro(source).cuda().train()
    opt=optimizer(model,'fixmatch')
    if cfg['arm']!='source_prepare':
        saved=torch.load(cfg['prepared'],map_location='cpu',weights_only=False)
        assert saved['step']==saved['physical_updates']==138
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        source=model.teacher()
    model.arm=cfg['arm']; model.geometry_step=0
    ema=model.teacher()
    def clip(*_):
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    opt.register_step_pre_hook(clip)
    budget_hooks(opt,cfg)
    return source,model,ema,opt


def rate(step,source=False):
    j=step if source else step+138
    return .03*math.cos(7*math.pi*j/(16*53300))


@torch.no_grad()
def update_ema(ema,model):
    if model.arm=='source_prepare':
        ema.load_state_dict(model.state_dict())
    else:
        model.parent.update_dense_ema(ema.parent,decay=.999)
        # Projector never participates in inference or any teacher loss.
        ema.projector.load_state_dict(model.projector.state_dict())
        ema.frames.copy_(model.frames)


def losses(arm,model,data,step):
    x,y,_=data.labeled(step)
    l=pixel_ce(model(x),y).mean(); counts=[0]*4;pixels=0
    if arm=='dense_fixmatch':
        u=torch.cat([data.unlabeled(step*7+i)[0] for i in range(7)])
        with torch.no_grad():
            conf,labels=model(u).softmax(1).max(1);keep=conf>=.95
        v=(pixel_ce(model(strong_view(u,step)),labels)*keep).mean()
        counts=[int(((labels==c)&keep).sum()) for c in range(4)];pixels=keep.numel()
    else:
        if arm=='source_prepare':
            images=data.pool.epoch_batch(step)
            ids=torch.zeros(16,device=x.device,dtype=torch.long)
            beta=0. if step<69 else 1/3
        else:
            if arm.endswith('_noNewU'):
                # Read image-only extra L slots: _batch geometry ignores labels via an image-only view.
                extra=data._l
                from types import SimpleNamespace
                imageonly=SimpleNamespace(**extra.__dict__)
                imageonly.role='train_unlabeled'
                # Bind the safe image-only read without modifying the frozen accessor.
                from ..main_style_ssl_v1.data import Accessor
                imageonly.read=lambda i,z: Accessor.read(imageonly,i,z)
                u=torch.cat([data._batch(imageonly,step*7+i,'unlabeled')[0] for i in range(7)])
            else:
                u=torch.cat([data.unlabeled(step*7+i)[0] for i in range(7)])
            images=torch.cat([x,u,historical(data,step)])
            ids=torch.tensor([1]*16+[0]*8,device=x.device)
            beta=1/3 if arm.startswith('G') else 0.
        strong=strong_view(images,step)
        # Encode both views together: BN observes the full 2P projector batch.
        z=model.project(torch.cat([images,strong]))
        z=torch.stack(z.chunk(2),dim=1)
        v=spectral(z,model.frames,ids,beta)
    return l,v,dict(supervised=float(l.detach()),unsupervised=float(v.detach()),
        accepted_class_pixels=counts,U_pixels=pixels)
