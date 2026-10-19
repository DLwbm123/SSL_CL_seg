"""Native U-Net SGD with a finite, revisable functional training state."""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .state import graph_sites,label_sites,rebuild
from ..space_lora_v1.core import source_model
from ..semantic_condition_v1.core import budget_hooks
from ..siemens_ssl_reproduction_v1.methods import optimizer,pixel_ce,strong_view
from ..di_dmpa_jascl.data import stable_seed
from ..main_style_ssl_v1.protocol import SHAPES

ARMS=('dense_lonly','native_lonly','functional_lonly','fixed_ssl','revisable_ssl')
STEPS=400
GRID=SHAPES['mnms'][1:]


def views(images,step):
    strong=strong_view(images,step)
    rng=np.random.default_rng(stable_seed(163,step,'fixmatch_strong'))
    masks=[]
    for image in images:
        for _ in range(2):
            rng.integers(8); rng.uniform(0,1); rng.choice([-1,1])
        h,w=image.shape[-2:]; side=min(h,w)//2
        cy,cx=int(rng.integers(h)),int(rng.integers(w))
        mask=torch.ones(h,w,device=images.device,dtype=torch.bool)
        mask[max(0,cy-side//2):min(h,cy+side//2),max(0,cx-side//2):min(w,cx+side//2)]=False
        masks.append(mask)
    return strong,torch.stack(masks)


def cached_features(parent,e1):
    m=parent.native
    e2=m.enc2(m.pool(e1)); e3=m.enc3(m.pool(e2)); b=m.bottleneck(m.pool(e3))
    return m.decoder.dec1(m.decoder.dec2(m.decoder.dec3(b,e3),e2),e1)


def contrasts(logits):
    return logits[:,1:]-logits[:,:1]


class Native(nn.Module):
    def __init__(self,source,arm,bank):
        super().__init__()
        self.parent=copy.deepcopy(source.parent)
        self.parent.configure_stage_training()
        self.arm=arm
        if arm!='dense_lonly': self.parent.native.enc1.requires_grad_(False)
        self.functional=arm not in ('dense_lonly','native_lonly')
        if self.functional:
            for key in ('features','points','labels','pairs','base_precision','base_natural','initial_relation'):
                self.register_buffer('memory_'+key,bank[key].cuda())
            self.register_buffer('state_precision',bank['base_precision'].cuda().clone())
            self.register_buffer('state_mean',bank['source_response'].cuda().double().flatten().clone())
            self.register_buffer('state_relation',bank['initial_relation'].cuda().clone())
            self.register_buffer('state_label_precision',torch.zeros_like(self.state_precision))
            self.register_buffer('state_label_natural',torch.zeros_like(self.state_mean))
            self.register_buffer('state_refresh_count',torch.zeros((),device='cuda',dtype=torch.long))
            self.last_edges=None
            self.graph_change=0.

    def forward(self,x):
        return self.parent.native_readout(self.parent.features(x),x.shape[-2:])

    def teacher(self):
        result=copy.deepcopy(self).eval().requires_grad_(False)
        # Immutable input/evidence buffers are one bank, including in saved
        # student/EMA snapshots. Mutable posterior buffers remain independent.
        if self.functional:
            for name,value in self.named_buffers():
                if name.startswith('memory_'): setattr(result,name,value)
        return result

    def deploy(self):
        return self.teacher()

    def query(self):
        values=[]; semantics=[]
        points=self.memory_points
        for i in range(len(self.memory_features)):
            where=(points[:,0]==i).nonzero().flatten()
            h=cached_features(self.parent,self.memory_features[i:i+1])
            z=self.parent.native_readout(h,GRID)
            semantic=F.interpolate(h,size=GRID,mode='bilinear',align_corners=True)
            y,x=points[where,1],points[where,2]
            values.append(contrasts(z)[0,:,y,x].T)
            semantics.append(semantic[0,:,y,x].T)
        return torch.cat(values),torch.cat(semantics)

    @torch.no_grad()
    def refresh(self,f,h):
        tp,hp=label_sites(f,self.memory_labels)
        relation=self.memory_initial_relation
        if self.arm=='functional_lonly': relation=torch.zeros_like(relation)
        elif self.arm=='revisable_ssl':
            relation,_=graph_sites(h,self.memory_pairs,True)
        self.graph_change=float((relation-self.state_relation).norm())
        precision,mean=rebuild(self.memory_base_precision,self.memory_base_natural,tp,hp,relation)
        self.state_precision.copy_(precision); self.state_mean.copy_(mean)
        self.state_label_precision.copy_(tp); self.state_label_natural.copy_(hp)
        self.state_relation.copy_(relation); self.state_refresh_count.add_(1)


def setup(cfg):
    if cfg['arm'] not in ARMS: raise PermissionError('unregistered arm')
    torch.manual_seed(163); torch.cuda.manual_seed_all(163)
    source=source_model(cfg)
    bank=torch.load(cfg['bank'],map_location='cpu',weights_only=False)
    if bank['points'].shape!=(128,3) or bank['optimizer_updates']!=0:
        raise ValueError('fixed memory receipt mismatch')
    model=Native(source,cfg['arm'],bank).cuda().train()
    ema=model.teacher(); opt=optimizer(model,'fixmatch')
    budget_hooks(opt,cfg)
    return source,model,ema,opt


def rate(step):
    if not 0<=step<STEPS: raise PermissionError('outside frozen pilot')
    return .03*math.cos(7*math.pi*step/(16*53300))


@torch.no_grad()
def update_ema(ema,model):
    current=dict(model.named_parameters())
    for name,p in ema.named_parameters():
        q=current[name]
        if q.requires_grad: p.mul_(.999).add_(q,alpha=.001)
        else: p.copy_(q)
    # Functional state is training metadata, not an independently averaged teacher.
    buffers=dict(model.named_buffers())
    for name,b in ema.named_buffers(): b.copy_(buffers[name])


def losses(arm,model,data,step):
    x,y,_=data.labeled(step)
    l=pixel_ce(model(x),y).mean()
    u=l*0; function=l*0; dense=l*0; mismatch=0.
    if model.functional:
        f,h=model.query()
        if step%25==0 or int(model.state_refresh_count)==0:
            model.refresh(f.detach(),h.detach())
        delta=f.double().flatten()-model.state_mean
        function=(.5*delta@(model.state_precision@delta)/delta.numel()).float()
        l=l+function
        mismatch=float(delta.detach().square().mean().sqrt())
    pixels=0
    if arm.endswith('_ssl'):
        images=torch.cat([data.unlabeled(step*7+i)[0] for i in range(7)])
        strong,visible=views(images,step)
        a=contrasts(model(images)); b=contrasts(model(strong))
        dense=(.5*(a-b).square().mean(1)*visible).sum()/visible.sum().clamp_min(1)
        u=dense
        pixels=int(visible.sum())
    return l,u,dict(supervised=float(l.detach()-function.detach()),
        unsupervised=float(u.detach()),functional_penalty=float(function.detach()),
        state_network_rms=mismatch,relation_change=model.graph_change if model.functional else 0.,
        accepted_class_pixels=[0]*4,U_pixels=pixels,
        state_refreshes=int(model.state_refresh_count) if model.functional else 0)
