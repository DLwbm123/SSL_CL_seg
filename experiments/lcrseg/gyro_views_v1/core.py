"""Original GYRO mean over four views; no extra semantic loss."""
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from ..gyro_memory_v1 import core as memory
from ..gyro_source_repair_v11 import core as base
from ..gyro_autonomy_12h.space_guard import SpaceGuard
from ..main_style_ssl_v1.runtime import append
from ..siemens_ssl_reproduction_v1.methods import pixel_ce,strong_view
ARMS=('G_U','M_U')
STEPS=2000
provider=memory.provider
configure_lr=memory.configure_lr
update_ema=memory.update_ema


def spectral_views(z,frames,ids,beta):
    assert z.ndim==3 and z.shape[1] in [2,4] and z.shape[2]==128
    c=F.normalize(z,dim=-1,eps=1e-12).mean(1)
    if beta:
        e=frames[ids].detach();j=torch.einsum('pd,pkd->pk',c.detach(),e).argmax(1)
        a=e[torch.arange(len(c),device=c.device),j];y=(1-beta)*c+beta*a
    else:y=c
    return -torch.linalg.svdvals(y.float()).sum()/math.sqrt(len(c)*min(len(c),128))


def setup(cfg):
    assert cfg['arm'] in ARMS and cfg['budget_cap']==4064 and cfg['views']==4
    guard=SpaceGuard(cfg['storage_authorization']);guard.check(300_000_000)
    source,model,ema,opt=base.setup(cfg);del source
    source=model.teacher();assert not opt.state and cfg['prepared']==cfg['calibrated']
    calls=0
    def check(*_):
        nonlocal calls
        if calls%50==0:guard.check(100_000_000)
        calls+=1
    opt.register_step_pre_hook(check)
    return source,model,ema,opt


def losses(arm,model,data,step):
    x,y,_=data.labeled(step);l=pixel_ce(model(x),y).mean()
    u=torch.cat([data.unlabeled(step*7+i)[0] for i in range(7)])
    images=torch.cat([x,u,memory.historical(data,step)])
    # First two views exactly reuse the V2 seeds; two more independent strong seeds.
    views=torch.cat([images,*[strong_view(images,step+offset) for offset in [0,2000,4000]]])
    z=model.project(views);z=torch.stack(z.chunk(4),dim=1)
    ids=torch.tensor([1]*16+[0]*8,device=z.device);beta=1/3 if arm=='G_U' else 0.
    weight=.45 if arm=='G_U' else .3
    raw=spectral_views(z,model.frames,ids,beta);v=weight*raw;g=base.gradients(model,l,v)
    stats=dict(supervised=float(l.detach()),unsupervised=float(v.detach()),raw_geometry=float(raw.detach()),
        accepted_class_pixels=[0]*4,U_pixels=0,gradient_groups=g,
        encoder_weighted_ratio=g['encoder_all']['weighted_G_norm']/max(g['encoder_all']['CE_norm'],1e-12),
        geometric_images=24,replay_images=8,new_U_images=14,bank_size=256,views=4,geometry_view_images=96,spectral_weight=weight,anchor_beta=beta)
    append(Path(model.cfg['output'])/'private_loss_diagnostics.jsonl',dict(step=step,arm=arm,**stats))
    return l,v,stats
