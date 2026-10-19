import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from ..gyro_source_repair_v11 import core as base
from ..gyro_native_v1 import core as raw_core
from ..gyro_native_v1.data import current
from ..gyro_autonomy_12h.space_guard import SpaceGuard
from ..five_frameworks_v1.recipes import generator
from ..single_teacher_scd_v0_1.data import geometry
from ..main_style_ssl_v1.runtime import append

ARMS=('G_U','M_U')
STEPS=2000
configure_lr=base.configure_lr
update_ema=base.update_ema


def replay_indices(step,size):
    return np.random.default_rng(88163+step).choice(size,8,replace=False).tolist()


def historical(data,step):
    values=[]
    for pos,i in enumerate(replay_indices(step,len(data.bank))):
        x=data.bank[i];valid=torch.ones(x.shape[-2:],dtype=torch.bool)
        x,_,_=geometry(x,None,valid,generator(163,1,1,step,'gyro_replay_'+str(pos)))
        values.append(x)
    return torch.stack(values).cuda()


def provider(cfg):
    data=current(cfg,'Philips',2000)
    bank=torch.load(cfg['bank'],map_location='cpu',weights_only=False)
    assert bank['size']==256 and bank['label_fields']==[] and bank['domain']=='Siemens'
    assert len(bank['images'])==256 and torch.isfinite(bank['images']).all()
    data.bank=bank['images'];return data


def setup(cfg):
    assert cfg['arm'] in ARMS and cfg['budget_cap']==4064 and cfg['bank_size']==256
    guard=SpaceGuard(cfg['storage_authorization']);guard.check(250_000_000)
    source,model,ema,opt=base.setup(cfg);del source
    source=model.teacher();assert not opt.state and cfg['prepared']==cfg['calibrated']
    count=0
    def storage(*_):
        nonlocal count
        if count%50==0:guard.check(100_000_000)
        count+=1
    opt.register_step_pre_hook(storage)
    return source,model,ema,opt


def losses(arm,model,data,step):
    with patch.object(raw_core,'historical',historical):l,raw,stats=raw_core.losses(arm,model,data,step)
    u=.1*raw;groups=base.gradients(model,l,u)
    stats.update(unsupervised=float(u.detach()),raw_geometry=float(raw.detach()),gradient_groups=groups,
        encoder_weighted_ratio=groups['encoder_all']['weighted_G_norm']/max(groups['encoder_all']['CE_norm'],1e-12),
        geometric_images=24,replay_images=8,new_U_images=14,bank_size=256,spectral_weight=.1)
    append(Path(model.cfg['output'])/'private_loss_diagnostics.jsonl',dict(step=step,arm=arm,**stats))
    return l,u,stats
