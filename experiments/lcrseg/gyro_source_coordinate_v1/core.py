"""Prepare the existing projector on source images; target native fully trains."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from ..gyro_source_repair_v11 import core as base
from ..gyro_memory_v1 import core as memory
from ..gyro_native_v1.module import spectral
from ..gyro_native_v1.data import current
from ..gyro_fixed_projector_v1.core import fixed_eval
from ..siemens_ssl_reproduction_v1.methods import strong_view
from ..gyro_autonomy_12h.space_guard import SpaceGuard
from ..main_style_ssl_v1.runtime import append
ARMS=('calibration','G_U','M_U')
configure_lr=base.configure_lr
update_ema=base.update_ema
projection=base.projection

class BankPool:
    def __init__(self,bank):self.bank=bank
    def epoch_batch(self,step):
        assert 0<=step<1000
        ids=np.random.default_rng(98163+step).choice(256,16,replace=False).tolist()
        return self.bank[ids].cuda()

def provider(cfg):
    if cfg['arm']!='calibration':return memory.provider(cfg)
    bank=torch.load(cfg['bank'],map_location='cpu',weights_only=False)
    assert bank['size']==256 and bank['domain']=='Siemens' and bank['label_fields']==[]
    assert torch.isfinite(bank['images']).all()
    data=current(cfg,'Siemens',1000);data.pool=BankPool(bank['images'])
    # CurrentDomain supplies only sampler/read counters. No labeled/unlabeled call.
    return data

def setup(cfg):
    assert cfg['budget_cap']==5096 and cfg['bank_size']==256
    guard=SpaceGuard(cfg['storage_authorization']);guard.check(250_000_000)
    source,model,ema,opt=base.setup(cfg);del source
    original=torch.load(cfg['prepared'],map_location='cpu',weights_only=False)
    assert original['step']==original['physical_updates']==69
    if cfg['arm']=='calibration':
        model.load_state_dict(original['model'])
        assert all(not p.requires_grad for p in model.parent.parameters())
        assert all(g['name'].startswith('projector') and g['fixed_lr']==.03 for g in opt.param_groups)
    else:
        cal=torch.load(cfg['coordinate_checkpoint'],map_location='cpu',weights_only=False)
        assert cal['step']==cal['physical_updates']==1000
        assert all(torch.equal(v,original['model'][k]) for k,v in cal['model'].items() if k.startswith('parent.') or k=='frames')
        model.projector.load_state_dict({k.removeprefix('projector.'):v for k,v in cal['model'].items() if k.startswith('projector.')})
        model.projector.eval().requires_grad_(False);model.projector.register_forward_pre_hook(fixed_eval)
        opt.param_groups[:]=[g for g in opt.param_groups if not g['name'].startswith('projector')]
        assert all(g['fixed_lr']==.001 for g in opt.param_groups)
    assert not opt.state
    source=model.teacher();ema=model.teacher()
    calls=0
    def check(*_):
        nonlocal calls
        if calls%50==0:guard.check(100_000_000)
        calls+=1
    opt.register_step_pre_hook(check)
    return source,model,ema,opt

def losses(arm,model,data,step):
    if arm=='calibration':
        assert data.l_reads==data.u_reads==0
        images=data.pool.epoch_batch(step)
        z=projection(model,torch.cat([images,strong_view(images,step)]))
        raw=spectral(torch.stack(z.chunk(2),1),model.frames,torch.zeros(16,dtype=torch.long,device=z.device),1/3)
        l=next(model.projector.parameters()).sum()*0;u=raw;g=base.gradients(model,l,u)
        assert g['encoder_all']['weighted_G_norm']==0 and g['projector']['weighted_G_norm']>0
        stats=dict(supervised=0.,unsupervised=float(u.detach()),accepted_class_pixels=[0]*4,U_pixels=0,
          gradient_groups=g,source_images=16,source_bank_size=256,views=2,anchor_beta=1/3,spectral_weight=1.,labels_read=False)
    else:
        with patch.object(memory.raw_core,'historical',memory.historical):l,raw,stats=memory.raw_core.losses(arm,model,data,step)
        weight=.45 if arm=='G_U' else .3;u=weight*raw;g=base.gradients(model,l,u)
        assert g['projector']['weighted_G_norm']==0 and not model.projector.training
        stats.update(unsupervised=float(u.detach()),raw_geometry=float(raw.detach()),gradient_groups=g,
          encoder_weighted_ratio=g['encoder_all']['weighted_G_norm']/max(g['encoder_all']['CE_norm'],1e-12),
          geometric_images=24,replay_images=8,new_U_images=14,bank_size=256,views=2,spectral_weight=weight,fixed_projector=True)
    append(Path(model.cfg['output'])/'private_loss_diagnostics.jsonl',dict(step=step,arm=arm,**stats))
    return l,u,stats
