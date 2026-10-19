"""Fixed tenfold spectral strength and a matched-entry CE reference."""
from pathlib import Path
import torch
from ..gyro_source_repair_v11 import core as base
from ..gyro_native_v1.core import losses as raw_losses
from ..gyro_transfer_v1.core import provider,configure_lr,update_ema
from ..main_style_ssl_v1.runtime import append
from ..gyro_autonomy_12h.space_guard import SpaceGuard

ARMS=('G_U','M_U','CE_only')
STEPS=2000


def setup(cfg):
    assert cfg['arm'] in ARMS and cfg['budget_cap']==6096 and cfg['spectral_weight']==.1
    guard=SpaceGuard(cfg['storage_authorization']);guard.check(250_000_000)
    source,model,ema,opt=base.setup(cfg);del source
    source=model.teacher();assert not opt.state and cfg['calibrated']==cfg['prepared']
    calls=0
    def storage(*_):
        nonlocal calls
        if calls%50==0:guard.check(100_000_000)
        calls+=1
    opt.register_step_pre_hook(storage)
    return source,model,ema,opt


def losses(arm,model,data,step):
    alias='G_U' if arm=='CE_only' else arm
    l,raw,stats=raw_losses(alias,model,data,step)
    u=l*0 if arm=='CE_only' else .1*raw
    g=base.gradients(model,l,u)
    ratio=g['encoder_all']['weighted_G_norm']/max(g['encoder_all']['CE_norm'],1e-12)
    stats.update(unsupervised=float(u.detach()),raw_geometry=float(raw.detach()),gradient_groups=g,encoder_weighted_ratio=ratio,
        geometric_images=24,replay_images=8,new_U_images=14,spectral_weight=0 if arm=='CE_only' else .1)
    append(Path(model.cfg['output'])/'private_loss_diagnostics.jsonl',dict(step=step,arm=arm,**stats))
    return l,u,stats
