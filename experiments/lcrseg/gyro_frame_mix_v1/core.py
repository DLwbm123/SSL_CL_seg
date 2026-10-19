"""Original GYRO fixed spectral weight0.9, beta2/3; same256-image replay."""
from pathlib import Path
from unittest.mock import patch
from ..gyro_memory_v1 import core as memory
from ..gyro_source_repair_v11 import core as base
from ..main_style_ssl_v1.runtime import append
from ..gyro_autonomy_12h.space_guard import SpaceGuard
ARMS=('G_U',)
STEPS=2000
provider=memory.provider
configure_lr=memory.configure_lr
update_ema=memory.update_ema
raw_core=memory.raw_core
replay_indices=memory.replay_indices
native_spectral=raw_core.spectral


def stronger_mix(z,frames,ids,beta):
    assert beta==1/3
    return native_spectral(z,frames,ids,2/3)


def setup(cfg):
    assert cfg['arm']=='G_U' and cfg['spectral_weight']==.9 and cfg['anchor_beta']==2/3 and cfg['budget_cap']==2032 and cfg['bank_size']==256
    guard=SpaceGuard(cfg['storage_authorization']);guard.check(250_000_000)
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
    with patch.object(raw_core,'historical',memory.historical),patch.object(raw_core,'spectral',stronger_mix):
        l,raw,stats=raw_core.losses(arm,model,data,step)
    u=.9*raw;groups=base.gradients(model,l,u)
    stats.update(unsupervised=float(u.detach()),raw_geometry=float(raw.detach()),gradient_groups=groups,
        encoder_weighted_ratio=groups['encoder_all']['weighted_G_norm']/max(groups['encoder_all']['CE_norm'],1e-12),
        geometric_images=24,replay_images=8,new_U_images=14,bank_size=256,spectral_weight=.9,anchor_beta=2/3)
    append(Path(model.cfg['output'])/'private_loss_diagnostics.jsonl',dict(step=step,arm=arm,**stats))
    return l,u,stats
