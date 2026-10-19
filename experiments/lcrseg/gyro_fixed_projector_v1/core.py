"""Original GYRO in a fixed existing projection coordinate; native fully trains."""
from pathlib import Path
from unittest.mock import patch
from ..gyro_memory_v1 import core as memory
from ..gyro_source_repair_v11 import core as base
from ..gyro_autonomy_12h.space_guard import SpaceGuard
from ..main_style_ssl_v1.runtime import append
ARMS=('G_U','M_U')
STEPS=2000
provider=memory.provider
configure_lr=memory.configure_lr
update_ema=memory.update_ema


def fixed_eval(module,inputs):
    module.eval()


def setup(cfg):
    assert cfg['fixed_projector'] is True and cfg['budget_cap']==4064
    guard=SpaceGuard(cfg['storage_authorization']);guard.check(250_000_000)
    source,model,ema,opt=base.setup(cfg);del source
    assert not opt.state
    model.projector.eval().requires_grad_(False)
    model.projector.register_forward_pre_hook(fixed_eval)
    opt.param_groups[:]=[g for g in opt.param_groups if not g['name'].startswith('projector')]
    assert all(g['fixed_lr']==.001 for g in opt.param_groups)
    source=model.teacher()
    calls=0
    def check(*_):
        nonlocal calls
        if calls%50==0:guard.check(100_000_000)
        calls+=1
    opt.register_step_pre_hook(check)
    return source,model,ema,opt


def losses(arm,model,data,step):
    with patch.object(memory.raw_core,'historical',memory.historical):l,raw,stats=memory.raw_core.losses(arm,model,data,step)
    weight=.45 if arm=='G_U' else .3
    u=weight*raw;g=base.gradients(model,l,u)
    assert g['projector']['weighted_G_norm']==0 and not model.projector.training
    stats.update(unsupervised=float(u.detach()),raw_geometry=float(raw.detach()),gradient_groups=g,
        encoder_weighted_ratio=g['encoder_all']['weighted_G_norm']/max(g['encoder_all']['CE_norm'],1e-12),
        geometric_images=24,replay_images=8,new_U_images=14,bank_size=256,views=2,spectral_weight=weight,fixed_projector=True)
    append(Path(model.cfg['output'])/'private_loss_diagnostics.jsonl',dict(step=step,arm=arm,**stats))
    return l,u,stats
