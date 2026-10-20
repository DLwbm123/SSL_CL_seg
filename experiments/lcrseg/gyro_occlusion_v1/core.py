"""Only target geometry Cutout is removed; original GYRO objective is retained."""
from pathlib import Path
from unittest.mock import patch
import torch
from .augmentation import strong_view
from ..gyro_source_coordinate_v1 import core as coordinate
from ..gyro_source_repair_v11 import core as base
from ..gyro_memory_v1 import core as memory
from ..gyro_fixed_projector_v1.core import fixed_eval
from ..gyro_autonomy_12h.space_guard import SpaceGuard
from ..main_style_ssl_v1.runtime import append
ARMS=('G_U','M_U')
provider=memory.provider
configure_lr=base.configure_lr
update_ema=base.update_ema

def setup(cfg):
    assert cfg['budget_cap']==4064 and cfg['bank_size']==256 and cfg['geometry_cutout'] is False
    guard=SpaceGuard(cfg['storage_authorization']);guard.check(250_000_000)
    source,model,ema,opt=base.setup(cfg);del source
    original=torch.load(cfg['prepared'],map_location='cpu',weights_only=False)
    cal=torch.load(cfg['coordinate_checkpoint'],map_location='cpu',weights_only=False)
    assert original['step']==original['physical_updates']==69 and cal['step']==cal['physical_updates']==1000
    assert all(torch.equal(v,original['model'][k]) for k,v in cal['model'].items() if k.startswith('parent.') or k=='frames')
    model.projector.load_state_dict({k.removeprefix('projector.'):v for k,v in cal['model'].items() if k.startswith('projector.')})
    model.projector.eval().requires_grad_(False);model.projector.register_forward_pre_hook(fixed_eval)
    opt.param_groups[:]=[g for g in opt.param_groups if not g['name'].startswith('projector')]
    assert not opt.state and all(g['fixed_lr']==.001 for g in opt.param_groups)
    source=model.teacher();ema=model.teacher()
    calls=0
    def check(*_):
        nonlocal calls
        if calls%50==0:guard.check(100_000_000)
        calls+=1
    opt.register_step_pre_hook(check)
    return source,model,ema,opt

def losses(arm,model,data,step):
    with patch.object(memory.raw_core,'strong_view',strong_view):l,u,stats=coordinate.losses(arm,model,data,step)
    stats.update(geometry_cutout=False,photometric_rng_matched=True)
    append(Path(model.cfg['output'])/'private_augmentation_contract.jsonl',dict(step=step,geometry_cutout=False,photometric_rng_matched=True))
    return l,u,stats
