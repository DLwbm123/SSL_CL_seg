from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from . import core
from ..gyro_native_v1.module import self_check
from ..main_style_ssl_v1.runtime import write
from ..gyro_autonomy_12h.space_guard import SpaceGuard


def main(cfg):
    assert self_check()['status']=='PASS'
    for step in [0,1,1999]:
        a=core.replay_indices(step,256)
        assert len(set(a))==8 and min(a)>=0 and max(a)<256 and max(a)>=16
        assert a==core.replay_indices(step,256)
        assert core.replay_indices(step,16)==np.random.default_rng(88163+step).choice(16,8,replace=False).tolist()
    for arm in core.ARMS:
        m=nn.Module();m.parent=nn.Module();m.parent.native=nn.Module()
        m.parent.native.enc1=nn.Linear(1,1,bias=False);m.projector=nn.Linear(1,1,bias=False);m.cfg={'output':cfg['root']}
        l=m.parent.native.enc1.weight.sum();raw=-(l+m.projector.weight.sum())
        with patch.object(core.raw_core,'losses',return_value=(l,raw,{})),patch.object(core,'append'):
            _,u,s=core.losses(arm,m,None,0)
        assert torch.allclose(u,.1*raw) and s['gradient_groups']['encoder_all']['weighted_G_norm']>0
    SpaceGuard(cfg['storage_authorization']).check(500_000_000)
    write(Path(cfg['root'])/'SYNTHETIC_CHECKS.json',dict(status='PASS',optimizer_calls=0,deterministic_256_sampling=True,legacy16_sampling_equal=True,fixed_weight_contract=True))
