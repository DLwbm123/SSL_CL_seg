from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from . import core
from ..gyro_native_v1.module import self_check
from ..main_style_ssl_v1.runtime import write
from ..gyro_autonomy_12h.space_guard import SpaceGuard


def main(cfg):
    assert self_check()['status']=='PASS'
    for arm in core.ARMS:
        m=nn.Module();m.parent=nn.Module();m.parent.native=nn.Module()
        m.parent.native.enc1=nn.Linear(1,1,bias=False);m.parent.native.decoder=nn.Linear(1,1,bias=False)
        m.projector=nn.Linear(1,1,bias=False);m.cfg={'output':cfg['root']}
        l=m.parent.native.enc1.weight.sum()+m.parent.native.decoder.weight.sum()
        raw=-(m.parent.native.enc1.weight.sum()+m.projector.weight.sum())
        with patch.object(core,'raw_losses',return_value=(l,raw,{})),patch.object(core,'append'):
            _,u,s=core.losses(arm,m,None,0)
        assert s['gradient_groups']['encoder_all']['CE_norm']>0
        assert (s['gradient_groups']['encoder_all']['weighted_G_norm']==0)==(arm=='CE_only')
        assert (s['gradient_groups']['projector']['weighted_G_norm']==0)==(arm=='CE_only')
        assert torch.allclose(u,l*0 if arm=='CE_only' else raw*.1)
    SpaceGuard(cfg['storage_authorization']).check(250_000_000)
    write(Path(cfg['root'])/'SYNTHETIC_CHECKS.json',dict(status='PASS',optimizer_calls=0,exact_weighted_gradient_contract=True,CE_geometry_gradient_zero=True))
