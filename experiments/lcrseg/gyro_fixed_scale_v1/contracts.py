from pathlib import Path
import torch
from torch import nn
from . import core
from ..gyro_native_v1.module import self_check
from ..main_style_ssl_v1.runtime import write
from ..gyro_autonomy_12h.space_guard import SpaceGuard


def main(cfg):
    assert cfg['fixed_projector'] and self_check()['status']=='PASS'
    m=nn.Sequential(nn.Linear(4,8),nn.BatchNorm1d(8),nn.ReLU(),nn.Linear(8,4))
    m.eval().requires_grad_(False);m.register_forward_pre_hook(core.fixed_eval)
    old={k:v.clone() for k,v in m.state_dict().items()};m.train()
    x=torch.randn(6,4,requires_grad=True);m(x).square().sum().backward()
    assert x.grad is not None and x.grad.norm()>0 and all(p.grad is None for p in m.parameters())
    assert not m.training and all(torch.equal(v,m.state_dict()[k]) for k,v in old.items())
    assert abs(.075*(1-1/3)-.05)<1e-12 and cfg['scale_divisor']==6
    SpaceGuard(cfg['storage_authorization']).check(300_000_000)
    write(Path(cfg['root'])/'SYNTHETIC_CHECKS.json',dict(status='PASS',optimizer_calls=0,frozen_projection_input_gradient=True,weights_BN_exact_after_train_call=True))
