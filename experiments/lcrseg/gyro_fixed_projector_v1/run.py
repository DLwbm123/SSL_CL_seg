from pathlib import Path
from unittest.mock import patch
import torch
from . import core
from ..gyro_strength_v1 import run as shared
from ..philips_transfer_v1 import run as engine
from ..main_style_ssl_v1.runtime import write


def preflight(cfg,out,source,model,ema,opt,data):
    initial=engine.snapshot(model,ema,opt,data,0);engine.atomic_save(initial,out/'initial.pt')
    assert not opt.state
    x,_,_=data.labeled(0)
    with torch.no_grad():assert torch.equal(model(x),source(x)) and torch.equal(ema(x),source(x))
    fixed={'native.decoder.conv_logit.sigma.weight','native.decoder.conv_logit.grad_update'}
    assert {n for n,p in model.parent.named_parameters() if not p.requires_grad}==fixed
    l,u,stats=core.losses(cfg['arm'],model,data,0);g=stats['gradient_groups']
    assert g['encoder_all']['CE_norm']>0
    assert g['encoder_all']['weighted_G_norm']>0
    assert g['projector']['weighted_G_norm']==0
    assert all(not p.requires_grad for p in model.projector.parameters()) and not model.projector.training
    assert all(not g['name'].startswith('projector') for g in opt.param_groups)
    pars=[p for p in model.parameters() if p.requires_grad]
    (l+u).backward();value=float(l+u);grads=[None if p.grad is None else p.grad.clone() for p in pars]
    assert not model.frames.requires_grad and all(p.grad is None for p in source.parameters())
    engine.restore(initial,model,ema,opt,data)
    l,u,_=core.losses(cfg['arm'],model,data,0);(l+u).backward()
    assert float(l+u)==value
    assert all((a is None and p.grad is None) or (a is not None and p.grad is not None and torch.equal(a,p.grad)) for p,a in zip(pars,grads))
    engine.restore(initial,model,ema,opt,data)
    engine.atomic_save(engine.snapshot(model,ema,opt,data,0),out/'latest.pt')
    write(out/'PREFLIGHT.json',dict(status='PASS',optimizer_calls=0,both_momenta_empty=True,common_entry_exact=True,golden_exact=True,**stats))



def main(cfg):
    with patch.multiple(shared,core=core,preflight=preflight):shared.main(cfg)


def qualify(cfg):
    with patch.multiple(shared,core=core,preflight=preflight):shared.qualify(cfg)
