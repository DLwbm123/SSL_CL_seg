from pathlib import Path
from unittest.mock import patch
import torch
from . import core
from ..gyro_transfer_v1.run import finalize
from ..gyro_source_repair_v11 import qualification as old_q,run as old_run
from ..philips_transfer_v1 import run as engine
from ..five_frameworks_v1.checkpoint import atomic_save as real_save
from ..gyro_autonomy_12h.space_guard import SpaceGuard
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
    assert (g['encoder_all']['weighted_G_norm']==0)==(cfg['arm']=='CE_only')
    assert (g['projector']['weighted_G_norm']==0)==(cfg['arm']=='CE_only')
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


def guarded_saver(cfg):
    guard=SpaceGuard(cfg['storage_authorization'])
    def save(state,path):
        guard.check(100_000_000);real_save(state,path)
    return save


def main(cfg):
    with patch.multiple(engine,ARMS=core.ARMS,STEPS=2000,GPU='3',setup=core.setup,provider=core.provider,
        preflight=preflight,losses=core.losses,rate=lambda step:.001,set_learning_rate=core.configure_lr,
        update_ema=core.update_ema,finalize=finalize,atomic_save=guarded_saver(cfg)):
        engine.main(cfg)


def qualify(cfg):
    with patch.object(old_q,'core',core),patch.object(old_q,'provider',core.provider),patch.object(old_run,'main',main),patch.object(old_q,'atomic_save',guarded_saver(cfg)):
        old_q.qualify(cfg)
