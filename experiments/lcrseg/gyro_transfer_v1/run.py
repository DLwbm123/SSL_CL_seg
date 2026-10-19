"""Register the frozen transfer objective in the shared training engine."""
import json
from pathlib import Path
from unittest.mock import patch
import torch
from . import core
from ..philips_transfer_v1 import run as engine
from ..gyro_source_repair_v11 import qualification as old_q,run as old_run
from ..five_frameworks_v1.checkpoint import atomic_save
from ..main_style_ssl_v1.runtime import write
from ..siemens_lonly_v1.run import aggregate,compare


def preflight(cfg,out,source,model,ema,opt,data):
    initial=engine.snapshot(model,ema,opt,data,0);atomic_save(initial,out/'initial.pt')
    assert not opt.state
    x,_,_=data.labeled(0)
    with torch.no_grad():assert torch.equal(model(x),source(x)) and torch.equal(ema(x),source(x))
    fixed={'native.decoder.conv_logit.sigma.weight','native.decoder.conv_logit.grad_update'}
    assert {n for n,p in model.parent.named_parameters() if not p.requires_grad}==fixed
    l,u,stats=core.losses(cfg['arm'],model,data,0)
    assert stats['gradient_groups']['encoder_all']['CE_norm']>0
    assert stats['gradient_groups']['encoder_all']['weighted_G_norm']>0
    if cfg['arm']!='dense_fixmatch':assert stats['gradient_groups']['projector']['weighted_G_norm']>0
    else:assert stats['gradient_groups']['projector']['weighted_G_norm']==0
    pars=[p for p in model.parameters() if p.requires_grad]
    (l+u).backward();value=float(l+u);gg=[None if p.grad is None else p.grad.clone() for p in pars]
    assert not model.frames.requires_grad and all(p.grad is None for p in source.parameters())
    engine.restore(initial,model,ema,opt,data)
    l,u,_=core.losses(cfg['arm'],model,data,0);(l+u).backward()
    assert float(l+u)==value
    assert all((g is None and p.grad is None) or (g is not None and p.grad is not None and torch.equal(g,p.grad)) for p,g in zip(pars,gg))
    engine.restore(initial,model,ema,opt,data)
    atomic_save(engine.snapshot(model,ema,opt,data,0),out/'latest.pt')
    write(out/'PREFLIGHT.json',dict(status='PASS',optimizer_updates=0,source_entry_max_abs_error=0,
        both_momenta_empty=True,native_fully_trainable=True,golden_exact=True,
        frames_not_parameters=True,**stats))


def finalize(cfg,out,model,ema,committed,physical):
    old,oldrows=engine.evaluate(model,cfg,'Siemens','val',out/'private_Siemens_val.jsonl')
    target,targetrows=engine.evaluate(model,cfg,'Philips','val',out/'private_Philips_val.jsonl')
    source_train,_=engine.evaluate(model,cfg,'Siemens','train_labeled',out/'private_source_train.jsonl')
    target_train,_=engine.evaluate(model,cfg,'Philips','train_labeled',out/'private_target_train.jsonl')
    ema_old,_=engine.evaluate(ema,cfg,'Siemens','val',out/'private_ema_Siemens_val.jsonl')
    ema_target,_=engine.evaluate(ema,cfg,'Philips','val',out/'private_ema_Philips_val.jsonl')
    base_old=[json.loads(x) for x in Path(cfg['source_val_rows']).read_text().splitlines()]
    base_target=[json.loads(x) for x in (Path(cfg['root'])/'preparation/private_Philips_val.jsonl').read_text().splitlines()]
    o=aggregate(base_old)['primary_dice'];n=aggregate(base_target)['primary_dice']
    assert abs(o-.7943371427121955)<1e-12
    write(out/'RESULTS.json',dict(status='COMPLETE',arm=cfg['arm'],step=committed,physical_updates=physical,
        primary='fixed final student',Siemens=old,Philips=target,source_train=source_train,target_train=target_train,
        EMA_diagnostic=dict(Siemens=ema_old,Philips=ema_target),two_domain_mean=(old['primary_dice']+target['primary_dice'])/2,
        old_delta_from_entry_pp=100*(old['primary_dice']-o),target_delta_from_entry_pp=100*(target['primary_dice']-n),
        paired_old_entry=compare(base_old,oldrows),paired_target_entry=compare(base_target,targetrows),validation_only=True))


def main(cfg):
    with patch.multiple(engine,ARMS=core.ARMS,STEPS=2000,GPU='3',setup=core.setup,provider=core.provider,
        preflight=preflight,losses=core.losses,rate=lambda step:.001,set_learning_rate=core.configure_lr,
        update_ema=core.update_ema,finalize=finalize):engine.main(cfg)


def qualify(cfg):
    # Reuse the charged32-call qualification: full objective check restores the clean initial state.
    with patch.object(old_q,'core',core),patch.object(old_q,'provider',core.provider),patch.object(old_run,'main',main):
        old_q.qualify(cfg)
