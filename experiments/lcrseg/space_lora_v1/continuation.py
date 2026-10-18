"""Separately authorized 4,000 -> 10,000 continuation; no training-loop fork."""
import math
from pathlib import Path
from unittest.mock import patch
import torch
from . import core
from .run import provider
from ..philips_transfer_v1 import run as engine
from ..main_style_ssl_v1.runtime import write

STEPS=10000
original_preflight=engine.preflight


def rate(step):
    if not 0<=step<STEPS: raise PermissionError('outside continuation')
    return .03*math.cos(7*math.pi*step/(16*53300))


def preflight(cfg,out,source,model,ema,opt,data):
    original_preflight(cfg,out,source,model,ema,opt,data)
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
    state=torch.load(cfg['resume_checkpoint'],map_location='cpu',weights_only=False)
    assert state['arm']==cfg['arm'] and state['step']==state['physical_updates']==4000
    assert state['optimizer']['state'] and state['reads']==[4000,28000]
    params=[p for p in model.parameters() if p.requires_grad]
    engine.restore(state,model,ema,opt,data)
    l,u,_=engine.losses(cfg['arm'],model,data,4000)
    loss=l+u; loss.backward()
    value=float(loss.detach()); grads=[p.grad.clone() for p in params]
    assert all(torch.isfinite(g).all() for g in grads)
    engine.restore(state,model,ema,opt,data)
    ll,uu,_=engine.losses(cfg['arm'],model,data,4000)
    (ll+uu).backward()
    assert float(ll+uu)==value and all(torch.equal(p.grad,g) for p,g in zip(params,grads))
    assert all(p.grad is None for p in source.parameters())
    write(out/'RESUME_PREFLIGHT.json',dict(status='PASS',resume_step=4000,
          optimizer_updates=0,nonempty_momentum=True,restored_loss_and_gradient_exact=True,
          prior_32_update_two_case_and_momentum_qualification_reused=True,
          next_sampler_step=4000,schedule_continued=True))
    engine.restore(initial,model,ema,opt,data)


def main(cfg):
    if cfg.get('protocol')!='space_lora_continuation_v1' or cfg.get('resume_step')!=4000:
        raise PermissionError('unregistered continuation')
    if cfg.get('preflight_only') or not Path(cfg['resume_checkpoint']).is_file():
        raise PermissionError('invalid continuation entry')
    replacements=dict(ARMS=core.ARMS,STEPS=STEPS,GPU='3',setup=core.setup,
                      rate=rate,update_ema=core.update_ema,provider=provider,preflight=preflight)
    with patch.object(core,'STEPS',STEPS), patch.multiple(engine,**replacements): engine.main(cfg)
