"""Separate, charged engineering updates, discarded before the formal pilot."""
import copy
from pathlib import Path
import torch
from . import core
from .run import main, provider
from ..philips_transfer_v1.run import snapshot, restore
from ..siemens_ssl_reproduction_v1.methods import pixel_ce
from ..five_frameworks_v1.checkpoint import atomic_save
from ..main_style_ssl_v1.runtime import write, append


def qualify(cfg):
    main(dict(cfg,preflight_only=True))
    out=Path(cfg['output'])
    source,model,ema,opt=core.setup(cfg)
    data=provider(cfg); data.arm=cfg['arm']
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
    restore(initial,model,ema,opt,data)
    x,y,_=data.labeled(0)
    calls=0
    def charge(opt,args,kwargs):
        nonlocal calls
        calls+=1
        append(out/'engineering_updates.jsonl',dict(physical_call=calls,optimizer_executed=True))
    def guard(opt,args,kwargs):
        if calls>=32: raise PermissionError('engineering cap exhausted')
    hook=opt.register_step_post_hook(charge); guard_hook=opt.register_step_pre_hook(guard)
    def step():
        opt.zero_grad(set_to_none=True)
        loss=pixel_ce(model(x),y).mean()
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite engineering loss')
        loss.backward(); opt.step(); core.update_ema(ema,model)
        return float(loss.detach())
    try:
        first=None
        for index in range(31):
            value=step()
            if first is None: first=value
            if index==14:
                atomic_save(snapshot(model,ema,opt,data,15),out/'resume_probe.pt')
            if index==15:
                expected={k:v.clone() for k,v in model.state_dict().items()}
                momentum=copy.deepcopy(opt.state_dict())
                ema_expected={k:v.clone() for k,v in ema.state_dict().items()}
                resume=torch.load(out/'resume_probe.pt',map_location='cpu',weights_only=False)
                restore(resume,model,ema,opt,data); again=step()
                assert again==value
                assert all(torch.equal(v,expected[k]) for k,v in model.state_dict().items())
                assert all(torch.equal(v,ema_expected[k]) for k,v in ema.state_dict().items())
                for key,state in opt.state_dict()['state'].items():
                    assert torch.equal(state['momentum_buffer'],momentum['state'][key]['momentum_buffer'])
        with torch.no_grad(): final=float(pixel_ce(model(x),y).mean())
        assert final<first-1e-4, ('two-case CE did not decrease',first,final)
        assert calls==32 and all(p.grad is None for p in source.parameters())
        write(out/'ENGINEERING.json',dict(status='PASS',physical_optimizer_updates=calls,
              two_case_initial_ce=first,two_case_final_ce=final,
              resume_nonempty_momentum_exact=True,discarded=True))
    finally:
        hook.remove(); guard_hook.remove()
        restore(initial,model,ema,opt,data)
