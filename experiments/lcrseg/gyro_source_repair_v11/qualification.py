import copy
from pathlib import Path
import torch
from . import core
from .data import provider
from ..gyro_native_v1.module import spectral
from ..philips_transfer_v1.run import snapshot,restore
from ..five_frameworks_v1.checkpoint import atomic_save
from ..main_style_ssl_v1.runtime import write,append
from ..siemens_ssl_reproduction_v1.methods import pixel_ce,strong_view


def preflight(cfg,out,source,model,ema,opt,data):
    atomic_save(snapshot(model,ema,opt,data,0),out/'initial.pt')
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
    cal=cfg['arm']=='calibration'
    x=data.pool.epoch_batch(0)[:2] if cal else data.labeled(69)[0]
    if not cal:
        x,y,_=data.labeled(69);model.query=(x,y)
        fixed={'native.decoder.conv_logit.sigma.weight','native.decoder.conv_logit.grad_update'}
        assert {n for n,p in model.parent.named_parameters() if not p.requires_grad}==fixed
    assert not opt.state,'stage-boundary momentum must be empty'
    with torch.no_grad():assert torch.equal(model(x),source(x)) and torch.equal(ema(x),source(x))
    l,u,stats=core.losses(cfg['arm'],model,data,0)
    norm=stats['gradient_groups']
    if cal:assert norm['projector']['weighted_G_norm']>0 and norm['encoder_all']['weighted_G_norm']==0
    else:
        assert norm['encoder_all']['CE_norm']>0 and norm['decoder']['CE_norm']>0
        if cfg['arm']=='GYRO_low':assert norm['encoder_all']['weighted_G_norm']>0
    params=[p for p in model.parameters() if p.requires_grad]
    (l+u).backward();value=float(l+u);gg=[None if p.grad is None else p.grad.clone() for p in params]
    assert all(p.grad is None for p in source.parameters())
    restore(initial,model,ema,opt,data)
    l,u,_=core.losses(cfg['arm'],model,data,0);(l+u).backward()
    assert float(l+u)==value
    assert all((g is None and p.grad is None) or (g is not None and p.grad is not None and torch.equal(g,p.grad)) for p,g in zip(params,gg))
    restore(initial,model,ema,opt,data)
    write(out/'PREFLIGHT.json',dict(status='PASS',optimizer_updates=0,source_entry_max_abs_error=0,
        momentum_empty=True,calibration=cal,native_fully_trainable=not cal,
        frames_not_parameters=not model.frames.requires_grad,golden_exact=True,**stats))


def qualify(cfg):
    from .run import main
    main(dict(cfg,preflight_only=True))
    out=Path(cfg['output']);source,model,ema,opt=core.setup(cfg)
    data=provider(cfg);data.arm=cfg['arm'];cal=cfg['arm']=='calibration'
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
    restore(initial,model,ema,opt,data)
    if cal:
        images=data.pool.epoch_batch(0)[:2];images=torch.cat([images,strong_view(images,0)])
    else:
        x,y,_=data.labeled(69);model.query=(x,y)
    def objective():
        if not cal:return pixel_ce(model(x),y).mean()
        z=core.projection(model,images)
        return spectral(torch.stack(z.chunk(2),1),model.frames,torch.zeros(2,dtype=torch.long,device=z.device),0)
    calls=0
    def guard(*_):
        if calls>=32:raise PermissionError('engineering cap32')
    def charge(*_):
        nonlocal calls
        calls+=1;append(out/'engineering_updates.jsonl',dict(physical_call=calls,optimizer_executed=True))
    opt.register_step_pre_hook(guard);opt.register_step_post_hook(charge)
    def step():
        opt.zero_grad(set_to_none=True);v=objective();v.backward();opt.step();core.update_ema(ema,model)
        return float(v.detach())
    try:
        first=float(objective().detach())
        # BN is part of the exact starting state, including this measurement forward.
        start=copy.deepcopy(snapshot(model,ema,opt,data,0))
        for i in range(30):
            if i==15:saved=copy.deepcopy(snapshot(model,ema,opt,data,15))
            v=step()
            if i==15:
                expected=copy.deepcopy(snapshot(model,ema,opt,data,16))
                restore(saved,model,ema,opt,data);assert step()==v
                assert all(torch.equal(a,model.state_dict()[k]) for k,a in expected['model'].items())
                assert all(torch.equal(a,ema.state_dict()[k]) for k,a in expected['ema'].items())
                for k,a in expected['optimizer']['state'].items():assert torch.equal(a['momentum_buffer'],opt.state_dict()['state'][k]['momentum_buffer'])
        final=float(objective().detach())
        write(out/'ENGINEERING_MEASUREMENT.json',dict(first=first,final=final,actual_calls=calls))
        assert final<first-1e-4,('two-case learning gate',first,final)
        # The real full-target check starts from the unmodified stage entry.
        # All31 already executed calls remain charged in the global ledger.
        restore(initial,model,ema,opt,data)
        opt.zero_grad(set_to_none=True);l,u,_=core.losses(cfg['arm'],model,data,0)
        (l+u).backward();opt.step();core.update_ema(ema,model)
        state=snapshot(model,ema,opt,data,1);state['engineering_actual_calls']=calls
        atomic_save(state,out/'engineering_terminal.pt')
        assert calls==32 and torch.equal(model.frames.cpu(),initial['model']['frames'])
        if cal:assert all(torch.equal(v.cpu(),initial['model'][k]) for k,v in model.state_dict().items() if k.startswith('parent.'))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        write(out/'ENGINEERING.json',dict(status='PASS',actual_calls=32,discarded=True,first=first,final=final,
            nonempty_momentum_resume_exact=True,full_target_from_entry=True,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    except BaseException as exc:
        write(out/'ENGINEERING_FAILURE.json',dict(status='FAILED',actual_calls=calls,error=str(exc)));raise
