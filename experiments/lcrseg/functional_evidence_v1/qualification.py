"""Discarded engineering calls, exact resume, and immutable state guards."""
import copy
from pathlib import Path
import torch
from . import core
from ..philips_transfer_v1.run import snapshot,restore
from ..semantic_condition_v1.run import provider
from ..five_frameworks_v1.checkpoint import atomic_save
from ..main_style_ssl_v1.runtime import write,append
from ..siemens_ssl_reproduction_v1.methods import pixel_ce


def preflight(cfg,out,source,model,ema,opt,data):
    atomic_save(snapshot(model,ema,opt,data,0),out/'initial.pt')
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
    x,_,_=data.labeled(0)
    with torch.no_grad():
        assert torch.equal(model(x),source(x))
        assert torch.equal(ema(x),source(x))
        if model.functional:
            for name,b in model.named_buffers():
                if name.startswith('memory_'): assert b.data_ptr()==dict(ema.named_buffers())[name].data_ptr()
            assert model.state_mean.data_ptr()!=ema.state_mean.data_ptr()
            retained={}
            for network in (model,ema):
                for name,b in network.named_buffers():
                    if name.startswith(('memory_','state_')):
                        retained[b.untyped_storage().data_ptr()]=b.untyped_storage().nbytes()
            assert sum(retained.values())<=256*1024**2
    params=[p for p in model.parameters() if p.requires_grad]
    l,u,stats=core.losses(cfg['arm'],model,data,0)
    norms={}
    for name,term in [('L',l)]+([('U',u)] if cfg['arm'].endswith('_ssl') else []):
        gradients=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        norms[name]=sum(float(g.double().square().sum()) for g in gradients)**.5
        assert norms[name]>0
    (l+u).backward(); loss=float(l+u)
    grads=[p.grad.clone() for p in params]
    assert all(p.grad is None for p in source.parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    restore(initial,model,ema,opt,data)
    l,u,_=core.losses(cfg['arm'],model,data,0); (l+u).backward()
    assert float(l+u)==loss and all(torch.equal(p.grad,g) for p,g in zip(params,grads))
    if model.functional:
        # The state and sites, not just network parameters, are checkpointed.
        updated=copy.deepcopy(model.state_dict())
        probe=model.teacher(); probe.load_state_dict(updated)
        assert all(torch.equal(v,probe.state_dict()[k]) for k,v in updated.items())
        before_p=model.state_precision.clone(); before_m=model.state_mean.clone()
        with torch.no_grad():
            f,h=model.query(); model.refresh(f,h)
        assert torch.equal(model.state_precision,before_p)
        assert torch.equal(model.state_mean,before_m)
    restore(initial,model,ema,opt,data)
    write(out/'PREFLIGHT.json',dict(status='PASS',optimizer_updates=0,
        source_entry_max_abs_error=0.,gradient_norms=norms,golden_loss_and_gradients_exact=True,
        persistent_state_checkpointed=model.functional,**stats))


def qualify(cfg):
    from .run import main
    main(dict(cfg,preflight_only=True))
    out=Path(cfg['output'])
    source,model,ema,opt=core.setup(cfg)
    data=provider(cfg); data.arm=cfg['arm']
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
    restore(initial,model,ema,opt,data)
    x,y,_=data.labeled(0)
    calls=0
    def charge(*_):
        nonlocal calls
        calls+=1
        append(out/'engineering_updates.jsonl',dict(physical_call=calls,optimizer_executed=True))
    def guard(*_):
        if calls>=32: raise PermissionError('engineering phase cap')
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    opt.register_step_pre_hook(guard); opt.register_step_post_hook(charge)
    def step():
        opt.zero_grad(set_to_none=True)
        value=pixel_ce(model(x),y).mean()
        value.backward(); opt.step(); core.update_ema(ema,model)
        return float(value.detach())
    try:
        with torch.no_grad(): first=float(pixel_ce(model(x),y).mean())
        for i in range(30):
            if i==15: saved=copy.deepcopy(snapshot(model,ema,opt,data,15))
            value=step()
            if i==15:
                expected=copy.deepcopy(snapshot(model,ema,opt,data,16))
                restore(saved,model,ema,opt,data); again=step()
                assert again==value
                assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected['model'].items())
                assert all(torch.equal(v,ema.state_dict()[k]) for k,v in expected['ema'].items())
                for k,v in expected['optimizer']['state'].items():
                    assert torch.equal(v['momentum_buffer'],opt.state_dict()['state'][k]['momentum_buffer'])
        with torch.no_grad(): final=float(pixel_ce(model(x),y).mean())
        # A real full L/U/state update is charged too, rather than relying only
        # on the existence of a backward graph for the new mechanism.
        opt.zero_grad(set_to_none=True)
        l,u,_=core.losses(cfg['arm'],model,data,0)
        assert torch.isfinite(l+u)
        (l+u).backward(); opt.step(); core.update_ema(ema,model)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        atomic_save(snapshot(model,ema,opt,data,31),out/'engineering_terminal.pt')
        write(out/'ENGINEERING_MEASUREMENT.json',dict(first_ce=first,final_ce=final,actual_calls=calls))
        assert calls==32 and final<first-1e-4,('two-case fitting gate',first,final)
        for k,v in initial['model'].items():
            if k.startswith('memory_') or (cfg['arm']!='dense_lonly' and k.startswith('parent.native.enc1.')):
                assert torch.equal(v,model.state_dict()[k].cpu()),k
                assert torch.equal(v,ema.state_dict()[k].cpu()),k
        write(out/'ENGINEERING.json',dict(status='PASS',physical_optimizer_updates=calls,
            first_ce=first,final_ce=final,nonempty_momentum_resume_exact=True,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),discarded=True))
    except BaseException as exc:
        write(out/'ENGINEERING_FAILURE.json',dict(status='FAILED',error=str(exc),actual_calls=calls))
        raise
