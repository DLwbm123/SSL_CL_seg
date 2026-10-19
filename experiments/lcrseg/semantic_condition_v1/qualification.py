"""Real entry/gradient/EMA checks and separately charged discarded updates."""
import copy
from pathlib import Path
import torch
from . import core
from ..philips_transfer_v1.run import snapshot, restore
from ..siemens_ssl_reproduction_v1.methods import pixel_ce
from ..five_frameworks_v1.checkpoint import atomic_save
from ..main_style_ssl_v1.runtime import write, append


def preflight(cfg,out,source,model,ema,opt,data):
    atomic_save(snapshot(model,ema,opt,data,0),out/'initial.pt')
    state=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
    x,_,_=data.labeled(0)
    report={}
    with torch.no_grad():
        reference=source(x)
        entry_error=float((model(x)-reference).abs().max())
        assert torch.equal(model(x),reference)
        assert torch.equal(ema(x),reference)
        probe=copy.deepcopy(model)
        for p in probe.parameters():
            if p.requires_grad:
                p.add_(torch.sin(torch.arange(p.numel(),device=p.device).reshape_as(p))*.0001)
        target=model.teacher()
        before={k:v.clone() for k,v in target.state_dict().items()}
        core.update_ema(target,probe)
        trainable={n for n,p in model.named_parameters() if p.requires_grad}
        for k,v in target.state_dict().items():
            if k in trainable:
                expected=before[k]*.999+probe.state_dict()[k]*.001
                assert torch.allclose(v,expected,atol=1e-7,rtol=1e-6),k
            elif isinstance(model,core.Conditioned):
                assert torch.equal(v,before[k]),k
        deployment=probe.deploy()
        assert torch.allclose(probe(x),deployment(x),atol=3e-5,rtol=3e-5)
        if isinstance(model,core.Conditioned):
            h=model.parent.features(x)
            r=model.conditioner.condition(h)
            # Same fixed spatial shift for both aligned/shuffled diagnostics.
            shifted=r.flatten(2).roll((r.shape[2]*r.shape[3])//2,dims=2).reshape_as(r)
            report=dict(spatial_class_agreement=float((r.argmax(1)==shifted.argmax(1)).float().mean()),
                        spatial_probability_mean_abs_change=float((r-shifted).abs().mean()),
                        condition_class_mass=r.mean((0,2,3)).tolist())
        del probe,target,before,deployment
    restore(state,model,ema,opt,data)
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if isinstance(model,core.Conditioned):
        assert all(n.startswith('conditioner.') for n,p in params)
        assert not model.conditioner.prototypes.requires_grad
    l,u,stats=core.losses(cfg['arm'],model,data,0)
    norms={}
    for name,term in [('L',l)]+([] if cfg['arm'].endswith('_lonly') else [('U',u)]):
        grads=torch.autograd.grad(term,[p for _,p in params],retain_graph=True,allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        norms[name]=sum(float(g.double().square().sum()) for g in grads)**.5
        assert norms[name]>0, (name,'no real gradient; stop qualification')
    (l+u).backward()
    saved=[p.grad.clone() for _,p in params]; first=float(l+u)
    assert all(p.grad is None for p in source.parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    del l,u
    restore(state,model,ema,opt,data)
    l,u,_=core.losses(cfg['arm'],model,data,0); (l+u).backward()
    assert float(l+u)==first
    assert all(torch.equal(p.grad,g) for (_,p),g in zip(params,saved))
    assert not opt.state
    restore(state,model,ema,opt,data)
    write(out/'PREFLIGHT.json',dict(status='PASS',optimizer_updates=0,
          source_entry_max_abs_error=entry_error,source_frozen=True,
          native_frozen=isinstance(model,core.Conditioned),gradient_norms=norms,
          residual_ema_and_immutable_buffers=True,nonzero_deployment_identity=True,
          checkpoint_restore_loss_and_gradient_exact=True,
          trainable_parameters=sum(p.numel() for _,p in params),**stats,**report))


def qualify(cfg):
    from .run import main,provider
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
        if calls>=32: raise PermissionError('engineering cap exhausted')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite engineering gradient')
    hook=opt.register_step_post_hook(charge); guard_hook=opt.register_step_pre_hook(guard)
    def step():
        opt.zero_grad(set_to_none=True)
        value=pixel_ce(model(x),y).mean()
        if not torch.isfinite(value): raise FloatingPointError('nonfinite CE')
        value.backward(); opt.step(); core.update_ema(ema,model)
        return float(value.detach())
    try:
        first=None
        for index in range(31):
            value=step()
            if first is None: first=value
            if index==14: atomic_save(snapshot(model,ema,opt,data,15),out/'resume_probe.pt')
            if index==15:
                expected={k:v.clone() for k,v in model.state_dict().items()}
                momentum=copy.deepcopy(opt.state_dict())
                expected_ema={k:v.clone() for k,v in ema.state_dict().items()}
                resume=torch.load(out/'resume_probe.pt',map_location='cpu',weights_only=False)
                restore(resume,model,ema,opt,data); again=step()
                assert again==value
                assert all(torch.equal(v,expected[k]) for k,v in model.state_dict().items())
                assert all(torch.equal(v,expected_ema[k]) for k,v in ema.state_dict().items())
                for key,state in opt.state_dict()['state'].items():
                    assert torch.equal(state['momentum_buffer'],momentum['state'][key]['momentum_buffer'])
        with torch.no_grad(): final=float(pixel_ce(model(x),y).mean())
        # Preserve the terminal evidence even when a numerical gate rejects it.
        atomic_save(snapshot(model,ema,opt,data,31),out/'engineering_terminal.pt')
        write(out/'ENGINEERING_MEASUREMENT.json',dict(first_ce=first,final_ce=final,
              physical_optimizer_updates=calls,trajectory_updates=31))
        assert final<first-1e-4,('two-case CE did not decrease',first,final)
        assert calls==32 and all(p.grad is None for p in source.parameters())
        if isinstance(model,core.Conditioned):
            for k,v in model.state_dict().items():
                if k.startswith('parent.') or k=='conditioner.prototypes':
                    assert torch.equal(v.cpu(),initial['model'][k]),k
                    assert torch.equal(ema.state_dict()[k].cpu(),initial['ema'][k]),k
        write(out/'ENGINEERING.json',dict(status='PASS',physical_optimizer_updates=calls,
              two_case_initial_ce=first,two_case_final_ce=final,
              nonempty_momentum_resume_exact=True,
              immutable_adapter_state_exact=True if isinstance(model,core.Conditioned) else None,
              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),discarded=True))
    except BaseException as exc:
        write(out/'ENGINEERING_FAILURE.json',dict(status='FAILED',error=str(exc),physical_optimizer_updates=calls))
        raise
    finally:
        hook.remove(); guard_hook.remove()
        restore(initial,model,ema,opt,data)
