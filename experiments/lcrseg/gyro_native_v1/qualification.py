"""Real gradients, golden batch, two-case learning, charged resume replay; all discarded."""
import copy
from pathlib import Path
import torch
from . import core
from .data import provider
from ..philips_transfer_v1.run import snapshot,restore
from ..five_frameworks_v1.checkpoint import atomic_save
from ..main_style_ssl_v1.runtime import write,append
from ..siemens_ssl_reproduction_v1.methods import pixel_ce


def preflight(cfg,out,source,model,ema,opt,data):
    atomic_save(snapshot(model,ema,opt,data,0),out/'initial.pt')
    state=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
    x,_,_=data.labeled(0)
    with torch.no_grad():
        assert torch.equal(model(x),source(x)) and torch.equal(ema(x),source(x))
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    fixed={'native.decoder.conv_logit.sigma.weight','native.decoder.conv_logit.grad_update'}
    assert {n for n,p in model.parent.named_parameters() if not p.requires_grad}==fixed
    l,u,stats=core.losses(cfg['arm'],model,data,0)
    norms={}
    for label,term in [('L',l),('U',u)]:
        grads=torch.autograd.grad(term,[p for n,p in params],allow_unused=True,retain_graph=True)
        assert all(g is None or torch.isfinite(g).all() for g in grads)
        norms[label]={}
        for key,prefix in [('encoder','parent.native.enc'),('decoder','parent.native.decoder'),('projector','projector.')]:
            norms[label][key]=sum(float(g.double().square().sum()) for (n,p),g in zip(params,grads) if g is not None and n.startswith(prefix))**.5
        assert norms[label]['encoder']>0
        if label=='L': assert norms[label]['decoder']>0
        if label=='U' and cfg['arm']!='dense_fixmatch': assert norms[label]['projector']>0
    (l+u).backward(); first=float(l+u)
    gg=[None if p.grad is None else p.grad.clone() for n,p in params]
    assert not model.frames.requires_grad and all(p.grad is None for p in source.parameters())
    restore(state,model,ema,opt,data)
    l,u,_=core.losses(cfg['arm'],model,data,0); (l+u).backward()
    assert first==float(l+u)
    assert all((p.grad is None and g is None) or (p.grad is not None and g is not None and torch.equal(p.grad,g)) for (n,p),g in zip(params,gg))
    restore(state,model,ema,opt,data)
    write(out/'PREFLIGHT.json',dict(status='PASS',optimizer_updates=0,source_entry_max_abs_error=0,
        source_reference_frozen=True,backbone_fully_trainable=True,frames_not_parameters=True,
        gradient_norms=norms,golden_loss_gradient_exact=True,**stats))


def qualify(cfg):
    from .run import main
    main(dict(cfg,preflight_only=True))
    out=Path(cfg['output']);source,model,ema,opt=core.setup(cfg)
    data=provider(cfg);data.arm=cfg['arm']
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
    restore(initial,model,ema,opt,data)
    x,y,_=data.labeled(0);calls=0
    def guard(*_):
        if calls>=32: raise PermissionError('engineering cap32')
    def record(*_):
        nonlocal calls
        calls+=1;append(out/'engineering_updates.jsonl',dict(physical_call=calls,optimizer_executed=True))
    opt.register_step_pre_hook(guard);opt.register_step_post_hook(record)
    def step():
        opt.zero_grad(set_to_none=True)
        loss=pixel_ce(model(x),y).mean();loss.backward();opt.step();core.update_ema(ema,model)
        return float(loss.detach())
    try:
        with torch.no_grad(): first=float(pixel_ce(model(x),y).mean())
        for i in range(30):
            if i==15: saved=copy.deepcopy(snapshot(model,ema,opt,data,15))
            value=step()
            if i==15:
                expected=copy.deepcopy(snapshot(model,ema,opt,data,16))
                restore(saved,model,ema,opt,data);assert step()==value
                assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected['model'].items())
                assert all(torch.equal(v,ema.state_dict()[k]) for k,v in expected['ema'].items())
                for k,v in expected['optimizer']['state'].items():
                    assert torch.equal(v['momentum_buffer'],opt.state_dict()['state'][k]['momentum_buffer'])
        with torch.no_grad(): final=float(pixel_ce(model(x),y).mean())
        opt.zero_grad(set_to_none=True)
        l,u,_=core.losses(cfg['arm'],model,data,0);(l+u).backward();opt.step();core.update_ema(ema,model)
        atomic_save(snapshot(model,ema,opt,data,31),out/'engineering_terminal.pt')
        write(out/'ENGINEERING_MEASUREMENT.json',dict(first_ce=first,final_ce=final,actual_calls=calls))
        assert calls==32 and final<first-1e-4,('two-case fitting gate',first,final)
        assert torch.equal(initial['model']['frames'],model.frames.cpu())
        assert all(torch.isfinite(p).all() for p in model.parameters())
        write(out/'ENGINEERING.json',dict(status='PASS',physical_optimizer_updates=32,
            discarded=True,first_ce=first,final_ce=final,nonempty_momentum_resume_exact=True,
            frames_exact=True,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    except BaseException as exc:
        write(out/'ENGINEERING_FAILURE.json',dict(status='FAILED',error=str(exc),actual_calls=calls));raise
