"""Reuse the project's native training engine; evaluation in a separate process."""
import copy,json
from pathlib import Path
from unittest.mock import patch
import torch
from . import core
from .data import Domain
from .runtime import save,exact,guard
from ..philips_transfer_v1 import run as engine
from ..siemens_ssl_reproduction_v1.methods import pixel_ce
from ..main_style_ssl_v1.runtime import write


def preflight(cfg,out,source,model,ema,opt,data):
    initial=copy.deepcopy(engine.snapshot(model,ema,opt,data,0));save(cfg,initial,out/'initial.pt')
    assert not opt.state
    fixed={'native.decoder.conv_logit.sigma.weight','native.decoder.conv_logit.grad_update'}
    assert {n for n,p in model.parent.named_parameters() if not p.requires_grad}==fixed
    x,_,_=data.labeled(0)
    with torch.no_grad():assert torch.equal(source(x),model(x)) and torch.equal(ema(x),model(x))
    l,u,_=core.losses(cfg['arm'],model,data,0)
    ep=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(('parent.native.enc','parent.native.bottleneck'))]
    g=torch.autograd.grad(u,ep,retain_graph=True,allow_unused=True);norm=sum(float(v.square().sum()) for v in g if v is not None)**.5
    assert norm>0 and all(v is None or torch.isfinite(v).all() for v in g)
    (l+u).backward();value=float(l+u);pars=[p for p in model.parameters() if p.requires_grad];gg=[p.grad.clone() for p in pars]
    engine.restore(initial,model,ema,opt,data);l,u,_=core.losses(cfg['arm'],model,data,0);(l+u).backward()
    assert float(l+u)==value and all(torch.equal(p.grad,a) for p,a in zip(pars,gg))
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('parent.'))
    engine.restore(initial,model,ema,opt,data);save(cfg,engine.snapshot(model,ema,opt,data,0),out/'latest.pt')
    write(out/'PREFLIGHT.json',dict(status='PASS',common_native_entry=True,empty_optimizer=True,full_native_trainable=True,frozen_memory=True,encoder_geometry_gradient=norm,golden_exact=True,target_only_HDF5=True))


def main(cfg):
    def saver(state,path):
        if Path(path).name=='exception_state.pt':
            from ..five_frameworks_v1.checkpoint import atomic_save
            guard(cfg,200_000_000);atomic_save(state,path)
        else:save(cfg,state,path)
    def finalize(cfg,out,*args):write(out/'TRAINED.json',dict(status='TRAINED',fixed_final_student=True,evaluation_separate_process=True))
    with patch.multiple(engine,ARMS=('A','B','C','D','E'),STEPS=2000,GPU='3',setup=core.target_setup,provider=lambda cfg:Domain(cfg,'Philips'),preflight=preflight,losses=core.losses,rate=lambda step:.001,set_learning_rate=lambda opt,step:None,update_ema=core.update_ema,atomic_save=saver,finalize=finalize):engine.main(cfg)


def qualify(cfg):
    main(dict(cfg,preflight_only=True))
    out=Path(cfg['output']);source,model,ema,opt=core.target_setup(cfg);data=Domain(cfg,'Philips')
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False);engine.restore(initial,model,ema,opt,data)
    x,y,_=data.labeled(0);calls=0
    def step():
        nonlocal calls
        opt.zero_grad(set_to_none=True);v=pixel_ce(model(x),y).mean();v.backward();opt.step();core.update_ema(ema,model);calls+=1;return float(v)
    first=float(pixel_ce(model(x),y).mean())
    for i in range(30):
        before=copy.deepcopy(engine.snapshot(model,ema,opt,data,i)) if i==15 else None
        v=step()
        if i==15:
            expected=copy.deepcopy(engine.snapshot(model,ema,opt,data,i+1));engine.restore(before,model,ema,opt,data)
            assert step()==v and exact(expected,engine.snapshot(model,ema,opt,data,i+1))
    last=float(pixel_ce(model(x),y).mean());assert last<first-1e-4
    engine.restore(initial,model,ema,opt,data);opt.zero_grad(set_to_none=True);l,u,_=core.losses(cfg['arm'],model,data,0);(l+u).backward();opt.step();core.update_ema(ema,model);calls+=1
    assert all(torch.equal(v.cpu(),initial['model'][k]) for k,v in model.state_dict().items() if not k.startswith('parent.'))
    save(cfg,engine.snapshot(model,ema,opt,data,1),out/'engineering_terminal.pt')
    assert calls==32
    write(out/'ENGINEERING.json',dict(status='PASS',discarded=True,actual_calls=calls,first=first,final=last,nonempty_restore_exact=True,full_target_from_empty_state=True,peak_bytes=torch.cuda.max_memory_allocated()))


def evaluate(cfg):
    from ..gyro_transfer_v1.run import finalize
    out=Path(cfg['output']);model=core.build(cfg);ema=copy.deepcopy(model)
    saved=torch.load(out/'final.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema'])
    assert saved['step']==saved['physical_updates']==2000
    # Original protocol uses all13 target labeled patients, not prior holdout subset.
    with patch.object(engine,'subset',lambda access,patients:access):finalize(cfg,out,model,ema,2000,2000)
    write(out/'EVALUATION_EXIT.json',dict(exit_code=0))
