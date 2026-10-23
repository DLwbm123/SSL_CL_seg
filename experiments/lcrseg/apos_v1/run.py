"""Register APOS in the existing training engine; do not duplicate its loop."""
import copy,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from . import model as alg
from .data import Current,fields
from .runtime import save,install
from ..philips_transfer_v1 import run as engine
from ..main_style_ssl_v1.runtime import write
from ..gyro_spatial_memory_v1.runtime import exact,finite


def terms(model,x,y):
    logp=model(x);target,patterns=fields(y);flat=target.flatten(1)
    ce=-logp.gather(1,y[:,None]).mean()
    p=logp.exp()[:,1:];one=F.one_hot(y,4).permute(0,3,1,2)[:,1:].to(p)
    dice=1-((2*(p*one).sum((2,3))+1e-5)/(p.sum((2,3))+one.sum((2,3))+1e-5)).mean()
    t,w,rawpi,_=model.last
    obs=.5*(w*(t-flat).square()-w.log()+np.log(2*np.pi)).mean()
    post=model.posterior_nll(flat);presence=F.cross_entropy(rawpi,patterns)
    total=ce+dice+.1*(obs+post+presence)
    assert torch.isfinite(total)
    return ce+dice,.1*(obs+post+presence),dict(CE=float(ce.detach()),Dice_loss=float(dice.detach()),observation_NLL=float(obs.detach()),posterior_NLL=float(post.detach()),presence_CE=float(presence.detach()),precision_min=float(w.detach().min()),precision_max=float(w.detach().max()),accepted_class_pixels=[0,0,0,0],U_pixels=0)


def losses(arm,model,data,step):
    x,y,_=data.labeled(step)
    return terms(model,x,y)


def check_trainability(model,opt):
    expected={id(p) for p in model.parameters() if p.requires_grad}
    assert expected=={id(p) for g in opt.param_groups for p in g['params']}
    assert not any(p.requires_grad for p in model.parent.native.decoder.conv_logit.parameters())
    if model.stage=='target':assert not any(p.requires_grad for p in model.parent.parameters()) and not model.parent.training
    alg.assert_frozen(model)


def preflight(cfg,out,source,model,ema,opt,data):
    model.frozen_reference=alg.frozen(model);ema.frozen_reference=alg.frozen(model)
    check_trainability(model,opt)
    state=copy.deepcopy(engine.snapshot(model,ema,opt,data,0))
    if cfg.get('repair_initial') and cfg['stage']=='source' and cfg['arm']=='D':
        clean=torch.load(cfg['repair_initial'],map_location='cpu',weights_only=False)
        assert exact(state,clean),'clean initial model/EMA/optimizer/RNG state differs'
    save(cfg,state,out/'initial.pt')
    assert not opt.state
    if cfg['phase']=='qualification':
        first=None;grads=None
        for attempt in range(2):
            engine.restore(state,model,ema,opt,data)
            l,u,_=losses(cfg['arm'],model,data,0);(l+u).backward()
            value=float((l+u).detach());current=[None if p.grad is None else p.grad.clone() for p in model.parameters() if p.requires_grad]
            assert all(g is None or torch.isfinite(g).all() for g in current)
            assert sum(float(g.square().sum()) for g in current if g is not None)>1e-12
            if attempt==0:first=value;grads=current
            else:assert value==first and exact(current,grads)
        engine.restore(state,model,ema,opt,data)
    save(cfg,engine.snapshot(model,ema,opt,data,0),out/'latest.pt')
    write(out/'PREFLIGHT.json',dict(status='PASS',empty_optimizer=True,exact_trainable_set=True,prior_and_classification_head_frozen=True,golden_exact=cfg['phase']=='qualification',target_train_L_only=cfg['stage']=='target'))


def trained(cfg,out,*args):
    write(out/'TRAINED.json',dict(status='TRAINED',metrics_sealed=True))


def train(cfg):
    install(cfg)
    with patch.multiple(engine,ARMS=('D','G','T'),STEPS=2000,GPU='3',setup=alg.setup,provider=Current,preflight=preflight,losses=losses,rate=lambda step:.001,set_learning_rate=lambda *args:None,update_ema=alg.ema_update,atomic_save=lambda state,path:save(cfg,state,path),finalize=trained):
        engine.main(cfg)


def qualify(cfg):
    # Engine preflight owns directory creation and the golden check; no training loop copy.
    train(dict(cfg,preflight_only=True))
    out=Path(cfg['output']);_,model,ema,opt=alg.setup(cfg);data=Current(cfg)
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
    engine.restore(initial,model,ema,opt,data)
    model.frozen_reference=alg.frozen(model);ema.frozen_reference=alg.frozen(model)
    x,y,_=data.labeled(0)
    def objective():
        l,u,_=terms(model,x,y);model.qualification_seg=float(l.detach());return l+u
    with torch.no_grad():first=float(objective());first_seg=model.qualification_seg
    calls=0
    def step():
        nonlocal calls
        opt.zero_grad(set_to_none=True);v=objective();v.backward();opt.step();alg.ema_update(ema,model);calls+=1
        return float(v.detach())
    for i in range(30):
        before=copy.deepcopy(engine.snapshot(model,ema,opt,data,i)) if i==15 else None
        value=step()
        if i==15:
            expected=copy.deepcopy(engine.snapshot(model,ema,opt,data,i+1));engine.restore(before,model,ema,opt,data)
            assert step()==value and exact(expected,engine.snapshot(model,ema,opt,data,i+1))
    with torch.no_grad():last=float(objective());last_seg=model.qualification_seg
    assert last<first-1e-4 and last_seg<first_seg-1e-4,(first,last,first_seg,last_seg)
    engine.restore(initial,model,ema,opt,data);opt.zero_grad(set_to_none=True)
    l,u,_=losses(cfg['arm'],model,data,0);(l+u).backward();opt.step();alg.ema_update(ema,model);calls+=1
    check_trainability(model,opt);finite(engine.snapshot(model,ema,opt,data,1));assert calls==32
    save(cfg,engine.snapshot(model,ema,opt,data,1),out/'engineering_terminal.pt')
    write(out/'ENGINEERING.json',dict(status='PASS',discarded=True,actual_optimizer_calls=calls,two_case_initial=first,two_case_final=last,seg_initial=first_seg,seg_final=last_seg,full_nonempty_restore_exact=True,frozen_state_exact=True,peak_GPU_bytes=torch.cuda.max_memory_allocated()))


def evaluate(cfg):
    install(cfg);model=alg.make(cfg)
    state=torch.load(Path(cfg['output'])/'final.pt',map_location='cpu',weights_only=False)
    assert state['step']==state['physical_updates']==2000
    model.load_state_dict(state['ema']);model.eval().requires_grad_(False)
    out=Path(cfg['output']);scores={}
    for domain in (('Siemens',) if cfg['stage']=='source' else ('Siemens','Philips')):
        scores[domain],_=engine.evaluate(model,cfg,domain,'val',out/('private_'+domain+'_val.jsonl'))
    write(out/'RESULTS.json',dict(status='COMPLETE',primary='fixed final EMA .999',stage=cfg['stage'],arm=cfg['arm'],metrics=scores,independent_test=False))
    write(out/'EVALUATION_EXIT.json',dict(exit_code=0))
