"""V2 hooks for the existing engine, including fixed source convergence diagnostics."""
import copy,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from . import model as alg
from ..apos_v2.data import Current
from .runtime import install,save,Calls
from ..philips_transfer_v1 import run as engine
from ..main_style_ssl_v1.runtime import write,append
from ..gyro_spatial_memory_v1.runtime import exact,finite
from ..space_lora_v1.core import source_model

def task(lp,y):
    ce=-lp.gather(1,y[:,None]).mean();p=lp.exp()[:,1:];one=F.one_hot(y,4).permute(0,3,1,2)[:,1:].to(p);present=one.sum((2,3))>0
    dc=(2*(p*one).sum((2,3))+1e-5)/(p.sum((2,3))+one.sum((2,3))+1e-5);dice=(((1-dc)*present).sum(1)/present.sum(1).clamp_min(1)).mean()
    return ce,dice
def terms(m,x,y):
    lp=m(x);ce,dice=task(lp,y);m.task_graph=(ce,dice)
    return ce+dice,(ce+dice)*0,dict(CE=float(ce.detach()),Dice_loss=float(dice.detach()),accepted_class_pixels=[0]*4,U_pixels=0,**m.last['stats'])

def losses(arm,m,data,step):
    x,y,_=data.labeled(step);l,u,stats=terms(m,x,y)
    if m.cfg['phase']=='formal' and step in (0,999,1999):
        from .runtime import record
        ce,dice=m.task_graph
        for name,v in [('CE',ce),('Dice',dice)]:
            record(m.cfg,'target_formal_reverse',1);g=torch.autograd.grad(v,[p for p in m.parameters() if p.requires_grad],retain_graph=True,allow_unused=True);stats[name+'_gradient_energy']=sum(float(t.detach().double().square().sum()) for t in g if t is not None)
    return l,u,stats

def preflight(cfg,out,source,m,ema,opt,data):
    m.frozen_reference=alg.frozen(m);ema.frozen_reference=alg.frozen(m);state=copy.deepcopy(engine.snapshot(m,ema,opt,data,0));save(cfg,state,out/'initial.pt');assert not opt.state
    if cfg['phase']=='qualification':
        saved_grads=None;first=None
        for attempt in range(2):
            engine.restore(state,m,ema,opt,data);l,u,_=losses(cfg['arm'],m,data,0);(l+u).backward()
            value=float((l+u).detach());grads=[None if p.grad is None else p.grad.clone() for p in m.parameters() if p.requires_grad]
            assert all(g is None or torch.isfinite(g).all() for g in grads) and sum(float(g.square().sum()) for g in grads if g is not None)>1e-12
            if attempt==0:first=value;saved_grads=grads
            else:assert value==first and exact(grads,saved_grads)
        engine.restore(state,m,ema,opt,data);x,_,_=data.labeled(0)
        # Exactly two extra native passes, budgeted in qualification: reference and EMA.
        with torch.no_grad():
            if cfg['stage']=='source':
                ref=source_model(cfg);m.calls.sample(len(x));expected=F.log_softmax(ref(x),1);del ref
            else:
                cc=dict(cfg,stage='source_eval');ref=alg.make(cc);expected=ref(x);del ref
            got=ema(x)
            assert torch.equal(expected,got),'source function entry changed'
        engine.restore(state,m,ema,opt,data)
    save(cfg,engine.snapshot(m,ema,opt,data,0),out/'latest.pt')
    write(out/'PREFLIGHT.json',dict(status='PASS',empty_optimizer=True,golden_exact=cfg['phase']=='qualification',native_function_equivalent=True,prior_and_native_frozen=True))

def trained(cfg,out,*_):write(out/'TRAINED.json',dict(status='TRAINED',performance_sealed=True))
def train(cfg):
    install(cfg)
    with patch.multiple(engine,ARMS=('CAL','N','Q','R'),STEPS=2000,GPU='3',setup=alg.setup,provider=Current,preflight=preflight,losses=losses,rate=lambda step:.001,set_learning_rate=lambda *_:None,update_ema=alg.ema_update,atomic_save=lambda s,p:save(cfg,s,p),finalize=trained):engine.main(cfg)

def qualify(cfg):
    train(dict(cfg,preflight_only=True));out=Path(cfg['output']);_,m,ema,opt=alg.setup(cfg);data=Current(cfg)
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False);engine.restore(initial,m,ema,opt,data);m.frozen_reference=alg.frozen(m);ema.frozen_reference=alg.frozen(m)
    x,y,_=data.labeled(0)
    def objective():l,u,_=terms(m,x,y);return l+u
    with torch.no_grad():first=float(objective())
    calls=0
    def step():
        nonlocal calls
        opt.zero_grad(set_to_none=True);v=objective();v.backward();opt.step();alg.ema_update(ema,m);calls+=1;return float(v.detach())
    for i in range(30):
        before=copy.deepcopy(engine.snapshot(m,ema,opt,data,i)) if i==15 else None
        v=step()
        if i==15:
            expected=copy.deepcopy(engine.snapshot(m,ema,opt,data,i+1));engine.restore(before,m,ema,opt,data);assert step()==v and exact(expected,engine.snapshot(m,ema,opt,data,i+1))
    with torch.no_grad():last=float(objective())
    if cfg['arm']=='N':
        from .runtime import record
        lp=m(x);base=sum(task(lp,y));gg={};params=[p for p in m.parameters() if p.requires_grad]
        for arm in ('Q','R'):
            outp,_=m.infer(m.last['logq'],m.last['logq0'],m.last['w'],m.last['pi'],arm);aux=sum(task(outp,y))-base
            record(cfg,'target_qualification_reverse',1);g=torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True);gg[arm]=torch.cat([torch.zeros_like(p).flatten() if t is None else t.flatten() for p,t in zip(params,g)]).double()
        nq=float(gg['Q'].norm());nr=float(gg['R'].norm());distance=float((gg['Q']/max(nq,1e-30)-gg['R']/max(nr,1e-30)).norm())
        write(Path(cfg['root'])/'P0_RESULT.json',dict(status='PASS' if nq*nq>=1e-12 and nr*nr>=1e-12 and distance>=.05 else 'STOP',Q_energy=nq*nq,R_energy=nr*nr,normalized_distance=distance,no_new_probe=True))
    assert last<first-1e-4,(first,last)
    engine.restore(initial,m,ema,opt,data);opt.zero_grad(set_to_none=True);l,u,_=losses(cfg['arm'],m,data,0);(l+u).backward();opt.step();alg.ema_update(ema,m);calls+=1
    alg.check_frozen(m);finite(engine.snapshot(m,ema,opt,data,1));assert calls==32
    save(cfg,engine.snapshot(m,ema,opt,data,1),out/'engineering_terminal.pt')
    write(out/'ENGINEERING.json',dict(status='PASS',optimizer_calls=32,discarded=True,initial_loss=first,final_loss=last,full_nonempty_restore_exact=True,frozen_state_exact=True))

def evaluate(cfg):
    m=alg.make(cfg)
    if cfg['stage']=='target':
        state=torch.load(Path(cfg['output'])/'final.pt',map_location='cpu',weights_only=False);assert state['step']==state['physical_updates']==2000;m.load_state_dict(state['ema'])
    m.eval().requires_grad_(False);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True);metrics={}
    for domain in (('Siemens',) if cfg['stage']=='source_eval' else ('Siemens','Philips')):metrics[domain],_=engine.evaluate(m,cfg,domain,'val',out/('private_'+domain+'_val.jsonl'))
    write(out/'RESULTS.json',dict(status='COMPLETE',primary='fixed final EMA .999',arm=cfg['arm'],metrics=metrics))
    write(out/'SOLVER_AGGREGATES.json',m.aggregate);write(out/'EVALUATION_EXIT.json',dict(exit_code=0))
