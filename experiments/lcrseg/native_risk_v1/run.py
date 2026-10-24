"""Two-view acquisition/anatomy hooks, reusing the original single engine."""
import copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from . import model as alg
from .appearance import transform
from .data import Current
from .runtime import install,save,record
from ..philips_transfer_v1 import run as engine
from ..main_style_ssl_v1.runtime import write,append
from ..gyro_spatial_memory_v1.runtime import exact,finite
from ..space_lora_v1.core import source_model

def task(lp,y):
    ce=-lp.gather(1,y[:,None]).mean();one=F.one_hot(y,4).permute(0,3,1,2)[:,1:].to(lp);p=lp.exp()[:,1:];present=one.sum((2,3))>0;dc=(2*(p*one).sum((2,3))+1e-5)/(p.sum((2,3))+one.sum((2,3))+1e-5);dice=(((1-dc)*present).sum(1)/present.sum(1).clamp_min(1)).mean();return ce+dice,ce,dice

def js(p,q):
    mid=(p+q)*.5
    return .5*((p*(p.clamp_min(1e-8).log()-mid.clamp_min(1e-8).log())).sum(1)+(q*(q.clamp_min(1e-8).log()-mid.clamp_min(1e-8).log())).sum(1))/np.log(2)

def terms(m,x,y,keys,cache,step=0,diagnostic=False):
    from .risk import penalty
    aug=transform(x,step,m.stage,diagnostic,m.calls);lp=m(torch.cat((x,aug)));n=len(x);base,ce,dice=task(lp,torch.cat((y,y)));probs=lp.exp();one=F.one_hot(y,4).permute(0,3,1,2).to(lp);anatomy=base*0
    for r in (32,64):
        if m.calls:m.calls.extra('anatomy_samples',len(lp))
        q=F.avg_pool2d(probs,384//r);gt=F.avg_pool2d(torch.cat((one,one)),384//r);anatomy=anatomy+.5*js(q,gt).mean()
    if m.calls:m.calls.extra('equivariance_samples',n)
    equiv=js(probs[:n],probs[n:]).mean();reg,violation=penalty(m,lp[:n],y,keys,cache)
    full=m.stage=='source' or m.arm=='C';aux=(anatomy+equiv+reg)*(1 if full else 0);m.task_graph=(base,anatomy,equiv,reg);audit=base+anatomy+equiv+violation
    assert torch.isfinite(base+aux)
    return base,aux,dict(CE=float(ce.detach()),Dice_loss=float(dice.detach()),task_loss=float(base.detach()),anatomy_loss=float(anatomy.detach()),equivariance_loss=float(equiv.detach()),risk_lagrangian=float(reg.detach()),positive_risk_residual=float(violation.detach()),audit_objective=float(audit.detach()),total_loss=float((base+aux).detach()),dual_max=float(m.dual.max()) if m.dual.numel() else 0,accepted_class_pixels=[0]*4,U_pixels=0)

def losses(arm,m,data,step):
    object.__setattr__(m,'current_data',data);x,y,keys=data.labeled(step);l,u,stats=terms(m,x,y,keys,data.cache,step)
    if m.cfg['phase']=='formal' and step in (0,999,1999):
        for name,v in zip(('task','anatomy','equivariance','risk'),m.task_graph):
            record(m.cfg,m.cfg['bucket']+'_reverse',1);g=torch.autograd.grad(v,[p for p in m.parameters() if p.requires_grad],retain_graph=True,allow_unused=True);stats[name+'_gradient_energy']=sum(float(t.detach().double().square().sum()) for t in g if t is not None)
    return l,u,stats

def preflight(cfg,out,source,m,ema,opt,data):
    m.frozen_reference=alg.frozen(m);ema.frozen_reference=alg.frozen(m);state=copy.deepcopy(engine.snapshot(m,ema,opt,data,0));save(cfg,state,out/'initial.pt');assert not opt.state
    if cfg['phase']=='qualification':
        first=None;saved=None
        for attempt in range(2):
            engine.restore(state,m,ema,opt,data);l,u,_=losses(cfg['arm'],m,data,0);(l+u).backward();value=float((l+u).detach());grads=[None if p.grad is None else p.grad.clone() for p in m.parameters() if p.requires_grad];assert sum(float(g.square().sum()) for g in grads if g is not None)>1e-12
            if attempt==0:first=value;saved=grads
            else:assert first==value and exact(saved,grads)
        engine.restore(state,m,ema,opt,data);x,_,_=data.labeled(0);mode=m.training;m.eval()
        with torch.no_grad():
            if cfg['stage']=='source':
                ref=source_model(cfg);m.calls.sample(len(x));expected=F.log_softmax(ref(x),1);del ref
            else:
                ref=alg.make(dict(cfg,stage='target',load='initial' if cfg['arm']=='B' else 'source'));expected=ref(x);del ref
            got=ema(x);assert torch.equal(expected,got),'native entry function changed'
        m.train(mode);engine.restore(state,m,ema,opt,data)
    save(cfg,engine.snapshot(m,ema,opt,data,0),out/'latest.pt');write(out/'PREFLIGHT.json',dict(status='PASS',empty_optimizer=True,golden_exact=cfg['phase']=='qualification',native_entry_equal=True,frozen_head=True))

def trained(cfg,out,*_):write(out/'TRAINED.json',dict(status='TRAINED',performance_sealed=True))
def train(cfg):
    install(cfg)
    with patch.multiple(engine,ARMS=('B','F','S','C'),STEPS=2000,GPU='3',setup=alg.setup,provider=Current,preflight=preflight,losses=losses,rate=lambda step:.001,set_learning_rate=lambda *_:None,update_ema=alg.ema_update,atomic_save=lambda s,p:save(cfg,s,p),finalize=trained):engine.main(cfg)

def diagnostic(m,ema,data,step):
    old=m.calls.bucket;olde=ema.calls.bucket;m.calls.bucket=ema.calls.bucket='source_diagnostic';mode=m.training;m.eval();rr={'student':[],'EMA':[]}
    with torch.no_grad():
        for j in range(8):
            x,y,keys=data.labeled(200000+j,diagnostic=True)
            for name,model in [('student',m),('EMA',ema)]:_,_,stats=terms(model,x,y,keys,data.cache,200000+j,True);rr[name].append(stats)
    out=dict(step=step,batch_reads=8,native_samples=64)
    for name,v in rr.items():out[name]={k:float(np.mean([a[k] for a in v])) for k in ('task_loss','anatomy_loss','equivariance_loss','risk_lagrangian','positive_risk_residual','audit_objective','total_loss')}
    append(Path(m.cfg['root'])/('SOURCE_DIAGNOSTICS_'+m.arm+'.jsonl'),out);m.calls.bucket=old;ema.calls.bucket=olde;m.train(mode);ema.eval()

def qualify(cfg):
    train(dict(cfg,preflight_only=True));out=Path(cfg['output']);_,m,ema,opt=alg.setup(cfg);data=Current(cfg);initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False);engine.restore(initial,m,ema,opt,data);m.frozen_reference=alg.frozen(m);ema.frozen_reference=alg.frozen(m);x,y,keys=data.labeled(0)
    def objective():l,u,_=terms(m,x,y,keys,data.cache);return l+u
    with torch.no_grad():first=terms(m,x,y,keys,data.cache)[2]['audit_objective']
    calls=0
    def step():
        nonlocal calls
        opt.zero_grad(set_to_none=True);v=objective();v.backward();opt.step();alg.ema_update(ema,m);calls+=1;state=engine.snapshot(m,ema,opt,data,int(m.local_updates));state['qualification_physical_calls']=calls;save(cfg,state,out/'qualification_latest.pt');return float(v.detach())
    for i in range(30):
        before=copy.deepcopy(engine.snapshot(m,ema,opt,data,i)) if i==15 else None;v=step()
        if i==15:
            expected=copy.deepcopy(engine.snapshot(m,ema,opt,data,i+1));engine.restore(before,m,ema,opt,data);assert step()==v and exact(expected,engine.snapshot(m,ema,opt,data,i+1))
    with torch.no_grad():last=terms(m,x,y,keys,data.cache)[2]['audit_objective']
    assert last<first-1e-4,(first,last)
    engine.restore(initial,m,ema,opt,data);opt.zero_grad(set_to_none=True);l,u,_=losses(cfg['arm'],m,data,0);(l+u).backward();opt.step();alg.ema_update(ema,m);calls+=1;alg.check_frozen(m);assert calls==32
    state=engine.snapshot(m,ema,opt,data,1);finite(state);save(cfg,state,out/'engineering_terminal.pt');write(out/'ENGINEERING.json',dict(status='PASS',optimizer_calls=32,discarded=True,initial_loss=first,final_loss=last,full_nonempty_restore_exact=True,frozen_state_exact=True))

def evaluate(cfg):
    m=alg.make(cfg)
    assert cfg['load']==('source' if cfg['stage']=='source_eval' else 'target')
    m.eval().requires_grad_(False);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True);metrics={}
    for domain in (('Siemens',) if cfg['stage']=='source_eval' else ('Siemens','Philips')):metrics[domain],_=engine.evaluate(m,cfg,domain,'val',out/('private_'+domain+'_val.jsonl'))
    write(out/'RESULTS.json',dict(status='COMPLETE',primary='fixed final EMA .999',arm=cfg['arm'],metrics=metrics));write(out/'EVALUATION_EXIT.json',dict(exit_code=0))
