"""Fixed coordinate structure with pure versus composite objectives."""
import json,os,time
from pathlib import Path
from functools import partial
from unittest.mock import patch
from dataclasses import replace
import torch
from .config import Config
from .budgets import ResourceLedger,Counters
from .stream_state import StreamState
from .semantic_initialization import SemanticCurriculum
from .semantic_joint import SemanticJoint
from .local_inference import Refiner
from .predictive import Predictor,InferenceBudget
from .real_diagnostic import prepare_data,audit,write,evaluate
from .qualification import code_fingerprint,fingerprint
from .training import run_stage


def configuration(p,arm):
    if (p['purpose']!='REAL_COORDINATE_JOINT' or p['arms']!=['pure','mixed'] or
        p['steps']!=1600 or p['warm_steps']!=1200 or p['optimizer_reset_step']!=1200 or
        p['use_mcmc_teacher'] or p['permit_formal_promotion'] or p['domain']!='RUNMC'):
        raise PermissionError('coordinate joint protocol mismatch')
    if arm not in p['arms']:raise ValueError('unknown arm')
    return replace(Config(**p['config']),coordinate_channels=True)


def initial(cfg):
    s=StreamState(cfg)
    if cfg.coordinate_channels:
        base=StreamState(replace(cfg,coordinate_channels=False))
        with torch.no_grad():
            for name,param in s.phi.named_parameters():
                ref=dict(base.phi.named_parameters())[name]
                if name=='enc.weight':
                    param.zero_();param[:,:cfg.channels].copy_(ref)
                else:param.copy_(ref)
    return s


def objective(state,refiner,arm):
    if arm=='mixed':return SemanticJoint(state,refiner,transition=1200)
    return SemanticCurriculum(state,refiner,warm_steps=1600,boundary=1200,balanced_ce=True)


def ready(root,p,cfg):
    pre=json.loads((root/'PREFLIGHT02.json').read_text());r=json.loads((root/'READY.json').read_text())
    expected={a:configuration(p,a).to_dict() for a in p['arms']}
    if (r['state']!='REAL_DIAGNOSTIC_READY' or r['preflight_fingerprint']!=fingerprint(pre) or
        pre['code_fingerprint']!=code_fingerprint() or pre['protocol_fingerprint']!=fingerprint(p) or
        fingerprint(pre['arm_configs'])!=fingerprint(expected) or
        fingerprint(cfg.to_dict()) not in [fingerprint(v) for v in expected.values()]):
        raise PermissionError('coordinate admission mismatch')
    return pre


def preflight(root,p,ledger,counter):
    from . import revision_checks
    original_begin=StreamState.begin_task;original_tiny=revision_checks.tiny
    def gpu_begin(s,*args,**kwargs):original_begin(s,*args,**kwargs);s.to_device('cuda')
    checks=[]
    for arm in p['arms']:
        factory=partial(SemanticJoint,transition=2) if arm=='mixed' else partial(SemanticCurriculum,warm_steps=4,boundary=2,balanced_ce=True)
        with patch.object(StreamState,'begin_task',gpu_begin),patch.object(revision_checks,'tiny',partial(original_tiny,coordinate_channels=True)),patch.object(revision_checks,'run_stage',partial(run_stage,data_seed=p['data_seed'],checkpoint_stride=2,objective_factory=factory)):
            checks+=revision_checks.run_checks(ledger,f'resume_{arm}',['R04'])
    if not all(r['status']=='PASS' for r in checks):raise RuntimeError('coordinate resume failed')
    with ledger.attempt('preflight',seconds=300,updates=0):
        configs={};evidence={};timings={};costs={};parameters={}
        audited=audit(p,configuration(p,'pure'))
        for arm in p['arms']:
            cfg=configuration(p,arm);configs[arm]=cfg.to_dict();data=prepare_data(p,cfg,arm)
            evidence[arm]=data.evidence['fingerprint'];s=initial(cfg);s.begin_task(arm,data.manifest_hash,data.evidence);s.to_device('cuda')
            L=data.batch('L',1,torch.Generator().manual_seed(p['data_seed']));L['x']=L['x'].cuda();L['observations']={k:v.cuda() for k,v in L['observations'].items()}
            U=data.batch('U',1,torch.Generator().manual_seed(p['data_seed']+1));U['x']=U['x'].cuda()
            s.steps=1200
            torch.cuda.synchronize();start=time.monotonic();loss,_=objective(s,Refiner(s.model,s.phi,counter),arm).estimate(L,U,data.counts,s.generator);loss.backward();torch.cuda.synchronize();timings[arm]=time.monotonic()-start
            assert all(v.grad is None or torch.isfinite(v.grad).all() for g in s.optimizer.param_groups for v in g['params'])
            assert s.phi.enc.weight.grad.abs().sum()>0
            assert s.phi.enc.weight.grad[:,1:].abs().sum()>0
            parameters[arm]=sum(v.numel() for v in s.phi.parameters())
            s.finalize_task();costs[arm]={}
            for K in (0,3):
                torch.cuda.synchronize();start=time.monotonic();prediction=Predictor(counter).predict_image_only(L['x'],s,InferenceBudget(K=K));torch.cuda.synchronize()
                costs[arm][str(K)]=time.monotonic()-start;assert torch.isfinite(prediction['probability']).all()
        estimated=2800*timings['pure']+400*timings['mixed']+audited['validation_slices']*sum(sum(c.values()) for c in costs.values())
        result=dict(arm_configs=configs,evidence=evidence,code_fingerprint=code_fingerprint(),protocol_fingerprint=fingerprint(p),tests=checks,data_audit=audited,
            train_step_seconds=timings,inference_seconds=costs,estimated_compute_seconds=estimated,estimated_bytes=350000000,recognition_parameters=parameters,real_optimizer_updates=0)
        write(root/'PREFLIGHT02.json',result)
        if estimated*1.5+ledger.data['seconds_charged']>5400:raise RuntimeError('cost admission')
        write(root/'READY.json',dict(state='REAL_DIAGNOSTIC_READY',preflight_fingerprint=fingerprint(result),scientific_state='NOT_QUALIFIED'))


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['preflight','train','evaluate']);ap.add_argument('--arm',choices=['pure','mixed']);ap.add_argument('--K',type=int,choices=[0,3],default=3);args=ap.parse_args()
    root=Path(os.environ['RUN_ROOT'])/'results';p=json.loads((root.parent/'PRIVATE_PROTOCOL.json').read_text())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root,max_seconds=5400,max_updates=3400,max_bytes=1024**3);counter=Counters();start=time.time()
    try:
        if args.command=='preflight':preflight(root,p,ledger,counter)
        else:
            cfg=configuration(p,args.arm)
            if args.command=='train':
                pre=ready(root,p,cfg);data=prepare_data(p,cfg,args.arm)
                if data.evidence['fingerprint']!=pre['evidence'][args.arm]:raise PermissionError('evidence changed')
                s=initial(cfg);s.begin_task(args.arm,data.manifest_hash,data.evidence);s.to_device('cuda')
                run_stage(s,data,ledger,counter,args.arm,1600,2100,fingerprint(p),data_seed=p['data_seed'],checkpoint_stride=100,objective_factory=partial(objective,arm=args.arm))
            else:evaluate(root,p,cfg,ledger,counter,args.arm,args.K,readiness_validator=ready)
    finally:write(root/f'COUNTERS_{args.command}_{args.arm}_{time.time_ns()}.json',dict(seconds=time.time()-start,counters=counter.export()))

if __name__=='__main__':main()
