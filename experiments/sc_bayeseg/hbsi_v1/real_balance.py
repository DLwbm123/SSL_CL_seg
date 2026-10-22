"""RUNMC balanced semantic training diagnostic: no formal learning admission."""
import io
import json
import os
from functools import partial
from pathlib import Path
import time
from unittest.mock import patch
import torch
from .budgets import Counters, ResourceLedger
from .config import Config
from .local_inference import Refiner
from .qualification import code_fingerprint, fingerprint
from .real_diagnostic import audit, prepare_data, write, require_ready, evaluate
from .semantic_initialization import SemanticCurriculum
from .stream_state import StreamState
from .training import run_stage


def validate(p):
    fixed = dict(purpose='REAL_BALANCED_SEMANTICS', dataset='prostate', domain='RUNMC',
        arms=['plain','balanced'], steps=1200, warm_steps=1200, optimizer_reset_step=800,
        use_mcmc_teacher=False, permit_formal_promotion=False, device='cuda',
        data_seed=163000, checkpoint_stride=100, training_seconds_per_arm=1200,
        evaluation_seconds_per_cell=300, batch_size=1)
    if any(p.get(k)!=v for k,v in fixed.items()):
        raise PermissionError('semantic protocol mismatch')
    cfg=Config(**p['config'])
    if (cfg.height,cfg.width,cfg.classes,cfg.inner_steps,cfg.local_parameterization)!=(128,128,2,3,'diagonal'):
        raise PermissionError('semantic model configuration mismatch')
    return cfg


def factory(p,arm):
    return partial(SemanticCurriculum,warm_steps=p['warm_steps'], balanced_ce=arm=='balanced',
                   boundary=p['optimizer_reset_step'])


def preflight(root,p,cfg,ledger,counter):
    from . import revision_checks
    original_begin=StreamState.begin_task
    def gpu_begin(s,*args,**kwargs):
        original_begin(s,*args,**kwargs);s.to_device('cuda')
    checks=[]
    # Resume precisely at the curriculum boundary, including original prior,
    # optimizer reset, model RNG and sparse checkpoint retention.
    for warm in (0,2):
        with patch.object(StreamState,'begin_task',gpu_begin), patch.object(revision_checks,'run_stage',
                partial(run_stage,data_seed=p['data_seed'],checkpoint_stride=2,
                        objective_factory=partial(SemanticCurriculum,warm_steps=2,boundary=2,balanced_ce=bool(warm)))):
            checks+=revision_checks.run_checks(ledger,f'resume_warm{warm}',['R04'])
    if not all(c['status']=='PASS' for c in checks):
        raise RuntimeError('curriculum resume failed')
    with ledger.attempt('preflight03',seconds=300,updates=0) as out:
        audited=audit(p,cfg);evidence={};timings={};checkpoint_bytes=0
        for arm in p['arms']:
            data=prepare_data(p,cfg,arm);evidence[arm]=data.evidence['fingerprint']
            s=StreamState(cfg);s.begin_task(arm,data.manifest_hash,data.evidence);s.to_device('cuda')
            L=data.batch('L',1,torch.Generator().manual_seed(p['data_seed']))
            U=data.batch('U',1,torch.Generator().manual_seed(p['data_seed']+1))
            for b in (L,U):
                b['x']=b['x'].cuda()
                if 'observations' in b:b['observations']={k:v.cuda() for k,v in b['observations'].items()}
            torch.cuda.synchronize();start=time.monotonic()
            loss,parts=factory(p,arm)(s,Refiner(s.model,s.phi,counter)).estimate(L,U,data.counts,s.generator)
            loss.backward();torch.cuda.synchronize();timings[arm]=time.monotonic()-start
            assert all(v.grad is None or torch.isfinite(v.grad).all() for g in s.optimizer.param_groups for v in g['params'])
            assert s.q.mean.grad is not None and s.phi.enc.weight.grad.abs().sum()>0
            if arm in p['arms']:
                assert s.q.raw_scale.grad is None
                assert all(v.grad is None for v in s.current.parameters())
                assert all(v.grad is None for h in (s.phi.s_head,s.phi.b_head) for v in h.parameters())
                assert all(v.abs().sum()==0 for k,v in s.q.unpack(s.q.mean.grad).items() if k.startswith('f.'))
            buffer=io.BytesIO();torch.save(s.payload(),buffer)
            moments=sum(2*v.numel()*v.element_size() for g in s.optimizer.param_groups for v in g['params'])
            checkpoint_bytes=max(checkpoint_bytes,buffer.tell()+moments+32768)
            write(out/f'{arm}_BACKWARD.json',dict(seconds=timings[arm],parts=parts,optimizer_updates=0))
        # Verified v1 prediction timings, same untouched predictor/config/hardware;
        # conservative 1.5 margin applied to the whole estimate below.
        old=json.loads(Path(p['prediction_cost_receipt']).read_text())
        assert fingerprint(old['config'])==fingerprint(cfg.to_dict())
        eval_seconds=audited['validation_slices']*2*sum(old['inference_slice_seconds'].values())
        estimated=1200*sum(timings.values())+eval_seconds
        size=32*checkpoint_bytes+4*audited['validation_slices']*3*2*128*128*8
        result=dict(config=cfg.to_dict(),protocol_fingerprint=fingerprint(p),code_fingerprint=code_fingerprint(),
            evidence=evidence,data_audit=audited,tests=checks,train_step_seconds=timings,
            estimated_compute_seconds=estimated,estimated_bytes=size,real_optimizer_updates=0,
            prediction_cost_source='v1 identical configuration / same GPU',scientific_state='NOT_QUALIFIED')
        write(root/'PREFLIGHT02.json',result)
        if estimated*1.5+ledger.data['seconds_charged']>3600 or size*1.25>0.9*1024**3:
            raise RuntimeError('cost admission failed')
        write(root/'READY.json',dict(state='REAL_DIAGNOSTIC_READY',preflight_fingerprint=fingerprint(result),
                                  scientific_state='NOT_QUALIFIED'))


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','train','evaluate'])
    parser.add_argument('--arm',choices=['plain','balanced']);parser.add_argument('--K',type=int,choices=[0,3],default=3)
    args=parser.parse_args();root=Path(os.environ['RUN_ROOT'])/'results'
    p=json.loads((root.parent/'PRIVATE_PROTOCOL.json').read_text());cfg=validate(p)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    counter=Counters();ledger=ResourceLedger(root,max_seconds=3600,max_updates=2600,max_bytes=1024**3)
    start=time.time()
    try:
        if args.command=='preflight':preflight(root,p,cfg,ledger,counter)
        elif args.command=='train':
            pre=require_ready(root,p,cfg);data=prepare_data(p,cfg,args.arm)
            if data.evidence['fingerprint']!=pre['evidence'][args.arm]:raise PermissionError('changed evidence')
            s=StreamState(cfg);s.begin_task(args.arm,data.manifest_hash,data.evidence);s.to_device('cuda')
            run_stage(s,data,ledger,counter,args.arm,p['steps'],p['training_seconds_per_arm'],fingerprint(p),
                      data_seed=p['data_seed'],checkpoint_stride=p['checkpoint_stride'],objective_factory=factory(p,args.arm))
        else:evaluate(root,p,cfg,ledger,counter,args.arm,args.K)
    finally:
        write(root/f'COUNTERS_{args.command}_{args.arm}_{args.K}_{time.time_ns()}.json',
              dict(seconds=time.time()-start,counters=counter.export()))


if __name__=='__main__':main()
