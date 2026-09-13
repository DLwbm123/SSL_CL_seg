"""Atomic synthetic checkpoints and append-only physical-update accounting."""
import json
import os
import random
from pathlib import Path
import torch


def atomic_save(value,path,fault=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    if fault:raise RuntimeError('injected during_checkpoint')
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    receipt=path.with_suffix(path.suffix+'.receipt.json')
    temp=receipt.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump({'committed':True,'bytes':path.stat().st_size,'step':value.get('step')},f)
        f.flush();os.fsync(f.fileno())
    os.replace(temp,receipt)


def save(trainer,path,identity,fault=False):
    value={'synthetic_only':True,'identity':identity,'student':trainer.model.state_dict(),
           'ema':trainer.ema.state_dict(),'optimizer':trainer.optimizer.state_dict(),
           'scheduler':trainer.scheduler.state_dict(),'scaler':trainer.scaler.state_dict(),
           'prototypes':trainer.prototypes.values,'support':trainer.prototypes.support,
           'step':trainer.step,'cursor':trainer.cursor,'epoch':trainer.epoch,
           'probe':trainer.probe,'telemetry':trainer.telemetry,'last':trainer.last,
           'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
           'provider_reads':[trainer.provider.l_reads,trainer.provider.u_reads],
           'stream_scheme':'stateless study/seed/order/stage/cursor/stream; arm excluded'}
    atomic_save(value,path,fault)


def restore(trainer,path,identity):
    # This API is deliberately synthetic-only. Real tensor reads require a new,
    # reviewed loader capability; naming a real file here is not authorization.
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value.get('synthetic_only') is not True or value['identity']!=identity:
        raise ValueError('checkpoint lineage/type mismatch')
    trainer.model.load_state_dict(value['student']);trainer.ema.load_state_dict(value['ema'])
    trainer.optimizer.load_state_dict(value['optimizer']);trainer.scheduler.load_state_dict(value['scheduler'])
    trainer.scaler.load_state_dict(value['scaler'])
    trainer.prototypes.values=value['prototypes'];trainer.prototypes.support=value['support']
    for name in ('step','cursor','epoch','probe','telemetry','last'):setattr(trainer,name,value[name])
    trainer.provider.l_reads,trainer.provider.u_reads=value['provider_reads']
    torch.set_rng_state(value['torch_rng']);random.setstate(value['python_rng'])
    return value


class PhysicalLedger:
    def __init__(self,path):self.path=Path(path)
    def append(self,step):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a') as f:
            f.write(json.dumps({'optimizer_executed':True,'scientific_step':step})+'\n')
            f.flush();os.fsync(f.fileno())
    def summary(self,committed_step):
        rows=[json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []
        return {'physical_optimizer_updates':len(rows),'committed_scientific_updates':committed_step,
                'physical_retry_or_uncommitted_updates':len(rows)-committed_step}


def require_predecessor(identity,predecessor):
    for key in ('family','candidate_id','seed','order'):
        if identity.get(key)!=predecessor.get(key):raise ValueError('cross-trajectory predecessor: '+key)
    if identity.get('stage')!=predecessor.get('stage',0)+1:raise ValueError('nonsequential stage')
