"""Atomic checkpoints; native access requires a sealed execution capability."""
import json
import os
import random
from pathlib import Path
import torch
from .semantics import binding
from .gate import digest
from .numerics import finite


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
    if trainer.requires_restore:raise RuntimeError('cannot checkpoint an uncommitted failed state')
    semantic=binding(trainer)
    for key,actual in {'family':trainer.model.family,'seed':trainer.provider.seed,'order':trainer.provider.order,'stage':trainer.provider.stage}.items():
        if key in identity and identity[key]!=actual:raise ValueError('identity contradicts actual '+key)
    value={'semantic':semantic,'semantic_sha256':digest(semantic),'synthetic_only':not getattr(trainer,'native',False),'identity':identity,'student':trainer.model.state_dict(),
           'ema':trainer.ema.state_dict(),'optimizer':trainer.optimizer.state_dict(),
           'scheduler':trainer.scheduler.state_dict(),'scaler':trainer.scaler.state_dict(),
           'prototypes':trainer.prototypes.values,'support':trainer.prototypes.support,
           'step':trainer.step,'cursor':trainer.cursor,'epoch':trainer.epoch,
           'probe':trainer.probe,'telemetry':trainer.telemetry,'last':trainer.last,
           'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
           'provider_reads':[trainer.provider.l_reads,trainer.provider.u_reads],
           'spectral':trainer.model.spectral,'adapter_new':[a.new for a in trainer.model.parent.adapters],
           'parent_projectors':None if getattr(trainer,'native',False) else [a.free_projector for a in trainer.model.parent.adapters],
           'cuda_rng':torch.cuda.get_rng_state_all() if getattr(trainer,'native',False) and torch.cuda.is_available() else None,
           'stream_scheme':'stateless study/seed/order/stage/cursor/stream; arm excluded'}
    atomic_save(value,path,fault)


def restore(trainer,path,identity):
    if getattr(trainer,"native",False):trainer.execution.validate()
    # Native resume validates its execution capability before tensor IO.
    # Synthetic callers still cannot load native checkpoint payloads.
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value.get('synthetic_only') is not (not getattr(trainer,'native',False)) or value['identity']!=identity:
        raise ValueError('checkpoint lineage/type mismatch')
    # Validate all semantic fields from constructed objects before mutating any state.
    actual=binding(trainer)
    if value.get('semantic_sha256')!=digest(value.get('semantic')) or digest(actual)!=value['semantic_sha256']:
        raise ValueError('resolved semantic/provider/source mismatch; state unchanged')
    for key in ('student','ema'):
        current=(trainer.model if key=='student' else trainer.ema).state_dict()
        if current.keys()!=value[key].keys() or any(current[n].shape!=value[key][n].shape or current[n].dtype!=value[key][n].dtype for n in current):
            raise ValueError('checkpoint tensor schema mismatch')
        finite(value[key],key)
    for key in ('optimizer','scheduler','scaler','prototypes','probe'):finite(value[key],key)
    if type(value['step']) is not int or value['step']<0 or value['cursor']!=value['step']:
        raise ValueError('checkpoint scientific cursor mismatch')
    trainer.model.load_state_dict(value['student']);trainer.ema.load_state_dict(value['ema'])
    trainer.optimizer.load_state_dict(value['optimizer']);trainer.scheduler.load_state_dict(value['scheduler'])
    trainer.scaler.load_state_dict(value['scaler'])
    trainer.prototypes.values=value['prototypes'];trainer.prototypes.support=value['support']
    for name in ('step','cursor','epoch','probe','telemetry','last'):setattr(trainer,name,value[name])
    trainer.model.spectral=value['spectral'];trainer.ema.spectral=value['spectral']
    if not getattr(trainer,'native',False):
        for a,b,new,projector in zip(trainer.model.parent.adapters,trainer.ema.parent.adapters,value['adapter_new'],value['parent_projectors']):
            a.new=b.new=new;a.free_projector=projector;b.free_projector=projector
    elif value.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([r.cpu() for r in value['cuda_rng']])
    trainer.requires_restore=False
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
