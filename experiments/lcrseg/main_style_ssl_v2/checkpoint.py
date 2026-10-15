"""Extend the existing atomic resume payload; old checkpoints cannot masquerade as V2."""
import copy
from pathlib import Path
import torch
from ..five_frameworks_v1 import checkpoint as base
from .memory import SiteMemory, finite_tree
from .protocol import VERSION
from .protection import bounded


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a, dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
    if isinstance(a, (list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return a == b


def save(trainer, path, identity, fault=False):
    path = Path(path)
    staging = path.with_suffix(path.suffix+'.v2_staging')
    base.save(trainer, staging, identity, fault)
    value = torch.load(staging, map_location='cpu', weights_only=False)
    value['v2'] = dict(version=VERSION, memory=trainer.memory.state(),
                      guard_totals=copy.deepcopy(trainer.guard_totals),
                      physical_optimizer_updates=trainer.physical_optimizer_updates)
    if not finite_tree(value['v2']):
        raise FloatingPointError('nonfinite V2 resume state')
    base.atomic_save(value, path)
    staging.unlink()
    staging.with_suffix(staging.suffix+'.receipt.json').unlink()


def restore(trainer, path, identity):
    value = torch.load(path, map_location='cpu', weights_only=False)
    extra = value.get('v2', {})
    if extra.get('version') != VERSION:
        raise ValueError('not a V2 resume checkpoint')
    candidate = SiteMemory(trainer.memory.classes, trainer.memory.width)
    candidate.load(extra['memory'])
    if [h['domain'] for h in candidate.history] != trainer.history_lineage:
        raise ValueError('historical memory lineage mismatch')
    if not finite_tree(extra) or set(extra['guard_totals']) != set(trainer.guard_totals):
        raise ValueError('invalid V2 guard state')
    if any(type(v) is not int or v<0 for v in extra['guard_totals'].values()):
        raise ValueError('invalid guard counters')
    if type(extra['physical_optimizer_updates']) is not int or extra['physical_optimizer_updates'] < value['step']:
        raise ValueError('physical/scientific counter mismatch')
    if not same_tree(candidate.history,trainer.memory.history):
        raise ValueError('immutable inherited memory changed')
    candidate.assert_coordinates(trainer.model.native_values())
    if trainer.model.sidecar is not None:
        for key in ('student','ema'):
            state=value[key]; q=state['sidecar.q']; g=torch.eye(len(q))+q@state['sidecar.r']@q.T
            if not bounded(g) or not bounded(state['sidecar.previous']@g):
                raise ValueError('unbounded checkpoint transform; state unchanged')
    base.restore(trainer, path, identity)
    trainer.memory = candidate
    trainer.guard_totals = copy.deepcopy(extra['guard_totals'])
    trainer.physical_optimizer_updates = extra['physical_optimizer_updates']
    trainer.draft = None; trainer.protect_grads = None; trainer.labeled_grads = None
    if trainer.model.sidecar is not None and (not trainer.model.sidecar.valid() or not trainer.ema.sidecar.valid()):
        trainer.requires_restore = True
        raise ValueError('unbounded checkpoint transform')
    return value
