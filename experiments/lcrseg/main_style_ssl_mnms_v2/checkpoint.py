"""Extend atomic StageTrainer checkpoints; validate candidate state before mutation."""
import copy
from pathlib import Path
import torch
from ..five_frameworks_v1 import checkpoint as base
from ..main_style_ssl_v2.checkpoint import same_tree
from .memory import History, finite
from .protocol import VERSION, ORTHOGONAL_TOLERANCE
from .protection import require_rotation


def save(trainer,path,identity,fault=False):
    path=Path(path);staging=path.with_suffix('.mnms_staging')
    base.save(trainer,staging,identity,fault)
    value=torch.load(staging,map_location='cpu',weights_only=False)
    value['mnms_v2']=dict(version=VERSION,history=trainer.history.state(),
        precision_counts=trainer.precision_counts.clone(),guard_totals=copy.deepcopy(trainer.guard_totals),
        physical_optimizer_updates=trainer.physical_optimizer_updates)
    if not finite(value['mnms_v2']):raise ValueError('nonfinite candidate checkpoint')
    base.atomic_save(value,path)
    staging.unlink();staging.with_suffix(staging.suffix+'.receipt.json').unlink()


def restore(trainer,path,identity):
    value=torch.load(path,map_location='cpu',weights_only=False);extra=value.get('mnms_v2',{})
    if extra.get('version')!=VERSION:raise ValueError('not an M&Ms V2 checkpoint')
    history=History(extra['history'],trainer.history_lineage)
    if not same_tree(history.state(),trainer.history.state()):raise ValueError('immutable predecessor history changed')
    history.terms(trainer.model.native_values())
    counts=extra['precision_counts'];guards=extra['guard_totals']
    if not finite(extra) or counts.shape!=(2,4) or counts.dtype!=torch.float64 or (counts<1).any() or (counts[0]>=counts[1]).any():raise ValueError('invalid precision state')
    if set(guards)!=set(trainer.guard_totals) or any(type(v)is not int or v<0 for v in guards.values()) or guards['calls']!=value['step']:raise ValueError('invalid guard counters')
    if type(extra['physical_optimizer_updates'])is not int or extra['physical_optimizer_updates']!=value['step']:raise ValueError('uncommitted physical checkpoint')
    if not 0<=value['step']<=trainer.options['total_steps'] or value['cursor']!=value['step']:raise ValueError('checkpoint budget/cursor mismatch')
    if len(value['provider_reads'])!=2 or any(type(v)is not int or v<0 for v in value['provider_reads']):raise ValueError('invalid provider counters')
    if value['scheduler'].get('last_epoch')!=value['step']:raise ValueError('scheduler/step mismatch')
    candidate_optimizer=copy.deepcopy(trainer.optimizer)
    candidate_optimizer.load_state_dict(value['optimizer'])
    for parameter,state in candidate_optimizer.state.items():
        for name in ('exp_avg','exp_avg_sq'):
            if name in state and state[name].shape!=parameter.shape:raise ValueError('optimizer moment shape mismatch')
    del candidate_optimizer
    for key in ('student','ema'):
        s=value[key];q=s['sidecar.q'];r=s['sidecar.r'];previous=s['sidecar.previous']
        if not torch.allclose(q.T@q,torch.eye(q.shape[1]),atol=ORTHOGONAL_TOLERANCE,rtol=0):raise ValueError('invalid Q')
        eye=torch.eye(len(r),dtype=torch.float64);k=(r.double()-r.T.double())/2
        o=torch.linalg.solve(eye-k/2,eye+k/2);g=torch.eye(len(q),dtype=torch.float64)+q.double()@(o-eye)@q.T.double()
        require_rotation(g);require_rotation(previous);require_rotation(previous.double()@g)
        if not torch.equal(previous,trainer.model.sidecar.previous.detach().cpu()):raise ValueError('immutable predecessor transform changed')
        current=(trainer.model if key=='student' else trainer.ema).state_dict()
        for name in s:
            if name.startswith('entry_anchor.') or name.startswith('parent.V64_'):
                if not torch.equal(s[name],current[name].detach().cpu()):raise ValueError('immutable entry/projection changed')
    base.restore(trainer,path,identity)
    trainer.history=history;trainer.precision_counts=counts.clone();trainer.guard_totals=copy.deepcopy(guards)
    trainer.physical_optimizer_updates=extra['physical_optimizer_updates']
    trainer.protect_loss=None;trainer.entry_grads=None;trainer.labeled_grads=None;trainer.calibration=None
    return value
