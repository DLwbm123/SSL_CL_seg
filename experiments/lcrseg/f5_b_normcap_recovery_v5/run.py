"""Use the existing training loop and checkpoint restore without method changes."""
import os
from pathlib import Path
from ..f5_b_normcap_v5 import run as original
from ..five_frameworks_v1 import checkpoint
from . import protocol
engine=original.engine

def construct(config,j,stage,device,permit,previous=None,native=None,source_id=None,allow_u=True):
    resume=stage==1 and j['id'] in protocol.RESUME
    trainer=original.construct(config,j,stage,device,permit,previous,native,source_id,allow_u,initialize=not resume)
    if resume:
        prior=Path(config['interrupted_root']);cfg=protocol.read(prior/'CONFIG.private.json')
        if cfg['execution_commit']!=protocol.PRIOR_COMMIT or Path(cfg['code'],'CODE_COMMIT').read_text().strip()!=protocol.PRIOR_COMMIT:
            raise PermissionError('wrong frozen V5 source')
        source=prior/'jobs'/j['id']/'stage1';identity=dict(family='F5',seed=j['seed'],order=j['order'],stage=1,arm=j['arm'])
        calls=sum(1 for _ in (source/'physical.jsonl').open())
        if calls!=protocol.RESUME[j['id']] or (source/'SEALED.json').exists():raise ValueError('unexpected interrupted ledger')
        checkpoint.restore(trainer,source/'latest.pt',identity)
        if trainer.step!=1200 or trainer.cursor!=1200 or trainer.provider.l_reads<0:raise ValueError('wrong saved checkpoint cursor')
        trainer.recovery_receipt=dict(original_execution_commit=protocol.PRIOR_COMMIT,checkpoint_step=1200,inherited_physical_calls=calls,
                                      missing_gradient_diagnostic_steps=[int(trainer.options['total_steps']*p) for p in (.25,.5,.75,1.) if int(trainer.options['total_steps']*p)<=1200],
                                      earlier_diagnostics='NOT_PERSISTED_IN_CHECKPOINT; no recomputation')
    return trainer

def bind():
    for name in ('plan','jobs','CAPS','ARMS','compare'):setattr(engine,name,getattr(protocol,name))
    engine.construct=construct

def main():
    bind();config=protocol.read(os.environ['EXEC_CONFIG'])
    if os.environ.get('EXEC_MODE')=='qualify':
        from .tests import qualify
        qualify(config)
    else:engine.main()
