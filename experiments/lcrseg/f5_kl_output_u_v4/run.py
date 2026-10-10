"""Bind the new study to the existing pilot orchestration and single training engine."""
import os
from ..f5_module_pilots_v1 import run as engine
from .model import SelectiveUModel,SelectiveUTrainer
from ..five_frameworks_v1.native_parent import NativeLRParent
from ..five_frameworks_v1.native_data import NativeCurrentDomain
from . import protocol


def construct(config,j,stage,device,permit,previous=None,native=None,source_id=None,allow_u=True):
    if native is None:
        native,sr=engine.source(config,j['seed'],device)
        source_id=dict(node_id=sr['node_id'],student_hash=sr['student_hash'],seed=j['seed'],domain='REFUGE')
    provider=NativeCurrentDomain(config['data'],j['seed'],j['order'],stage,source_id,device,permit,allow_u=allow_u)
    options={**protocol.OPTIONS,'total_steps':protocol.STEPS[provider.domain]}
    model=SelectiveUModel(NativeLRParent(native,j['seed'],source_id),'F5',ratio=options['rank_ratio'],previous=previous).to(device)
    return SelectiveUTrainer(model,provider,options,execution=permit)


def bind():
    # Process-local study binding; the archived V1 code and outputs remain immutable.
    for name in ('plan','jobs','CAPS','ARMS','compare'):setattr(engine,name,getattr(protocol,name))
    engine.construct=construct


def main():
    bind();config=protocol.read(os.environ['EXEC_CONFIG'])
    if os.environ.get('EXEC_MODE')=='qualify':
        from .tests import qualify
        qualify(config)
    elif os.environ.get('EXEC_MODE')=='cpu_qualify':
        from .tests import cpu_qualification
        engine.admission(config,'qualification');cpu_qualification(config)
    else:engine.main()
