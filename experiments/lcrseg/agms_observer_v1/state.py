"""Native resume preserves heads/EMA/risk; deployment is separate."""
import copy
import random
from pathlib import Path
import torch
from .protocol import STUDY,digest,read,write
from .trainer import ObserverTrainer
from ..five_frameworks_v1 import checkpoint
from ..five_frameworks_v1.semantics import binding,tensor_fingerprint
from ..five_frameworks_v1.native_parent import build,NativeLRParent,lr
from ..five_frameworks_v1.model import Model


def metadata_path(path):return Path(str(path)+'.agms.json')


def save(t,path,identity):
    if type(t) is not ObserverTrainer or t.requires_restore:raise ValueError('uncommitted/non-AGMS trainer')
    for key,actual in dict(study_id=STUDY,arm=t.arm,family=t.model.family,seed=t.provider.seed,order=t.provider.order,stage=t.provider.stage).items():
        if identity.get(key)!=actual:raise ValueError('identity contradicts actual '+key)
    semantic=binding(t);agms=t.semantic_record()
    value=dict(semantic=semantic,semantic_sha256=digest(semantic),synthetic_only=False,identity=identity,
        student=t.model.state_dict(),ema=t.ema.state_dict(),optimizer=t.optimizer.state_dict(),
        scheduler=t.scheduler.state_dict(),scaler=t.scaler.state_dict(),prototypes=t.prototypes.values,
        support=t.prototypes.support,step=t.step,cursor=t.cursor,epoch=t.epoch,probe=t.probe,
        telemetry=t.telemetry,last=t.last,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
        provider_reads=[t.provider.l_reads,t.provider.u_reads],spectral=t.model.spectral,
        adapter_new=[a.new for a in t.model.parent.adapters],parent_projectors=None,
        cuda_rng=torch.cuda.get_rng_state_all() if next(t.model.parameters()).is_cuda else None,
        agms=agms,risk=t.risk,risk_updates=t.risk_updates,extra_cost=t.extra_cost,diagnostics=t.diagnostics,
        support_summary=t.support_summary,physical_optimizer_updates=t.physical_optimizer_updates)
    checkpoint.atomic_save(value,path)
    # Missing/mismatched sidecar after interruption is a stop, never implicit replay.
    write(metadata_path(path),dict(step=t.step,identity=identity,study_id=STUDY,execution_scope=t.execution.bindings['execution_scope'],agms_sha256=digest(agms),
        diagnostics_sha256=digest(t.diagnostics),cost_sha256=digest(t.extra_cost),support_sha256=digest(t.support_summary),risk_sha256=tensor_fingerprint({'risk':t.risk}),risk_updates=t.risk_updates))


def validate_payload(v,meta,identity,arm,options,provider):
    from .protocol import METHOD,GEOMETRY,ARMS
    agms=v['agms']
    if (v['identity']!=identity or meta['identity']!=identity or meta['step']!=v['step']
        or v['step']!=v['cursor'] or agms['study_id']!=STUDY or agms['arm']!=arm
        or agms['options']!=options or agms['prefix']!=provider.stage_source
        or agms['method']!=METHOD or agms['geometry']!=GEOMETRY or agms['arm_options']!=ARMS[arm]
        or digest(agms)!=meta['agms_sha256'] or digest(v['diagnostics'])!=meta['diagnostics_sha256']
        or digest(v['extra_cost'])!=meta['cost_sha256'] or digest(v['support_summary'])!=meta['support_sha256']):
        raise ValueError('study/arm/prefix/options/committed evidence mismatch')
    if tensor_fingerprint({'risk':v['risk']})!=meta['risk_sha256'] or v['risk_updates']!=meta['risk_updates']:raise ValueError('risk state mismatch')
    if v['risk'].shape!=(3,) or v['risk'].requires_grad or not torch.isfinite(v['risk']).all():raise ValueError('invalid risk state')
    if v['synthetic_only'] is not False:raise ValueError('native checkpoint required')


def resume(path,reference,device,provider,options,identity,permit,arm,node=None):
    from .authority import Capability
    from .protocol import canonical_plan
    from ..five_frameworks_v1.recipes import SyntheticCurrentDomain
    if not isinstance(permit,Capability):raise PermissionError('new study capability required')
    permit.validate();scope=permit.bindings['execution_scope']
    if scope in ('cpu_synthetic','synthetic'):
        if not isinstance(provider,SyntheticCurrentDomain):raise PermissionError('generated provider required')
    else:
        from ..five_frameworks_v1.native_data import NativeCurrentDomain
        if (scope!='formal' or node not in permit.bindings['nodes'] or options!=permit.bindings['options'][node['domain']]
            or arm!=node['arm'] or type(provider) is not NativeCurrentDomain
            or (provider.seed,provider.order,provider.stage)!=(node['seed'],node['order'],2)
            or provider.stage_source.get('prefix_binding_sha256')!=node['prefix_binding_sha256']):
            raise PermissionError('noncanonical restore request')
        from .execution import identity as actual_identity
        if identity!=actual_identity(node,{'execution_commit':permit.bindings['reviewed_code_commit']}):
            raise PermissionError('noncanonical runtime identity')
    meta=read(metadata_path(path))
    if meta['study_id']!=STUDY or meta['execution_scope']!=scope or meta['identity']!=identity:raise ValueError('checkpoint metadata scope mismatch')
    v=torch.load(path,map_location=device,weights_only=False)
    validate_payload(v,meta,identity,arm,options,provider)
    native=build(reference,device,provider.seed)
    parent=NativeLRParent(native,provider.seed,provider.stage_source,adapt=False)
    state=v['student']
    # Same native reconstruction as the shared resume helper, but construct the
    # correct trainer directly. Never change an existing object's Python class.
    for i,name in enumerate(lr.LAYERS):
        module=name[:-7];prefix='parent.native.'+module;conv=native.get_submodule(module)
        key=state['parent.V64_'+str(i)]
        tensors={'W0':state[prefix+'.weight'],'V':key,'G':state[prefix+'.A'].double()}
        layer=lr.LowRankConv(conv,tensors,'LR_SRC_A')
        par,_,child=module.rpartition('.');native.get_submodule(par)._modules[child]=layer
        parent.register_buffer('V64_'+str(i),key.clone())
    from .model import ObserverModel
    model=ObserverModel(parent,arm,provider.seed,provider.order,provider.stage,initialize_heads=False).to(device)
    model.load_state_dict(state)
    t=ObserverTrainer(model,provider,options,arm,permit,initialize=False,node=node)
    t.entry_fingerprint=v['semantic']['stage_entry_tensors']
    if t.semantic_record()!=v['agms']:raise ValueError('actual restored AGMS semantics differ')
    checkpoint.restore(t,path,identity)
    t.risk=v['risk'].to(device);t.risk_updates=v['risk_updates'];t.extra_cost=copy.deepcopy(v['extra_cost']);t.diagnostics=copy.deepcopy(v['diagnostics'])
    t.support_summary=copy.deepcopy(v['support_summary']);t.physical_optimizer_updates=v['physical_optimizer_updates']
    return t
