"""Native resume reconstructs saved adapters without source SVD or F2 probes."""
import torch
from . import checkpoint
from .native_parent import NativeLRParent,build,lr
from .model import Model
from .train_stage import StageTrainer
from .gate import digest


def resume(path,reference,device,provider,options,cwmi,identity,permit):
    permit.validate()
    value=torch.load(path,map_location=device,weights_only=False)
    if value.get('synthetic_only') is not False or value['identity']!=identity:raise ValueError('native checkpoint lineage')
    if digest(value['semantic'])!=value['semantic_sha256']:raise ValueError('native checkpoint metadata')
    native=build(reference,device,provider.seed)
    parent=NativeLRParent(native,provider.seed,provider.stage_source,adapt=False)
    state=value['student']
    for i,name in enumerate(lr.LAYERS):
        path_name=name[:-7];prefix='parent.native.'+path_name
        conv=native.get_submodule(path_name)
        v=state['parent.V64_'+str(i)]
        tensors={'W0':state[prefix+'.weight'],'V':v,'G':state[prefix+'.A'].double()}
        module=lr.LowRankConv(conv,tensors,'LR_SRC_A')
        par,_,child=path_name.rpartition('.');native.get_submodule(par)._modules[child]=module
        parent.register_buffer('V64_'+str(i),v.clone())
    previous=state.get('sidecar.previous')
    model=Model.for_resume(parent,identity['family'],options.get('rank_ratio',.5),previous).to(device)
    model.load_state_dict(state)
    trainer=StageTrainer.for_resume(model,provider,options,cwmi,execution=permit)
    trainer.entry_fingerprint=value['semantic']['stage_entry_tensors']
    del value
    checkpoint.restore(trainer,path,identity)
    return trainer
