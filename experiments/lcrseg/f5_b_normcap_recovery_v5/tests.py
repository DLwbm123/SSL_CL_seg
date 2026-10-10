"""Four counted generated updates plus zero-update native restore admission."""
import copy
from pathlib import Path
import torch
from . import protocol
from .run import engine,construct
from ..f5_b_normcap_v5.cap import NormCapTrainer
from ..f5_kl_output_u_v4.model import SelectiveUModel
from ..f5_module_pilots_v1.tests import GeneratedDomain,snapshot,equal
from ..five_frameworks_v1.native_parent import build,NativeLRParent
from ..five_frameworks_v1.native_runner import Counter
from ..five_frameworks_v1 import checkpoint

def qualify(config):
    root=Path(config['run_root']);counter=Counter(root/'qualification_cuda_physical.jsonl',4)
    if (root/'QUALIFICATION.json').exists():raise FileExistsError('no qualification retry')
    try:
        torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);permit=engine.admission(config,'qualification')
        options={**protocol.OPTIONS,'total_steps':4,'warmup_fraction':.25,'U_ramp_fraction':.25,'PAS_confidence':0.,'PAS_cosine':-1.}
        p=GeneratedDomain(seed=163,size=384,device=device,stage_source=dict(kind='generated',seed=163,domain='REFUGE'))
        native=build(config['reference'],device,163)
        from ..f5_confirmation_v1.native_qualification import foreground_fixture
        foreground_fixture(native)
        m=SelectiveUModel(NativeLRParent(native,163,p.stage_source),'F5',ratio=options['rank_ratio']).to(device)
        t=NormCapTrainer(m,p,options,execution=permit);counter.wrap(t.optimizer);t.update();t.update()
        ident=dict(family='F5',seed=163,order=1,stage=1,arm='B_NORMCAP');checkpoint.save(t,root/'qualification.pt',ident)
        other=NormCapTrainer.for_resume(copy.deepcopy(t.model),copy.deepcopy(t.provider),options,execution=permit)
        other.entry_fingerprint=t.entry_fingerprint;checkpoint.restore(other,root/'qualification.pt',ident);counter.wrap(other.optimizer)
        t.update();other.update();equal(snapshot(t),snapshot(other));assert t.telemetry==other.telemetry
        assert all(p.grad is None for p in t.ema.parameters());assert counter.count==4
        del t,other,m,p,native
        restored=[]
        for j in protocol.jobs()[1:]:
            t=construct(config,j,1,device,permit);assert t.step==1200 and t.cursor==1200
            assert all(p.grad is None for p in t.ema.parameters())
            restored.append(dict(job=j['id'],step=t.step,inherited_physical=protocol.RESUME[j['id']],semantic_and_identity='PASS',new_optimizer_calls=0))
            del t
        protocol.write(root/'QUALIFICATION.json',dict(status='PASS',execution_commit=config['execution_commit'],plan_id=protocol.plan()['plan_id'],
                       costs=dict(cpu_optimizer_updates=0,synthetic_cuda_updates=counter.count,real_L_smoke_updates=0),native_restores=restored,
                       checks=['current-runtime generated checkpoint continuation incl optimizer/RNG/counters','teacher frozen','all three archived native checkpoints admit unchanged semantics/identity and cursor1200; no current patient sample reads']))
    except BaseException as e:
        protocol.write(root/'QUALIFICATION.json',dict(status='FAIL',execution_commit=config['execution_commit'],costs=dict(cpu_optimizer_updates=0,synthetic_cuda_updates=counter.count,real_L_smoke_updates=0)))
        protocol.write(root/'QUALIFICATION_FAILURE.private.json',dict(error=repr(e),traceback=__import__('traceback').format_exc()));raise
