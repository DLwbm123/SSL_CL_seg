import copy
from pathlib import Path
import torch
from ..f5_kl_output_u_v4 import tests as base
from ..f5_module_pilots_v1.tests import snapshot,equal
from ..five_frameworks_v1.native_runner import Counter
from ..five_frameworks_v1.native_operations import NativeOperations
from ..five_frameworks_v1 import checkpoint
from .cap import cap_group,NormCapTrainer
from .protocol import CAPS,OPTIONS,plan,write,read


def cpu_qualification(config):
    base.plan=plan;base.CAPS=CAPS
    return base.cpu_qualification(config)


def mathematical_checks(counter):
    for l,u in [(torch.tensor([1.,0.]),torch.tensor([-4.,8.])),(torch.zeros(2),torch.ones(2)),
                (torch.ones(2),torch.zeros(2)),(torch.tensor([1e-30,2e-30]),torch.tensor([3e-29,1e-29]))]:
        f,s=cap_group(l,u);assert f.double().norm()<=l.double().norm()*(1+1e-6)+1e-300
        assert float(l.double()@(l+f).double())>=-1e-6*float(l.double().norm())**2
    for value in [float('nan'),float('inf')]:
        try:cap_group(torch.ones(1),torch.tensor([value]))
        except FloatingPointError:pass
        else:raise AssertionError('nonfinite accepted')
    w=torch.nn.Parameter(torch.tensor([-.5,.8]));target=torch.tensor([1.,-1.]);opt=torch.optim.SGD([w],lr=.2)
    start=float((w-target).square().sum())
    for i in range(32):
        opt.zero_grad();l=torch.autograd.grad((w-target).square().sum(),w)[0];u=torch.autograd.grad(100*(w-target).square().sum(),w)[0]
        w.grad=l+cap_group(l,u)[0];counter.call(i+1);opt.step()
    assert float((w-target).square().sum())<start*1e-5


def qualify(config):
    root=Path(config['run_root']);base.plan=plan;base.CAPS=CAPS
    try:
        base.qualify(config)
        cpu=Counter(root/'qualification_cpu_physical.jsonl',CAPS['cpu_optimizer_updates']);mathematical_checks(cpu)
        write(root/'NORMCAP_CPU_QUALIFICATION.json',dict(status='PASS',additional_optimizer_calls=32,cumulative_cpu_calls=cpu.count,checks=['zero/tiny/nonfinite and norm bound','oversized-U two-case SGD convergence'],execution_commit=config['execution_commit']))
        cuda=Counter(root/'qualification_cuda_physical.jsonl',CAPS['synthetic_cuda_updates'])
        from .run import engine
        from ..f5_kl_output_u_v4.model import SelectiveUModel
        from ..f5_module_pilots_v1.tests import GeneratedDomain
        from ..five_frameworks_v1.native_parent import build,NativeLRParent
        permit=engine.admission(config,'qualification');device=torch.device('cuda:0')
        options={**OPTIONS,'total_steps':4,'warmup_fraction':.25,'U_ramp_fraction':.25,'PAS_confidence':0.,'PAS_cosine':-1.}
        def generated(enabled):
            p=GeneratedDomain(seed=163,size=384,device=device,stage_source=dict(kind='generated',seed=163,domain='REFUGE'))
            native=build(config['reference'],device,163)
            from ..f5_confirmation_v1.native_qualification import foreground_fixture
            foreground_fixture(native)
            m=SelectiveUModel(NativeLRParent(native,163,p.stage_source),'F5',ratio=options['rank_ratio']).to(device)
            return NormCapTrainer(m,p,options,execution=permit,cap_enabled=enabled)
        with NativeOperations(root/'cap_qualification_operations') as operations:
            a=generated(False);legacy=Path(config['v4_parity_code'])/'experiments/lcrseg/five_frameworks_v1/train_stage.py'
            ns=dict(__name__='experiments.lcrseg.five_frameworks_v1.v4_frozen_stage',__package__='experiments.lcrseg.five_frameworks_v1')
            exec(compile(legacy.read_text(),str(legacy),'exec'),ns)
            b=ns['StageTrainer'](copy.deepcopy(a.model),copy.deepcopy(a.provider),options,execution=permit)
            cuda.wrap(a.optimizer);cuda.wrap(b.optimizer)
            for _ in range(2):a.update();b.update();equal(snapshot(a),snapshot(b))
            del a,b
            t=generated(True);cuda.wrap(t.optimizer);t.update();grads=t.update()
            assert all(p.grad is None for p in t.ema.parameters())
            params=t.model.parent.parameter_groups()['output_factors'];forced={k:dict(v) for k,v in grads.items()}
            for p in params:
                if forced[id(p)]['U'] is not None:forced[id(p)]['U']=forced[id(p)]['U']*1e6
            t.transform_gradients(forced);assert t.module_stats['capped'] and t.module_stats['U_B_norm_after']<=t.module_stats['L_B_norm']*(1+1e-6)
            ident=dict(family='F5',seed=163,order=1,stage=1,arm='B_NORMCAP')
            checkpoint.save(t,root/'cap_qualification.pt',ident)
            other=NormCapTrainer.for_resume(copy.deepcopy(t.model),copy.deepcopy(t.provider),options,execution=permit)
            other.entry_fingerprint=t.entry_fingerprint;checkpoint.restore(other,root/'cap_qualification.pt',ident);cuda.wrap(other.optimizer)
            t.update();other.update();equal(snapshot(t),snapshot(other));assert t.telemetry==other.telemetry
            try:t.update(fault='after_optimizer')
            except RuntimeError as e:assert str(e)=='injected after_optimizer'
            else:raise AssertionError('expected failure')
            assert t.requires_restore and t.step==3
        assert cpu.count==64 and cuda.count==27
        q=read(root/'QUALIFICATION.json');q['checks']+=['norm bound, zero/tiny/nonfinite and oversized-U two-case optimization','disabled B cap matches archived V4','native cap slicing/teacher freeze/forced-cap branch','cap-enabled checkpoint continuation/counters and failed-call accounting']
        q['costs'].update(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,cap_operations=dict(operations.counts))
        write(root/'QUALIFICATION.json',q)
    except BaseException:
        failure=dict(status='FAIL',execution_commit=config['execution_commit'],costs={})
        for name,file in [('cpu_optimizer_updates','qualification_cpu_physical.jsonl'),('synthetic_cuda_updates','qualification_cuda_physical.jsonl'),('real_L_smoke_updates','smoke_physical.jsonl')]:
            p=root/file;failure['costs'][name]=sum(1 for _ in p.open()) if p.exists() else 0
        write(root/'QUALIFICATION.json',failure);raise
