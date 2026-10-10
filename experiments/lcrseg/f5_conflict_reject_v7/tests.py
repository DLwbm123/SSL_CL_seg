import copy
from pathlib import Path
import torch
from . import protocol
from .run import engine,GatedTrainer,construct
from .gate import reject
from ..f5_module_pilots_v1.tests import GeneratedDomain,snapshot,equal
from ..f5_module_pilots_v1.modules import make_model
from ..five_frameworks_v1.native_parent import build,NativeLRParent
from ..five_frameworks_v1.recipes import generator
from ..five_frameworks_v1.native_runner import Counter
from ..five_frameworks_v1 import checkpoint

def qualify(config):
    root=Path(config['run_root']);cpu=Counter(root/'qualification_cpu_physical.jsonl',96);cuda=Counter(root/'qualification_cuda_physical.jsonl',32);smoke=Counter(root/'smoke_physical.jsonl',4)
    if (root/'QUALIFICATION.json').exists():raise FileExistsError('no qualification retry')
    try:
        torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);permit=engine.admission(config,'qualification')
        w=torch.nn.Parameter(torch.tensor([-.5,.8]));target=torch.tensor([1.,-1.]);opt=torch.optim.SGD([w],lr=.1);start=float((w-target).square().sum())
        for i in range(32):
            opt.zero_grad();l=(w-target).square().sum();u=-5*(w-target).square().sum();gl=torch.autograd.grad(l,w,retain_graph=True)[0];gu=torch.autograd.grad(u,w)[0];filtered,_=reject(gl,gu);w.grad=gl+filtered;cpu.call(i+1);opt.step()
        assert float((w-target).square().sum())<start*1e-5
        options={**protocol.OPTIONS,'total_steps':4,'warmup_fraction':.25,'U_ramp_fraction':.25,'PAS_confidence':0.,'PAS_cosine':-1.}
        def generated(enabled):
            p=GeneratedDomain(seed=163,size=384,device=device,stage_source=dict(kind='generated',seed=163,domain='REFUGE'));native=build(config['reference'],device,163)
            from ..f5_confirmation_v1.native_qualification import foreground_fixture
            foreground_fixture(native)
            o=dict(options);m=make_model(NativeLRParent(native,163,p.stage_source),'F5',o,None,generator(163,1,1,0,'pilot_adapter_initialization')).to(device)
            return GatedTrainer(m,p,o,execution=permit,gate_enabled=enabled)
        a=generated(False);legacy=Path(config['parity_code'])/'experiments/lcrseg/five_frameworks_v1/train_stage.py';ns=dict(__name__='experiments.lcrseg.five_frameworks_v1.v1_frozen_stage',__package__='experiments.lcrseg.five_frameworks_v1');exec(compile(legacy.read_text(),str(legacy),'exec'),ns)
        b=ns['StageTrainer'](copy.deepcopy(a.model),copy.deepcopy(a.provider),a.options,execution=permit);cuda.wrap(a.optimizer);cuda.wrap(b.optimizer)
        for _ in range(2):a.update();b.update();equal(snapshot(a),snapshot(b))
        del a,b
        t=generated(True);cuda.wrap(t.optimizer);t.update();grads=t.update();assert t.telemetry['R_nonzero_U_calls']>0
        assert all(p.grad is None for p in t.ema.parameters())
        gl=torch.tensor([1.,0.]);gu=torch.tensor([-1.,2.]);z,st=reject(gl,gu);assert st['blocked'] and torch.equal(z,torch.zeros_like(gu))
        for v in (torch.tensor([1.,2.]),torch.tensor([0.,2.]),torch.zeros(2)):
            z,st=reject(gl,v);assert not st['blocked'] and torch.equal(z,v)
        z,st=reject(torch.zeros(2),gu);assert not st['blocked'] and torch.equal(z,gu)
        try:reject(gl,torch.tensor([float('nan'),0.]))
        except FloatingPointError:pass
        else:raise AssertionError('nonfinite rejection input accepted')
        try:reject(torch.ones(3),torch.ones(2))
        except ValueError:pass
        else:raise AssertionError('shape mismatch accepted')
        try:reject(torch.full((2,),1e308,dtype=torch.float64),torch.full((2,),1e308,dtype=torch.float64))
        except FloatingPointError:pass
        else:raise AssertionError('nonfinite dot accepted')
        p=t.model.sidecar.r;gl=grads[id(p)]['L'];gu=grads[id(p)]['U'];filtered,_=reject(gl,gu);assert torch.allclose(p.grad,gl+filtered,atol=1e-7,rtol=1e-5)
        assert t.telemetry['R_gate_calls']>0
        ident=dict(family='F5',seed=163,order=1,stage=1,arm='U_REJECT');checkpoint.save(t,root/'qualification.pt',ident)
        other=GatedTrainer.for_resume(copy.deepcopy(t.model),copy.deepcopy(t.provider),options,execution=permit);other.entry_fingerprint=t.entry_fingerprint;checkpoint.restore(other,root/'qualification.pt',ident);cuda.wrap(other.optimizer);t.update();other.update();equal(snapshot(t),snapshot(other));assert t.telemetry==other.telemetry;del other
        with torch.no_grad():before=t.model(torch.zeros(2,3,384,384,device=device));deployed=t.model.deploy();after=deployed(torch.zeros(2,3,384,384,device=device))
        assert torch.allclose(before,after,atol=2e-5,rtol=2e-5)
        p=GeneratedDomain(seed=163,stage=2,size=384,device=device,stage_source=dict(kind='own_generated_predecessor',seed=163,domain='RIM_ONE_r3'));m=make_model(NativeLRParent(deployed.parent.native,163,p.stage_source),'F5',options,deployed.transform.detach().clone(),generator(163,1,2,0,'pilot_adapter_initialization')).to(device)
        nxt=GatedTrainer(m,p,options,execution=permit);cuda.wrap(nxt.optimizer);nxt.update();del nxt,m,p,deployed
        try:t.update(fault='after_optimizer')
        except RuntimeError as e:assert str(e)=='injected after_optimizer'
        else:raise AssertionError('expected injected failure')
        assert t.requires_restore and t.step==3;del t
        j=protocol.jobs()[0];t=construct(config,j,1,device,engine.admission(config,'smoke'),allow_u=False);smoke.wrap(t.optimizer);t.update();t.update();assert t.provider.u_reads==0;del t
        assert (cpu.count,cuda.count,smoke.count)==(32,10,2)
        protocol.write(root/'QUALIFICATION.json',dict(status='PASS',execution_commit=config['execution_commit'],plan_id=protocol.plan()['plan_id'],costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count),checks=['two-case conflict-rejection CPU optimization','disabled gate matches archived F5 state','conflict/aligned/orthogonal/zero/nonfinite vectors and native merged R gradient; A/B U forbidden','teacher freeze and checkpoint continuation incl telemetry','deployment/own-prefix stage transition','deliberate failed optimizer call accounted','discarded current-L smoke with zero U reads'],matched_control_reuse_qualified=True,historical_B2_reuse='unchanged previous parity retained; historical only'))
    except BaseException as e:
        protocol.write(root/'QUALIFICATION.json',dict(status='FAIL',execution_commit=config['execution_commit'],costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count)));protocol.write(root/'QUALIFICATION_FAILURE.private.json',dict(error=repr(e),traceback=__import__('traceback').format_exc()));raise
