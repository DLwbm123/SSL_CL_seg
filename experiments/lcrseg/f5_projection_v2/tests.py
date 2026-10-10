import copy
from pathlib import Path
import torch
from .projection import project,ProjectionTrainer
from .protocol import CAPS,OPTIONS,write,read,plan
from ..f5_module_pilots_v1.tests import GeneratedDomain,snapshot,equal
from ..f5_module_pilots_v1.modules import make_model
from ..five_frameworks_v1.native_parent import build,NativeLRParent
from ..five_frameworks_v1.native_runner import Counter
from ..five_frameworks_v1.native_operations import NativeOperations
from ..five_frameworks_v1.recipes import generator
from ..five_frameworks_v1 import checkpoint


def mathematical_checks(counter):
    for l,u in [(torch.tensor([1.,0.]),torch.tensor([-1.,2.])),
                (torch.tensor([1.,2.]),torch.tensor([3.,4.])),
                (torch.zeros(2),torch.tensor([-2.,1.])),
                (torch.tensor([1e-8,2e-8]),torch.tensor([-3e-8,1e-8]))]:
        f,s=project(l,u)
        if s['projected']:
            assert float(l.double()@f.double())>=-1e-6*float(l.norm()*u.norm())
            assert f.norm()<=u.norm()+1e-6
        else:assert torch.equal(f,u)
    w=torch.nn.Parameter(torch.tensor([-.5,-.8]));opt=torch.optim.SGD([w],lr=.2)
    start=float((w-1).square().mean())
    for i in range(32):
        opt.zero_grad();l=torch.autograd.grad((w-1).square().sum(),w)[0]
        u=torch.autograd.grad((w+3).square().sum(),w)[0]
        filtered=torch.cat([project(l[k:k+1],u[k:k+1])[0] for k in (0,1)])
        w.grad=l+filtered;counter.call(i+1);opt.step()
    assert float((w-1).square().mean())<start*1e-5
    return dict(status='PASS',checks=['conflict half-space and norm','aligned/zero unchanged','tiny finite gradients','two-case projected optimization'],optimizer_calls=32)


def qualify(config):
    from .run import engine,construct
    root=Path(config['run_root'])
    if (root/'QUALIFICATION.json').exists():
        prior=read(root/'QUALIFICATION.json')
        if prior['status']=='PASS' or prior['execution_commit']==config['execution_commit']:
            raise FileExistsError('qualification already attempted for this code')
    out=root/'qualification_attempts'/config['execution_commit'];out.mkdir(parents=True,exist_ok=False)
    cpu=Counter(root/'qualification_cpu_physical.jsonl',CAPS['cpu_optimizer_updates'])
    cuda=Counter(root/'qualification_cuda_physical.jsonl',CAPS['synthetic_cuda_updates'])
    smoke=Counter(root/'smoke_physical.jsonl',CAPS['real_smoke_updates'])
    start_cuda,start_smoke=cuda.count,smoke.count
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    permit=engine.admission(config,'qualification');rows=[]
    options={**OPTIONS,'total_steps':4,'warmup_fraction':.25,'U_ramp_fraction':.25,'PAS_confidence':0.,'PAS_cosine':-1.}
    def generated(enabled=True):
        p=GeneratedDomain(seed=163,size=384,device=device,stage_source=dict(kind='generated',seed=163,domain='REFUGE'))
        native=build(config['reference'],device,163)
        from ..f5_confirmation_v1.native_qualification import foreground_fixture
        foreground_fixture(native)
        m=make_model(NativeLRParent(native,163,p.stage_source),'F5',options,None,generator(163,1,1,0,'pilot_adapter_initialization')).to(device)
        return ProjectionTrainer(m,p,options,arm='F5',execution=permit,project_enabled=enabled)
    try:
        math=mathematical_checks(cpu);write(out/'MATHEMATICAL.json',math)
        with NativeOperations(out/'operations') as operations:
            a=generated(False);m=copy.deepcopy(a.model);p=copy.deepcopy(a.provider)
            legacy=Path(config['parity_code'])/'experiments/lcrseg/five_frameworks_v1/train_stage.py'
            ns=dict(__name__='experiments.lcrseg.five_frameworks_v1.v1_frozen_stage',__package__='experiments.lcrseg.five_frameworks_v1')
            exec(compile(legacy.read_text(),str(legacy),'exec'),ns)
            b=ns['StageTrainer'](m,p,options,execution=permit)
            cuda.wrap(a.optimizer);cuda.wrap(b.optimizer)
            for _ in range(2):a.update();b.update();equal(snapshot(a),snapshot(b))
            rows.append('disabled projection parity with archived V1 F5');del a,b,m,p
            t=generated();cuda.wrap(t.optimizer);t.update();gradients=t.update()
            assert all(p.grad is None for p in t.ema.parameters())
            for p in t.model.parameters():
                if id(p) in gradients and p is not t.model.sidecar.r:assert gradients[id(p)]['U'] is None
            ident=dict(family='F5',seed=163,order=1,stage=1,arm='PROJECT')
            checkpoint.save(t,out/'qualification.pt',ident)
            other=ProjectionTrainer.for_resume(copy.deepcopy(t.model),copy.deepcopy(t.provider),options,execution=permit)
            other.entry_fingerprint=t.entry_fingerprint;checkpoint.restore(other,out/'qualification.pt',ident)
            cuda.wrap(other.optimizer);t.update();other.update();equal(snapshot(t),snapshot(other))
            assert t.telemetry['projection_calls']==other.telemetry['projection_calls'];del other
            with torch.no_grad():before=t.model(torch.zeros(2,3,384,384,device=device))
            deployed=t.model.deploy()
            with torch.no_grad():after=deployed(torch.zeros(2,3,384,384,device=device))
            assert torch.allclose(before,after,atol=2e-5,rtol=2e-5)
            native=deployed.parent.native;previous=deployed.transform.detach().clone();del deployed
            p=GeneratedDomain(seed=163,stage=2,size=384,device=device,stage_source=dict(kind='own_generated_predecessor',seed=163,domain='RIM_ONE_r3'))
            m=make_model(NativeLRParent(native,163,p.stage_source),'F5',options,previous,generator(163,1,2,0,'pilot_adapter_initialization')).to(device)
            nxt=ProjectionTrainer(m,p,options,arm='F5',execution=permit);assert torch.count_nonzero(nxt.model.sidecar.r)==0
            cuda.wrap(nxt.optimizer);nxt.update();del nxt,m,native,p
            try:t.update(fault='after_optimizer')
            except RuntimeError as e:assert str(e)=='injected after_optimizer'
            else:raise AssertionError('expected injected failure')
            assert t.requires_restore and t.step==3;del t
            rows+=['native backward and U permissions','teacher frozen','checkpoint continuation including counters','deployment merge','own-prefix stage transition','failed invocation remains uncommitted']
            for seed in (163,164):
                native,_=engine.source(config,seed,device)
                with torch.no_grad():value=native(torch.zeros(2,3,384,384,device=device),stochastic_classifier=False)[0]
                assert torch.isfinite(value).all();del native
            j=dict(arm='PROJECT',seed=163,order=1)
            t=construct(config,j,1,device,engine.admission(config,'smoke'),allow_u=False);smoke.wrap(t.optimizer)
            t.update();t.update();assert t.provider.u_reads==0;del t
            rows.append('discarded actual current-L smoke; no U reads')
        assert cuda.count-start_cuda==10 and smoke.count-start_smoke==2
        report=dict(status='PASS',plan_id=plan()['plan_id'],execution_commit=config['execution_commit'],checks=rows,
                    costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count,
                               operations=dict(operations.counts)),matched_control_reuse_qualified=True)
        write(out/'QUALIFICATION.json',report);write(root/'QUALIFICATION.json',report)
    except BaseException as e:
        write(out/'FAILURE.private.json',dict(error=repr(e),traceback=__import__('traceback').format_exc()))
        failure=dict(status='FAIL',execution_commit=config['execution_commit'],costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count))
        write(out/'QUALIFICATION.json',failure);write(root/'QUALIFICATION.json',failure);raise
