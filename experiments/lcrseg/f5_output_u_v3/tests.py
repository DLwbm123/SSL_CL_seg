import copy
from pathlib import Path
import torch
from .model import OutputUModel,OutputUTrainer
from ..five_frameworks_v1.train_stage import split_gradients,StageTrainer
from ..five_frameworks_v1.model import Model
from .protocol import CAPS,OPTIONS,write,read,plan
from ..f5_module_pilots_v1.tests import GeneratedDomain,snapshot,equal
from ..five_frameworks_v1.native_parent import build,NativeLRParent
from ..five_frameworks_v1.native_runner import Counter
from ..five_frameworks_v1.native_operations import NativeOperations
from ..five_frameworks_v1 import checkpoint


def mathematical_checks(counter):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a=torch.nn.Parameter(torch.eye(2));self.b=torch.nn.Parameter(torch.zeros(2,2))
            self.r=torch.nn.Parameter(torch.zeros(2,2));self.register_buffer('free',torch.diag(torch.tensor([0.,1.])))
        def forward(self,x):return (self.b@(self.a@self.free)+self.r)@x
        def u_parameters(self):return [self.b,self.r]
    m=Toy();x=torch.eye(2);target=torch.tensor([[0.,.2],[0.,.4]])
    opt=torch.optim.SGD(m.parameters(),lr=.15)
    initial=float((m(x)-target).square().sum())
    for i in range(32):
        opt.zero_grad();l=(m(x)-target).square().sum();u=(m(x)-target).square().sum()
        g=split_gradients(m,l,u)
        assert g[id(m.a)]['U'] is None and g[id(m.b)]['U'] is not None
        if i==0:assert g[id(m.b)]['U'].norm()>0
        counter.call(i+1);opt.step()
        assert torch.count_nonzero((m.b@(m.a@m.free))[:,0])==0
    assert float((m(x)-target).square().sum())<initial*1e-5
    return dict(status='PASS',checks=['U B/R routing with A forbidden','two-case projected-factor optimization','protected input coordinate unchanged'],optimizer_calls=32)


def cpu_qualification(config):
    from ..five_frameworks_v1.gate import digest
    signature=digest(dict(function=__import__('inspect').getsource(mathematical_checks),
                          output_u=Path(__file__).with_name('model.py').read_text()))
    root=Path(config['run_root']);path=root/'CPU_QUALIFICATION.json'
    if path.exists() and read(path).get('tested_code')==signature:return read(path)
    counter=Counter(root/'qualification_cpu_physical.jsonl',CAPS['cpu_optimizer_updates'])
    result={**mathematical_checks(counter),'tested_code':signature,'execution_commit':config['execution_commit'],
            'cumulative_cpu_optimizer_updates':counter.count}
    write(path,result);return result


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
        m=OutputUModel(NativeLRParent(native,163,p.stage_source),'F5',ratio=options['rank_ratio'],output_u=enabled).to(device)
        return OutputUTrainer(m,p,options,execution=permit)
    try:
        math=cpu_qualification(config);cpu.count=sum(1 for _ in cpu.path.open());write(out/'MATHEMATICAL.json',math)
        with NativeOperations(out/'operations') as operations:
            a=generated(False);m=copy.deepcopy(a.model);p=copy.deepcopy(a.provider)
            legacy=Path(config['parity_code'])/'experiments/lcrseg/five_frameworks_v1/train_stage.py'
            ns=dict(__name__='experiments.lcrseg.five_frameworks_v1.v1_frozen_stage',__package__='experiments.lcrseg.five_frameworks_v1')
            exec(compile(legacy.read_text(),str(legacy),'exec'),ns)
            b=ns['StageTrainer'](m,p,options,execution=permit)
            cuda.wrap(a.optimizer);cuda.wrap(b.optimizer)
            for _ in range(2):a.update();b.update();equal(snapshot(a),snapshot(b))
            rows.append('disabled output-U parity with archived V1 F5');del a,b,m,p
            p=GeneratedDomain(seed=163,size=384,device=device,stage_source=dict(kind='generated',seed=163,domain='REFUGE'))
            native=build(config['reference'],device,163)
            from ..f5_confirmation_v1.native_qualification import foreground_fixture
            foreground_fixture(native)
            m=Model(NativeLRParent(native,163,p.stage_source),'B2_PARENT_PAS_KL').to(device)
            opt={**options,'lambda_U':1.,'PAS_confidence':.6,'PAS_cosine':.7}
            a=StageTrainer(m,p,opt,execution=permit)
            b=ns['StageTrainer'](copy.deepcopy(a.model),copy.deepcopy(a.provider),opt,execution=permit)
            cuda.wrap(a.optimizer);cuda.wrap(b.optimizer)
            for _ in range(2):a.update();b.update();equal(snapshot(a),snapshot(b))
            rows.append('unchanged historical B2 state parity');del a,b,m,p,native

            t=generated();cuda.wrap(t.optimizer);t.update();gradients=t.update()
            assert all(p.grad is None for p in t.ema.parameters())
            groups=t.model.parent.parameter_groups()
            assert all(gradients[id(p)]['U'] is None for p in groups['input_factors'])
            assert sum(float(gradients[id(p)]['U'].norm()) for p in groups['output_factors'])>0
            assert t.telemetry['nonzero_output_U_calls']>0
            # Check each existing U component reaches B, without optimizer calls or patient data.
            saved_swd=t.options['lambda_SWD'];saved_kl=t.fine_kl
            for component in ('KL','SWD'):
                t.options['lambda_SWD']=0. if component=='KL' else saved_swd
                if component=='SWD':t.fine_kl=lambda logits,target,valid:logits.sum()*0
                _,u,_=t.losses()
                params=groups['output_factors']+[t.model.sidecar.r]
                gs=torch.autograd.grad(u,params,allow_unused=True)
                b_norm=sum(float(g.norm()) for g in gs[:-1] if g is not None)
                if component=='SWD' and t.model.detach_u_parent(clean=True):
                    assert b_norm==0 and gs[-1] is not None and gs[-1].norm()>0
                else:assert b_norm>0,component+' B gradient missing'
                t.fine_kl=saved_kl
            t.options['lambda_SWD']=saved_swd
            ident=dict(family='F5',seed=163,order=1,stage=1,arm='OUTPUT_U')
            checkpoint.save(t,out/'qualification.pt',ident)
            other=OutputUTrainer.for_resume(copy.deepcopy(t.model),copy.deepcopy(t.provider),options,execution=permit)
            other.entry_fingerprint=t.entry_fingerprint;checkpoint.restore(other,out/'qualification.pt',ident)
            cuda.wrap(other.optimizer);t.update();other.update();equal(snapshot(t),snapshot(other))
            assert t.telemetry['output_U_calls']==other.telemetry['output_U_calls'];del other
            with torch.no_grad():before=t.model(torch.zeros(2,3,384,384,device=device))
            deployed=t.model.deploy()
            with torch.no_grad():after=deployed(torch.zeros(2,3,384,384,device=device))
            assert torch.allclose(before,after,atol=2e-5,rtol=2e-5)
            native=deployed.parent.native;previous=deployed.transform.detach().clone();del deployed
            p=GeneratedDomain(seed=163,stage=2,size=384,device=device,stage_source=dict(kind='own_generated_predecessor',seed=163,domain='RIM_ONE_r3'))
            m=OutputUModel(NativeLRParent(native,163,p.stage_source),'F5',ratio=options['rank_ratio'],previous=previous).to(device)
            nxt=OutputUTrainer(m,p,options,execution=permit);assert torch.count_nonzero(nxt.model.sidecar.r)==0
            cuda.wrap(nxt.optimizer);nxt.update();del nxt,m,native,p
            try:t.update(fault='after_optimizer')
            except RuntimeError as e:assert str(e)=='injected after_optimizer'
            else:raise AssertionError('expected injected failure')
            assert t.requires_restore and t.step==3;del t
            rows+=['native backward: KL and SWD each reach B; U A forbidden','teacher frozen','checkpoint continuation including counters','deployment merge','own-prefix stage transition','failed invocation remains uncommitted']
            for seed in (163,164):
                native,_=engine.source(config,seed,device)
                with torch.no_grad():value=native(torch.zeros(2,3,384,384,device=device),stochastic_classifier=False)[0]
                assert torch.isfinite(value).all();del native
            j=dict(arm='OUTPUT_U',seed=163,order=1)
            t=construct(config,j,1,device,engine.admission(config,'smoke'),allow_u=False);smoke.wrap(t.optimizer)
            t.update();t.update();assert t.provider.u_reads==0;del t
            rows.append('discarded actual current-L smoke; no U reads')
        assert cuda.count-start_cuda==14 and smoke.count-start_smoke==2
        report=dict(status='PASS',plan_id=plan()['plan_id'],execution_commit=config['execution_commit'],checks=rows,
                    costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count,
                               operations=dict(operations.counts)),matched_control_reuse_qualified=True)
        write(out/'QUALIFICATION.json',report);write(root/'QUALIFICATION.json',report)
    except BaseException as e:
        write(out/'FAILURE.private.json',dict(error=repr(e),traceback=__import__('traceback').format_exc()))
        failure=dict(status='FAIL',execution_commit=config['execution_commit'],costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count))
        write(out/'QUALIFICATION.json',failure);write(root/'QUALIFICATION.json',failure);raise
