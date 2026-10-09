import copy
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from .modules import ARMS,SPEC,boundary_interior,distribution_triplet,sinkhorn_divergence,make_model,PilotTrainer
from .protocol import OPTIONS,CAPS,read,write,plan
from ..five_frameworks_v1.recipes import SyntheticCurrentDomain,generator
from ..five_frameworks_v1 import checkpoint


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert a.shape==b.shape and a.dtype==b.dtype
        assert torch.allclose(a.cpu(),b.cpu(),atol=2e-6,rtol=1e-5) if a.is_floating_point() else torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def snapshot(t):
    return copy.deepcopy(dict(student=t.model.state_dict(),ema=t.ema.state_dict(),optimizer=t.optimizer.state_dict(),
                              scheduler=t.scheduler.state_dict(),step=t.step,cursor=t.cursor,
                              values=t.prototypes.values,support=t.prototypes.support))


class GeneratedDomain(SyntheticCurrentDomain):
    def __init__(self,*a,device='cpu',**kw):super().__init__(*a,**kw);self.device=device
    def labeled(self,*a,**kw):
        x,_,ids=super().labeled(*a,**kw)
        yy,xx=torch.meshgrid(torch.arange(self.size),torch.arange(self.size),indexing='ij')
        distance=(xx-self.size/2).square()+(yy-self.size/2).square()
        y=(distance<(self.size*.36)**2).long()+(distance<(self.size*.2)**2).long()
        y=y[None].expand(2,-1,-1).clone();x=x*.03+y[:,None]*.3
        return x.to(self.device),y.to(self.device),ids
    def unlabeled(self,*a,**kw):
        x,g,ids=super().unlabeled(*a,**kw)
        return (x*1.7+.3).to(self.device),g.to(self.device),ids


def mathematical_tests(physical=None):
    torch.manual_seed(9);calls=0;checks=[]
    y=GeneratedDomain(size=20).labeled(0)[1]
    logits=torch.randn(2,3,20,20,requires_grad=True)
    bl,il,counts=boundary_interior(logits.log_softmax(1),y)
    (bl+il).backward();assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    assert counts['boundary_pixels']>0 and counts['interior_classes']>0
    ignore=torch.full_like(y,255);z=sum(boundary_interior(logits.log_softmax(1),ignore)[:2])
    assert z==0
    checks.append('boundary GT/ignore/grid and nonzero gradient')
    # The two spatial marginals agree for a constant confidence on the GT interior.
    uniform=torch.zeros(2,3,20,20).log_softmax(1)
    assert float(boundary_interior(uniform,y)[1])<1e-12
    mix=torch.zeros_like(y,dtype=torch.bool);mix[:,:,:10]=True
    masked=boundary_interior(logits.log_softmax(1),y,mix)[2]
    assert masked['boundary_pixels']<counts['boundary_pixels']
    checks.append('interior CDF zero case and seam exclusion')
    x=torch.randn(8,4,dtype=torch.double);x=F.normalize(x,dim=1)
    assert abs(float(sinkhorn_divergence(x,x)))<1e-10
    target=F.normalize(torch.randn(8,4,dtype=torch.double),dim=1)
    test=x.clone().requires_grad_(True);loss=sinkhorn_divergence(test,target);g,=torch.autograd.grad(loss,test)
    eps=1e-5;hi=x.clone();lo=x.clone();hi[0,0]+=eps;lo[0,0]-=eps
    numeric=(sinkhorn_divergence(hi,target)-sinkhorn_divergence(lo,target))/(2*eps)
    assert torch.allclose(g[0,0],numeric,atol=2e-4,rtol=2e-3)
    checks.append('Sinkhorn self-zero and finite-difference envelope gradient')
    # Two-case module overfit checks, with explicit optimization accounting.
    fit=torch.nn.Parameter(torch.randn(2,3,20,20));opt=torch.optim.Adam([fit],lr=.15)
    start=None
    for step in range(64):
        opt.zero_grad();a,b,_=boundary_interior(fit.log_softmax(1),y);loss=a+b
        if start is None:start=float(loss)
        loss.backward()
        if physical:physical(calls+1)
        opt.step();calls+=1
    assert float(loss)<start*.6
    checks.append('two-case boundary/interior overfit')
    classes=torch.ones(2,4,4,dtype=torch.long);lc=classes.clone();lc[:,:,2:]=2
    teacher=torch.zeros(2,4,4,4);teacher[:,0,:,:2]=1;teacher[:,1,:,2:]=1
    student=torch.nn.Parameter(torch.randn(2,4,4,4));opt=torch.optim.Adam([student],lr=.1);start=None
    for step in range(64):
        opt.zero_grad();loss,stats=distribution_triplet(student,teacher,classes,lc,classes.bool(),generator(163,1,1,0,'test_ot'))
        if start is None:start=float(loss)
        loss.backward()
        if physical:physical(calls+1)
        opt.step();calls+=1
    assert stats['valid_classes']==1 and float(loss)<max(start*.2,1e-5)
    checks.append('two-case distribution triplet overfit and class support')
    from ..five_frameworks_v1.kernels import StageSubspaceAdapter
    from .modules import LossIndexedAdapter
    sidecar=LossIndexedAdapter(StageSubspaceAdapter(torch.eye(4)),generator(163,1,1,0,'test_adapter'))
    desired=torch.eye(4)*.15;opt=torch.optim.Adam(sidecar.optimizer_parameters(),lr=.05)
    for _ in range(64):
        opt.zero_grad();loss=(sidecar.current_matrix()-desired).square().mean();loss.backward()
        if physical:physical(calls+1)
        opt.step();calls+=1
    assert float(loss)<1e-4
    teacher=copy.deepcopy(sidecar).requires_grad_(False);teacher.up.data.zero_()
    before=teacher.current_matrix().clone();value=sidecar.current_matrix().clone();sidecar.update_teacher(teacher)
    assert torch.allclose(teacher.current_matrix(),before*.99+value*.01,atol=1e-6)
    checks.append('two-case matrix overfit and effective-matrix EMA')
    assert calls==192 and calls<=CAPS['cpu_optimizer_updates']
    return dict(status='PASS',checks=checks,cpu_optimizer_updates=calls)


def qualify(config):
    from .run import admission,construct,source
    from ..five_frameworks_v1.native_parent import build,NativeLRParent
    from ..five_frameworks_v1.native_runner import Counter
    from ..five_frameworks_v1.native_operations import NativeOperations
    from ..five_frameworks_v1.train_stage import StageTrainer
    root=Path(config['run_root'])
    if (root/'QUALIFICATION.json').exists():raise FileExistsError('qualification already attempted')
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    permit=admission(config,'qualification')
    cpu_counter=Counter(root/'qualification_cpu_physical.jsonl',CAPS['cpu_optimizer_updates'])
    cuda_counter=Counter(root/'qualification_cuda_physical.jsonl',CAPS['synthetic_cuda_updates'])
    smoke_counter=Counter(root/'smoke_physical.jsonl',CAPS['real_smoke_updates']);rows=[]
    options={**OPTIONS,'total_steps':4,'warmup_fraction':.25,'U_ramp_fraction':.25,'PAS_confidence':0.,'PAS_cosine':-1.}
    fixture=GeneratedDomain(seed=163,size=384,device=device,stage_source=dict(kind='generated',seed=163,domain='REFUGE'))
    def generated(arm):
        provider=GeneratedDomain(seed=163,size=384,device=device,stage_source=fixture.stage_source)
        native=build(config['reference'],device,163)
        from ..f5_confirmation_v1.native_qualification import foreground_fixture
        foreground_fixture(native)
        model=make_model(NativeLRParent(native,163,provider.stage_source),arm,options,None,generator(163,1,1,0,'pilot_adapter_initialization')).to(device)
        return PilotTrainer(model,provider,options,arm=arm,execution=permit)
    try:
        cpu=mathematical_tests(cpu_counter.call);write(root/'CPU_TEST_REPORT.json',cpu)
        with NativeOperations(root/'qualification_operations') as operations:
            # Baseline parity with the actual historical trainer source, not a fabricated approval.
            legacy_path=Path(config['legacy_code'])/'experiments/lcrseg/five_frameworks_v1/train_stage.py'
            namespace=dict(__name__='experiments.lcrseg.five_frameworks_v1.legacy_stage',__package__='experiments.lcrseg.five_frameworks_v1')
            exec(compile(legacy_path.read_text(),str(legacy_path),'exec'),namespace)
            a=generated('F5');model=copy.deepcopy(a.model);provider=copy.deepcopy(a.provider)
            b=namespace['StageTrainer'](model,provider,options,execution=permit)
            cuda_counter.wrap(a.optimizer);cuda_counter.wrap(b.optimizer)
            for _ in range(2):a.update();b.update();equal(snapshot(a),snapshot(b))
            rows.append(dict(check='native historical F5 warmup/active full-state parity',status='PASS'))
            del a,b,model,provider
            for arm in ARMS:
                t=generated(arm);cuda_counter.wrap(t.optimizer);t.update();t.update()
                assert all(p.grad is None for p in t.ema.parameters())
                labeled,unlabeled,_=t.losses();g=t.model.u_parameters()
                if arm=='OT_TRIPLET':assert t.module_stats['valid_classes']>=1
                if arm=='BOUNDARY':assert t.module_stats['boundary_pixels']>0
                gu=torch.autograd.grad(unlabeled,[p for p in t.model.parameters() if p.requires_grad],allow_unused=True)
                for p,v in zip([p for p in t.model.parameters() if p.requires_grad],gu):
                    if id(p) not in {id(x) for x in g}:assert v is None or torch.count_nonzero(v)==0
                del labeled,unlabeled,g,gu
                before=t.model(torch.zeros(2,3,384,384,device=device)).detach()
                deployed=t.model.deploy()
                after=deployed(torch.zeros(2,3,384,384,device=device)).detach()
                assert torch.allclose(before,after,atol=2e-5,rtol=2e-5);del deployed
                ident=dict(family='F5',seed=163,order=1,stage=1,arm=arm)
                checkpoint.save(t,root/f'{arm}_qualification.pt',ident)
                second=PilotTrainer.for_resume(copy.deepcopy(t.model),copy.deepcopy(t.provider),options,arm=arm,execution=permit)
                second.entry_fingerprint=t.entry_fingerprint
                checkpoint.restore(second,root/f'{arm}_qualification.pt',ident);cuda_counter.wrap(second.optimizer)
                t.update();second.update();equal(snapshot(t),snapshot(second))
                del second
                deployment=t.model.deploy();native=deployment.parent.native;previous=deployment.transform.detach().clone();del deployment
                provider=GeneratedDomain(seed=163,stage=2,size=384,device=device,
                                         stage_source=dict(kind='generated_own_predecessor',seed=163,domain='RIM_ONE_r3'))
                next_model=make_model(NativeLRParent(native,163,provider.stage_source),arm,options,previous,
                                      generator(163,1,2,0,'pilot_adapter_initialization')).to(device)
                next_trainer=PilotTrainer(next_model,provider,options,arm=arm,execution=permit)
                assert torch.count_nonzero(next_trainer.model.sidecar.current_matrix())==0
                cuda_counter.wrap(next_trainer.optimizer);next_trainer.update();del next_trainer,next_model,native
                try:t.update(fault='after_optimizer')
                except RuntimeError as e:assert str(e)=='injected after_optimizer'
                else:raise AssertionError('failure injection did not fail')
                assert t.requires_restore and t.step==3
                rows.append(dict(arm=arm,check='backward, U permissions, teacher frozen, merge, checkpoint continuation',status='PASS'))
                rows.append(dict(arm=arm,check='own-prefix transition and failed invocation remains uncommitted',status='PASS'))
                del t
            # Actual source tensors loaded under the original native schema; generated forward only.
            sources=[]
            for seed in (163,164):
                native,sr=source(config,seed,device)
                with torch.no_grad():result=native(torch.zeros(2,3,384,384,device=device),stochastic_classifier=False)[0]
                assert torch.isfinite(result).all();sources.append(seed);del native
            for arm in ARMS:
                for stage in (1,2):
                    j=dict(arm=arm,seed=163,order=1)
                    t=construct(config,j,stage,device,admission(config,'smoke'),allow_u=False)
                    smoke_counter.wrap(t.optimizer)
                    t.update();t.update()
                    assert t.provider.u_reads==0
                    rows.append(dict(arm=arm,stage=stage,check='discarded actual current-L smoke',status='PASS'))
                    del t
        report=dict(status='PASS',execution_commit=config['execution_commit'],plan_id=plan()['plan_id'],checks=rows,
                    source_schema_generated_forward=sources,costs=dict(cpu_optimizer_updates=cpu_counter.count,
                    synthetic_cuda_updates=cuda_counter.count,real_L_smoke_updates=smoke_counter.count,
                    operations=dict(operations.counts),qualification_extra_gradient_vjps=4))
        assert cuda_counter.count==28 and smoke_counter.count==16
        write(root/'QUALIFICATION.json',report)
    except BaseException as e:
        write(root/'QUALIFICATION_FAILURE.private.json',dict(error=repr(e),traceback=__import__('traceback').format_exc()))
        write(root/'QUALIFICATION.json',dict(status='FAIL',execution_commit=config['execution_commit'],
             costs=dict(cpu_optimizer_updates=cpu_counter.count,synthetic_cuda_updates=cuda_counter.count,
                        real_L_smoke_updates=smoke_counter.count)))
        raise
