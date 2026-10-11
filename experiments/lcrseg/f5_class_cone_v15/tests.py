import copy
from pathlib import Path
import torch
from . import protocol
from .run import engine,ClassConeTrainer,construct,make_model
from ..f5_entry_gradient_v12.model import halfspace
from ..f5_class_reference_v13.model import class_reference
from .model import class_cone
from ..f5_anchor_guard_v10.model import guarded_mask
from ..f5_module_pilots_v1.tests import GeneratedDomain,snapshot,equal
from ..five_frameworks_v1.native_parent import build,NativeLRParent
from ..five_frameworks_v1.recipes import generator
from ..five_frameworks_v1.native_runner import Counter
from ..five_frameworks_v1 import checkpoint

def qualify(config):
    root=Path(config['run_root']);cpu=Counter(root/'qualification_cpu_physical.jsonl',96);cuda=Counter(root/'qualification_cuda_physical.jsonl',32);smoke=Counter(root/'smoke_physical.jsonl',4)
    if (root/'QUALIFICATION.json').exists():raise FileExistsError('no qualification retry')
    try:
        torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);permit=engine.admission(config,'qualification')
        anchor=torch.tensor([[[[.8,.8,.4,.8]],[[.1,.1,.3,.1]],[[.1,.1,.3,.1]]]])
        target=torch.tensor([[[[.8,.1,.1,.1]],[[.1,.8,.8,.8]],[[.1,.1,.1,.1]]]])
        valid=torch.tensor([[[True,True,True,False]]])
        assert torch.equal(guarded_mask(anchor,target,valid,.7),torch.tensor([[[True,False,True,False]]]))
        for invalid in (anchor[:,:,:,:2],anchor*float('nan')):
            try:guarded_mask(invalid,target,valid,.7)
            except (ValueError,FloatingPointError):pass
            else:raise AssertionError('invalid guard input accepted')
        for merged,ref,wanted in [([-1.,2.],[1.,0.],[0.,2.]),([1.,2.],[1.,0.],[1.,2.]),([0.,2.],[1.,0.],[0.,2.]),([-1.,2.],[0.,0.],[-1.,2.])]:
            out,stats=halfspace([torch.tensor(merged)],[torch.tensor(ref)])
            assert torch.allclose(out[0],torch.tensor(wanted)) and stats['dot_after']>=-1e-10
        for merged,ref in [([torch.zeros(2)],[torch.zeros(3)]),([torch.tensor([float('nan')])],[torch.ones(1)]),([torch.tensor([1e308],dtype=torch.double)],[torch.tensor([1e308],dtype=torch.double)])]:
            try:halfspace(merged,ref)
            except (ValueError,FloatingPointError):pass
            else:raise AssertionError('invalid half-space input accepted')
        # A nonnegative balanced dot can hide a negative class dot; enumerate the full intersection.
        cases=[([-1.,2.],[[1.,0.],[0.,1.]],[0.,2.]),([-1.,-2.,-3.],[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],[0.,0.,0.]),([1.,2.],[[1.,0.],[0.,1.]],[1.,2.]),([-1.,2.],[[1.,0.],[-1.,0.]],[0.,2.]),([-1.,2.],[[0.,0.]],[-1.,2.]),([-1.,2.],[[1.,0.],[1.,0.]],[0.,2.]),([-1.,2.],[],[-1.,2.])]
        for g,refs,wanted in cases:
            out,stats=class_cone([torch.tensor(g,dtype=torch.double)],[[torch.tensor(r,dtype=torch.double)] for r in refs])
            assert torch.allclose(out[0],torch.tensor(wanted,dtype=torch.double),atol=1e-10) and all(x>=-1e-10 for x in stats['dots_after'])
        g=torch.tensor([-1.,2.],dtype=torch.double)
        original,_=halfspace([g],[torch.tensor([.5,.5],dtype=torch.double)])
        separate,stats=class_cone([g],[[torch.tensor([1.,0.],dtype=torch.double)],[torch.tensor([0.,1.],dtype=torch.double)]])
        assert torch.equal(original[0],g) and stats['aggregate_safe_class_conflict'] and not torch.equal(separate[0],g)
        for g,refs in [([torch.zeros(2)],[[torch.zeros(3)]]),([torch.tensor([float('nan')])],[[torch.ones(1)]]),([torch.tensor([1e308],dtype=torch.double)],[[torch.ones(1,dtype=torch.double)]]),([torch.ones(1)],[[torch.ones(1)]]*4)]:
            try:class_cone(g,refs)
            except (ValueError,FloatingPointError):pass
            else:raise AssertionError('invalid class cone accepted')
        rng=torch.Generator().manual_seed(150)
        for _ in range(128):
            g=torch.randn(7,generator=rng,dtype=torch.double);refs=[torch.randn(7,generator=rng,dtype=torch.double) for _ in range(3)]
            out,stats=class_cone([g],[[r] for r in refs]);d=out[0]
            assert all(d@r>=-1e-9 for r in refs) and (d-g).square().sum()<=g.square().sum()+1e-8
            normals=[r for r in refs if abs(float(d@r))<1e-8]
            if normals:
                matrix=torch.stack(normals,1);coef=torch.linalg.lstsq(matrix,(d-g).unsqueeze(1)).solution[:,0]
                assert (coef>=-1e-8).all() and torch.allclose(matrix@coef,d-g,atol=1e-8,rtol=1e-8)
            else:assert torch.allclose(d,g,atol=1e-8,rtol=1e-8)
        # Replicating only background changes pixel mean, but preserves the balanced scalar and aggregate gradient.
        q=torch.tensor([[[[.8,.1,.1]],[[.1,.8,.1]],[[.1,.1,.8]]]],dtype=torch.double,requires_grad=True)
        z=torch.tensor([[[[.2,.1,-.2]],[[.1,-.3,.2]],[[.4,.5,.6]]]],dtype=torch.double,requires_grad=True)
        mask=torch.ones(1,1,3,dtype=torch.bool)
        value,counts=class_reference(z,q,mask);assert counts==[1,1,1]
        idx=torch.tensor([0,0,0,0,0,0,0,0,0,0,1,2]);zr=z.detach()[...,idx].requires_grad_(True);qr=q.detach()[...,idx]
        replicated,_=class_reference(zr,qr,mask[...,idx]);assert torch.allclose(value,replicated,atol=1e-12,rtol=1e-12)
        a_grad,=torch.autograd.grad(value,z,retain_graph=True);b_grad,=torch.autograd.grad(replicated,zr)
        for c in range(3):assert torch.allclose(a_grad[...,c],b_grad[...,idx==c].sum(-1),atol=1e-12,rtol=1e-12)
        assert torch.autograd.grad(value,q,allow_unused=True)[0] is None
        empty,counts=class_reference(z,q,torch.zeros_like(mask));assert counts==[0,0,0] and empty==0
        assert torch.count_nonzero(torch.autograd.grad(empty,z)[0])==0
        single=mask.clone();single[...,1:]=False
        from ..five_frameworks_v1.kernels import masked_kl
        assert torch.allclose(class_reference(z,q,single)[0],masked_kl(z,q,single))
        # Two conflicting cases retain the entry direction while learning its orthogonal component.
        fit=torch.nn.Parameter(torch.tensor([[.3,0.],[-.3,0.]],dtype=torch.double));opt=torch.optim.SGD([fit],lr=.2)
        target=torch.tensor([[1.,1.],[-1.,1.]],dtype=torch.double)
        old=float(fit[:,0].square().mean().detach());start=float((fit-target).square().sum().detach())
        for i in range(32):
            opt.zero_grad();loss=.5*(fit-target).square().sum();g,=torch.autograd.grad(loss,fit)
            ref=torch.zeros_like(fit);ref[:,0]=fit[:,0].detach()
            projected,stats=class_cone([g],[[ref]]);fit.grad=projected[0]
            assert stats['projected'] and stats['dots_after'][0]>=-1e-10
            cpu.call(i+1);opt.step()
        assert float((fit-target).square().sum().detach())<start*.6
        assert abs(float(fit[:,0].square().mean().detach())-old)<1e-12
        del fit,opt
        options={**protocol.OPTIONS,'total_steps':4,'warmup_fraction':.25,'U_ramp_fraction':.25,'PAS_confidence':0.,'PAS_cosine':-1.}
        def generated(enabled):
            p=GeneratedDomain(seed=163,size=384,device=device,stage_source=dict(kind='generated',seed=163,domain='REFUGE'));native=build(config['reference'],device,163)
            from ..f5_confirmation_v1.native_qualification import foreground_fixture
            foreground_fixture(native)
            o=dict(options);m=make_model(NativeLRParent(native,163,p.stage_source),o,None,generator(163,1,1,0,'pilot_adapter_initialization'),enabled=enabled).to(device)
            return ClassConeTrainer(m,p,o,execution=permit)
        a=generated(False)
        legacy_root=Path(config['v13_parity_code'])/'experiments/lcrseg'
        legacy=legacy_root/'f5_class_reference_v13/model.py'
        ns=dict(__name__='experiments.lcrseg.f5_class_reference_v13.v13_frozen_model',__package__='experiments.lcrseg.f5_class_reference_v13')
        exec(compile(legacy.read_text(),str(legacy),'exec'),ns)
        b=ns['ClassReferenceTrainer'](copy.deepcopy(a.model),copy.deepcopy(a.provider),a.options,execution=permit)
        legacy=legacy_root/'five_frameworks_v1/train_stage.py'
        old=dict(__name__='experiments.lcrseg.five_frameworks_v1.v13_frozen_stage',__package__='experiments.lcrseg.five_frameworks_v1')
        exec(compile(legacy.read_text(),str(legacy),'exec'),old)
        b.losses=__import__('types').MethodType(old['StageTrainer'].losses,b)
        cuda.wrap(a.optimizer);cuda.wrap(b.optimizer)
        for _ in range(2):a.update();b.update();equal(snapshot(a),snapshot(b))
        disabled_vjps=a.telemetry['entry_reference_vjps']+b.telemetry['entry_reference_vjps'];del a,b
        t=generated(True);cuda.wrap(t.optimizer);t.update();anchor_state={k:v.detach().clone() for k,v in t.model.entry_anchor.state_dict().items()};before=t.ema.sidecar.r.detach().clone();grads=t.update()
        assert all(t.telemetry[n+'_nonzero_U_calls']>0 for n in ('A','B','R'))
        assert t.module_stats['class_cone'] and all(x>=-1e-6 for x in t.module_stats['entry_dots_after'])
        assert all(torch.equal(v,t.model.entry_anchor.state_dict()[k]) for k,v in anchor_state.items())
        assert not t.model.entry_anchor.training and all(not p.requires_grad and p.grad is None for p in t.model.entry_anchor.parameters())
        assert t.model.entry_anchor is t.ema.entry_anchor
        dummy=torch.zeros(2,3,384,384,device=device)
        with torch.no_grad():
            anchor_probs=t.model.entry_anchor(dummy).softmax(1);labels=(anchor_probs.argmax(1)+1)%3
            forced=torch.nn.functional.one_hot(labels,3).permute(0,3,1,2).float();valid=torch.ones_like(labels,dtype=torch.bool)
        logits=torch.zeros_like(forced,requires_grad=True)
        loss=t.kl_objective(dummy,logits,forced,valid)
        # Main KL remains V10 rejection, while the entry reference retains a real gradient.
        assert loss==0 and torch.count_nonzero(torch.autograd.grad(loss,logits,retain_graph=True)[0])==0
        ref_logits,=torch.autograd.grad(t.entry_reference_loss,logits)
        assert ref_logits.abs().sum()>0
        assert all(p.grad is None for p in t.ema.parameters())
        assert torch.allclose(t.ema.sidecar.r,before*.99+t.model.sidecar.r.detach()*.01,atol=1e-7,rtol=1e-5)
        groups=t.model.parent.parameter_groups()
        # Exercise actual native merged-gradient projection with a forced conflicting direction.
        labeled,unlabeled,_=t.losses();t.gradient_audit(labeled,unlabeled)
        params=[p for p in t.model.parameters() if p.requires_grad]
        refs=[torch.zeros_like(p) if g is None else g.detach().clone() for p,g in zip(params,t.entry_reference_grads)]
        assert sum(float(g.square().sum()) for g in refs)>0
        forced_grads={}
        for p,g in zip(params,refs):
            p.grad=-g.clone();forced_grads[id(p)]={'L':-g,'U':torch.zeros_like(g)}
        t.transform_gradients(forced_grads)
        assert t.module_stats['entry_projected'] and t.module_stats['entry_dot_after']>=-1e-7
        assert all(x>=-1e-7 for x in t.module_stats['entry_dots_after']) and t.telemetry['class_cone_calls']>0
        assert sum(float(p.grad.square().sum()) for p in params)<1e-12
        assert t.entry_reference_grads is None and t.entry_class_grads is None
        # Same clean feature route remains R-only; no new patient probe or optimizer call.
        _,clean,_=t.model.parts(torch.zeros(2,3,384,384,device=device),detach_parent=t.model.detach_u_parent(clean=True))
        clean_grads=torch.autograd.grad(clean.square().mean(),groups['input_factors']+groups['output_factors']+t.model.sidecar.optimizer_parameters(),allow_unused=True)
        assert all(g is None for g in clean_grads[:-1]) and clean_grads[-1] is not None
        assert not t.model.sidecar.q.requires_grad and not t.model.sidecar.previous.requires_grad
        ident=dict(family='F5',seed=163,order=1,stage=1,arm='CLASS_CONE');checkpoint.save(t,root/'qualification.pt',ident)
        other=ClassConeTrainer.for_resume(copy.deepcopy(t.model),copy.deepcopy(t.provider),options,execution=permit);other.entry_fingerprint=t.entry_fingerprint;checkpoint.restore(other,root/'qualification.pt',ident);cuda.wrap(other.optimizer);restored_before=other.telemetry['entry_reference_vjps'];t.update();other.update();equal(snapshot(t),snapshot(other));assert t.telemetry==other.telemetry;assert all(torch.equal(v,t.model.entry_anchor.state_dict()[k]) for k,v in anchor_state.items());resume_vjps=other.telemetry['entry_reference_vjps']-restored_before;del other
        wrong_model=copy.deepcopy(t.model);wrong_model.class_cone=False
        wrong=ClassConeTrainer.for_resume(wrong_model,copy.deepcopy(t.provider),options,execution=permit);wrong.entry_fingerprint=t.entry_fingerprint
        try:checkpoint.restore(wrong,root/'qualification.pt',ident)
        except ValueError as e:assert 'semantic' in str(e)
        else:raise AssertionError('changed actual module flag accepted')
        del wrong,wrong_model
        with torch.no_grad():before=t.model(torch.zeros(2,3,384,384,device=device));deployed=t.model.deploy();after=deployed(torch.zeros(2,3,384,384,device=device))
        assert torch.allclose(before,after,atol=2e-5,rtol=2e-5)
        p=GeneratedDomain(seed=163,stage=2,size=384,device=device,stage_source=dict(kind='own_generated_predecessor',seed=163,domain='RIM_ONE_r3'));m=make_model(NativeLRParent(deployed.parent.native,163,p.stage_source),options,deployed.transform.detach().clone(),generator(163,1,2,0,'pilot_adapter_initialization')).to(device)
        nxt=ClassConeTrainer(m,p,options,execution=permit);cuda.wrap(nxt.optimizer);nxt.update();next_vjps=nxt.telemetry['entry_reference_vjps'];del nxt,m,p,deployed
        try:t.update(fault='after_optimizer')
        except RuntimeError as e:assert str(e)=='injected after_optimizer'
        else:raise AssertionError('expected injected failure')
        assert t.requires_restore and t.step==3;enabled_vjps=t.telemetry['entry_reference_vjps']+resume_vjps+next_vjps;del t
        j=protocol.jobs()[0];t=construct(config,j,1,device,engine.admission(config,'smoke'),allow_u=False);smoke.wrap(t.optimizer);t.update();t.update();assert t.provider.u_reads==0;del t
        assert (cpu.count,cuda.count,smoke.count)==(32,10,2)
        protocol.write(root/'QUALIFICATION.json',dict(status='PASS',execution_commit=config['execution_commit'],plan_id=protocol.plan()['plan_id'],costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count),checks=['half-space conflict/aligned/orthogonal/zero/grid/nonfinite/overflow; two-case orthogonal adaptation and entry retention','disabled class cone matches archived V13 native complete state/loss path; equal present-class reference with absent/empty/replicated-class gradient checks','actual native A/B/R U gradients and summed L/U; clean SWD parent-detached feature route','immutable shared entry teacher, original V10 zero rejected KL retained; nonzero reference; actual A/B/R merged conflicting direction projected against every present class, linear R EMA; checkpoint continuation incl anchor/telemetry and reject wrong actual flag','deployment/own-prefix stage transition','deliberate failed optimizer call accounted','discarded current-L smoke with zero U reads'],qualification_enabled_class_reference_vjps=enabled_vjps,qualification_disabled_reference_vjps=disabled_vjps,qualification_extra_gradient_vjps=3,qualification_extra_cpu_vjps=4,qualification_class_cone_math_cases=136,qualification_extra_generated_loss_calls=1,qualification_extra_anchor_full_forwards=2,matched_control_reuse_qualified=True,matched_V10_reuse_qualified=True,matched_V13_reuse_qualified=True,matched_V12_reuse_qualified=True,historical_B2_reuse='unchanged previous parity retained; historical only'))
    except BaseException as e:
        protocol.write(root/'QUALIFICATION.json',dict(status='FAIL',execution_commit=config['execution_commit'],costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count)));protocol.write(root/'QUALIFICATION_FAILURE.private.json',dict(error=repr(e),traceback=__import__('traceback').format_exc()));raise
