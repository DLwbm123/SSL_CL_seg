import copy
from pathlib import Path
import torch
from . import protocol
from .run import engine,AnchorRestoreTrainer,construct,make_model
from .model import routed_target
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
        # Two explicit teacher-conflict cases fit the routed soft targets, not a surrogate factor task.
        routed,restored=routed_target(anchor,target,valid,.7)
        assert torch.equal(restored,torch.tensor([[[False,True,False,False]]]))
        assert torch.equal(routed[:,:,:,1],anchor[:,:,:,1]) and torch.equal(routed[:,:,:,0],target[:,:,:,0])
        assert not routed.requires_grad
        from ..five_frameworks_v1.kernels import masked_kl
        q=routed.expand(2,-1,-1,-1);v=valid.expand(2,-1,-1)
        fit=torch.nn.Parameter(torch.zeros_like(q,dtype=torch.float64));opt=torch.optim.Adam([fit],lr=.3)
        start=float(masked_kl(fit,q,v).detach())
        for i in range(32):
            opt.zero_grad();loss=masked_kl(fit,q,v);loss.backward()
            assert torch.isfinite(fit.grad).all()
            cpu.call(i+1);opt.step()
        assert float(masked_kl(fit,q,v).detach())<start*.1
        del fit,opt
        options={**protocol.OPTIONS,'total_steps':4,'warmup_fraction':.25,'U_ramp_fraction':.25,'PAS_confidence':0.,'PAS_cosine':-1.}
        def generated(enabled):
            p=GeneratedDomain(seed=163,size=384,device=device,stage_source=dict(kind='generated',seed=163,domain='REFUGE'));native=build(config['reference'],device,163)
            from ..f5_confirmation_v1.native_qualification import foreground_fixture
            foreground_fixture(native)
            o=dict(options);m=make_model(NativeLRParent(native,163,p.stage_source),o,None,generator(163,1,1,0,'pilot_adapter_initialization'),enabled=enabled).to(device)
            return AnchorRestoreTrainer(m,p,o,execution=permit)
        a=generated(False)
        legacy_root=Path(config['v10_parity_code'])/'experiments/lcrseg'
        legacy=legacy_root/'f5_anchor_guard_v10/model.py'
        ns=dict(__name__='experiments.lcrseg.f5_anchor_guard_v10.v10_frozen_model',__package__='experiments.lcrseg.f5_anchor_guard_v10')
        exec(compile(legacy.read_text(),str(legacy),'exec'),ns)
        b=ns['AnchorGuardTrainer'](copy.deepcopy(a.model),copy.deepcopy(a.provider),a.options,execution=permit)
        legacy=legacy_root/'five_frameworks_v1/train_stage.py'
        old=dict(__name__='experiments.lcrseg.five_frameworks_v1.v10_frozen_stage',__package__='experiments.lcrseg.five_frameworks_v1')
        exec(compile(legacy.read_text(),str(legacy),'exec'),old)
        b.losses=__import__('types').MethodType(old['StageTrainer'].losses,b)
        cuda.wrap(a.optimizer);cuda.wrap(b.optimizer)
        for _ in range(2):a.update();b.update();equal(snapshot(a),snapshot(b))
        del a,b
        t=generated(True);cuda.wrap(t.optimizer);t.update();anchor_state={k:v.detach().clone() for k,v in t.model.entry_anchor.state_dict().items()};before=t.ema.sidecar.r.detach().clone();grads=t.update()
        assert all(t.telemetry[n+'_nonzero_U_calls']>0 for n in ('A','B','R'))
        assert all(torch.equal(v,t.model.entry_anchor.state_dict()[k]) for k,v in anchor_state.items())
        assert not t.model.entry_anchor.training and all(not p.requires_grad and p.grad is None for p in t.model.entry_anchor.parameters())
        assert t.model.entry_anchor is t.ema.entry_anchor
        dummy=torch.zeros(2,3,384,384,device=device)
        with torch.no_grad():
            anchor_probs=t.model.entry_anchor(dummy).softmax(1);labels=(anchor_probs.argmax(1)+1)%3
            forced=torch.nn.functional.one_hot(labels,3).permute(0,3,1,2).float();valid=torch.ones_like(labels,dtype=torch.bool)
        logits=torch.zeros_like(forced,requires_grad=True)
        loss=t.kl_objective(dummy,logits,forced,valid)
        restored_gradient=torch.autograd.grad(loss,logits)[0]
        reference_logits=torch.zeros_like(forced,requires_grad=True)
        reference_loss=t.fine_kl(reference_logits,anchor_probs,valid)
        assert torch.allclose(restored_gradient,torch.autograd.grad(reference_loss,reference_logits)[0],atol=1e-7,rtol=1e-5)
        assert restored_gradient.abs().sum()>0 and t.telemetry['anchor_restored_pixels']>0
        # V10 dropped the same pixels; restoring their soft target creates a corrective student gradient.
        kept=guarded_mask(anchor_probs,forced,valid,options['PAS_confidence'])
        assert not kept.any()
        old_logits=torch.zeros_like(forced,requires_grad=True)
        old_loss=t.fine_kl(old_logits,forced,kept)
        assert old_loss==0 and torch.count_nonzero(torch.autograd.grad(old_loss,old_logits)[0])==0
        assert all(p.grad is None for p in t.ema.parameters())
        assert torch.allclose(t.ema.sidecar.r,before*.99+t.model.sidecar.r.detach()*.01,atol=1e-7,rtol=1e-5)
        groups=t.model.parent.parameter_groups()
        for p in groups['input_factors']+groups['output_factors']+t.model.sidecar.optimizer_parameters():
            gl,gu=grads[id(p)]['L'],grads[id(p)]['U']
            expected=(torch.zeros_like(p) if gl is None else gl)+(0 if gu is None else gu)
            assert torch.allclose(p.grad,expected,atol=1e-7,rtol=1e-5)
        # Validate the same clean feature route consumed by actual class-SWD; no optimizer call or patient data.
        _,clean,_=t.model.parts(torch.zeros(2,3,384,384,device=device),detach_parent=t.model.detach_u_parent(clean=True))
        params=groups['input_factors']+groups['output_factors']+t.model.sidecar.optimizer_parameters()
        clean_grads=torch.autograd.grad(clean.square().mean(),params,allow_unused=True)
        assert all(g is None for g in clean_grads[:-1]) and clean_grads[-1] is not None
        assert not t.model.sidecar.q.requires_grad and not t.model.sidecar.previous.requires_grad
        ident=dict(family='F5',seed=163,order=1,stage=1,arm='ANCHOR_RESTORE');checkpoint.save(t,root/'qualification.pt',ident)
        other=AnchorRestoreTrainer.for_resume(copy.deepcopy(t.model),copy.deepcopy(t.provider),options,execution=permit);other.entry_fingerprint=t.entry_fingerprint;checkpoint.restore(other,root/'qualification.pt',ident);cuda.wrap(other.optimizer);t.update();other.update();equal(snapshot(t),snapshot(other));assert t.telemetry==other.telemetry;assert all(torch.equal(v,t.model.entry_anchor.state_dict()[k]) for k,v in anchor_state.items());del other
        wrong_model=copy.deepcopy(t.model);wrong_model.anchor_restore=False
        wrong=AnchorRestoreTrainer.for_resume(wrong_model,copy.deepcopy(t.provider),options,execution=permit);wrong.entry_fingerprint=t.entry_fingerprint
        try:checkpoint.restore(wrong,root/'qualification.pt',ident)
        except ValueError as e:assert 'semantic' in str(e)
        else:raise AssertionError('changed actual module flag accepted')
        del wrong,wrong_model
        with torch.no_grad():before=t.model(torch.zeros(2,3,384,384,device=device));deployed=t.model.deploy();after=deployed(torch.zeros(2,3,384,384,device=device))
        assert torch.allclose(before,after,atol=2e-5,rtol=2e-5)
        p=GeneratedDomain(seed=163,stage=2,size=384,device=device,stage_source=dict(kind='own_generated_predecessor',seed=163,domain='RIM_ONE_r3'));m=make_model(NativeLRParent(deployed.parent.native,163,p.stage_source),options,deployed.transform.detach().clone(),generator(163,1,2,0,'pilot_adapter_initialization')).to(device)
        nxt=AnchorRestoreTrainer(m,p,options,execution=permit);cuda.wrap(nxt.optimizer);nxt.update();del nxt,m,p,deployed
        try:t.update(fault='after_optimizer')
        except RuntimeError as e:assert str(e)=='injected after_optimizer'
        else:raise AssertionError('expected injected failure')
        assert t.requires_restore and t.step==3;del t
        j=protocol.jobs()[0];t=construct(config,j,1,device,engine.admission(config,'smoke'),allow_u=False);smoke.wrap(t.optimizer);t.update();t.update();assert t.provider.u_reads==0;del t
        assert (cpu.count,cuda.count,smoke.count)==(32,10,2)
        protocol.write(root/'QUALIFICATION.json',dict(status='PASS',execution_commit=config['execution_commit'],plan_id=protocol.plan()['plan_id'],costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count),checks=['two-case routed soft-target overfit; same/different/low-confidence/invalid support boundaries','disabled restoration matches archived V10 native model/EMA/optimizer/cursor/prototypes, including frozen training loss path','actual native A/B/R U gradients and summed L/U; clean SWD parent-detached feature route','immutable shared entry teacher, actual native rejected conflict restored to entry soft target and nonzero corrective gradient; original V10 zero on identical conflict, linear R EMA; checkpoint continuation incl anchor/telemetry and reject wrong actual flag','deployment/own-prefix stage transition','deliberate failed optimizer call accounted','discarded current-L smoke with zero U reads'],qualification_extra_gradient_vjps=4,qualification_extra_anchor_full_forwards=2,matched_control_reuse_qualified=True,matched_V10_reuse_qualified=True,historical_B2_reuse='unchanged previous parity retained; historical only'))
    except BaseException as e:
        protocol.write(root/'QUALIFICATION.json',dict(status='FAIL',execution_commit=config['execution_commit'],costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count)));protocol.write(root/'QUALIFICATION_FAILURE.private.json',dict(error=repr(e),traceback=__import__('traceback').format_exc()));raise
