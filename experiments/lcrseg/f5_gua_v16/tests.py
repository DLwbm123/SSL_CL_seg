import copy
from pathlib import Path
import torch
from . import protocol
from .run import engine,GUATrainer,construct,make_model
from ..f5_entry_gradient_v12.model import halfspace
from .model import adam_metric
from ..f5_gua_candidate.alignment import align_displacement
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
        # New two-case actual-Adam check; reuse previously passed solver algebra checks.
        fit=torch.nn.Parameter(torch.tensor([[.5,0.],[-.5,0.]],dtype=torch.double))
        opt=torch.optim.Adam([fit],lr=.05);target=torch.tensor([[1.5,1.],[-1.5,1.]],dtype=torch.double)
        initial=fit.detach().clone();start=float((fit-target).square().sum().detach());projected_calls=0
        for i in range(32):
            opt.zero_grad();g,=torch.autograd.grad(.5*(fit-target).square().sum(),fit)
            h=torch.zeros_like(fit);h[:,0]=fit[:,0].detach()
            fit.grad=halfspace([g],[h])[0][0];before=fit.detach().clone()
            cpu.call(i+1);opt.step();state=copy.deepcopy(opt.state_dict())
            corrected,st=align_displacement([before-fit.detach()],[ [h] ],adam_metric(opt,[fit]))
            with torch.no_grad():fit.copy_(before-corrected[0])
            equal(state,opt.state_dict());projected_calls+=st['projected']
            if i==0:opt.state[fit]['exp_avg'][:,0]=-initial[:,0]
        assert projected_calls>0 and torch.allclose(fit[:,0],initial[:,0],atol=1e-12,rtol=0)
        assert float((fit-target).square().sum().detach())<start*.65
        del fit,opt
        options={**protocol.OPTIONS,'total_steps':7,'warmup_fraction':1/7,'U_ramp_fraction':.25,'PAS_confidence':0.,'PAS_cosine':-1.}
        def generated(enabled):
            p=GeneratedDomain(seed=163,size=384,device=device,stage_source=dict(kind='generated',seed=163,domain='REFUGE'));native=build(config['reference'],device,163)
            from ..f5_confirmation_v1.native_qualification import foreground_fixture
            foreground_fixture(native)
            o=dict(options);m=make_model(NativeLRParent(native,163,p.stage_source),o,None,generator(163,1,1,0,'pilot_adapter_initialization'),enabled=enabled).to(device)
            return GUATrainer(m,p,o,execution=permit)
        a=generated(False)
        legacy_root=Path(config['v13_parity_code'])/'experiments/lcrseg'
        legacy=legacy_root/'f5_class_reference_v13/model.py'
        ns=dict(__name__='experiments.lcrseg.f5_class_reference_v13.v13_frozen_model',__package__='experiments.lcrseg.f5_class_reference_v13')
        exec(compile(legacy.read_text(),str(legacy),'exec'),ns)
        b=ns['ClassReferenceTrainer'](copy.deepcopy(a.model),copy.deepcopy(a.provider),a.options,execution=permit)
        legacy=legacy_root/'five_frameworks_v1/train_stage.py'
        old=dict(__name__='experiments.lcrseg.five_frameworks_v1.v10_frozen_stage',__package__='experiments.lcrseg.five_frameworks_v1')
        exec(compile(legacy.read_text(),str(legacy),'exec'),old)
        b.losses=__import__('types').MethodType(old['StageTrainer'].losses,b)
        b._update=__import__('types').MethodType(old['StageTrainer']._update,b)
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
        # Main KL remains V10 rejection, while the entry reference retains a real gradient.
        assert loss==0 and torch.count_nonzero(torch.autograd.grad(loss,logits,retain_graph=True)[0])==0
        ref_logits,=torch.autograd.grad(t.entry_reference_loss,logits)
        assert ref_logits.abs().sum()>0
        assert all(p.grad is None for p in t.ema.parameters())
        assert torch.allclose(t.ema.sidecar.r,before*.99+t.model.sidecar.r.detach()*.01,atol=1e-7,rtol=1e-5)
        natural=[dict(t.module_stats)]
        t.update();natural.append(dict(t.module_stats))
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
        assert sum(float(p.grad.square().sum()) for p in params)<1e-12
        assert t.entry_reference_grads is None
        # One counted native Adam invocation with injected historical moments.
        # Existing generated current batch is reused; this is a construction, not a natural frequency.
        t.optimizer.zero_grad(set_to_none=True)
        effective_before=[a.effective_weight().detach().clone() for a in t.model.parent.adapters]
        from unittest.mock import patch
        from . import model as implementation
        captured={}
        def observe(proposal,references,metric):
            captured['optimizer']=copy.deepcopy(t.optimizer.state_dict())
            captured['effective_proposed']=[a.effective_weight().detach().clone() for a in t.model.parent.adapters]
            return align_displacement(proposal,references,metric)
        original_step=t.optimizer.step
        by_id={id(p):h for p,h in zip(params,refs)}
        ref_norm=sum(h.double().square().sum() for h in refs).sqrt()
        def injected_momentum_step(*a,**kw):
            # Set a bounded -1e-3 normalized-reference proposal, through a real Adam call.
            with torch.no_grad():
                for group in t.optimizer.param_groups:
                    beta1,beta2=group['betas']
                    for p in group['params']:
                        st=t.optimizer.state[p];k=float(st['step'])+1
                        grad=p.grad+group['weight_decay']*p
                        vnext=beta2*st['exp_avg_sq']+(1-beta2)*grad.square()
                        den=(vnext/(1-beta2**k)).sqrt()+group['eps']
                        wanted=-1e-3*by_id[id(p)]/ref_norm
                        mnext=wanted*den*(1-beta1**k)/group['lr']
                        st['exp_avg'].copy_((mnext-(1-beta1)*grad)/beta1)
            return original_step(*a,**kw)
        with patch.object(implementation,'align_displacement',observe),patch.object(t.optimizer,'step',injected_momentum_step):t.update()
        equal(captured['optimizer'],t.optimizer.state_dict())
        assert t.module_stats['gua_proposal_dot']<0 and t.module_stats['gua_projected']
        assert t.module_stats['gua_actual_cosine']>=-1e-5
        native_diagnostic=dict(constructed=True,**t.module_stats,
            effective_proposal_change_norm=sum(float((a-b).double().square().sum()) for a,b in zip(captured['effective_proposed'],effective_before))**.5,
            effective_final_change_norm=sum(float((a.effective_weight().detach()-b).double().square().sum()) for a,b in zip(t.model.parent.adapters,effective_before))**.5)
        # Same clean feature route remains R-only; no new patient probe or optimizer call.
        _,clean,_=t.model.parts(torch.zeros(2,3,384,384,device=device),detach_parent=t.model.detach_u_parent(clean=True))
        clean_grads=torch.autograd.grad(clean.square().mean(),groups['input_factors']+groups['output_factors']+t.model.sidecar.optimizer_parameters(),allow_unused=True)
        assert all(g is None for g in clean_grads[:-1]) and clean_grads[-1] is not None
        assert not t.model.sidecar.q.requires_grad and not t.model.sidecar.previous.requires_grad
        ident=dict(family='F5',seed=163,order=1,stage=1,arm='GUA');checkpoint.save(t,root/'qualification.pt',ident)
        other=GUATrainer.for_resume(copy.deepcopy(t.model),copy.deepcopy(t.provider),options,execution=permit);other.entry_fingerprint=t.entry_fingerprint;checkpoint.restore(other,root/'qualification.pt',ident);cuda.wrap(other.optimizer);t.update();other.update();equal(snapshot(t),snapshot(other));assert t.telemetry==other.telemetry;assert all(torch.equal(v,t.model.entry_anchor.state_dict()[k]) for k,v in anchor_state.items());del other
        wrong_model=copy.deepcopy(t.model);wrong_model.gua=False
        wrong=GUATrainer.for_resume(wrong_model,copy.deepcopy(t.provider),options,execution=permit);wrong.entry_fingerprint=t.entry_fingerprint
        try:checkpoint.restore(wrong,root/'qualification.pt',ident)
        except ValueError as e:assert 'semantic' in str(e)
        else:raise AssertionError('changed actual module flag accepted')
        del wrong,wrong_model
        with torch.no_grad():before=t.model(torch.zeros(2,3,384,384,device=device));deployed=t.model.deploy();after=deployed(torch.zeros(2,3,384,384,device=device))
        assert torch.allclose(before,after,atol=2e-5,rtol=2e-5)
        p=GeneratedDomain(seed=163,stage=2,size=384,device=device,stage_source=dict(kind='own_generated_predecessor',seed=163,domain='RIM_ONE_r3'));m=make_model(NativeLRParent(deployed.parent.native,163,p.stage_source),options,deployed.transform.detach().clone(),generator(163,1,2,0,'pilot_adapter_initialization')).to(device)
        nxt=GUATrainer(m,p,options,execution=permit);cuda.wrap(nxt.optimizer);nxt.update();del nxt,m,p,deployed
        try:t.update(fault='after_optimizer')
        except RuntimeError as e:assert str(e)=='injected after_optimizer'
        else:raise AssertionError('expected injected failure')
        assert t.requires_restore and t.step==5;del t
        j=protocol.jobs()[0];t=construct(config,j,1,device,engine.admission(config,'smoke'),allow_u=False);smoke.wrap(t.optimizer);t.update();t.update();assert t.provider.u_reads==0;del t
        assert (cpu.count,cuda.count,smoke.count)==(32,12,2)
        advance=any(v['gua_raw_cosine']>=-1e-6 and v['gua_proposal_dot']<0 for v in natural)
        protocol.write(root/'QUALIFICATION.json',dict(status='PASS',execution_commit=config['execution_commit'],plan_id=protocol.plan()['plan_id'],costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count),formal_advancement=advance,
            checks=['32 actual Adam two-case generated CPU updates; direction corrected and moments unchanged','disabled complete native archived V13 state/old update and loss parity','actual native A/B/R reference and raw projection retained; main KL rejection; clean SWD parent detach','natural generated-native proposals recorded separately from injected historical-moment native Adam construction; effective BA changes measured','shared immutable entry and EMA frozen; linear R EMA; real checkpoint continuation telemetry and wrong actual flag rejection','deploy own-prefix next stage; after-optimizer failure counted and blocked pending restore; zeroU L smoke'],
            natural_native_diagnostics=natural,constructed_native_diagnostic=native_diagnostic,
            qualification_reference_vjps=7,qualification_extra_gradient_vjps=3,qualification_extra_generated_loss_calls=1,qualification_extra_anchor_full_forwards=2,
            matched_V13_reuse_qualified=True,historical_B2_reuse='historical only; unchanged prior route/data parity'))
    except BaseException as e:
        protocol.write(root/'QUALIFICATION.json',dict(status='FAIL',execution_commit=config['execution_commit'],costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count)));protocol.write(root/'QUALIFICATION_FAILURE.private.json',dict(error=repr(e),traceback=__import__('traceback').format_exc()));raise
