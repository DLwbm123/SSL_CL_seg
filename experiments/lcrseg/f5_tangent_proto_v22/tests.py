import copy
from pathlib import Path
import torch
from . import protocol
from .run import engine,HierarchyProtoTrainer,construct,make_model
from ..f5_class_reference_v13.model import class_reference
from .math_checks import check
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
        math_result=check(cpu)
        options={**protocol.OPTIONS,'total_steps':4,'warmup_fraction':.25,'U_ramp_fraction':.25,'PAS_confidence':0.,'PAS_cosine':-1.}
        def generated(enabled):
            p=GeneratedDomain(seed=163,size=384,device=device,stage_source=dict(kind='generated',seed=163,domain='REFUGE'));native=build(config['reference'],device,163)
            from ..f5_confirmation_v1.native_qualification import foreground_fixture
            foreground_fixture(native)
            o=dict(options);m=make_model(NativeLRParent(native,163,p.stage_source),o,None,generator(163,1,1,0,'pilot_adapter_initialization'),enabled=enabled).to(device)
            return HierarchyProtoTrainer(m,p,o,execution=permit)
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
        t=generated(True);cuda.wrap(t.optimizer);t.update();anchor_state={k:v.detach().clone() for k,v in t.model.entry_anchor.state_dict().items()};before=t.ema.sidecar.r.detach().clone();grads=t.update();native_cases=[copy.deepcopy(t.module_stats)]
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
        groups=t.model.parent.parameter_groups()
        # Exercise actual native merged-gradient projection with a forced conflicting direction.
        t.capture_qualification_terms=True
        labeled,unlabeled,_=t.losses()
        params=[p for p in t.model.parameters() if p.requires_grad]
        hg=torch.autograd.grad(sum(t.qualification_terms.values()),params,allow_unused=True,retain_graph=True)
        position={id(p):i for i,p in enumerate(params)}
        hierarchy_gradients={name:sum(float(hg[position[id(p)]].square().sum()) for p in group if hg[position[id(p)]] is not None)**.5
                            for name,group in [('A',groups['input_factors']),('B',groups['output_factors']),('R',t.model.sidecar.optimizer_parameters())]}
        for key in ('coarse','parts'):
            gg=torch.autograd.grad(t.qualification_terms[key],params,allow_unused=True,retain_graph=True)
            assert sum(float(v.square().sum()) for v in gg if v is not None)>0
        assert all(p.grad is None for p in t.ema.parameters())
        t.capture_qualification_terms=False;t.qualification_terms=None
        assert not t.model._capture_hierarchy and not t.model._labeled_features
        t.gradient_audit(labeled,unlabeled)
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
        # Same clean feature route remains R-only; no new patient probe or optimizer call.
        _,clean,_=t.model.parts(torch.zeros(2,3,384,384,device=device),detach_parent=t.model.detach_u_parent(clean=True))
        clean_grads=torch.autograd.grad(clean.square().mean(),groups['input_factors']+groups['output_factors']+t.model.sidecar.optimizer_parameters(),allow_unused=True)
        assert all(g is None for g in clean_grads[:-1]) and clean_grads[-1] is not None
        assert not t.model.sidecar.q.requires_grad and not t.model.sidecar.previous.requires_grad
        ident=dict(family='F5',seed=163,order=1,stage=1,arm='TANGENT_PROTO');checkpoint.save(t,root/'qualification.pt',ident)
        other=HierarchyProtoTrainer.for_resume(copy.deepcopy(t.model),copy.deepcopy(t.provider),options,execution=permit);other.entry_fingerprint=t.entry_fingerprint;checkpoint.restore(other,root/'qualification.pt',ident);cuda.wrap(other.optimizer);t.update();native_cases.append(copy.deepcopy(t.module_stats));other.update();equal(snapshot(t),snapshot(other));assert t.telemetry==other.telemetry;assert all(torch.equal(v,t.model.entry_anchor.state_dict()[k]) for k,v in anchor_state.items());del other
        for attr,value in [('hierarchy',False),('parent_map',(1,0,0)),('hierarchy_weight',.2),('hierarchy_temperature',.2),('hierarchy_margin',.5)]:
            wrong_model=copy.deepcopy(t.model);setattr(wrong_model,attr,value)
            wrong=HierarchyProtoTrainer.for_resume(wrong_model,copy.deepcopy(t.provider),options,execution=permit);wrong.entry_fingerprint=t.entry_fingerprint
            try:checkpoint.restore(wrong,root/'qualification.pt',ident)
            except ValueError as e:assert 'semantic' in str(e)
            else:raise AssertionError('changed actual hierarchy setting accepted')
            del wrong,wrong_model
        with torch.no_grad():before=t.model(torch.zeros(2,3,384,384,device=device));deployed=t.model.deploy();after=deployed(torch.zeros(2,3,384,384,device=device))
        assert torch.allclose(before,after,atol=2e-5,rtol=2e-5)
        p=GeneratedDomain(seed=163,stage=2,size=384,device=device,stage_source=dict(kind='own_generated_predecessor',seed=163,domain='RIM_ONE_r3'));m=make_model(NativeLRParent(deployed.parent.native,163,p.stage_source),options,deployed.transform.detach().clone(),generator(163,1,2,0,'pilot_adapter_initialization')).to(device)
        nxt=HierarchyProtoTrainer(m,p,options,execution=permit);cuda.wrap(nxt.optimizer);nxt.update();del nxt,m,p,deployed
        try:t.update(fault='after_optimizer')
        except RuntimeError as e:assert str(e)=='injected after_optimizer'
        else:raise AssertionError('expected injected failure')
        assert t.requires_restore and t.step==3;del t
        j=protocol.jobs()[0];t=construct(config,j,1,device,engine.admission(config,'smoke'),allow_u=False);smoke.wrap(t.optimizer);t.update();t.update();assert t.provider.u_reads==0;del t
        assert (cpu.count,cuda.count,smoke.count)==(32,10,2)
        formal_advancement=all(v['hierarchy_stats']['coarse_classes']>=2 and v['hierarchy_stats']['fine_groups']>0 and v['hierarchy_stats']['tangent_supported_children']>=2 for v in native_cases) and hierarchy_gradients['A']>0 and hierarchy_gradients['B']>0 and hierarchy_gradients['R']>0
        protocol.write(root/'QUALIFICATION.json',dict(status='PASS',execution_commit=config['execution_commit'],plan_id=protocol.plan()['plan_id'],costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count),formal_advancement=formal_advancement,
            checks=['label-generic current-L hierarchy, detached targets/live alignment, bounded 32CPU two-case fit','disabled archived V13 native full-state and frozen loss/update parity','one unchanged V13 raw halfspace; hierarchy gradients A/B/R and unforced coarse/sibling activation; main U and SWD routes retained','shared frozen entry/EMA, linear R EMA, original rejected main KL and active reference','checkpoint continuation and flag rejection; deployment own-prefix next stage; deliberate failed optimizer ledger; current L zeroU smoke'],
            math_result=math_result,native_hierarchy_cases=native_cases,hierarchy_gradient_norms=hierarchy_gradients,
            qualification_reference_vjps=5,qualification_extra_gradient_vjps=6,qualification_extra_cpu_vjps=6,
            qualification_extra_generated_loss_calls=1,qualification_extra_anchor_full_forwards=2,
            matched_V13_reuse_qualified=True,historical_B2_reuse='unchanged route/data parity; historical only'))
    except BaseException as e:
        protocol.write(root/'QUALIFICATION.json',dict(status='FAIL',execution_commit=config['execution_commit'],costs=dict(cpu_optimizer_updates=cpu.count,synthetic_cuda_updates=cuda.count,real_L_smoke_updates=smoke.count)));protocol.write(root/'QUALIFICATION_FAILURE.private.json',dict(error=repr(e),traceback=__import__('traceback').format_exc()));raise
