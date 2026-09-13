import copy,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.lcrseg.five_frameworks_v1.model import Model
from experiments.lcrseg.five_frameworks_v1.parent_bridge import SyntheticParentBridge
from experiments.lcrseg.five_frameworks_v1.recipes import SyntheticCurrentDomain
from experiments.lcrseg.five_frameworks_v1.train_stage import StageTrainer
from experiments.lcrseg.five_frameworks_v1.checkpoint import save,restore,atomic_save,PhysicalLedger
from experiments.lcrseg.five_frameworks_v1.kernels import gradient_seed_basis,InsufficientProbeSupport
from experiments.lcrseg.five_frameworks_v1.registry import FAMILIES,BASELINES
from experiments.lcrseg.five_frameworks_v1.evaluate import segmentation_metrics
from experiments.lcrseg.tests.five_frameworks_v1.test_integration import assert_state_equal,trainer

OPTIONS={'total_steps':6,'PAS_confidence':0.,'PAS_cosine':-1.}


def pack(t):
    return {'student':t.model.state_dict(),'EMA':t.ema.state_dict(),'optimizer':t.optimizer.state_dict(),
            'scheduler':t.scheduler.state_dict(),'scaler':t.scaler.state_dict(),'probes':t.probe,
            'prototypes':t.prototypes.values,'support':t.prototypes.support,'cursor':t.cursor,'step':t.step,
            'flags':{n:p.requires_grad for n,p in t.model.named_parameters()},
            'modes':{n:m.training for n,m in t.model.named_modules()},
            'EMA_modes':{n:m.training for n,m in t.ema.named_modules()},
            'torch_rng':torch.get_rng_state(),'augmentation_rng':t.rng('UL').get_state(),
            'reads':[t.provider.l_reads,t.provider.u_reads],'telemetry':t.telemetry}


@pytest.mark.parametrize('family',FAMILIES+BASELINES)
def test_stage2_independent_process_resume_complete_state(family,cwmi,tmp_path):
    torch.manual_seed(41)
    size=128 if family=='F2' else 16
    source=StageTrainer(Model(SyntheticParentBridge(),family),SyntheticCurrentDomain(size=size),OPTIONS,cwmi)
    source.update();source.update();entry=source.model.deploy()
    atomic_save({'synthetic_only':True,'parent':entry.parent.state_dict(),'F':entry.transform.detach()},tmp_path/'entry.pt')
    parent=copy.deepcopy(entry.parent)
    a=StageTrainer(Model(parent,family,previous=entry.transform.detach()),SyntheticCurrentDomain(stage=2,size=size),OPTIONS,cwmi)
    a.update();a.update();identity={'family':family,'stage':2}
    save(a,tmp_path/'resume.pt',identity)
    a.update();a.update();expected=pack(a)
    # Neutral child command line; all task paths and source travel over stdin.
    code='''import os,json,torch
from pathlib import Path
from experiments.lcrseg.tests.five_frameworks_v1.cost_audit import install
install()
from experiments.lcrseg.tests.five_frameworks_v1.test_revision_r2 import pack,OPTIONS
from experiments.lcrseg.five_frameworks_v1.losses import CWMI
from experiments.lcrseg.five_frameworks_v1.parent_bridge import SyntheticParentBridge
from experiments.lcrseg.five_frameworks_v1.model import Model
from experiments.lcrseg.five_frameworks_v1.recipes import SyntheticCurrentDomain
from experiments.lcrseg.five_frameworks_v1.train_stage import StageTrainer
from experiments.lcrseg.five_frameworks_v1.checkpoint import restore
torch.set_num_threads(1)
root=Path(ROOT);e=torch.load(root/'entry.pt',weights_only=True);assert e['synthetic_only']
p=SyntheticParentBridge();p.load_state_dict(e['parent']);p.eval().requires_grad_(False)
m=Model.for_resume(p,FAMILY,previous=e['F']);assert m.spectral.get('eigensolves',0)==0
cw=CWMI(Path(os.environ['SSLCL5_DEP_ROOT'])/'CWMI')
t=StageTrainer.for_resume(m,SyntheticCurrentDomain(stage=2,size=SIZE),OPTIONS,cw)
assert t.provider.l_reads==0 and t.telemetry['probe_vjps']==0
restore(t,root/'resume.pt',{'family':FAMILY,'stage':2})
t.update();t.update();torch.save(pack(t),root/'actual.pt')
'''
    code='ROOT='+repr(str(tmp_path))+'\nFAMILY='+repr(family)+'\nSIZE='+repr(size)+'\n'+code
    result=subprocess.run([sys.executable,'-'],input=code,text=True,capture_output=True,env=os.environ.copy())
    assert result.returncode==0,result.stderr
    actual=torch.load(tmp_path/'actual.pt',weights_only=False)
    assert_state_equal(expected,actual)


@pytest.mark.parametrize('key,value',[('lambda_U',0.),('lambda_JML',1.),('lambda_structure',1.),('lambda_shape',1.),
 ('lambda_SWD',.2),('total_steps',8),('warmup_fraction',.1),('U_ramp_fraction',.1),('lr',.02)])
def test_resume_semantic_changes_rejected_before_mutation(tmp_path,key,value):
    a=trainer();a.update();save(a,tmp_path/'a.pt',{'family':'F1'})
    b=trainer(initialize=False,options={key:value});before=copy.deepcopy(pack(b))
    with pytest.raises(ValueError,match='semantic'):restore(b,tmp_path/'a.pt',{'family':'F1'})
    assert_state_equal(before,pack(b))


@pytest.mark.parametrize('change',[{'seed':162},{'order':2},{'stage':2},{'manifest_id':'different'},
                                  {'split_id':'changed'},{'stage_source':{'kind':'wrong source'}},{'size':18}])
def test_provider_actual_binding_rejected(tmp_path,change):
    a=trainer();save(a,tmp_path/'a.pt',{'family':'F1'})
    torch.manual_seed(4)
    b=StageTrainer.for_resume(Model.for_resume(SyntheticParentBridge(),'F1'),SyntheticCurrentDomain(**change),a.options)
    with pytest.raises(ValueError):restore(b,tmp_path/'a.pt',{'family':'F1'})
    assert b.step==0 and b.provider.l_reads==0


def test_resume_family_rank_parent_entry_and_nonsemantic_whitelist(tmp_path):
    a=trainer();save(a,tmp_path/'a.pt',{'family':'F1'})
    for family,ratio in [('F3',.5),('F1',.25)]:
        torch.manual_seed(4)
        b=StageTrainer.for_resume(Model.for_resume(SyntheticParentBridge(),family,ratio),SyntheticCurrentDomain(),a.options)
        with pytest.raises(ValueError):restore(b,tmp_path/'a.pt',{'family':'F1'})
    torch.manual_seed(5)
    b=StageTrainer.for_resume(Model.for_resume(SyntheticParentBridge(),'F1'),SyntheticCurrentDomain(),a.options)
    with pytest.raises(ValueError):restore(b,tmp_path/'a.pt',{'family':'F1'})
    b=trainer(initialize=False,options={'output_path':'not-semantic','log_every':2})
    restore(b,tmp_path/'a.pt',{'family':'F1'})
    assert b.cursor==0


@pytest.mark.parametrize('kind',['optimizer','constraint','optimizer_state','effective_overflow'])
def test_post_update_invalid_state_uncommitted_and_resume(kind,tmp_path):
    t=trainer();path=tmp_path/'entry.pt';save(t,path,{'family':'F1'})
    ledger=PhysicalLedger(tmp_path/'physical.jsonl');teacher=copy.deepcopy(t.ema.state_dict());proto=t.prototypes.values.clone()
    # Explicit fault injection only; these replacements never enter production.
    step=t.optimizer.step
    def invalid_step(*a,**kw):
        value=step(*a,**kw)
        with torch.no_grad():
            if kind=='optimizer':t.model.parent.adapters[-1].b.fill_(float('inf'))
            if kind=='optimizer_state':next(iter(t.optimizer.state.values()))['exp_avg'].fill_(float('nan'))
            if kind=='effective_overflow':
                t.model.sidecar.previous  # shared frozen matrix is not mutated
                t.model.parent.adapters[-1].a.fill_(1e30);t.model.parent.adapters[-1].b.fill_(1e30)
        return value
    if kind=='constraint':
        def bad_constraint():
            with torch.no_grad():t.model.parent.adapters[-1].b.fill_(float('inf'))
        t.model.parent.apply_constraints=bad_constraint
    else:t.optimizer.step=invalid_step
    with pytest.raises(FloatingPointError):t.update(physical=ledger.append)
    assert t.step==t.cursor==0 and ledger.summary(0)['physical_optimizer_updates']==1
    assert_state_equal(teacher,t.ema.state_dict());assert torch.equal(proto,t.prototypes.values)
    with pytest.raises(RuntimeError,match='restore'):t.update()
    good=trainer(initialize=False);restore(good,path,{'family':'F1'});good.update(physical=ledger.append)
    assert ledger.summary(1)['physical_retry_or_uncommitted_updates']==1


def test_skip_then_fault_no_scientific_commit(tmp_path):
    t=trainer();before=copy.deepcopy(pack(t));ledger=PhysicalLedger(tmp_path/'physical.jsonl')
    t.update(skip=True,physical=ledger.append)
    assert t.cursor==0 and ledger.summary(0)['physical_optimizer_updates']==0
    assert_state_equal(before['EMA'],t.ema.state_dict())
    with pytest.raises(RuntimeError):t.update(fault='after_optimizer',physical=ledger.append)
    assert t.step==0 and ledger.summary(0)['physical_optimizer_updates']==1


@pytest.mark.parametrize('invalid',['double_identity','wrong_shape','asymmetric','nan','bad_gradient'])
def test_F2_invalid_probe_propagates_without_any_A_mutation(invalid):
    class ToyStructure:
        def __call__(self,p,y):return (p[:,1]-(y==1).to(p)).square().mean()
    torch.manual_seed(6);p=SyntheticParentBridge();old=[a.a.detach().clone() for a in p.adapters]
    if invalid=='bad_gradient':
        with pytest.raises(ValueError):gradient_seed_basis([torch.full((4,4),float('nan'))],2)
        return
    matrices={'double_identity':2*torch.eye(4),'wrong_shape':torch.eye(3),
              'asymmetric':torch.triu(torch.ones(4,4)),'nan':torch.full((4,4),float('nan'))}
    p.adapters[-1].free_projector=matrices[invalid]
    t=StageTrainer.for_resume(Model(p,'F2'),SyntheticCurrentDomain(),cwmi=ToyStructure())
    with pytest.raises(ValueError):t.prepare_stage()
    assert not t.probe['complete']
    assert all(torch.equal(a.a,v) for a,v in zip(p.adapters,old))


def test_ignore_support_and_auxiliary_metric_contract():
    target=np.array([[1,255],[0,0]]);a=np.array([[1,0],[0,0]]);b=np.array([[1,2],[0,0]])
    ma=segmentation_metrics(a,target);mb=segmentation_metrics(b,target)
    assert ma==mb and ma['rim']['Dice']==1 and ma['rim']['HD95'] is None
    empty=segmentation_metrics(a,np.full((2,2),255))
    assert empty['rim']['Dice'] is None and empty['rim']['valid_pixels']==0
    with pytest.raises(ValueError):segmentation_metrics(a,np.zeros((3,3)))
    with pytest.raises(ValueError):segmentation_metrics(a,np.full((2,2),3))


def test_CWMI_device_local_masks_and_CPU_author_parity(cwmi,dependency_root):
    from experiments.lcrseg.five_frameworks_v1.losses import MissingBackend
    p=cwmi.pyramid_for(torch.device('cpu'))
    assert p is cwmi.pyramid_for(torch.device('cpu'))
    masks=p.get_mask((128,128))
    assert all(v is None or v.device.type=='cpu' for values in masks.values() for v in values)
    import importlib.util
    path=dependency_root/'CWMI/model/CWMI_loss/ComplexSteerablePyramid.py'
    spec=importlib.util.spec_from_file_location('author_device_parity',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.ComplexSteerablePyramid(complex=True,N=2,K=4,device='cpu')
    x=torch.rand(1,2,128,128,requires_grad=True)
    a=p(x);b=original(x)
    assert all(torch.equal(u,v) for u,v in zip(a,b))
    ga,=torch.autograd.grad(sum(v.abs().sum() for v in a),x,retain_graph=True)
    gb,=torch.autograd.grad(sum(v.abs().sum() for v in b),x)
    assert torch.equal(ga,gb)
    with pytest.raises(MissingBackend):cwmi.pyramid_for('meta')


@pytest.mark.parametrize('family',FAMILIES)
def test_behavioral_bridge_modes_all_paths(family,cwmi):
    class BehavioralParent(SyntheticParentBridge):
        def __init__(self):
            super().__init__();self.calls=[]
        def features(self,x,mode='student',overrides=None):
            assert self.training==(mode=='student')
            if mode!='student':assert not any(p.requires_grad for p in self.parameters())
            self.calls.append(('features',mode))
            h=super().features(x,mode,overrides)
            return h+{'student':0.,'teacher':.2,'eval':.4}[mode]
        def native_readout(self,h,shape,mode='student'):
            assert self.training==(mode=='student')
            self.calls.append(('readout',mode))
            logits=super().native_readout(h,shape,mode)
            return logits+logits.new_tensor([0.,{'student':0.,'teacher':.3,'eval':.6}[mode],0.])[None,:,None,None]
    p=BehavioralParent();size=128 if family=='F2' else 16
    t=StageTrainer(Model(p,family),SyntheticCurrentDomain(size=size),OPTIONS,cwmi)
    assert t.ema.parent.calls[-1]==('features','teacher')
    t.ema.parent.calls.clear();t.model.parent.calls.clear()
    t.update();t.update();t.update()
    assert set(mode for _,mode in t.ema.parent.calls)=={'teacher'}
    assert set(mode for _,mode in t.model.parent.calls)=={'student'}
    assert ('readout','teacher') in t.ema.parent.calls
    deploy=t.model.deploy();deploy.parent.calls.clear();deploy(torch.randn(2,3,size,size))
    assert set(mode for _,mode in deploy.parent.calls)=={'eval'}
    with pytest.raises(ValueError):t.model(torch.randn(2,3,size,size),mode='teacher')


def test_native_bridge_delegates_actual_named_parameters_and_geometry():
    from experiments.lcrseg.five_frameworks_v1.integration import NativeParentBridge
    native=SyntheticParentBridge();byid={id(p):n for n,p in native.named_parameters()}
    groups={k:[byid[id(p)] for p in values] for k,values in native.parameter_groups().items()}
    hooks={'features':lambda n,x,mode,overrides:n.features(x,mode,overrides),
           'readout':lambda n,h,shape,mode:n.native_readout(h,shape,mode),
           'readout_kernel':lambda n:n.effective_readout_kernel_at_entry(),
           'supervised':lambda n,p,y:n.supervised(p,y),'constraint_loss':lambda n:n.constraint_loss(),
           'apply_constraints':lambda n:n.apply_constraints(),'stage_entry':lambda n:n.stage_entry(),
           'stage_exit':lambda n:n.stage_exit(),'configure_modes':lambda n,mode:n.configure_stage_training(),
           'output_to_feature':lambda n,x,shape,cat:n.output_to_feature(x,shape,cat),
           'feature_to_output':lambda n,x,shape:n.feature_to_output(x,shape),
           'optimizer_groups':lambda n,options:n.optimizer_groups(options)}
    metadata={'source_identity':'SYNTHETIC_CALLBACK_FIXTURE_ONLY','delta_order':'B @ A',
              'constraint_kind':'none_verified','parameter_names':groups,'feature_width':4}
    bridge=NativeParentBridge(native,metadata,hooks);bridge.configure_stage_training()
    x=torch.rand(2,3,16,16)
    assert torch.equal(bridge.native_readout(bridge.features(x,'student'),(16,16),'student'),native.native(x))
    assert bridge.parameter_groups()['input_factors'][0] is native.adapters[0].a
    assert not native.readout.weight.requires_grad
    before=copy.deepcopy(native.state_dict())
    with pytest.raises(RuntimeError):StageTrainer(Model(bridge,'F1'),SyntheticCurrentDomain())
    assert_state_equal(before,native.state_dict())
    with pytest.raises(ValueError):NativeParentBridge(native,{**metadata,'delta_order':'A @ B'},hooks)


def test_manifest_U_cannot_receive_GT_and_preflight_unbound(tmp_path):
    from experiments.lcrseg.five_frameworks_v1.integration import validate_current_manifest,CurrentDomainDataAdapter,ExecutionPermit,_PERMIT_SEAL,preflight
    from experiments.lcrseg.five_frameworks_v1.gate import ReviewRequired
    manifest={'domain':'synthetic_current','seed':161,'order':1,'stage':1,'split_id':'fixture',
              'L':[{'domain':'synthetic_current','image':'l-image','label':'l-label','patient_id':'toy-patient'}],
              'U':[{'domain':'synthetic_current','image':'u-image','geometry':'u-geometry','source_id':'toy-source'}]}
    fingerprint=validate_current_manifest(manifest)
    calls=[]
    permit=ExecutionPermit({'authorized_manifest_digests':[fingerprint]},('B',),{},_PERMIT_SEAL)
    adapter=CurrentDomainDataAdapter(manifest,permit,lambda *a:calls.append(('L',a)),lambda *a:calls.append(('U',a)))
    adapter.unlabeled(0);assert calls==[('U',('u-image','u-geometry','toy-source'))]
    bad=copy.deepcopy(manifest);bad['U'][0]['label']='must-not-be-read'
    with pytest.raises(ValueError):validate_current_manifest(bad)
    bad=copy.deepcopy(manifest);bad['L'][0]['domain']='future'
    with pytest.raises(ValueError):validate_current_manifest(bad)
    forged=ExecutionPermit(permit.bindings,('B',),{},object())
    with pytest.raises(ReviewRequired):CurrentDomainDataAdapter(manifest,forged,None,None)
    review=tmp_path/'experiments/lcrseg/docs/five_frameworks_v1/review';review.mkdir(parents=True)
    (review/'PARENT_BINDING.json').write_text(json.dumps({'status':'BOUND_VERIFIED','real_parent_identity':'manual-edit-cannot-register-code'}))
    with pytest.raises(ReviewRequired,match='PARENT_BINDING_REQUIRED'):preflight(tmp_path,tmp_path/'absent-approval',['B'])


def test_finite_DAG_metadata_ready_without_dispatch():
    from experiments.lcrseg.five_frameworks_v1.integration import ready_nodes
    from experiments.lcrseg.five_frameworks_v1.planner import dag
    plan=json.loads(Path('experiments/lcrseg/docs/five_frameworks_v1/delivery/configs/study_plan.json').read_text())
    nodes=dag(plan);ready=ready_nodes(nodes,{})
    assert len(ready)==4 and all(n['kind']=='source' for n in ready)
    receipts={n['id']:{'node_id':n['id'],'status':'SEALED'} for n in ready}
    ready=ready_nodes(nodes,receipts)
    assert len(ready)==8 and all(n['family']=='PARENT' and n['stage']==1 for n in ready)
    with pytest.raises(ValueError):ready_nodes(nodes,{'SELECT_PARENT':{'node_id':'SELECT_PARENT','status':'RESOLVED'}})


def test_partial_optimizer_exception_retains_physical_invocation(tmp_path):
    t=trainer();before=copy.deepcopy(t.ema.state_dict());original=t.optimizer.step
    def partial():original();raise RuntimeError('injected partial optimizer failure')
    t.optimizer.step=partial;ledger=PhysicalLedger(tmp_path/'cost.jsonl')
    with pytest.raises(RuntimeError):t.update(physical=ledger.append)
    assert t.cursor==0 and ledger.summary(0)['physical_optimizer_updates']==1
    assert_state_equal(before,t.ema.state_dict())


def test_nonfinite_candidate_scheduler_state_cannot_commit():
    t=trainer();before=copy.deepcopy(t.ema.state_dict());original=t.scheduler.step
    def invalid():
        original();t.optimizer.param_groups[0]['lr']=float('inf')
    t.scheduler.step=invalid
    with pytest.raises(FloatingPointError):t.update()
    assert t.step==0 and t.physical_optimizer_updates==1
    assert_state_equal(before,t.ema.state_dict())
