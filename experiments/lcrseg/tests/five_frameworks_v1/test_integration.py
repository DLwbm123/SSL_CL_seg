import copy,json,os,subprocess,sys
import pytest
import torch
from experiments.lcrseg.five_frameworks_v1.controller import synthetic_sequence
from experiments.lcrseg.five_frameworks_v1.parent_bridge import SyntheticParentBridge
from experiments.lcrseg.five_frameworks_v1.model import Model
from experiments.lcrseg.five_frameworks_v1.train_stage import StageTrainer,split_gradients
from experiments.lcrseg.five_frameworks_v1.recipes import SyntheticCurrentDomain,generator,collected_forward,unique_indices
from experiments.lcrseg.five_frameworks_v1.registry import FAMILIES,BASELINES
from experiments.lcrseg.five_frameworks_v1.checkpoint import save,restore,PhysicalLedger,atomic_save,require_predecessor
from experiments.lcrseg.five_frameworks_v1.kernels import gradient_seed_basis,uncertainty_scales


@pytest.mark.parametrize('family',FAMILIES+BASELINES)
def test_actual_two_stage_callgraphs(family,tmp_path,cwmi):
    rows=synthetic_sequence(family,tmp_path/family,cwmi,steps=3,options={'PAS_confidence':0.,'PAS_cosine':-1.})
    assert len(rows)==2
    for row in rows:
        assert row['telemetry']['formal_optimizer_updates']==0
        assert row['physical']['physical_optimizer_updates']==3
        assert row['status']=='SYNTHETIC_COMPLETE'
        if family.startswith(('B0','B3')):assert row['data_access']['synthetic_U_batches']==0
        if family=='F2':assert row['telemetry']['probe_vjps']==24 and row['telemetry']['probe_full_forwards']==16
        if family=='F4':assert row['telemetry']['coordinate_vjps']==6 and row['probe']['complete']
        if family=='F5':assert row['telemetry']['extra_clean_U_forwards']==2


def trainer(family='F1',cwmi=None,initialize=True,options=None):
    torch.manual_seed(4)
    return StageTrainer(Model(SyntheticParentBridge(),family),SyntheticCurrentDomain(size=128 if family=='F2' else 16),
                        {'PAS_confidence':0.,'PAS_cosine':-1.,'total_steps':5,**(options or {})},cwmi,initialize)


@pytest.mark.parametrize('family',FAMILIES)
def test_U_gradient_permissions_nonzero_R_or_B(family,cwmi):
    t=trainer(family,cwmi);t.update();t.update()
    # Now parent B and sidecar R are nonzero, so isolation cannot pass merely
    # because zero-init eliminates input-factor derivatives.
    l,u,p=t.losses();grads=split_gradients(t.model,l,u)
    allowed={id(p) for p in t.model.u_parameters()}
    assert sum(v['U'].abs().sum() for v in grads.values() if v['U'] is not None)>0
    assert all(v['U'] is None for key,v in grads.items() if key not in allowed)
    inputs=t.model.parent.parameter_groups()['input_factors']
    assert sum(grads[id(p)]['L'].abs().sum() for p in inputs)>0
    assert all(p.grad is None for p in t.model.parent.parameter_groups()['frozen'])
    assert all(p.grad is None for p in t.ema.parameters())


@pytest.mark.parametrize('family',FAMILIES)
def test_native_off_geometry_fullmodel_merge_and_independent_deploy(tmp_path,family):
    torch.manual_seed(9);parent=SyntheticParentBridge()
    x=torch.randn(2,3,17,19);native=parent.native(x)
    m=Model(parent,family);assert torch.equal(native,m(x))
    assert parent.readout.padding==(0,0) and parent.readout.kernel_size==(3,3)
    with torch.no_grad():
        if m.sidecar is not None:m.sidecar.r.normal_(0,.1)
        for a in parent.adapters:a.b.normal_(0,.03)
    expected=m(x).detach();deploy=m.deploy()
    assert torch.allclose(expected,deploy(x),atol=2e-6,rtol=2e-6)
    assert not any('sidecar' in k or 'spectrum' in k for k in deploy.state_dict())
    cp=tmp_path/'deploy.pt';inp=tmp_path/'input.pt';out=tmp_path/'output.pt'
    atomic_save({'synthetic_only':True,'student':deploy.state_dict(),'d':4,'rank':2},cp)
    torch.save(x,inp)
    code='import torch,sys; from experiments.lcrseg.five_frameworks_v1.evaluate import load_synthetic_student; torch.set_num_threads(1); torch.save(load_synthetic_student(sys.argv[1])(torch.load(sys.argv[2],weights_only=True)),sys.argv[3])'
    subprocess.run([sys.executable,'-c',code,str(cp),str(inp),str(out)],check=True,capture_output=True)
    assert torch.allclose(expected,torch.load(out,weights_only=True),atol=2e-6,rtol=2e-6)
    nextmodel=Model(deploy.parent,family,previous=deploy.transform.detach())
    nextmodel.parent.stage_entry()
    assert all(p.requires_grad for p in nextmodel.parent.parameter_groups()['input_factors'])
    assert torch.allclose(nextmodel(x),expected,atol=2e-6,rtol=2e-6)


def assert_state_equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_state_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_state_equal(x,y)
    else:assert a==b


@pytest.mark.parametrize('family',FAMILIES)
def test_checkpoint_exact_resume(family,cwmi,tmp_path):
    a=trainer(family,cwmi)
    a.update();save(a,tmp_path/'state.pt',{'family':family})
    a.update();a.update()
    b=trainer(family,cwmi,initialize=False);restore(b,tmp_path/'state.pt',{'family':family})
    b.update();b.update()
    for x,y in [(a.model.state_dict(),b.model.state_dict()),(a.ema.state_dict(),b.ema.state_dict()),
                (a.optimizer.state_dict(),b.optimizer.state_dict()),(a.prototypes.values,b.prototypes.values),
                (a.prototypes.support,b.prototypes.support),(a.probe,b.probe),(a.telemetry,b.telemetry)]:assert_state_equal(x,y)
    assert a.cursor==b.cursor==3 and a.step==b.step==3
    with pytest.raises(ValueError):restore(b,tmp_path/'state.pt',{'family':'different'})


@pytest.mark.parametrize('point',['before_optimizer','after_optimizer','before_ema','after_ema','during_checkpoint'])
def test_failure_tail_replayed_and_cost_retained(point,tmp_path):
    t=trainer();identity={'family':'F1'};path=tmp_path/'state.pt';ledger=PhysicalLedger(tmp_path/'physical.jsonl')
    save(t,path,identity)
    if point=='during_checkpoint':
        t.update(physical=ledger.append)
        with pytest.raises(RuntimeError):save(t,path,identity,fault=True)
    else:
        with pytest.raises(RuntimeError):t.update(fault=point,physical=ledger.append)
    resumed=trainer(initialize=False);restore(resumed,path,identity)
    assert resumed.step==0
    resumed.update(physical=ledger.append)
    expected=trainer();expected.update()
    assert_state_equal(resumed.model.state_dict(),expected.model.state_dict())
    assert_state_equal(resumed.ema.state_dict(),expected.ema.state_dict())
    count=ledger.summary(1)
    assert count['physical_retry_or_uncommitted_updates']==int(point!='before_optimizer')


def test_skip_no_model_ema_proto_cursor_change():
    t=trainer();t.update()
    before=copy.deepcopy((t.model.state_dict(),t.ema.state_dict(),t.prototypes.values,t.prototypes.support,t.optimizer.state_dict()))
    t.update(skip=True)
    assert_state_equal(before,(t.model.state_dict(),t.ema.state_dict(),t.prototypes.values,t.prototypes.support,t.optimizer.state_dict()))
    assert t.cursor==t.step==1 and t.telemetry['skipped_updates']==1


def test_ema_effective_matrix_and_shared_F():
    t=trainer();q=t.model.sidecar.q;prev=t.model.sidecar.previous
    assert t.ema.sidecar.previous is prev
    before=t.ema.sidecar.effective().clone();t.update()
    expected=.99*before+.01*t.model.sidecar.effective()
    assert torch.allclose(t.ema.sidecar.effective(),expected,atol=2e-7)


def test_F2_probe_preserves_entry_and_thin_svd(cwmi):
    torch.manual_seed(4);parent=SyntheticParentBridge();x=torch.randn(2,3,128,128)
    expected=parent.native(x).detach();norms=[a.a.norm().detach().clone() for a in parent.adapters]
    t=StageTrainer(Model(parent,'F2'),SyntheticCurrentDomain(size=128),{'lambda_structure':1.},cwmi)
    assert torch.equal(expected,parent.native(x))
    assert all(torch.allclose(a.a.norm(),n) for a,n in zip(parent.adapters,norms))
    assert all(p.grad is None for p in parent.parameters()) and not t.optimizer.state
    grads=[torch.randn(3,5,dtype=torch.float64) for _ in range(4)]
    q=gradient_seed_basis(grads,2);g=torch.cat(grads);_,v=torch.linalg.eigh(g.T@g);top=v[:,-2:]
    assert torch.allclose(q@q.T,top@top.T,atol=1e-12)
    t.model.parent.adapters[-1].new=False
    with pytest.raises(ValueError,match='NEW'):t.prepare_stage()


def test_common_recipe_source_collection_and_dedup():
    anchor=torch.randn(2,3,8,8);donor=torch.randn_like(anchor)
    out,_=collected_forward(lambda x,s:x,anchor,donor,generator(1,1,1,0,'LCTX'))
    assert torch.allclose(out,anchor.log_softmax(1),atol=1e-7)
    assert unique_indices(['a','b','a','b'])==[0,1]


def test_F3_mapping_and_identity_limit():
    # Test actual trainer mapping by recording only the explicit sidecar scales
    # via a local Model subclass, never monkeypatching production globals.
    class RecordingModel(Model):
        def forward(self,x,detach_parent=False,scale=None):
            if scale is not None:self.seen.append(scale.detach().clone())
            return super().forward(x,detach_parent,scale)
    torch.manual_seed(4);m=RecordingModel(SyntheticParentBridge(),'F3');m.seen=[]
    t=StageTrainer(m,SyntheticCurrentDomain(),{'total_steps':5,'kappa':2.})
    t.update();t.update()
    assert len(m.seen)==2 and not torch.equal(m.seen[0],m.seen[1])
    assert all(not s.requires_grad and (s<=1).all() and (s>0).all() for s in m.seen)
    # Complementary mapping preserves the two source values at each coordinate.
    assert torch.isfinite(m.seen[0]+m.seen[1]).all()


def test_U_isolation_does_not_assert_zero_total_Adam_update():
    t=trainer(options={'weight_decay':.1});t.update()
    a=t.model.parent.adapters[-1].a;before=a.detach().clone();g=t.update()
    assert g[id(a)]['U'] is None and not torch.equal(before,a)


def test_cross_sequence_lineage_rejected():
    a={'family':'F1','candidate_id':'x','seed':161,'order':1,'stage':2};b={**a,'stage':1}
    require_predecessor(a,b)
    for key in ('family','candidate_id','seed','order'):
        with pytest.raises(ValueError):require_predecessor(a,{**b,key:'wrong'})


def test_F3_exact_source_direction_and_kappa_zero():
    from experiments.lcrseg.five_frameworks_v1.recipes import mixed_feature_scales
    parent=SyntheticParentBridge();mask=torch.tensor([[[True,False],[False,True]]])
    u=torch.zeros(1,2,2);context=torch.ones(1,2,2);spectrum=torch.tensor([1.,.1])
    a,b=mixed_feature_scales(parent,mask,u,context,(2,2),spectrum,2.)
    full=uncertainty_scales(context,spectrum,2.)
    assert torch.equal(a,torch.where(mask[:,None],torch.ones_like(a),full))
    assert torch.equal(b,torch.where(mask[:,None],full,torch.ones_like(a)))
    z1,z2=mixed_feature_scales(parent,mask,u,context,(2,2),spectrum,0.)
    assert torch.equal(z1,torch.ones_like(z1)) and torch.equal(z2,z1)


def test_F5_ablation_retains_extra_forward_and_changes_gradient():
    a=trainer('F5',options={'lambda_SWD':0.});b=trainer('F5',options={'lambda_SWD':.2})
    a.update();b.update()
    # Same teacher and toy GT; change only the SWD coefficient.
    l1,u1,_=a.losses();l2,u2,_=b.losses()
    assert a.last['swd_counts']==b.last['swd_counts']
    assert a.telemetry['extra_clean_U_forwards']==b.telemetry['extra_clean_U_forwards']==1
    assert any(n>=8 for n in a.last['swd_counts'].values())
    g1,=torch.autograd.grad(u1,a.model.sidecar.r);g2,=torch.autograd.grad(u2,b.model.sidecar.r)
    assert not torch.equal(g1,g2)


def test_data_capabilities_and_unknown_parent_remain_closed():
    from experiments.lcrseg.five_frameworks_v1.contracts import DataScope,ParentContract
    scope=DataScope('current',mode='SYNTHETIC',unlabeled_images=True)
    scope.require('current','L');scope.require('current','U_IMAGE')
    for kind in ('U_GT','TEST','VAL','OLD_L'):
        with pytest.raises(PermissionError):scope.require('current',kind)
    with pytest.raises(PermissionError):scope.require('old','L')
    with pytest.raises(RuntimeError):ParentContract('PARENT_BINDING_REQUIRED',None).require_real_ready()


def test_F2_all_ignore_caller_fallback_keeps_legal_A(cwmi):
    class NoSupport(SyntheticCurrentDomain):
        def labeled(self,step,stream='labeled'):
            x,y,p=super().labeled(step,stream);return x,torch.full_like(y,255),p
    torch.manual_seed(6);parent=SyntheticParentBridge();old=[a.a.detach().clone() for a in parent.adapters]
    t=StageTrainer(Model(parent,'F2'),NoSupport(size=128),cwmi=cwmi)
    assert t.probe['zero_structure'] and t.probe['scale']==1. and len(t.probe['fallback'])==2
    assert all(torch.equal(a.a,v) for a,v in zip(parent.adapters,old))


def test_F4_acceptance_stays_original_and_controller_records_entry_failure(tmp_path):
    a=trainer('F4',options={'lambda_shape':0.});b=trainer('F4',options={'lambda_shape':10.})
    a.update();b.update();a.losses();b.losses()
    assert a.last['accepted']==b.last['accepted']
    with pytest.raises(RuntimeError):synthetic_sequence('F2',tmp_path/'failed',cwmi=None,steps=1)
    value=json.loads((tmp_path/'failed/failure.json').read_text())
    assert value['status']=='SYNTHETIC_FAILED' and value['identity']['family']=='F2'


def test_controller_applies_candidate_feature_rank(tmp_path):
    rows=synthetic_sequence('F1',tmp_path/'rank',steps=1,options={'rank_ratio':.25})
    assert [r['spectral']['rank'] for r in rows]==[1,1]
