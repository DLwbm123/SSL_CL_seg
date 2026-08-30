"""Synthetic inputs only: the 44 preregistered categories and failure boundaries."""
import copy
import csv
import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from di_dmpa_gate1.spherical_kmeans import clustering_seed
from di_dmpa_gate1.sampling import sample_layout
from di_dmpa_gate1_v2.features import split_support, ImmutableModels, weight_hash, validate_cache
from di_dmpa_gate1b_v2 import binding, plan, pairs, transport, evaluator, reporting, runner

ROOT=Path(__file__).resolve().parents[2]
DOCS=ROOT/'docs/di_dmpa_jascl'


def prereg():return binding.read_json(DOCS/'DI_DMPA_GATE1B_V2_PREREGISTRATION.json')
def frozen():return binding.read_json(DOCS/'GATE1A_V2_FREEZE.json')
def identity():return dict(kind='T0',optimizer_steps=0)
def residual():return dict(kind='T2',W=np.zeros((16,16)).tolist(),b=np.zeros(16).tolist())


def data():
    x=np.zeros((4,16));y=x.copy();x[:2,0]=1;y[[0,2],1]=1
    sa=np.array([1,1,0,0],dtype=bool);ta=np.array([1,0,1,0],dtype=bool)
    return dict(x=x,y=y,source_active=sa,target_active=ta,pair_state=pairs.pair_states(sa,ta),weights=np.full(4,.25),
        seed=0,stage_index=1,domain='RIM_ONE_r3',role='train_unlabeled',partition='fit')


def synthetic_plan():
    coords=[[i//384,i%384] for i in range(2048)]
    case=dict(case_id='SYNTHETIC_CASE',image_h5_relpath='synthetic/image.h5',image_sha256='a'*64,
        coordinates=coords,coordinate_uid_sha256=binding.H([['SYNTHETIC_CASE',y,x] for y,x in coords]))
    unit=dict(seed=0,stage_index=1,domain='RIM_ONE_r3',role='train_unlabeled',partition='fit',split_seed=20261831,
        split_hash='b'*64,cases=[case],case_count=1,registered_count=2048)
    lay=plan.layout(unit);unit.update(coordinate_uid_hash=binding.H(lay['uids']),original_weight_hash=weight_hash(lay['weights']))
    return unit


def evidence():
    transports=[];oracles=[]
    for s in range(3):
        for t in (1,2):
            transports.append(dict(seed=s,stage_index=t,feature_errors={'holdout':{m:dict(full_null_aware_support_error=e) for m,e in [('T0',1.),('T1',0.),('T2',.8)]}}))
        for a,b in ((0,1),(1,2),(0,2)):
            metrics={m:dict(class_angles=[dict(class_id=c,mean_angular_error=v) for c in range(3)],accuracy=dict(macro_accuracy=.8)) for m,v in [('T0',1.),('T1',0.),('T2',.8)]}
            oracles.append(dict(seed=s,source_stage=a,target_stage=b,metrics=metrics))
    return transports,oracles


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    torch.set_num_threads(1);d=data();before={k:v.copy() for k,v in d.items() if isinstance(v,np.ndarray)}
    path=tmp_path_factory.mktemp('synthetic_transport')/'trace.csv';calls=[];native=torch.optim.Adam.step
    def counted(opt,*args,**kwargs):
        calls.append(dict(defaults={k:v for k,v in opt.defaults.items()},shapes=[tuple(p.shape) for g in opt.param_groups for p in g['params']],
            dtypes=[p.dtype for g in opt.param_groups for p in g['params']],devices=[p.device.type for g in opt.param_groups for p in g['params']]))
        return native(opt,*args,**kwargs)
    with patch.object(torch.optim.Adam,'step',counted):model,trace=transport.fit_residual(d,np.eye(16)[:6],trace_path=path)
    assert all(np.array_equal(d[k],v) for k,v in before.items())
    return model,trace,calls,path


def test_01_freeze_sha_and_original_fit_binding():
    binding.check_hash(DOCS/'GATE1A_V2_FREEZE.json',binding.FREEZE_SHA)
    assert len(binding.validate_freeze(frozen(),ROOT.parents[1]))==27


def test_02_selected_K_exactly_two():
    f=frozen();f['selected_K']=3
    with pytest.raises(binding.ProtocolError):binding.validate_freeze(f)


@pytest.mark.parametrize('field,value',[('panel','B0-student'),('baseline','C0'),('feature_source','student'),('K',3)])
def test_03_only_B0_EMA_K2_accepted(field,value):
    f=frozen();f['prototype_records'][0][field]=value
    with pytest.raises(binding.ProtocolError):binding.validate_freeze(f)


@pytest.mark.parametrize('field,value',[('converged',False),('active_mask',[True,False]),('training_source','val')])
def test_04_frozen_input_convergence_active_role(field,value):
    f=frozen();f['prototype_records'][0][field]=value
    with pytest.raises(binding.ProtocolError):binding.validate_freeze(f)


def test_05_exact_inherited_case_splits():
    p=prereg();old=binding.read_json(DOCS/'DI_DMPA_GATE1_PREREGISTRATION.json')
    assert p['transport']['split_plans']==old['gate1b']['split_plans']
    for split in p['transport']['split_plans']:plan.validate_split(split)
    split=copy.deepcopy(p['transport']['split_plans'][0]);split['fit_case_ids'].reverse()
    with pytest.raises(binding.ProtocolError):plan.validate_split(split)


def test_06_coordinate_hash_rank_deterministic_synthetic_case():
    args=(0,1,'SYNTHETIC_NOT_A_MEDICAL_CASE');a=plan.coordinates(*args);b=plan.coordinates(*args)
    expected=sorted((binding.H(['transport-pixel-v1',*args,y,x]),y,x) for y in range(384) for x in range(384))[:2048]
    assert a==b==[[y,x] for _,y,x in expected]


def test_07_fit_holdout_disjoint():
    split=copy.deepcopy(prereg()['transport']['split_plans'][0]);split['holdout_case_ids'][0]=split['fit_case_ids'][0]
    with pytest.raises(binding.ProtocolError):plan.validate_split(split)


def test_08_plan_rejects_labels_and_never_opens_h5():
    import h5py
    record=dict(case_id='SYNTHETIC',image_h5_relpath='fake/image.h5',image_sha256='a'*64)
    with patch.object(h5py,'File',side_effect=AssertionError('label/image read forbidden')):
        with patch.object(plan,'coordinates',return_value=[[0,0]]):assert plan.case_plan((0,1,record))['coordinates']==[[0,0]]
        with pytest.raises(binding.ProtocolError):plan.case_plan((0,1,dict(record,label_h5_relpath='hidden.h5')))


@pytest.fixture
def cache(tmp_path):
    unit=synthetic_plan();raw=np.zeros((2048,16));raw[1:,0]=1;a=split_support(raw)
    arrays={side+'_'+k:v.copy() for side in ('source','target') for k,v in a.items()};arrays['pair_state']=pairs.pair_states(a['active_mask'],a['active_mask'])
    lay=plan.layout(unit)
    entry={k:unit[k] for k in ('seed','stage_index','domain','role','partition','registered_count','coordinate_uid_hash','original_weight_hash','split_hash')}
    entry.update(arrays=pairs.save_arrays(tmp_path,'arrays',arrays),all_rows_preserved=True,labels_read=False,support=pairs.support(arrays['pair_state'],lay['weights']))
    return tmp_path,unit,entry


def test_09_coordinate_alignment_hash(cache):
    output,u,e=cache;pairs.load_pair(output,e,u);e['coordinate_uid_hash']='0'*64
    with pytest.raises(binding.ProtocolError):pairs.load_pair(output,e,u)


def test_10_exact_cache_row_count(cache):
    output,u,e=cache;assert len(pairs.load_pair(output,e,u)['x'])==2048
    e['arrays']['source_directions']['shape'][0]-=1
    with pytest.raises(binding.ProtocolError):pairs.load_pair(output,e,u)


@pytest.mark.parametrize('state,expected',[(0,1.),(1,2.),(2,2.),(3,0.)])
def test_11_to_14_all_support_state_penalties(state,expected):
    d=data();d['weights']=np.array([1.,0.,0.,0.]) if state==0 else np.eye(4)[state]+np.array([1e-3,0,0,0])
    value=transport.feature_error(d,identity())
    expected_full=expected if state==0 else (expected+1e-3)/(1+1e-3)
    assert value['full_null_aware_support_error']==pytest.approx(expected_full)


def test_15_null_not_epsilon_normalized():
    raw=np.zeros((3,16));raw[1,0]=1e-13;raw[2,0]=1
    with np.errstate(divide='raise',invalid='raise'):a=split_support(raw)
    assert a['active_mask'].tolist()==[False,False,True] and np.all(a['directions'][:2]==0)


def test_16_null_rows_not_dropped(cache):
    output,u,e=cache;d=pairs.load_pair(output,e,u)
    assert d['x'].shape==(2048,16) and len(d['pair_state'])==2048 and d['pair_state'][0]==3


def test_17_full_objective_keeps_constants(fitted):
    _,trace,_,_=fitted
    assert all(r['support_constant_term']==1 for r in trace)
    for r in trace:assert r['full_objective']==pytest.approx(r['AA_weighted_term']+1+r['regularization'])
    assert trace[0]['full_support_error']==1.25


def test_18_T0_is_identity():
    d=data();out,_=transport.apply_model(d['x'],d['source_active'],identity());assert np.array_equal(out,d['x'])


def test_19_T2_zero_init_matches_T0(fitted):
    d=data();out,_=transport.apply_model(d['x'],d['source_active'],residual())
    assert np.array_equal(out,d['x']) and fitted[0]['step0_identity_pass'] and fitted[1][0]['W_frobenius_norm']==fitted[1][0]['b_norm']==0


def rotation_data():
    d=data();d['x']=np.eye(16);d['y']=np.eye(16);d['y'][0,0]=-1
    d.update(source_active=np.ones(16,dtype=bool),target_active=np.ones(16,dtype=bool),pair_state=np.zeros(16,dtype=np.uint8),weights=np.arange(1.,17.))
    return d


def test_20_weighted_procrustes_orthogonal():
    d=rotation_data();m=transport.procrustes(d);R=np.array(m['R'])
    assert np.allclose(R@R.T,np.eye(16),atol=1e-12,rtol=0)
    assert np.allclose(m['svd_singular_values'],np.linalg.svd((d['y']*d['weights'][:,None]).T@d['x'],compute_uv=False))


def test_21_procrustes_reflection_is_not_removed():
    m=transport.procrustes(rotation_data());assert m['determinant']<0 and m['optimizer_steps']==0


def test_22_exact_1000_updates(fitted):
    model,trace,calls,path=fitted
    assert model['optimizer_steps']==len(calls)==1000 and [r['step'] for r in trace]==list(range(1001))
    with path.open() as f:assert len(list(csv.DictReader(f)))==1001


def test_23_only_272_W_b_parameters_fixed_Adam(fitted):
    c=fitted[2][0];assert c['shapes']==[(16,16),(16,)] and c['dtypes']==[torch.float64]*2 and c['devices']==['cpu']*2
    for k,v in dict(lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False).items():assert c['defaults'][k]==v
    assert fitted[0]['model_optimizer_steps']==0


def test_24_segmentation_optimizer_construction_forbidden():
    W=torch.nn.Parameter(torch.zeros(16,16));b=torch.nn.Parameter(torch.zeros(16));model=torch.nn.Linear(16,3)
    with transport.only_transport_optimizer([W,b]):
        with pytest.raises(binding.ProtocolError):torch.optim.Adam(model.parameters())


def test_25_model_and_checkpoint_guard(tmp_path):
    cp_path=tmp_path/'synthetic_checkpoint';binding.write_text(cp_path,'synthetic immutable bytes')
    cp=dict(path=str(cp_path),sha256=binding.sha256(cp_path),checkpoint_id='B0/seed0/stage0')
    model=torch.nn.Linear(16,3).eval().requires_grad_(False)
    with ImmutableModels({'ema_teacher':model},cp,tmp_path/'pass',{}):model(torch.zeros(1,16))
    assert binding.read_json(tmp_path/'pass/immutability/B0_seed0_stage0.json')['bitwise_unchanged']
    with pytest.raises(binding.ModelMutation):
        with ImmutableModels({'ema_teacher':model},cp,tmp_path/'mutated',{}):model.weight.add_(1)


def test_26_transport_unit_output_and_null_extension(fitted):
    d=data();out,_=transport.apply_model(d['x'],d['source_active'],fitted[0])
    assert np.allclose(np.linalg.norm(out[d['source_active']],axis=1),1,atol=1e-12,rtol=0) and np.all(out[~d['source_active']]==0)


def test_27_zero_active_output_blocks():
    d=data();m=residual();m['W']=(-np.eye(16)).tolist()
    with pytest.raises(binding.InvalidTransportOutput):transport.apply_model(d['x'],d['source_active'],m)


@pytest.mark.parametrize('change',[{'domain':'REFUGE'},{'role':'val'},{'role':'test'},{'partition':'holdout'},{'image_path':'historical.h5'},{'label':'hidden'}])
def test_28_fit_api_rejects_historical_images_GT_and_holdout(change):
    d=data();d.update(change)
    with pytest.raises(binding.ProtocolError):transport.validate_data(d,fit=True)


def test_29_oracle_is_after_all_transforms_and_not_in_fit_module():
    source=inspect.getsource(transport.fit_residual)
    assert 'oracle' not in source and 'h5py' not in source and 'sample_layout' not in source
    assert 'ORACLE_START_BARRIER.json' in inspect.getsource(evaluator.evaluate_unit)
    assert '6000' in inspect.getsource(evaluator.evaluate_unit)


def test_30_operational_prototypes_are_original_without_refit():
    f=frozen()
    with patch.object(evaluator,'fit',side_effect=AssertionError('operational refit forbidden')):
        p=evaluator.operational(f,0,0)
    assert p.shape==(3,2,16)
    for c in range(3):assert p[c].tolist()==next(r['centers'] for r in f['prototype_records'] if r['seed']==r['stage_index']==0 and r['class_id']==c)


def test_31_oracle_K2_source_stage_original_five_seeds():
    a=split_support(np.eye(16)[:6]);out=evaluator.oracle_fit(a,np.full(6,1/6),np.arange(6),seed=2,source_stage=1,class_id=2)
    assert out['K']==2 and [r['seed'] for r in out['restarts']]==[clustering_seed(2,1,2,2,-1,r) for r in range(5)]


def test_32_Hungarian_minimum_angular_matching_and_inactive_penalty():
    p=np.eye(16)[:2];r=evaluator.angular_match(p,p[::-1],np.ones(2,dtype=bool))
    assert r['mean_angular_error']==0 and r['matching']==[(0,1),(1,0)]
    q=p.copy();q[1]=0;r=evaluator.angular_match(p,q,np.array([True,False]));assert r['mean_angular_error']==np.pi/2


def queries():
    p=np.repeat(np.eye(16)[:3,None,:],2,axis=1)
    return {c:split_support(np.stack([p[c,0],np.zeros(16)])) for c in range(3)},{c:np.array([.5,.5]) for c in range(3)},p


def test_33_null_query_incorrect_and_conditional_accuracy():
    q,w,p=queries();r=evaluator.prototype_accuracy(q,w,p)
    assert r['macro_accuracy']==.5 and r['directional_conditional_macro_accuracy']==1 and all(c['null_count']==1 for c in r['classes'])


def test_34_class_case_balanced_accuracy():
    unit=dict(cases=[dict(case_id='a',classes=[dict(sampled_pixels=1,coordinates=[[0,0]],boundary=[False]) for _ in range(3)]),
        dict(case_id='b',classes=[dict(sampled_pixels=3,coordinates=[[1,i] for i in range(3)],boundary=[False]*3) for _ in range(3)])])
    weights=sample_layout(unit,0)['weights'];assert np.allclose(weights,[.5,1/6,1/6,1/6])
    _,_,p=queries();q={c:split_support(np.stack([p[c,0],*[p[(c+1)%3,0]]*3])) for c in range(3)}
    r=evaluator.prototype_accuracy(q,{c:weights for c in range(3)},p)
    assert r['macro_accuracy']==pytest.approx(.5) and r['foreground_macro_accuracy']==pytest.approx(.5)


def test_35_chain_preserves_frozen_original():
    p=evaluator.operational(frozen(),0,0);before=p.copy();m=residual();m['b'][0]=.1
    evaluator.transported(p,[m,m]);assert np.array_equal(p,before)


def test_36_chain_order_is_T12_of_T01():
    p=evaluator.operational(frozen(),0,0);a=residual();b=residual();a['b'][0]=.4;b['b'][1]=.7
    ordered=evaluator.transported(p,[a,b]);wrong=evaluator.transported(p,[b,a])
    first,_=transport.apply_model(p.reshape(-1,16),np.ones(6,dtype=bool),a)
    second,_=transport.apply_model(first,np.ones(6,dtype=bool),b)
    assert np.array_equal(ordered,second.reshape(3,2,16)) and not np.allclose(ordered,wrong)


@pytest.mark.parametrize('gate',['B1','B2','B3','B4','B5','B6','B7'])
def test_37_B1_B7_boundaries(gate):
    ts,os=evidence()
    if gate=='B1':
        for t in ts:t['feature_errors']['holdout']['T2']['full_null_aware_support_error']=.85
    elif gate=='B2':
        for t in ts:
            if t['seed']==2:t['feature_errors']['holdout']['T2']['full_null_aware_support_error']=1.
    elif gate=='B3':
        for o in os:
            if o['target_stage']==o['source_stage']+1:
                for a in o['metrics']['T2']['class_angles'][1:]:a['mean_angular_error']=.9
    elif gate in ('B4','B6'):os[0 if gate=='B4' else 2]['metrics']['T2']['class_angles'][1]['mean_angular_error']=1.05
    elif gate=='B5':os[0]['metrics']['T2']['accuracy']['macro_accuracy']=.8-.005
    assert reporting.admission(ts,os)['B1_B7'][gate]['pass_']
    if gate=='B1':
        for t in ts:t['feature_errors']['holdout']['T2']['full_null_aware_support_error']=.85+1e-12
    elif gate=='B2':ts[0]['feature_errors']['holdout']['T2']['full_null_aware_support_error']=1.
    elif gate=='B3':
        for o in os:
            if o['target_stage']==o['source_stage']+1:
                for a in o['metrics']['T2']['class_angles'][1:]:a['mean_angular_error']=.9+1e-12
    elif gate in ('B4','B6'):os[0 if gate=='B4' else 2]['metrics']['T2']['class_angles'][1]['mean_angular_error']=np.nextafter(1.05,np.inf)
    elif gate=='B5':os[0]['metrics']['T2']['accuracy']['macro_accuracy']=np.nextafter(.8-.005,-np.inf)
    else:
        os[0]['metrics']['T1']['accuracy']['macro_accuracy']=float('nan')
        with pytest.raises(binding.InvalidTransportOutput):reporting.admission(ts,os)
        return
    assert not reporting.admission(ts,os)['B1_B7'][gate]['pass_']


def test_38_T1_cannot_rescue_T2():
    ts,os=evidence()
    for t in ts:t['feature_errors']['holdout']['T2']['full_null_aware_support_error']=1.
    r=reporting.admission(ts,os);assert r['transport_status']=='FAIL_TRANSPORT_NOT_SUPPORTED' and r['selected_transport']=='T0_identity'


@pytest.mark.parametrize('role',['val','test','train_labeled'])
def test_39_plan_forbids_non_unlabeled_roles(role):
    u=synthetic_plan();u['role']=role
    with pytest.raises(binding.ProtocolError):plan.layout(u)


def test_40_hidden_GT_rejected_in_plan():
    u=synthetic_plan();u['cases'][0]['label_sha256']='c'*64
    with pytest.raises(binding.ProtocolError):plan.layout(u)


@pytest.mark.parametrize('what',['transport','oracle','class'])
def test_41_incomplete_evidence_fail_closed(what):
    ts,os=evidence()
    if what=='transport':ts.pop()
    elif what=='oracle':os.pop()
    else:os[0]['metrics']['T2']['class_angles'].pop()
    with pytest.raises(binding.IncompleteEvidence):reporting.admission(ts,os)


def test_42_artifact_manifest_exact_hashes(tmp_path):
    binding.write_json(tmp_path/'a.json',dict(synthetic=True));binding.write_text(tmp_path/'b.txt','unit-test bytes')
    manifest=reporting.artifact_manifest(tmp_path)
    assert manifest['file_count']==2
    for r in manifest['artifacts']:assert binding.sha256(tmp_path/r['path'])==r['sha256']
    with pytest.raises(FileExistsError):reporting.artifact_manifest(tmp_path)


def test_43_report_compiler_complete_synthetic_evidence(tmp_path,fitted):
    ts,os=evidence();meta=dict(synthetic=True,gate1a_v2_freeze_commit='synthetic',preregistration_commit='synthetic',authorization_commit='synthetic',diagnostic_code_commit='synthetic')
    for name,value in [('TRANSFORM_START_BARRIER.json',dict(status='PASS',transport_optimizer_steps=0,evidence_sha256={})),
        ('GATE1B_V2_MODEL_IMMUTABILITY_AUDIT.json',dict(status='PASS',all9_B0_disk_hashes_unchanged=True,all_model_states_unchanged=True))]:binding.write_json(tmp_path/name,value)
    full=transport.feature_error(data(),identity())
    for t in ts:
        trace_path=f"transport_models/seed{t['seed']}_stage{t['stage_index']}_trace.csv"
        binding.write_text(tmp_path/trace_path,fitted[3].read_text())
        t.update(metadata=meta,role='train_unlabeled',domain=binding.DOMAINS[t['stage_index']],oracle_or_GT_access=False,model_optimizer_steps=0,
            models={'T0':identity(),'T1':dict(optimizer_steps=0),'T2':fitted[0]},trace_rows=1001,trace_path=trace_path,trace_sha256=binding.sha256(tmp_path/trace_path),
            feature_errors={r:{m:copy.deepcopy(full) for m in ('T0','T1','T2')} for r in ('fit','holdout')},spectra={m:transport.spectrum(identity()) for m in ('T0','T1','T2')})
    model_hashes={}
    for t in ts:
        name=f"seed{t['seed']}_stage{t['stage_index']}.json";model_hashes[name]=binding.write_json(tmp_path/'transport_models'/name,t)
    binding.write_json(tmp_path/'ORACLE_START_BARRIER.json',dict(transport_optimizer_steps=6000,model_sha256=model_hashes))
    synthetic=split_support(np.eye(16)[:2]);descriptors=pairs.save_arrays(tmp_path,'synthetic_oracle_cache',synthetic)
    cache=dict(registered_count=2,active_count=2,null_count=0,arrays=descriptors)
    for o in os:
        o.update(metadata=meta,role='val',gt_consumer='diagnostic_evaluator_only',transform_fit_called=False,operational_refit=False,
            oracle_GT_used_for_transform_fit=False,model_optimizer_steps=0,transport_optimizer_steps_in_evaluator=0,test_gt_usage='none',
            kind='chain' if o['target_stage']==2 and o['source_stage']==0 else 'immediate',source_domain=binding.DOMAINS[o['source_stage']],
            class_caches=[cache]*3,oracle_fits=[dict(K=2,source_stage_for_clustering_seeds=o['source_stage'],restarts=[dict(restart=i,converged=True) for i in range(5)],active=[True,True]) for _ in range(3)])
        for v in o['metrics'].values():v['foreground_macro_angular_error']=float(np.mean([a['mean_angular_error'] for a in v['class_angles'][1:]]))
    census=dict(metadata=meta,paired_units_completed=12,registered_count=638976,all_finite=True,all_rows_preserved=True,labels_read=False,counts=dict(AA=638976,A_NULL=0,NULL_A=0,NULL_NULL=0),units=[])
    reporting.report(tmp_path,meta,ts,os,census)
    result=binding.read_json(tmp_path/'GATE1B_V2_STATUS.json')
    assert result['transport_optimizer_steps']==6000 and result['transport_status']=='FAIL_TRANSPORT_NOT_SUPPORTED'
    with (tmp_path/'transport_fit_trace.csv').open() as stream:assert len(list(csv.DictReader(stream)))==6006
    for name in ('transport_feature_error_v2.csv','transport_prototype_error_v2.csv','transport_chain_error_v2.csv','transport_prototype_accuracy_v2.csv','transport_spectrum_v2.csv'):
        assert (tmp_path/name).is_file()


def test_44_no_Gate1C_or_training_flags():
    p=prereg();assert not any(p['method_flags'].values())
    for k in ('Gate1C','reliability','gradient_conflict','teacher_noise','theory_final','Gate2','Prostate','MnMS','main_merge','full_sweep'):assert p['limits'][k] is False
    assert p['next_action']=='STOP_FOR_INDEPENDENT_REVIEW'


def test_no_AA_mass_is_scientific_fail_not_epsilon():
    d=data();d['weights'][0]=0
    with pytest.raises(binding.NoDirectionalPairs):transport.validate_data(d)


def test_cache_corruption_blocks(cache):
    output,u,e=cache;e['arrays']['source_directions']['sha256']='0'*64
    with pytest.raises(binding.ProtocolError):pairs.load_pair(output,e,u)


def test_nonfinite_raw_feature_blocks():
    z=np.zeros((1,16));z[0,0]=np.nan
    with pytest.raises(binding.NonfiniteFeature):split_support(z)


def test_spectrum_finite_and_singular_blocks():
    assert transport.spectrum(identity())['condition_number']==1
    m=residual();m['W']=(-np.eye(16)).tolist()
    with pytest.raises(binding.InvalidTransportOutput):transport.spectrum(m)


def test_no_active_accuracy_explicit_undefined():
    q,w,p=queries();q[0]=split_support(np.zeros((2,16)));r=evaluator.prototype_accuracy(q,w,p)
    assert r['classes'][0]['directional_conditional_accuracy'] is None and not r['directional_conditional_macro_defined']


def test_tie_lowest_class_ID():
    q,w,p=queries();p[:]=p[0];r=evaluator.prototype_accuracy(q,w,p)
    assert r['classes'][0]['accuracy']==.5 and r['classes'][1]['accuracy']==r['classes'][2]['accuracy']==0


def test_zero_reference_relative_policy():
    assert not reporting.relative(0,0,.15,improvement=True)['pass_']
    assert reporting.relative(0,0,.05,improvement=False)['pass_']
    r=reporting.relative(0,.1,.05,improvement=False);assert r['relative_change'] is None and not r['pass_']


def test_incomplete_census_blocks_before_any_fit(tmp_path):
    with pytest.raises(binding.IncompleteEvidence):pairs.census(tmp_path,[],dict(units=[]),{})


def test_guard_records_state_even_on_forward_error(tmp_path):
    cp=tmp_path/'checkpoint';binding.write_text(cp,'synthetic')
    record=dict(path=str(cp),sha256=binding.sha256(cp),checkpoint_id='B0/seed0/stage0')
    model=torch.nn.Linear(16,3).eval().requires_grad_(False)
    with pytest.raises(ValueError):
        with ImmutableModels({'ema_teacher':model},record,tmp_path,{}):raise ValueError('synthetic forward failure')
    audit=binding.read_json(tmp_path/'immutability/B0_seed0_stage0.json')
    assert audit['bitwise_unchanged'] and not audit['extraction_completed']


def test_Hungarian_angular_not_maximum_cosine():
    from scipy.optimize import linear_sum_assignment
    p=np.zeros((2,16));q=p.copy();p[0,0]=q[0,0]=1
    p[1,:2]=[np.cos(.65),np.sin(.65)]
    phi=(np.cos(1.2)-np.cos(.65)**2)/np.sin(.65)**2
    q[1,:3]=[np.cos(.65),np.sin(.65)*phi,np.sin(.65)*np.sqrt(1-phi**2)]
    result=evaluator.angular_match(p,q,np.ones(2,dtype=bool))
    assert result['matching']==[(0,0),(1,1)] and result['mean_angular_error']==pytest.approx(.6)
    assert linear_sum_assignment(-(p@q.T))[1].tolist()==[1,0]


def test_complete_zero_AA_census_stops_without_fitting(tmp_path):
    units=[];entries=[]
    for s in range(3):
        for t in (1,2):
            for partition in ('fit','holdout'):
                key=dict(seed=s,stage_index=t,partition=partition,domain=binding.DOMAINS[t])
                units.append(dict(key,cases=[dict(case_id='SYNTHETIC')]))
                summary=pairs.support(np.full(2048,3,dtype=np.uint8),np.ones(2048))
                entries.append(dict(key,metadata={},all_finite=True,old_raw_cache_reused=False,support=summary,
                    coordinate_uid_hash='a'*64,original_weight_hash='b'*64,case_support=[dict(case_id='SYNTHETIC',counts=summary['counts'],mismatch_fraction=0.)]))
    with patch.object(pairs,'load_pair',return_value=None),patch.object(transport,'fit_residual',side_effect=AssertionError('no fits')):
        with pytest.raises(binding.NoDirectionalPairs):pairs.census(tmp_path,entries,dict(units=units),{})
    report=binding.read_json(tmp_path/'PAIRED_FEATURE_SUPPORT_CENSUS.json')
    assert report['paired_units_completed']==12 and report['transport_optimizer_steps']==0 and report['counts']['AA']==0
