"""Synthetic geometry/binding tests only; no optimizer or formal-result trial."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from di_dmpa_gate1.binding import (H,S,FILE_HASHES,PANELS,ProtocolError,NumericalError,
    check_hash,gate1a_records,read_json,run_metadata,sha256,verify_ancestor,write_json)
from di_dmpa_gate1.bootstrap import multiplicity_weights,matched_cosines,registered_draws
from di_dmpa_gate1.geometry_metrics import normalize,weighted_ecdf,boundary_band,geometry
from di_dmpa_gate1.sampling import plan_from_labels,sample_layout
from di_dmpa_gate1.spherical_kmeans import fit,weighted_initialization,_iterate,clustering_seed
from di_dmpa_gate1.gate1a_reporting import adjudicate,artifact_manifest,report,primary_conditions,relative_reduction

ROOT=Path(__file__).resolve().parents[2]
PREREG=ROOT/'docs/di_dmpa_jascl/DI_DMPA_GATE1_PREREGISTRATION.json'


@pytest.mark.parametrize('parts',[[1,'é',False],['abc',0],['geometry-pixel-v1',20262830,'val','case',1,4,7]])
def test_hash_seed_contract(parts):
    expected=hashlib.sha256(json.dumps(parts,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    assert H(parts)==expected and S(parts)==int(expected[:8],16)&0x7fffffff


@pytest.mark.parametrize('suffix',['md','json'])
def test_raw_registration_hash_and_tamper(tmp_path,suffix):
    source=PREREG.with_suffix('.'+suffix)
    assert check_hash(source,FILE_HASHES[suffix])==FILE_HASHES[suffix]
    altered=tmp_path/source.name
    altered.write_bytes(source.read_bytes()+b' ')
    with pytest.raises(ProtocolError,match='SHA mismatch'):
        check_hash(altered,FILE_HASHES[suffix])


def test_ancestor_direction(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    def commit():
        subprocess.run(['git','-C',str(tmp_path),'-c','user.name=synthetic-test','-c','user.email=synthetic@example.invalid','commit','--allow-empty','-qm','fixture'],check=True)
        return subprocess.check_output(['git','-C',str(tmp_path),'rev-parse','HEAD'],text=True).strip()
    old=commit();new=commit()
    verify_ancestor(tmp_path,old,new)
    with pytest.raises(ProtocolError):
        verify_ancestor(tmp_path,new,old)


def test_checkpoint_sha_validation(tmp_path):
    path=tmp_path/'synthetic.pt';path.write_bytes(b'not-a-real-checkpoint')
    check_hash(path,hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ProtocolError):check_hash(path,'0'*64)


@pytest.mark.parametrize('role',['test','train_unlabeled','hidden','train'])
def test_forbidden_role_rejected_before_any_asset_access(tmp_path,role):
    with pytest.raises(ProtocolError,match='forbidden role'):
        gate1a_records(tmp_path,{},0,'REFUGE',role)


def synthetic_plan():
    a=np.zeros((12,12),dtype=np.uint8);a[2:8,2:8]=1;a[5:7,5:7]=2
    b=a.copy();b[4:8,4:8]=2
    rows=[{'case_id':'a'},{'case_id':'b'}]
    return plan_from_labels(rows,[a,b],seed=0,stage=0,domain='synthetic',role='train_labeled',pixel_seed=20262830)


def test_balancing_quota_and_coordinate_hash():
    plan=synthetic_plan();assert plan['common_quota']==4
    for c in range(3):
        layout=sample_layout(plan,c)
        assert len(layout['uids'])==8
        assert np.isclose(layout['weights'].sum(),1)
        for case in ('a','b'):
            assert np.isclose(layout['weights'][layout['case_ids']==case].sum(),.5)
        for case in plan['cases']:
            group=case['classes'][c]
            assert group['coordinate_sha256']==H([[case['case_id'],c,*xy] for xy in group['coordinates']])


def test_panel_independent_all_K_bootstrap_sampling():
    plan=synthetic_plan()
    hashes={p:{k:H(plan) for k in (1,2,3,5)} for p in PANELS}
    assert len({v for d in hashes.values() for v in d.values()})==1
    assert H(synthetic_plan())==H(plan)


def test_foreground_missing_not_smaller_denominator():
    with pytest.raises(ProtocolError,match='foreground unit missing'):
        plan_from_labels([{'case_id':'a'}],[np.zeros((4,4),dtype=np.uint8)],seed=0,stage=0,domain='synthetic',role='val',pixel_seed=0)


def test_boundary_constant_false_exterior_and_ignore():
    labels=np.ones((11,11),dtype=np.uint8)
    band=boundary_band(labels,1)
    assert band[0,0] and band[2,5] and not band[3,5] and not band[5,5]
    labels[0,0]=255
    assert not boundary_band(labels,1)[0,0]
    assert labels[0,0]==255


def points():
    rng=np.random.Generator(np.random.PCG64(11))
    return normalize(rng.normal(size=(64,4))+np.array([1.,0,0,0]))


@pytest.mark.parametrize('K',[1,2,3,5])
def test_spherical_determinism_norms_and_five_restarts(K):
    x=points();weights=np.ones(len(x))/len(x)
    a=fit(x,weights,K,seed=0,stage=1,class_id=2)
    b=fit(x,weights,K,seed=0,stage=1,class_id=2)
    np.testing.assert_array_equal(a['centers'],b['centers'])
    assert a['restarts']==b['restarts'] and len(a['restarts'])==5
    np.testing.assert_allclose(np.linalg.norm(a['centers'][a['active']],axis=1),1,atol=1e-14)
    assert all(r['iterations']<=100 for r in a['restarts'])


def test_K1_normalized_weighted_mean_and_restart_tie():
    x=normalize(np.array([[1.,0.],[0.,1.]]));weights=np.array([.25,.75])
    result=fit(x,weights,1,seed=1,stage=2,class_id=1)
    np.testing.assert_allclose(result['centers'][0],normalize(np.array([.25,.75])))
    assert result['selected_restart']==0


def test_weighted_kmeans_plus_plus_distance_not_squared():
    x=np.array([[1.,0.],[0.,1.],[-1.,0.]])
    class RNG:
        def __init__(self):self.probabilities=[]
        def choice(self,n,p):
            self.probabilities.append(p.copy());return len(self.probabilities)-1
    rng=RNG()
    weighted_initialization(x,np.array([1.,2.,3.]),2,rng)
    np.testing.assert_allclose(rng.probabilities[0],[1/6,2/6,3/6])
    np.testing.assert_allclose(rng.probabilities[1],[0,.25,.75])


def test_empty_cluster_weighted_residual_and_uid_tie():
    x=normalize(np.array([[1.,0.],[0.,1.],[1.,1.]]))
    centers=np.array([[1.,0.],[1.,0.]])
    centers,active,*_= _iterate(x,np.ones(3),centers,np.ones(2,dtype=bool),np.array([1,0,2]),1,1e-6)
    assert active.all()
    np.testing.assert_allclose(centers[1],[0.,1.])


def test_inactive_slots_not_lower_K():
    x=np.tile([1.,0.],(6,1))
    result=fit(x,np.ones(6),5,seed=0,stage=0,class_id=0)
    assert result['centers'].shape==(5,2) and result['active'].sum()==1
    assert len(result['restarts'])==5


def test_lowest_cluster_index_assignment_tie():
    result=geometry(np.array([[1.,0.]]),np.ones(1),np.array([[1.,0.],[1.,0.]]),np.ones(2,dtype=bool))
    assert result['occupancy']==[1.,0.]


def test_ecdf_not_interpolated_and_unit_sphere_radius():
    assert weighted_ecdf([3.,0.,2.],[.04,.95,.01],.95)==0
    result=geometry(np.array([[0.,1.]]),np.ones(1),np.array([[1.,0.]]),np.ones(1,dtype=bool))
    assert result['Q_K']==1 and result['R95']==np.sqrt(2)


def test_registered_bootstraps_not_rerolled():
    prereg=read_json(PREREG)
    draws=registered_draws(prereg,0,0)
    assert len(draws)==5
    assert all(H(d['case_ids_with_replacement'])==d['case_draw_sha256'] for d in draws)


def test_multiplicity_case_weight_and_zero_class_case():
    weights=multiplicity_weights(np.array(['a','a','b']),['a','a','b'])
    np.testing.assert_allclose(weights,[1/3,1/3,1/3])
    # The same common draw may contain a case with zero pixels of this class.
    np.testing.assert_allclose(multiplicity_weights(np.array(['a']),['a','absent_class_case']),[1])


def test_hungarian_matching_and_inactive_zero():
    x=np.eye(3)
    values=matched_cosines(x,[True,True,False],x[[1,0,2]],[True,True,False])
    np.testing.assert_array_equal(values,[1,1,0])


def test_panel_cannot_enter_clustering_seed():
    assert clustering_seed(0,1,2,5,-1,0)==S(['kmeans-v1',0,1,2,5,-1,0])
    with pytest.raises(TypeError):clustering_seed(0,1,2,5,-1,0,panel='C0-student')


@pytest.mark.parametrize('x',[np.array([[np.nan,1.]]),np.array([[np.inf,1.]]),np.zeros((1,2))])
def test_numerical_failure_closed(x):
    with pytest.raises(NumericalError):normalize(x)


def test_zero_resultant_center_closed():
    with pytest.raises(NumericalError):fit(np.array([[1.,0.],[-1.,0.]]),np.ones(2),1,seed=0,stage=0,class_id=1)


def fake_results(primary_factor=.8,control_factor=1.1):
    output=[]
    for panel in PANELS:
        for seed in range(3):
            for stage in range(3):
                for c in range(3):
                    for K in (1,2,3,5):
                        radius=1. if K==1 else (primary_factor if panel=='B0-EMA' else control_factor)
                        metrics=dict(Q_K=radius,cosine_distance_p95=radius,R95=radius,occupancy=[1/K]*K,
                                     active_assignment_slots=[True]*K,inactive_count=0,dormant_assignment_count=0,
                                     minimum_inter_prototype_euclidean=None,minimum_inter_prototype_angular=None)
                        output.append(dict(panel_id=panel,seed=seed,stage_index=stage,domain=['REFUGE','RIM_ONE_r3','Drishti_GS'][stage],class_id=c,K=K,
                            metadata={'sampling_plan_sha256':'a'*64},metrics={r:dict(metrics) for r in ('train_labeled','val')},
                            bootstrap=[{'replicate':b,'matched_cosines':[.95]*K} for b in range(5)],
                            boundary_interior={r:{'boundary':dict(metrics),'interior':dict(metrics)} for r in ('train_labeled','val')}))
    return output


def test_only_primary_selects_smallest_K_despite_control_failure():
    result=adjudicate(fake_results())
    assert result['passing_K']==[2,3,5] and result['selected_K']==2
    assert result['control_thresholds_applied'] is False


def test_control_cannot_rescue_primary():
    result=adjudicate(fake_results(1.1,.5))
    assert result['passing_K']==[] and result['selected_K']==1
    assert result['prototype_geometry_status']=='FAIL_MULTI_MODALITY_NOT_SUPPORTED'
    assert result['selected_K_role']=='EXPLICIT_DOWNSTREAM_FALLBACK_ONLY'


def test_incomplete_control_blocks_final_adjudication():
    rows=fake_results();rows.pop()
    with pytest.raises(NumericalError,match='incomplete'):adjudicate(rows)


def test_duplicate_unit_blocks():
    rows=fake_results();rows.append(rows[-1])
    with pytest.raises(ProtocolError,match='duplicate'):adjudicate(rows)


def test_any_panel_nan_blocks():
    rows=fake_results();rows[-1]['metrics']['val']['R95']=float('nan')
    with pytest.raises(NumericalError):adjudicate(rows)


def test_any_panel_sampling_mismatch_blocks():
    rows=fake_results();rows[-1]['metadata']['sampling_plan_sha256']='b'*64
    with pytest.raises(ProtocolError,match='sampling hash'):adjudicate(rows)


def test_background_cannot_drive_admission():
    rows=fake_results(1.1,.8)
    for r in rows:
        if r['class_id']==0 and r['K']>1:r['metrics']['val']['R95']=0.0
    assert adjudicate(rows)['selected_K']==1


def test_reference_zero_and_inclusive_boundaries():
    assert relative_reduction(0,0)==0 and relative_reduction(0,1) is None
    conditions=primary_conditions(dict(A1_improving_units=12,A2_median_relative_fg_macro_R95_reduction=.10,
        A3_active_cluster_fraction_occupancy_at_least_005=.90,A4_matched_cosine_median=.85,A5_improving_domain_count=2,A6_background_excluded=True))
    assert all(conditions.values())


def test_run_binding_fields_and_zero_optimizer_counts():
    p=read_json(PREREG)
    receipt={'preregistration_git_commit':'r','preregistration_remote_verified_commit':'r','authorization_git_commit':'a','diagnostic_code_git_commit':'c'}
    metadata=run_metadata(p,receipt,'a'*64,panel_id='B0-EMA')
    assert set(p['runtime_binding']['required_in_every_run_metadata']).issubset(metadata)
    assert metadata['model_optimizer_steps']==metadata['transport_optimizer_steps']==0
    assert metadata['method_registered'] is metadata['di_dmpa_training_launched'] is False
    with pytest.raises(ProtocolError):run_metadata(p,receipt,'pending',panel_id='B0-EMA')


def test_report_schema_and_hash_manifest(tmp_path):
    p=read_json(PREREG)
    metadata=run_metadata(p,dict(preregistration_git_commit='r',preregistration_remote_verified_commit='r',authorization_git_commit='a',diagnostic_code_git_commit='c'),'a'*64,panel_id='ALL_FOUR_SEPARATE')
    status=report(tmp_path,metadata,fake_results())
    assert status['gate1_overall_status']=='INCOMPLETE_GATE1B_GATE1C_NOT_RUN'
    assert status['next_action']=='STOP_FOR_INDEPENDENT_REVIEW'
    artifacts=artifact_manifest(tmp_path)
    assert all(sha256(tmp_path/r['path'])==r['sha256'] for r in artifacts['files'])
    assert all(r['path']!='GATE1A_ARTIFACT_MANIFEST.json' for r in artifacts['files'])


def test_synthetic_feature_forward_state_and_optimizer_absence(monkeypatch):
    import torch
    from di_dmpa_gate1 import feature_extraction as ex
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(16));self.calls=[]
        def forward(self,image,*,stochastic_classifier):
            self.calls.append((stochastic_classifier,torch.is_grad_enabled(),torch.is_autocast_enabled()))
            feat=self.weight[None,:,None,None].expand(len(image),16,384,384)
            return feat[:,:3],feat
    model=Model().eval()
    unit={'role':'train_labeled','pixel_sampling_seed':20262830,'cases':[{'classes':[{'coordinates':[[0,0],[191,191]]} for _ in range(3)]}]}
    monkeypatch.setattr(ex,'_images',lambda cases,root:torch.ones(len(cases),3,384,384))
    before=ex.state_groups(model)
    arrays,details=ex.extract_unit(model,unit,'unused',device='cpu')
    assert before==ex.state_groups(model)
    assert all(a.shape==(2,16) for a in arrays.values())
    assert model.calls==[(False,False,False)]
    assert details['optimizer_construction_guard'] and all(p.grad is None for p in model.parameters())


def test_production_package_has_no_optimizer_or_training_runner_calls():
    import ast
    package=ROOT/'di_dmpa_gate1'
    for path in package.glob('*.py'):
        tree=ast.parse(path.read_text())
        assert 'Gate0RepairedRunner' not in path.read_text()
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in {'backward','step','update_teacher','update_gas_from_supervised_gradient'}


def test_immutability_evidence_preserved_on_numerical_failure(tmp_path):
    import torch
    from di_dmpa_gate1.feature_extraction import ImmutabilityGuard
    checkpoint_file=tmp_path/'synthetic.pt';checkpoint_file.write_bytes(b'synthetic state only')
    checkpoint={'checkpoint_id':'synthetic','path':str(checkpoint_file),'sha256':sha256(checkpoint_file)}
    with pytest.raises(NumericalError):
        with ImmutabilityGuard({'student':torch.nn.Linear(2,2)},checkpoint,tmp_path,{}):
            raise NumericalError('synthetic zero norm')
    audit=read_json(tmp_path/'immutability/synthetic.json')
    assert audit['bitwise_unchanged'] and audit['extraction_completed'] is False
