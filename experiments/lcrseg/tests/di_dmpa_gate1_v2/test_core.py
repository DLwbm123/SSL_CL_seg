"""Synthetic-only null-aware support, geometry, provenance and compiler gates."""
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from di_dmpa_gate1.binding import H, ProtocolError, read_json, sha256, write_json
from di_dmpa_gate1.geometry_metrics import geometry as v1_geometry, normalize
from di_dmpa_gate1.gate1a_reporting import artifact_manifest
from di_dmpa_gate1_v2.binding import FILE_HASHES, PLAN_SHA, PANELS, NonfiniteFeature, InvalidCenter, IncompletePanel
from di_dmpa_gate1_v2.features import split_support, validate_cache, extract_unit, ImmutableModels
from di_dmpa_gate1_v2.geometry import metrics, fit, matched_cosines, verify_null_identity
from di_dmpa_gate1_v2.census import complete_feature_keys
from di_dmpa_gate1_v2.reporting import adjudicate, validate_rows

ROOT=Path(__file__).resolve().parents[2]


def fixture():
    raw=np.zeros((3,16));raw[0,0]=1;raw[1,1]=1
    return split_support(raw),np.array([.4,.4,.2])


def measurement(a,w,K=1):
    f=fit(a['directions'],a['active_mask'],w,K,seed=0,stage=0,class_id=1)
    return metrics(a['directions'],a['active_mask'],w,f['centers'],f['active'])


def test_null_vector_is_preserved_not_dropped():
    a,w=fixture();assert len(a['directions'])==3 and not a['active_mask'][2] and np.all(a['directions'][2]==0)


def test_null_vector_is_never_normalized():
    with np.errstate(divide='raise',invalid='raise'):
        a=split_support(np.zeros((4,16)))
    assert np.all(a['directions']==0) and not a['active_mask'].any()


def test_active_vector_is_unit_normalized():
    a=split_support(np.arange(1,65).reshape(4,16))
    np.testing.assert_allclose(np.linalg.norm(a['directions'],axis=1),1,atol=1e-15)


def test_cache_row_count_equals_registered_coordinates():
    a,w=fixture();validate_cache(a,3)
    with pytest.raises(ProtocolError):validate_cache({k:v[:2] for k,v in a.items()},3)


def test_active_mask_matches_threshold_exactly():
    raw=np.zeros((4,16));raw[:,0]=[0,1e-13,1e-12,np.nextafter(1e-12,np.inf)]
    a=split_support(raw);assert a['active_mask'].tolist()==[False,False,False,True]
    a['active_mask'][0]=True
    with pytest.raises(ProtocolError):validate_cache(a,4)


def test_null_worst_case_distance_is_two():
    a=split_support(np.zeros((3,16)));m=measurement(a,np.ones(3)/3)
    assert m['Q_null_worst_case']==m['R95_null_worst_case']==m['cosine_distance_p95_null_worst_case']==2


def test_null_weight_enters_mean_distortion():
    a,w=fixture();m=measurement(a,w)
    assert np.isclose(m['Q_null_worst_case'],.8*m['Q_directional_conditional']+.4)
    assert m['null_count']==1 and m['null_mass']==.2


def test_null_weight_enters_weighted_ecdf():
    a,w=fixture();m=measurement(a,w)
    assert m['R95_null_worst_case']==2 and m['R95_directional']<2


def test_null_term_is_K_independent():
    a,w=fixture();m1=measurement(a,w,1);m2=measurement(a,w,2)
    assert verify_null_identity(m1,m2)['status']=='PASS'


def test_null_cannot_create_K_improvement():
    a=split_support(np.zeros((3,16)));w=np.ones(3)/3
    assert measurement(a,w,1)['R95_null_worst_case']==measurement(a,w,5)['R95_null_worst_case']


def test_no_null_matches_v1_geometry_exactly():
    rng=np.random.default_rng(4);a=split_support(rng.uniform(.01,1,(30,16)));w=np.ones(30)/30
    result=fit(a['directions'],a['active_mask'],w,3,seed=1,stage=2,class_id=1)
    new=metrics(a['directions'],a['active_mask'],w,result['centers'],result['active'])
    old=v1_geometry(a['directions'],w,result['centers'],result['active'])
    for n,o in [('Q_null_worst_case','Q_K'),('R95_null_worst_case','R95'),('cosine_distance_p95_null_worst_case','cosine_distance_p95')]:assert new[n]==old[o]
    assert new['occupancy']==old['occupancy']


def test_high_null_mass_makes_R95_worst_case_two():
    a,w=fixture();assert measurement(a,np.array([.02,.02,.96]))['R95_null_worst_case']==2


def test_all_null_original_unit_is_scientific_failure_not_exception():
    a=split_support(np.zeros((3,16)));f=fit(a['directions'],a['active_mask'],np.ones(3),5,seed=0,stage=0,class_id=1)
    assert f['directional_support']=='NONE' and not f['active'].any() and f['centers'].shape==(5,16)


def test_all_null_bootstrap_has_inactive_slots_and_zero_stability():
    a,w=fixture();original=fit(a['directions'],a['active_mask'],w,3,seed=0,stage=0,class_id=1)
    boot=fit(a['directions'],a['active_mask'],np.array([0.,0.,1.]),3,seed=0,stage=0,class_id=1,replicate=2)
    assert len(boot['restarts'])==5 and np.all(matched_cosines(original['centers'],original['active'],boot['centers'],boot['active'])==0)


@pytest.mark.parametrize('value',[np.nan,np.inf])
def test_nonfinite_full_map_still_blocks(monkeypatch,value):
    from di_dmpa_gate1_v2 import features
    class Model(torch.nn.Module):
        def forward(self,image,**kw):
            f=torch.ones(1,16,384,384);f[0,0,383,383]=value
            return f[:,:3],f
    unit=dict(role='val',pixel_sampling_seed=0,cases=[dict(case_id='a',classes=[dict(coordinates=[[0,0]],coordinate_sha256=H([['a',c,0,0]])) for c in range(3)])])
    monkeypatch.setattr(features,'_images',lambda *args:torch.ones(1,3,384,384))
    with pytest.raises(NonfiniteFeature):extract_unit(Model().eval(),unit,'unused',{},device='cpu')


def test_nonfinite_registered_feature_still_blocks():
    raw=np.zeros((2,16));raw[0,0]=np.nan
    with pytest.raises(NonfiniteFeature):split_support(raw)


def test_active_zero_center_still_blocks():
    a,w=fixture()
    with pytest.raises(InvalidCenter):metrics(a['directions'],a['active_mask'],w,np.zeros((1,16)),[True])


def test_same_sampling_plan_bytes_as_v1():
    path=ROOT/'docs/di_dmpa_jascl/gate1a_results/gate1a_formal_8f4a71a_attempt1/SHARED_GEOMETRY_SAMPLING_PLAN.json'
    assert sha256(path)==PLAN_SHA


def test_attempt1_and_attempt2_artifacts_unchanged():
    paths=[('gate1a_results/gate1a_formal_8f4a71a_attempt1','c26edceea102da568421e0327a7cc10fabb2ceee16fc936dd8adeb439eab8ee9'),
           ('gate1a_recovery_results/gate1a_formal_a89716ddbd2eccbe76c574e97e520d424aa923ab_attempt2','15e7beaf67ad55bd1c18b494f53c7f06fc6ea92ff161f99ccf62b6a648736ea3')]
    for relative,expected in paths:
        folder=ROOT/'docs/di_dmpa_jascl'/relative
        assert sha256(folder/'GATE1A_ARTIFACT_MANIFEST.json')==expected
        assert read_json(folder/'GATE1A_STATUS.json')['prototype_geometry_status']=='BLOCKED_NUMERICAL_FAILURE'


def test_all72_features_required_before_first_geometry_job():
    entries=[dict(panel_id=p,seed=s,stage_index=t,role=r) for p in PANELS for s in range(3) for t in range(3) for r in ('train_labeled','val')]
    complete_feature_keys(entries)
    with pytest.raises(IncompletePanel):complete_feature_keys(entries[:-1])
    with pytest.raises(IncompletePanel):complete_feature_keys(entries[:-1]+[entries[0]])


def fake_rows(primary=.7,control=1.1):
    rows=[]
    for p in PANELS:
        for s in range(3):
            for t in range(3):
                for c in range(3):
                    for K in (1,2,3,5):
                        radius=1. if K==1 else primary if p=='B0-EMA' else control
                        m=dict(metric_schema='NULL_AWARE_SPHERE_V2',admission_radius_field='R95_null_worst_case',registered_count=100,
                            active_count=100,null_count=0,full_uid_count_used=100,null_rows_retained=0,null_mass=0.,active_direction_mass=1.,
                            Q_directional_conditional=radius,Q_null_worst_case=radius,R95_null_worst_case=radius,occupancy=[1/K]*K)
                        rows.append(dict(panel_id=p,seed=s,stage_index=t,class_id=c,K=K,admission_radius_field='R95_null_worst_case',
                            metrics={r:dict(m) for r in ('train_labeled','val')},expected_registered_counts={r:100 for r in ('train_labeled','val')},
                            expected_null_counts={r:0 for r in ('train_labeled','val')},fit={'directional_support':'PRESENT'},
                            bootstrap=[dict(matched_cosines=[.9]*K) for _ in range(5)]))
    return rows


def test_primary_only_adjudication():
    result=adjudicate(fake_rows());assert result['selected_K']==2 and not result['control_thresholds_applied']


def test_controls_cannot_rescue():
    result=adjudicate(fake_rows(1.1,.5));assert result['selected_K']==1 and result['prototype_geometry_status']=='FAIL_MULTI_MODALITY_NOT_SUPPORTED'


def test_no_optimizer_construction(monkeypatch):
    from di_dmpa_gate1_v2 import features
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
        def forward(self,*args,**kw):torch.optim.SGD(self.parameters(),lr=.1)
    monkeypatch.setattr(features,'_images',lambda *args:torch.ones(1,3,384,384))
    unit=dict(role='val',pixel_sampling_seed=0,cases=[dict(classes=[dict(coordinates=[[0,0]])]*3)])
    with pytest.raises(ProtocolError,match='optimizer construction'):extract_unit(Model().eval(),unit,'unused',{},device='cpu')


def test_model_and_checkpoint_immutability(tmp_path):
    path=tmp_path/'synthetic.pt';path.write_bytes(b'synthetic')
    cp=dict(path=str(path),checkpoint_id='synthetic',sha256=sha256(path))
    with ImmutableModels({'student':torch.nn.Linear(16,3)},cp,tmp_path,{}):pass
    a=read_json(tmp_path/'immutability/synthetic.json');assert a['bitwise_unchanged'] and a['checkpoint_sha256_before']==a['checkpoint_sha256_after']


def test_null_aware_boundary_interior():
    a,w=fixture();f=fit(a['directions'],a['active_mask'],w,1,seed=0,stage=0,class_id=1)
    mask=np.array([True,False,True])
    m=metrics(a['directions'][mask],a['active_mask'][mask],np.ones(2)/2,f['centers'],f['active'])
    assert m['null_count']==1 and m['null_mass']==.5 and m['R95_null_worst_case']==2


def test_report_compiler_rejects_conditional_R95_used_for_A1_A2_A5():
    rows=fake_rows();rows[0]['admission_radius_field']='R95_directional'
    with pytest.raises(ProtocolError,match='conditional R95'):validate_rows(rows)


def test_report_compiler_rejects_missing_null_rows():
    rows=fake_rows();rows[0]['metrics']['val']['registered_count']=99
    with pytest.raises(ProtocolError,match='missing UID'):validate_rows(rows)


def test_report_compiler_rejects_hidden_null_drop():
    rows=fake_rows();rows[0]['expected_null_counts']['val']=1
    with pytest.raises(ProtocolError,match='hidden null'):validate_rows(rows)


def test_artifact_manifest(tmp_path):
    write_json(tmp_path/'census.json',{'rows':3,'null_rows':1})
    m=artifact_manifest(tmp_path);assert all(sha256(tmp_path/f['path'])==f['sha256'] for f in m['files'])


def test_v2_registration_hash_binding():
    for suffix,digest in FILE_HASHES.items():assert sha256(ROOT/f'docs/di_dmpa_jascl/DI_DMPA_GATE1A_V2_PREREGISTRATION.{suffix}')==digest


def test_all_primary_null_is_directional_scientific_failure():
    rows=fake_rows()
    for r in rows:
        r['fit']['directional_support']='NONE'
        for m in r['metrics'].values():
            m.update(active_count=0,null_count=100,null_rows_retained=100,null_mass=1.,active_direction_mass=0.,
                Q_directional_conditional=None,Q_null_worst_case=2.,R95_null_worst_case=2.,occupancy=[0.]*r['K'])
        r['expected_null_counts']={k:100 for k in ('train_labeled','val')}
        for b in r['bootstrap']:b['matched_cosines']=[0.]*r['K']
    result=adjudicate(rows);assert result['prototype_geometry_status']=='FAIL_DIRECTIONAL_SUPPORT_NOT_SUPPORTED' and result['selected_K']==1
