"""Scope-only synthetic recovery tests. No real checkpoint/forward by default."""
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from di_dmpa_gate1.binding import H, FILE_HASHES, NumericalError, ProtocolError, read_json, sha256
from di_dmpa_gate1.feature_extraction import ImmutabilityGuard, extract_unit, state_groups
from di_dmpa_gate1.geometry_metrics import normalize
from di_dmpa_gate1.registered_features import RegisteredFeatureNumericalError, inspect_registered_case
from di_dmpa_gate1.recovery import PLAN_SHA, request_stop, reuse_sampling_plan, verify_localization
from di_dmpa_gate1.gate1a_runner import checkpoint_sequence

ROOT=Path(__file__).resolve().parents[2]


def fixture():
    feature=np.arange(16*4*4,dtype=np.float32).reshape(16,4,4)+1
    case={'case_id':'case-a','classes':[{'coordinates':[[c,0],[c,1]]} for c in range(3)]}
    context=dict(panel_id='B0-EMA',baseline='B0',feature_source='ema_teacher',seed=1,stage_index=0,
        domain='REFUGE',role='val',checkpoint_id='B0/seed1/stage0',checkpoint_sha256='a'*64,
        sampling_unit_sha256='b'*64,sampling_plan_sha256=PLAN_SHA)
    return feature,case,context


def test_unsampled_full_map_zero_is_diagnostic_only():
    feature,case,context=fixture();feature[:,3,3]=0
    arrays,audit=inspect_registered_case(feature,case,context)
    assert audit['full_map_exact_zero_count']==audit['full_map_norm_le_1e12_count']==1
    assert not any(audit['registered_zero_count_by_class'].values())
    assert audit['first_zero_coordinates']==[[3,3]] and audit['full_map_zero_coordinate_sha256']==H([[3,3]])
    assert all(len(x)==2 for x in arrays.values())


def test_unsampled_zero_does_not_change_selected_cache_hash():
    feature,case,context=fixture()
    original,_=inspect_registered_case(feature,case,context)
    feature[:,3,3]=0
    altered,_=inspect_registered_case(feature,case,context)
    assert [H(x.tolist()) for x in original.values()]==[H(x.tolist()) for x in altered.values()]


def test_registered_zero_feature_blocks():
    feature,case,context=fixture();feature[:,1,0]=0
    with pytest.raises(RegisteredFeatureNumericalError) as caught:
        inspect_registered_case(feature,case,context)
    assert caught.value.case_audits[0]['registered_zero_count_by_class']=={'0':0,'1':1,'2':0}


def test_registered_zero_error_has_complete_provenance():
    feature,case,context=fixture();feature[:,1,0]=0
    with pytest.raises(RegisteredFeatureNumericalError) as caught:inspect_registered_case(feature,case,context)
    p=caught.value.provenance
    assert all(p[k]==v for k,v in context.items())
    assert (p['case_id'],p['class_id'],p['coordinate_y'],p['coordinate_x'])==('case-a',1,1,0)
    assert p['total_invalid_count']==1 and p['minimum_selected_norm']==0


def test_full_map_nan_blocks_even_if_unsampled():
    feature,case,context=fixture();feature[0,3,3]=np.nan
    with pytest.raises(NumericalError):inspect_registered_case(feature,case,context)


def test_full_map_inf_blocks_even_if_unsampled():
    feature,case,context=fixture();feature[0,3,3]=np.inf
    with pytest.raises(NumericalError):inspect_registered_case(feature,case,context)


def test_cluster_center_zero_still_blocks():
    with pytest.raises(NumericalError):normalize(np.zeros((1,16)))


def test_selected_features_are_unit_norm_after_normalization():
    feature,case,context=fixture()
    arrays,audit=inspect_registered_case(feature,case,context)
    for c,array in arrays.items():
        assert array.dtype==np.float64
        np.testing.assert_allclose(np.linalg.norm(array,axis=1),1,atol=2e-15)
        assert audit['normalized_norm_max_abs_error_by_class'][str(c)]<=2e-15


def test_no_eps_no_drop_no_resample():
    feature,case,context=fixture()
    original=H(case)
    feature[:,1,0]=1e-14
    with pytest.raises(RegisteredFeatureNumericalError):inspect_registered_case(feature,case,context)
    assert H(case)==original
    source=inspect.getsource(inspect_registered_case)
    assert 'selected / sn[:, None]' in source
    assert 'normalize(' not in source and 'random' not in source


def test_attempt1_sampling_plan_reused_byte_exact(tmp_path):
    source=ROOT/'docs/di_dmpa_jascl/gate1a_results/gate1a_formal_8f4a71a_attempt1/SHARED_GEOMETRY_SAMPLING_PLAN.json'
    plan,digest=reuse_sampling_plan(source,tmp_path/'reused')
    copied=tmp_path/'reused/SHARED_GEOMETRY_SAMPLING_PLAN.json'
    assert digest==sha256(source)==sha256(copied)==PLAN_SHA and len(plan['units'])==18
    assert copied.stat().st_mode & 0o222==0
    assert not read_json(tmp_path/'reused/SAMPLING_PLAN_AUDIT.json')['coordinates_rematerialized']
    with pytest.raises(ProtocolError):reuse_sampling_plan(source,tmp_path/'reused')


def test_original_preregistration_unchanged():
    for suffix,digest in FILE_HASHES.items():
        assert sha256(ROOT/f'docs/di_dmpa_jascl/DI_DMPA_GATE1_PREREGISTRATION.{suffix}')==digest


@pytest.mark.parametrize('failure',[False,True],ids=['success','registered-feature-failure'])
def test_model_state_unchanged_on_success_or_registered_feature_failure(tmp_path,failure):
    feature,case,context=fixture()
    if failure:feature[:,1,0]=0
    model=torch.nn.Linear(16,3).eval()
    path=tmp_path/'synthetic.pt';path.write_bytes(b'synthetic')
    checkpoint=dict(checkpoint_id='synthetic',path=str(path),sha256=sha256(path))
    def run():
        with ImmutabilityGuard({'ema_teacher':model},checkpoint,tmp_path,{}):
            inspect_registered_case(feature,case,context)
    if failure:
        with pytest.raises(RegisteredFeatureNumericalError):run()
    else:run()
    assert read_json(tmp_path/'immutability/synthetic.json')['bitwise_unchanged']


def test_no_optimizer_construction(monkeypatch):
    from di_dmpa_gate1 import feature_extraction as ex
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
        def forward(self,image,**kwargs):
            torch.optim.SGD(self.parameters(),lr=.1)
    unit=dict(role='val',pixel_sampling_seed=20262930,cases=[dict(classes=[{'coordinates':[[0,0]]}]*3)])
    monkeypatch.setattr(ex,'_images',lambda cases,root:torch.ones(1,3,384,384))
    with pytest.raises(ProtocolError,match='optimizer construction forbidden'):
        extract_unit(Model().eval(),unit,'unused',device='cpu')


def test_graceful_shard_cancellation_preserves_current_checkpoint_audit(tmp_path):
    path=tmp_path/'synthetic.pt';path.write_bytes(b'synthetic')
    checkpoints=[dict(checkpoint_id=name,path=str(path),sha256=sha256(path)) for name in ('current','next')]
    executed=[]
    def execute(checkpoint):
        with ImmutabilityGuard({'ema_teacher':torch.nn.Linear(2,2)},checkpoint,tmp_path,{}):
            request_stop(tmp_path,'another shard failed during current checkpoint')
            executed.append(checkpoint['checkpoint_id'])
    checkpoint_sequence(checkpoints,tmp_path,0,execute)
    assert executed==['current']
    assert read_json(tmp_path/'immutability/current.json')['bitwise_unchanged']
    assert read_json(tmp_path/'shards/extract_0_cancelled.json')['current_checkpoint_guard_completed']
    assert not (tmp_path/'immutability/next.json').exists()


def test_registered_zero_localization_cannot_authorize_attempt2(tmp_path):
    from di_dmpa_gate1.binding import write_json
    report=tmp_path/'localization.json'
    write_json(report,{'localization_status':'BLOCKED_REGISTERED_ZERO_FEATURE'})
    with pytest.raises(ProtocolError,match='attempt2 forbidden'):verify_localization(report,'unused')


def test_all_registered_cases_audited_without_invalid_cache(monkeypatch):
    from di_dmpa_gate1 import feature_extraction as ex
    class Model(torch.nn.Module):
        def forward(self,image,**kwargs):
            feature=torch.ones(len(image),16,384,384);feature[:,:,1,0]=0
            return feature[:,:3],feature
    cases=[dict(case_id=f'case-{i}',classes=[{'coordinates':[[c,0]]} for c in range(3)]) for i in range(9)]
    unit=dict(role='val',pixel_sampling_seed=20262930,cases=cases)
    monkeypatch.setattr(ex,'_images',lambda cases,root:torch.ones(len(cases),3,384,384))
    with pytest.raises(RegisteredFeatureNumericalError) as caught:
        extract_unit(Model().eval(),unit,'unused',device='cpu',batch_size=8,collect_all_invalid=True)
    assert [c['case_id'] for c in caught.value.case_audits]==[c['case_id'] for c in cases]
    assert len(caught.value.all_failures)==9


def test_full_map_zero_examples_are_bounded():
    feature,case,context=fixture()
    large=np.zeros((16,64,64));large[:,:4,:4]=feature
    _,audit=inspect_registered_case(large,case,context)
    assert audit['full_map_exact_zero_count']>32 and len(audit['first_zero_coordinates'])==32
