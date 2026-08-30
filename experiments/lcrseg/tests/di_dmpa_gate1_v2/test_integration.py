"""Opt-in exact-code checkpoint probe of the historical registered zero."""
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from di_dmpa_gate1.feature_extraction import load_models
from di_dmpa_gate1_v2.binding import H, ATTEMPT1, PLAN_SHA, ProtocolError, check_hash, read_json, verify, write_json
from di_dmpa_gate1_v2.features import ImmutableModels, extract_unit, validate_cache
from di_dmpa_gate1_v2.geometry import metrics


def test_known_registered_zero_preserved_with_worst_case_two():
    destination=os.environ.get('GATE1A_V2_INTEGRATION_OUTPUT')
    if not destination:pytest.skip('Explicit exact-code integration output required')
    output=Path(destination);output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    p,old,metadata=verify(root,os.environ['GATE1A_CODE_COMMIT'])
    check_hash(ATTEMPT1/'SHARED_GEOMETRY_SAMPLING_PLAN.json',PLAN_SHA)
    plan=read_json(ATTEMPT1/'SHARED_GEOMETRY_SAMPLING_PLAN.json')
    unit=next(u for u in plan['units'] if u['seed']==2 and u['stage_index']==0 and u['role']=='train_labeled')
    case_index=next(i for i,c in enumerate(unit['cases']) if c['case_id']=='REFUGE_test_n0128')
    start=(case_index//8)*8;batch=unit['cases'][start:start+8]
    probe={**unit,'cases':batch}
    checkpoint=next(c for c in old['immutable_baseline']['checkpoint_inputs'] if c['checkpoint_id']=='B0/seed2/stage0')
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=ProtocolError('optimizer construction forbidden')):
        models,payload=load_models(root,checkpoint,device='cuda:0')
        assert payload['config_hash']==p['immutable_baseline']['configs']['B0']['resolved_config_sha256']
        with ImmutableModels(models,checkpoint,output,metadata):
            arrays,census=extract_unit(models['ema_teacher'],probe,'/root/LCRSeg',{},device='cuda:0',batch_size=8)
    c=1;offset=0;selected=None
    for case in batch:
        coords=case['classes'][c]['coordinates']
        if case['case_id']=='REFUGE_test_n0128':selected=offset+coords.index([125,212]);break
        offset+=len(coords)
    a=arrays[c];validate_cache(a,len(a['active_mask']))
    assert selected is not None and not a['active_mask'][selected] and a['raw_norms'][selected]==0
    assert np.all(a['directions'][selected]==0)
    # A fixed synthetic unit center, not any prototype fit/clustering.
    center=np.zeros((1,16));center[0,0]=1
    m=metrics(a['directions'][selected:selected+1],a['active_mask'][selected:selected+1],np.ones(1),center,np.ones(1,dtype=bool))
    assert m['Q_null_worst_case']==m['R95_null_worst_case']==2
    audit=read_json(output/'immutability/B0_seed2_stage0.json')
    write_json(output/'GATE1A_V2_REAL_INTEGRATION.json',dict(status='PASS',metadata=metadata,
        checkpoint_id=checkpoint['checkpoint_id'],checkpoint_sha256=checkpoint['sha256'],sampling_unit_sha256=H(unit),
        case_id='REFUGE_test_n0128',class_id=1,coordinate=[125,212],registered_row_index=selected,
        known_null_row_retained=True,active_mask=False,raw_norm=0.,null_normalization_performed=False,
        original_registered_count=len(a['active_mask']),cache_rows=len(a['directions']),worst_case_distance=2.,
        batch_case_ids=[c['case_id'] for c in batch],batch_size=8,clustering_jobs=0,
        model_unchanged=audit['bitwise_unchanged'],model_optimizer_steps=0,transport_optimizer_steps=0,
        hidden_gt_training_usage='none',test_gt_usage='none'))
