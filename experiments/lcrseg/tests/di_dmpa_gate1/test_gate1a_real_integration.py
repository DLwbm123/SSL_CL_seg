"""Explicit opt-in, read-only checkpoint/coordinate integration; no geometry fit."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from di_dmpa_gate1.binding import (H,check_hash,gate1a_records,run_metadata,verify_registration,write_json)
from di_dmpa_gate1.feature_extraction import _images,load_models,seed_after_load,state_groups
from di_dmpa_gate1.geometry_metrics import normalize


def test_real_checkpoint_fixed_coordinates_read_only():
    destination=os.environ.get('GATE1A_REAL_CHECKPOINT_REPORT')
    if not destination:
        pytest.skip('Real checkpoint integration requires explicit report destination and committed source')
    root=Path(__file__).resolve().parents[2]
    code_commit=os.environ['GATE1A_CODE_COMMIT']
    prereg,receipt=verify_registration(root,code_commit)
    checkpoint=prereg['immutable_baseline']['checkpoint_inputs'][0]
    assert checkpoint['checkpoint_id']=='B0/seed0/stage0'
    check_hash(root.parents[1]/prereg['immutable_baseline']['freeze_path'],prereg['immutable_baseline']['freeze_sha256'])
    cases=gate1a_records('/root/LCRSeg',prereg,0,'REFUGE','train_labeled')
    case=cases[0]
    coords=[[0,0],[0,383],[96,96],[191,191],[192,192],[287,287],[383,0],[383,383]]
    probe_plan={'kind':'INTEGRATION_ONLY_NOT_SHARED_GEOMETRY_PLAN','case_id':case['case_id'],'coordinates':coords}
    metadata=run_metadata(prereg,receipt,H(probe_plan),panel_id='ALL_FOUR_SEPARATE')
    metadata['execution_scope']='READ_ONLY_INTEGRATION_TEST_NOT_ADMISSION'
    metadata['sources_tested']=['B0-EMA','B0-student']
    models,payload=load_models(root,checkpoint,device='cuda:0')
    assert payload['config_hash']==prereg['immutable_baseline']['configs']['B0']['resolved_config_sha256']
    before={k:state_groups(v) for k,v in models.items()}
    seed=prereg['shared_sampling']['plans'][0]['pixel_sampling_seed']
    seed_after_load(seed)
    image=_images([case],'/root/LCRSeg').to('cuda:0')
    outputs={}
    for source,model in models.items():
        with torch.no_grad():
            _,features=model(image,stochastic_classifier=False)
            _,repeat=model(image,stochastic_classifier=False)
        assert features.shape==(1,16,384,384) and torch.equal(features,repeat)
        y,x=np.asarray(coords).T
        selected=features[0,:,y,x].T.cpu().numpy()
        unit=normalize(selected)
        assert unit.shape==(8,16) and np.isfinite(unit).all()
        assert all(parameter.grad is None for parameter in model.parameters())
        outputs[source]={'selected_shape':[8,16],'dtype':str(unit.dtype),'repeat_exact':True,
                         'selected_normalized_feature_sha256':H(unit.tolist()),'no_geometry_metrics_computed':True}
    after={k:state_groups(v) for k,v in models.items()}
    assert before==after
    check_hash(checkpoint['path'],checkpoint['sha256'])
    write_json(destination,dict(status='PASS',metadata=metadata,checkpoint_sha256=checkpoint['sha256'],
                 probe_plan=probe_plan,outputs=outputs,before=before,after=after,bitwise_unchanged=True,
                 label_arrays_read=0,segmentation_optimizer_constructed=False,transport_optimizer_constructed=False,
                 model_optimizer_steps=0,transport_optimizer_steps=0,admission_inference_performed=False))
