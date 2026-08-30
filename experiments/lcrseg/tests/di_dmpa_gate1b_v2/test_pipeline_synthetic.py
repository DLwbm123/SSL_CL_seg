"""Exercise real orchestration with synthetic images/models, never a real checkpoint."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from di_dmpa_gate1_v2 import features
from di_dmpa_gate1_v2.features import weight_hash
from di_dmpa_gate1b_v2 import binding, plan, pairs, transport, evaluator, runner


class SyntheticEncoder(torch.nn.Module):
    def __init__(self,stage):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(16),requires_grad=False)
        self.weight[stage]=2
        self.eval()

    def forward(self,images,stochastic_classifier=False):
        assert not stochastic_classifier and not self.training and not torch.is_grad_enabled()
        x=self.weight[None,:,None,None].expand(len(images),16,384,384).clone();x[:,:,0,0]=0
        return torch.zeros((len(images),3,384,384)),x


def test_synthetic_paired_census_six_fits_and_chain_oracle(tmp_path):
    torch.set_num_threads(1);meta=dict(synthetic=True,diagnostic_code_commit='synthetic',selected_K=2)
    p={'immutable_baseline':{'checkpoint_inputs':[]}};records=[]
    for s in range(3):
        for t in range(3):
            path=tmp_path/f'fake_B0_seed{s}_stage{t}.pt';binding.write_text(path,'SYNTHETIC NOT A TORCH CHECKPOINT')
            p['immutable_baseline']['checkpoint_inputs'].append(dict(checkpoint_id=f'B0/seed{s}/stage{t}',seed=s,stage_index=t,domain=binding.DOMAINS[t],baseline='B0',path=str(path),sha256=binding.sha256(path)))
            for c in range(3):records.append(dict(seed=s,stage_index=t,class_id=c,panel='B0-EMA',K=2,active_mask=[True,True],converged=True,
                training_source='train_labeled',operational_refit_allowed=False,centers=np.eye(16)[2*c:2*c+2].tolist()))
    units=[];coords=[[i//384,i%384] for i in range(2048)]
    for s in range(3):
        for t in (1,2):
            for partition in ('fit','holdout'):
                name=f'SYNTHETIC_{s}_{t}_{partition}'
                case=dict(case_id=name,image_h5_relpath='synthetic/image.h5',image_sha256='a'*64,coordinates=coords,coordinate_uid_sha256=binding.H([[name,y,x] for y,x in coords]))
                u=dict(seed=s,stage_index=t,domain=binding.DOMAINS[t],role='train_unlabeled',partition=partition,split_seed=20261830+100*s+t,
                    split_hash='b'*64,cases=[case],case_count=1,registered_count=2048)
                lay=plan.layout(u);u.update(coordinate_uid_hash=binding.H(lay['uids']),original_weight_hash=weight_hash(lay['weights']));units.append(u)
    shared={'units':units};digest=binding.write_json(tmp_path/'SHARED_TRANSPORT_COORDINATE_PLAN.json',shared)
    meta['transport_coordinate_plan_sha256']=digest;binding.write_json(tmp_path/'GATE1B_V2_RUN_METADATA.json',meta)
    def models(root,p,seed,stage,device):return {name:SyntheticEncoder(stage) for name in ('student','ema_teacher')}
    def images(cases,root):return torch.zeros((len(cases),3,384,384),dtype=torch.float32)
    with patch.object(pairs,'load_b0',models),patch.object(pairs,'_images',images):
        for seed in range(3):
            for stage in (1,2):pairs.extract_transition('.', '.', p,[u for u in units if u['seed']==seed and u['stage_index']==stage],tmp_path,meta,device='cpu')
    entries=[binding.read_json(path) for path in sorted((tmp_path/'paired_units').glob('*.json'))]
    counts=pairs.census(tmp_path,entries,shared,meta)
    assert counts['paired_units_completed']==12 and counts['registered_count']==24576 and counts['counts']['NULL_NULL']==12
    with patch.object(runner,'disk_audit',return_value={'synthetic':'not real checkpoint'}):
        audit=runner.model_audit(tmp_path,p,{},meta)
    assert audit['model_load_guards']==12
    binding.write_json(tmp_path/'TRANSFORM_START_BARRIER.json',dict(status='PASS',transport_optimizer_steps=0,paired_units=12,
        evidence_sha256={'PAIRED_FEATURE_SUPPORT_CENSUS.json':binding.sha256(tmp_path/'PAIRED_FEATURE_SUPPORT_CENSUS.json')}))
    with patch.object(runner,'operational',return_value=np.eye(16)[:6].reshape(3,2,16)):
        paths=[runner.fit_job((str(tmp_path),seed,stage)) for seed in range(3) for stage in (1,2)]
    assert runner.actual_steps(tmp_path)==6000
    binding.write_json(tmp_path/'ORACLE_START_BARRIER.json',dict(six_transports_complete=True,transport_optimizer_steps=6000,
        model_sha256={Path(path).name:binding.sha256(path) for path in paths}))
    classes=[dict(class_id=c,coordinates=[[c,0],[c,1]],sampled_pixels=2,boundary=[False,False],
        coordinate_sha256=binding.H([['SYNTHETIC_VAL',c,c,x] for x in range(2)])) for c in range(3)]
    oracle_unit=dict(seed=0,stage_index=0,domain='REFUGE',role='val',gt_consumer='diagnostic_evaluator_only',pixel_sampling_seed=1,
        cases=[dict(case_id='SYNTHETIC_VAL',image_h5_relpath='synthetic/image.h5',image_sha256='a'*64,classes=classes)])
    digest=binding.write_json(tmp_path/'FROZEN_GEOMETRY_SAMPLING_PLAN.json',dict(units=[oracle_unit]))
    with patch.object(evaluator,'PLAN_SHA',digest),patch.object(evaluator,'load_b0',models),patch.object(features,'_images',images):
        out=evaluator.evaluate_unit('.','.',p,dict(prototype_records=records),tmp_path,meta,0,0,2,device='cpu')
    assert out['kind']=='chain' and not out['transform_fit_called'] and not out['operational_refit']
    assert out['class_caches'][0]['null_count']==1 and all(f['source_stage_for_clustering_seeds']==0 for f in out['oracle_fits'])
    assert set(out['metrics'])=={'T0','T1','T2'}
