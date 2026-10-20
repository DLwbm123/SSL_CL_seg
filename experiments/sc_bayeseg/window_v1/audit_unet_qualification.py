"""Audit the fixed U-Net run and its sealed metrics and read-only summaries."""
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset, collate_labeled
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import metrics, write
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.unet_qualification import build_model


def audit(s):
    torch.set_num_threads(4)
    root=Path(s['output'])
    load=lambda name:json.loads((root/name).read_text())
    pt=lambda path:torch.load(path,map_location='cpu',weights_only=False)
    assert load('RESULTS.json')['status']=='COMPLETE' and load('EXIT.json')['exit_code']==0
    assert load('STATUS.json')['status']=='COMPLETE' and not (root/'FAILURE.json').exists()
    costs=dict(optimizer=3000,reverse=3000,train_image_forward=24000,prediction_image_forward=48,
               generator_forward=0,diagnostic_encoder_image_forward=72,diagnostic_posterior_batch_forward=9)
    assert load('RESULTS.json')['costs']==costs
    updates=[json.loads(x) for x in (root/'physical_updates.jsonl').read_text().splitlines()]
    steps=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()]
    assert [r['index'] for r in updates]==[r['step'] for r in updates]==list(range(1,3001))
    assert [r['step'] for r in steps]==list(range(1,3001))
    torch.manual_seed(20261020)
    initial=build_model().state_dict()
    for step in range(0,3001,250):
        state=pt(root/f'S{step:04}.pt')
        assert state['step']==step and 'rng' in state
        assert all(torch.isfinite(v).all() for v in state['model'].values())
        assert not any(k.startswith('projection_head.') for k in state['model'])
        if step==0:
            assert same(state['model'],initial) and not state['optimizer']['state']
        else:
            assert all(int(v['step'])==step for v in state['optimizer']['state'].values())
    assert load('DESIGN.json')['parameters']==sum(v.numel() for v in initial.values())
    sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
    ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
    lb=collate_labeled([ds[i] for i in sel['train_indices']])
    y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
    y=torch.cat([flip(y,k) for k in range(4)])
    assert torch.bincount(y.flatten(),minlength=4).tolist()==load('DESIGN.json')['class_counts_all_views']
    rows=[json.loads(x) for x in (root/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==[0,250,500,1000,2000,3000]
    error=[0.]
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,z in zip(a,b):compare(x,z)
        elif isinstance(a,float):
            error[0]=max(error[0],abs(a-b))
            assert abs(a-b)<2e-6
        else:assert a==b
    for row in rows:
        p=pt(root/f"PRED_{row['step']:04}.pt")
        assert list(p)==['probability']
        compare(row['metrics'],metrics(p['probability'],y))
    q=load('POSTERIOR_SCALE.json')
    assert q['status']=='COMPLETE'
    assert [(r['arm'],r['step']) for r in q['rows']]==[(a,k) for a in ('R1','R2','R2-SG') for k in (0,200,600)]
    for row in q['rows']:
        for key,d in [('c',16),('R',4096)]:
            v=row['variables'][key];floor=d/(128**2)*.5*(math.exp(-2)+1)
            assert abs(v['family_KL_infimum_per_pixel']-floor)<1e-12
            assert abs(v['mean_KL_per_pixel']+v['scale_KL_per_pixel']-v['total_KL_per_pixel'])<1e-12
            assert abs(v['mean_squared']*.5*d/(128**2)-v['mean_KL_per_pixel'])<1e-12
            assert abs(v['total_KL_per_pixel']-floor-v['KL_above_infimum_per_pixel'])<1e-12
            assert 0<=v['near_upper_logsd_minus1_fraction']<=1
            assert -7<=min(v['logsd_quantiles'])<=max(v['logsd_quantiles'])<=-1
    assert q['costs']==dict(encoder_image_forward=72,posterior_batch_forward=9,optimizer=0,reverse=0,generator_forward=0)
    original=json.loads((Path(s['r_start'])/'CONTEXT.json').read_text())
    assert s['original_started_unix']==original['original_started_unix']
    assert load('RESULTS.json')['original_elapsed_seconds']<86400
    assert load('RESULTS.json')['parameter_displacement_L2']>0
    result=dict(status='PASS',optimizer=3000,reverse=3000,complete_step_coverage=True,
                complete_model_Adam_RNG_checkpoints=13,Adam_steps=[0,3000],fresh_initialization_exact=True,
                sealed_prediction_records_checked=6,maximum_metric_recompute_error=error[0],
                posterior_summary_records_checked=9,KL_decomposition_closed=True,original_clock_preserved=True,
                no_generator_or_H_training=True,no_patient_generalization_claim=True)
    write(root/'AUDIT.json',result)
    print(json.dumps(result))
    return result


if __name__=='__main__':
    audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
