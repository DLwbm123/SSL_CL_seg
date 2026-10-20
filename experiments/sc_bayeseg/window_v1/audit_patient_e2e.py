"""Audit full end-to-end training states and sealed patient-level metrics."""
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.patient_validation import phase_metrics,aggregate
from sc_bayeseg.window_v1.r_start import write


def audit(s):
    torch.set_num_threads(4);root=Path(s['output']);parent=Path(s['patient_control'])
    read=lambda p:json.loads(p.read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE'
    assert read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    c=r['costs'];assert (c['optimizer'],c['reverse'],c['training_encoder_images'],c['train_theta'],c['train_g_forward'],c['train_g_image_forward'])==(10000,10000,20000,10000,20000,40000)
    assert (c['validation_encoder_images'],c['validation_theta'],c['validation_g_forward'],c['validation_g_image_forward'],c['validation_label_phase_reads'])==(1088,1112,4448,8704,92)
    stages=('DIRECT_E2E','CONDITIONAL_G_E2E')
    updates=[json.loads(x) for x in (root/'physical_updates.jsonl').read_text().splitlines()]
    steps=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()]
    oldsteps=[json.loads(x) for x in (parent/'steps.jsonl').read_text().splitlines() if json.loads(x)['stage']=='DIRECT']
    assert len(updates)==len(steps)==10000 and len(oldsteps)==5000
    for j,stage in enumerate(stages):
        for i in range(5000):
            a=steps[j*5000+i];u=updates[j*5000+i];b=oldsteps[i]
            assert a['stage']==u['stage']==stage and a['step']==u['step']==i+1 and u['index']==j*5000+i+1
            assert all(a[k]==b[k] for k in ('step','absolute_schedule_step','indices','flips'))
    shared=load(parent/'PRETRAIN/S5000.pt');source=load(s['checkpoint'])['model'];changed={}
    for stage,oldstage in zip(stages,('DIRECT','CONDITIONAL_G')):
        old=load(parent/oldstage/'S0000.pt')
        for step in range(0,5001,1000):
            q=load(root/stage/f'S{step:04}.pt');assert q['step']==step and q['stage']==stage
            states=q['optimizer']['state'];assert (not states) if step==0 else all(int(v['step'])==step for v in states.values())
            if step==0:assert same(q['model'],old['model']) and same(q['rng'],shared['rng'])
            backbone={k[9:]:v for k,v in q['model'].items() if k.startswith('backbone.')}
            # The unused original segmentation head stays frozen; feature weights update.
            assert same({k:v for k,v in backbone.items() if k.startswith('segmentation_head.')},{k:v for k,v in shared['model'].items() if k.startswith('segmentation_head.')})
            if step:assert not same(backbone,shared['model'])
            if stage=='CONDITIONAL_G_E2E':assert same({k[10:]:v for k,v in q['model'].items() if k.startswith('generator.')},source)
            if step==5000:changed[stage]=sum(not same(v,shared['model'][k]) for k,v in backbone.items())
        assert r['stages'][stage]['trainable_parameters']==(482500 if stage=='DIRECT_E2E' else 484576)
    seal=read(root/'PREDICTIONS_SEALED.json');assert seal['costs']['validation_label_phase_reads']==0 and seal['phases_each']==46
    records=read(root/'PRIVATE_EVALUATION_MAP.json');assert same(records,read(parent/'PRIVATE_EVALUATION_MAP.json'))
    private=load(root/'PRIVATE_PHASE_RESULTS.pt');maxerr=[0.];aggregates={}
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,float):
            e=abs(a-b);assert e<1e-10;maxerr[0]=max(maxerr[0],e)
        else:assert a==b
    for stage in stages:
        rows=[]
        for i,row in enumerate(records):
            p=load(root/stage/'predictions'/f'CASE{i:03}.pt');prob=p['probability']
            assert prob.shape[1:]==(4,384,384) and torch.isfinite(prob).all() and (prob.sum(1)-1).abs().max()<2e-6
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(prob,y,p['spacing'])))
        compare(private[stage],rows);aggregates[stage]=aggregate(rows);compare(r['aggregates'][stage],aggregates[stage]);print('recomputed',stage,flush=True)
    arrays={name:np.array([v['macro_Dice'] for v in agg['patients']]) for name,agg in aggregates.items()}
    frozen=read(parent/'RESULTS.json')['aggregates'];controls={name:np.array([v['macro_Dice'] for v in agg['patients']]) for name,agg in frozen.items()}
    d=dict(G_minus_DIRECT=arrays['CONDITIONAL_G_E2E']-arrays['DIRECT_E2E'],DIRECT_unfreeze=arrays['DIRECT_E2E']-controls['DIRECT'],G_unfreeze=arrays['CONDITIONAL_G_E2E']-controls['CONDITIONAL_G'])
    d['unfreeze_interaction']=d['G_unfreeze']-d['DIRECT_unfreeze']
    for name,v in d.items():
        rng=np.random.default_rng(163);boot=np.mean(v[rng.integers(0,23,size=(10000,23))],axis=1)
        compare(r['paired'][name],dict(mean_Dice_difference=float(v.mean()),patient_bootstrap95=np.quantile(boot,[.025,.975]).tolist()))
    assert r['source_frozen'] and r['validation_predictions_before_labels'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    result=dict(status='PASS',physical_updates=10000,matched_schedule_pairs=5000,full_states_checked=12,fresh_Adam_steps=True,
      common_initial_models_and_RNG=True,G_frozen_all_snapshots=True,unused_segmentation_head_frozen=True,
      changed_backbone_state_tensors_at_endpoint=changed,sealed_prediction_sets_recomputed=92,patients=23,maximum_metric_error=maxerr[0],
      four_patient_bootstrap_contrasts_recomputed=True,metric_formula='same frozen evaluator, independent saved-prediction recomputation',
      old_controls_not_rerun=True,test_H5_opened=0,new_network_forwards=0,new_parameter_updates=0)
    write(root/'AUDIT.json',result);print(json.dumps(result))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
