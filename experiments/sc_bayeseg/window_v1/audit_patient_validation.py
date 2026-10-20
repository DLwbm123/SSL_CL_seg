"""Post-run state/ledger audit and recomputation of sealed patient predictions.

No training, new predictions, or test access. Metrics reuse the frozen evaluator;
independence is from saved predictions and saved full training states.
"""
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
    torch.set_num_threads(4);root=Path(s['output'])
    read=lambda p:json.loads(p.read_text())
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE'
    assert read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    c=r['costs'];assert c['optimizer']==c['reverse']==15000 and c['training_encoder_images']==30000
    assert (c['train_theta'],c['train_g_forward'],c['train_g_image_forward'])==(10000,20000,40000)
    assert (c['validation_encoder_images'],c['validation_g_forward'],c['validation_g_image_forward'],c['validation_theta'],c['validation_label_phase_reads'])==(1088,4448,8704,1112,92)
    stages=('PRETRAIN','DIRECT','CONDITIONAL_G')
    physical=[json.loads(x) for x in (root/'physical_updates.jsonl').read_text().splitlines()]
    steps=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()]
    assert len(steps)==len(physical)==15000
    assert [v['index'] for v in physical]==list(range(1,15001))
    for j,stage in enumerate(stages):
        for i in range(5000):
            a=physical[j*5000+i];b=steps[j*5000+i]
            assert a['stage']==b['stage']==stage and a['step']==b['step']==i+1
            assert b['absolute_schedule_step']==i+1+(0 if j==0 else 5000)
    for a,b in zip(steps[5000:10000],steps[10000:]):
        assert all(a[k]==b[k] for k in ('step','absolute_schedule_step','indices','flips'))
    source=load(s['checkpoint'])['model'];initial=load(s['unet_initial'])['model'];hinit=load(s['H_initial'])['model']
    shared=load(root/'PRETRAIN/S5000.pt');snapshots=0
    assert same(load(root/'PRETRAIN/S0000.pt')['model'],initial)
    for stage in stages:
        for step in range(0,5001,1000):
            q=load(root/stage/f'S{step:04}.pt');snapshots+=1
            assert q['stage']==stage and q['step']==step
            states=q['optimizer']['state']
            assert (not states) if step==0 else all(int(v['step'])==step for v in states.values())
            if stage!='PRETRAIN':
                m=q['model'];backbone={k[9:]:v for k,v in m.items() if k.startswith('backbone.')}
                assert same(backbone,shared['model'])
                if step==0:assert same(q['rng'],shared['rng'])
                if stage=='DIRECT' and step==0:
                    assert same({k[5:]:v for k,v in m.items() if k.startswith('head.')},{k[18:]:v for k,v in shared['model'].items() if k.startswith('segmentation_head.')})
                if stage=='CONDITIONAL_G':
                    assert same({k[10:]:v for k,v in m.items() if k.startswith('generator.')},source)
                    if step==0:
                        for prefix in ('hc.','hr.'):
                            assert same({k:v for k,v in m.items() if k.startswith(prefix)},{k:v for k,v in hinit.items() if k.startswith(prefix)})
    sealed=read(root/'PREDICTIONS_SEALED.json');assert sealed['costs']['validation_label_phase_reads']==0 and sealed['phases_each']==46
    records=read(root/'PRIVATE_EVALUATION_MAP.json');assert len(records)==46 and len({v['patient_id'] for v in records})==23
    private=load(root/'PRIVATE_PHASE_RESULTS.pt');maxerror=[0.]
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,float):
            error=abs(a-b);assert error<1e-10;maxerror[0]=max(error,maxerror[0])
        else:assert a==b
    aggregates={}
    for stage in stages[1:]:
        rows=[]
        for i,row in enumerate(records):
            p=load(root/stage/'predictions'/f'CASE{i:03}.pt');prob=p['probability']
            assert prob.ndim==4 and prob.shape[1:]==(4,384,384) and torch.isfinite(prob).all()
            assert (prob.sum(1)-1).abs().max()<2e-6
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(prob,y,p['spacing'])))
        compare(private[stage],rows);aggregates[stage]=aggregate(rows);compare(r['aggregates'][stage],aggregates[stage])
        print('recomputed',stage,flush=True)
    a=np.array([v['macro_Dice'] for v in aggregates['DIRECT']['patients']]);b=np.array([v['macro_Dice'] for v in aggregates['CONDITIONAL_G']['patients']])
    rng=np.random.default_rng(163);boot=np.mean((b-a)[rng.integers(0,23,size=(10000,23))],axis=1)
    compare(r['paired'],dict(mean_Dice_difference=float((b-a).mean()),patient_bootstrap95=np.quantile(boot,[.025,.975]).tolist()))
    assert r['source_frozen'] and r['validation_predictions_before_labels'] and not r['test_access']
    assert r['original_elapsed_seconds']<86400
    result=dict(status='PASS',physical_updates=15000,matched_head_schedule_pairs=5000,full_states_checked=snapshots,
      fresh_Adam_and_steps=True,frozen_backbone_and_G_all_snapshots=True,initial_sources_and_RNG_checked=True,
      sealed_prediction_sets_recomputed=92,patients=23,phases_per_patient=2,maximum_metric_error=maxerror[0],
      paired_patient_bootstrap_recomputed=True,metric_formula='same frozen evaluator, independent saved-prediction recomputation',
      prediction_seal_record_before_label_scoring=True,test_H5_opened=0,new_network_forwards=0,new_parameter_updates=0)
    write(root/'AUDIT.json',result);print(json.dumps(result))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
