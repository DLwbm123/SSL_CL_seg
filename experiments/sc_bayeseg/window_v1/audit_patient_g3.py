"""Audit matched G3 states, updates, sealed patient metrics and reused controls."""
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
    c=r['costs'];assert (c['optimizer'],c['reverse'],c['training_encoder_images'],c['train_theta'],c['train_g_forward'],c['train_g_image_forward'])==(5000,5000,10000,10000,20000,40000)
    assert (c['validation_encoder_images'],c['validation_theta'],c['validation_g_forward'],c['validation_g_image_forward'],c['validation_label_phase_reads'])==(544,1112,4448,8704,46)
    updates=[json.loads(x) for x in (root/'physical_updates.jsonl').read_text().splitlines()]
    steps=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()]
    controls=[json.loads(x) for x in (parent/'steps.jsonl').read_text().splitlines() if json.loads(x)['stage']=='CONDITIONAL_G']
    assert len(updates)==len(steps)==len(controls)==5000
    for i,(a,b,u) in enumerate(zip(steps,controls,updates),1):
        assert u['index']==u['step']==a['step']==i
        assert all(a[k]==b[k] for k in ('step','absolute_schedule_step','indices','flips'))
    original=load(s['checkpoint'])['model'];shared=load(parent/'PRETRAIN/S5000.pt')
    initial=load(parent/'CONDITIONAL_G/S0000.pt');delta=read(Path(s['g_bias_source'])/'RESULTS.json')['arms']['ZERO']['readouts']['G3']['delta']
    expected={k:v.clone() for k,v in original.items()};expected['anatomy.2.mean_b']+=torch.tensor(delta,dtype=expected['anatomy.2.mean_b'].dtype)
    assert not same(original['anatomy.2.mean_b'],expected['anatomy.2.mean_b'])
    for step in range(0,5001,1000):
        q=load(root/f'S{step:04}.pt');assert q['step']==step
        states=q['optimizer']['state'];assert (not states) if step==0 else all(int(v['step'])==step for v in states.values())
        m=q['model'];assert same({k[9:]:v for k,v in m.items() if k.startswith('backbone.')},shared['model'])
        assert same({k[10:]:v for k,v in m.items() if k.startswith('generator.')},expected)
        if step==0:
            assert same(q['rng'],initial['rng'])
            assert same({k:v for k,v in m.items() if k.startswith(('hc.','hr.'))},{k:v for k,v in initial['model'].items() if k.startswith(('hc.','hr.'))})
    seal=read(root/'PREDICTIONS_SEALED.json');assert seal['costs']['validation_label_phase_reads']==0 and seal['phases']==46
    records=read(parent/'PRIVATE_EVALUATION_MAP.json');saved=load(root/'PRIVATE_PHASE_RESULTS.pt');rows=[];maxerr=[0.]
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
    for i,row in enumerate(records):
        p=load(root/'predictions'/f'CASE{i:03}.pt');prob=p['probability']
        assert prob.shape[1:]==(4,384,384) and torch.isfinite(prob).all() and (prob.sum(1)-1).abs().max()<2e-6
        with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
        rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(prob,y,p['spacing'])))
    compare(saved,rows);a=aggregate(rows);compare(r['aggregate'],a)
    b=np.array([p['macro_Dice'] for p in a['patients']])
    for name,v in read(parent/'RESULTS.json')['aggregates'].items():
        assert [p['patient'] for p in a['patients']]==[p['patient'] for p in v['patients']]
        x=np.array([p['macro_Dice'] for p in v['patients']]);rng=np.random.default_rng(163)
        boot=np.mean((b-x)[rng.integers(0,23,size=(10000,23))],axis=1)
        compare(r['paired'][name],dict(mean_Dice_difference=float((b-x).mean()),patient_bootstrap95=np.quantile(boot,[.025,.975]).tolist()))
    assert r['source_frozen'] and r['validation_predictions_before_labels'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    result=dict(status='PASS',physical_updates=5000,matched_G0_schedule_pairs=5000,full_states_checked=6,fresh_Adam_steps=True,
      G_only_three_mean_bias_changes_and_frozen=True,backbone_frozen=True,H_initial_and_RNG_match_G0=True,
      sealed_predictions_recomputed=46,patients=23,maximum_metric_error=maxerr[0],paired_patient_bootstrap_recomputed=True,
      metric_formula='same frozen evaluator, independent saved-prediction recomputation',old_controls_not_rerun=True,
      test_H5_opened=0,new_network_forwards=0,new_parameter_updates=0)
    write(root/'AUDIT.json',result);print(json.dumps(result))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
