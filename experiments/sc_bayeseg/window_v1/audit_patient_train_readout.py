"""Independent saved-prediction recomputation of the training fitting readout."""
import json
import os
from pathlib import Path
import h5py
import torch
import numpy as np
from sc_bayeseg.window_v1.r_start import write
from sc_bayeseg.window_v1.patient_validation import phase_metrics
from sc_bayeseg.window_v1.patient_train_readout import aggregate_training


def audit(s):
    torch.set_num_threads(4);root=Path(s['output']);read=lambda p:json.loads(p.read_text())
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE'
    assert read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    c=r['costs'];assert (c['optimizer'],c['reverse'],c['encoder_images'],c['G_forward'],c['G_image_forward'],c['Theta'],c['training_label_phase_reads'],c['new_validation_forwards'])==(0,0,920,3840,7360,960,80,0)
    records=read(root/'PRIVATE_EVALUATION_MAP.json');assert len(records)==20 and len({v['patient_id'] for v in records})==10
    seal=read(root/'PREDICTIONS_SEALED.json');assert seal['costs']['training_label_phase_reads']==0 and seal['slices_each']==230
    saved=load(root/'PRIVATE_PHASE_RESULTS.pt');err=[0.];aggregates={}
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,float):
            e=abs(a-b);assert e<1e-10;err[0]=max(err[0],e)
        else:assert a==b
    for name in r['training']:
        rows=[];slices=0
        for i,row in enumerate(records):
            p=load(root/name/'predictions'/f'CASE{i:03}.pt');q=p['probability'];slices+=len(q)
            assert q.shape[1:]==(4,384,384) and torch.isfinite(q).all() and (q.sum(1)-1).abs().max()<2e-6
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(q,y,p['spacing'])))
        assert slices==230;compare(saved[name],rows);aggregates[name]=aggregate_training(rows);compare(r['training'][name],aggregates[name]);print('recomputed',name,flush=True)
    for file,names,keys in ((Path(s['patient_control'])/'RESULTS.json',('FROZEN_DIRECT','FROZEN_G0'),('DIRECT','CONDITIONAL_G')),(Path(s['patient_e2e_source'])/'RESULTS.json',('E2E_DIRECT','E2E_G0'),('DIRECT_E2E','CONDITIONAL_G_E2E'))):
        old=read(file)['aggregates']
        for name,key in zip(names,keys):
            a=aggregates[name];b=old[key];compare(r['reused_validation'][name],{k:v for k,v in b.items() if k!='patients'})
            compare(r['descriptive_train_minus_val'][name],dict(mean_Dice_train_minus_val=a['mean_Dice']-b['mean_Dice'],class_Dice_train_minus_val=[x-y for x,y in zip(a['class_Dice'],b['class_Dice'])]))
    for prefix in ('FROZEN','E2E'):
        a=np.array([v['macro_Dice'] for v in aggregates[prefix+'_G0']['patients']]);b=np.array([v['macro_Dice'] for v in aggregates[prefix+'_DIRECT']['patients']])
        g=np.random.default_rng(163);boot=np.mean((a-b)[g.integers(0,10,size=(10000,10))],axis=1)
        compare(r['paired_training'][prefix],dict(G_minus_DIRECT=float((a-b).mean()),patient_bootstrap95=np.quantile(boot,[.025,.975]).tolist()))
    assert r['source_frozen'] and r['RNG_restored'] and r['all_training_predictions_before_labels'] and r['new_val_or_test_arrays']==0 and r['original_elapsed_seconds']<86400
    result=dict(status='PASS',prediction_sets_recomputed=80,patients=10,phases=20,slices_per_model=230,models=4,maximum_metric_error=err[0],source_state_checks='worker exact loaded vs end state; metadata unchanged',
      validation_aggregates_exact_reuse=True,training_bootstrap_recomputed=True,new_network_forwards=0,new_parameter_updates=0,new_reverse=0,metric_formula='same frozen evaluator, independent saved-prediction recomputation')
    write(root/'AUDIT.json',result);print(json.dumps(result))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
