"""Read-only image-only training-patient fitting diagnostic, no new validation."""
import json
import os
import time
import traceback
from pathlib import Path
from collections import defaultdict
import h5py
import numpy as np
import torch
from lcrseg.data.h5_dataset import load_training_records
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.patient_validation import DirectHead,phase_metrics
from sc_bayeseg.window_v1.patient_image_interface import PatientImageInterface
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.r_start import write,save
from sc_bayeseg.window_v1.common import storage

def aggregate_training(records):
    grouped=defaultdict(list)
    for v in records:grouped[v['patient']].append(v)
    assert len(grouped)==10 and all(len(v)==2 for v in grouped.values())
    patients=[]
    for patient,phases in sorted(grouped.items()):
        classes=[]
        for k in (1,2,3):
            cells=[p['metrics']['classes'][k-1] for p in phases];valid=[v for v in cells if not v['GT_absent']]
            classes.append(dict(class_index=k,Dice=float(np.mean([v['Dice'] for v in valid])) if valid else None,
              HD95_penalized_mm=float(np.mean([v['HD95_penalized_mm'] for v in valid])) if valid else None,
              HD95_success_only_mm=float(np.mean([v['HD95_mm'] for v in valid if v['HD95_mm'] is not None])) if any(v['HD95_mm'] is not None for v in valid) else None,
              empty_prediction_failures=sum(v['empty_prediction_failure'] for v in cells),GT_absent_phases=sum(v['GT_absent'] for v in cells)))
        dice=[v['Dice'] for v in classes if v['Dice'] is not None]
        patients.append(dict(patient=patient,classes=classes,macro_Dice=float(np.mean(dice)) if dice else None,
          NLL=float(np.mean([v['metrics']['NLL'] for v in phases])),Brier=float(np.mean([v['metrics']['Brier'] for v in phases]))))
    values=np.array([v['macro_Dice'] for v in patients]);assert np.isfinite(values).all()
    return dict(patients=patients,mean_Dice=float(values.mean()),median_Dice=float(np.median(values)),std_Dice=float(values.std(ddof=1)),
      class_Dice=[float(np.mean([p['classes'][k]['Dice'] for p in patients if p['classes'][k]['Dice'] is not None])) for k in range(3)],
      empty_prediction_failures=sum(c['empty_prediction_failures'] for p in patients for c in p['classes']))


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time()
    costs=dict(optimizer=0,reverse=0,encoder_images=0,G_forward=0,G_image_forward=0,Theta=0,training_label_phase_reads=0,new_validation_forwards=0)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=capture_rng_state()
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=4_000_000_000)
        records=load_training_records(Path(s['data']),seed=0,dataset='mnms',sites=['Siemens'],roles=['train_labeled'])
        records=sorted(records,key=lambda r:(r['patient_id'],r['case_id']))
        assert len(records)==20 and len({r['patient_id'] for r in records})==10
        assert all(r['cohort']=='canonical320' and r['primary_20pct_split']=='train_labeled' for r in records)
        write(root/'PRIVATE_EVALUATION_MAP.json',records)
        parent=Path(s['patient_control']);e2e=Path(s['patient_e2e_source'])
        sources=dict(FROZEN_DIRECT=parent/'DIRECT/S5000.pt',FROZEN_G0=parent/'CONDITIONAL_G/S5000.pt',E2E_DIRECT=e2e/'DIRECT_E2E/S5000.pt',E2E_G0=e2e/'CONDITIONAL_G_E2E/S5000.pt')
        valfiles=[parent/'RESULTS.json',e2e/'RESULTS.json'];meta={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in list(sources.values())+valfiles}
        slices=0;batches=0
        for row in records:
            with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:
                shape=f['image'].shape
            assert shape[1:]==(384,384);slices+=shape[0];batches+=(shape[0]+1)//2
        assert slices==230 and batches<=125
        for name,source in sources.items():
            payload=torch.load(source,map_location='cpu',weights_only=False);assert payload['step']==5000
            encoder=build_model().cuda()
            if name.endswith('DIRECT'):model=DirectHead(encoder).cuda()
            else:model=PatientImageInterface(encoder,SpatialResidualModel().cuda()).cuda()
            model.load_state_dict(payload['model']);model.eval().requires_grad_(False)
            assert same(payload['model'],model.state_dict())
            folder=root/name/'predictions';folder.mkdir(parents=True,exist_ok=False)
            for index,row in enumerate(records):
                assert time.time()-s['original_started_unix']<85500
                with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:
                    volume=f['image'][...];spacing=json.loads(f.attrs['processed_spacing'])
                output=[]
                for offset in range(0,len(volume),2):
                    x=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda()
                    p=model.predict_probabilities(x,seed=620163+1000*index+offset//2,theta_draws=4,local_draws=4)
                    assert torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6
                    output.append(p.cpu());costs['encoder_images']+=len(x)
                    if name.endswith('G0'):
                        costs['G_forward']+=16;costs['G_image_forward']+=16*len(x);costs['Theta']+=4
                save(folder/f'CASE{index:03}.pt',dict(probability=torch.cat(output),spacing=spacing))
                storage(s,reserve=500_000_000);write(root/'STATUS.json',dict(status='PREDICTING',stage=name,phase_index=index,costs=costs))
            assert same(payload['model'],model.state_dict());del model,payload;torch.cuda.empty_cache()
        assert costs['encoder_images']==920 and costs['G_forward']==32*batches and costs['G_image_forward']==7360 and costs['Theta']==8*batches
        assert costs['training_label_phase_reads']==0
        write(root/'PREDICTIONS_SEALED.json',dict(models=list(sources),phases_each=20,slices_each=230,costs=costs,time_unix=time.time()))
        aggregates={};all_rows={}
        for name in sources:
            rows=[]
            for index,row in enumerate(records):
                p=torch.load(root/name/'predictions'/f'CASE{index:03}.pt',map_location='cpu',weights_only=False)
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                costs['training_label_phase_reads']+=1
                rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p['probability'],y,p['spacing'])))
                write(root/'STATUS.json',dict(status='SCORING',stage=name,phase_index=index,costs=costs))
            aggregates[name]=aggregate_training(rows);all_rows[name]=rows
        assert costs['training_label_phase_reads']==80
        vals={};descriptive={};paired={}
        for file,names in zip(valfiles,(('FROZEN_DIRECT','FROZEN_G0'),('E2E_DIRECT','E2E_G0'))):
            old=json.loads(file.read_text())['aggregates']
            for name,key in zip(names,('DIRECT','CONDITIONAL_G') if names[0]=='FROZEN_DIRECT' else ('DIRECT_E2E','CONDITIONAL_G_E2E')):
                value=old[key]
                vals[name]={k:v for k,v in value.items() if k!='patients'}
                descriptive[name]=dict(mean_Dice_train_minus_val=aggregates[name]['mean_Dice']-value['mean_Dice'],class_Dice_train_minus_val=[a-b for a,b in zip(aggregates[name]['class_Dice'],value['class_Dice'])])
        for prefix in ('FROZEN','E2E'):
            a=np.array([p['macro_Dice'] for p in aggregates[prefix+'_G0']['patients']]);b=np.array([p['macro_Dice'] for p in aggregates[prefix+'_DIRECT']['patients']])
            g=np.random.default_rng(163);boot=np.mean((a-b)[g.integers(0,10,size=(10000,10))],axis=1)
            paired[prefix]=dict(G_minus_DIRECT=float((a-b).mean()),patient_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in meta.items())
        save(root/'PRIVATE_PHASE_RESULTS.pt',all_rows);restore_rng_state(rng)
        write(root/'RESULTS.json',dict(status='COMPLETE',training=aggregates,reused_validation=vals,descriptive_train_minus_val=descriptive,paired_training=paired,costs=costs,source_frozen=True,RNG_restored=True,all_training_predictions_before_labels=True,new_val_or_test_arrays=0,phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        restore_rng_state(rng);write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
