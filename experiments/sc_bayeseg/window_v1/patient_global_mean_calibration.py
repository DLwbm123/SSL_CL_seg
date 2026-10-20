"""Matched crossfit temperature closure for weight-mean readout and training."""
import json,os,time,traceback
from pathlib import Path
import torch,numpy as np,h5py
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability
from sc_bayeseg.window_v1.patient_pro_diagnostics import risk,fit_temperature
from sc_bayeseg.window_v1.r_start import write,save,append
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);start=time.time();load=lambda p:torch.load(p,map_location='cpu',weights_only=False);read=lambda p:json.loads(Path(p).read_text())
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    costs=dict(network_optimizer=0,reverse=0,encoder_images=0,G_forward=0,Theta=0,raw_reconstruction_phases=0,temperature_reconstruction_phases=0,label_phase_reads=0,temperature_fits=0,temperature_function_evaluations=0,argmax_changes=0)
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=50_000_000)
        source=Path(s['mean_training_source']);mean=Path(s['mean_source']);diagnostic=Path(s['diagnostic_source']);folders={'MEAN_READ':mean/'val','Mmu':source/'Mmu/val'}
        assert read(mean/'AUDIT.json')['status']=='PASS' and read(source/'AUDIT.json')['status']=='PASS' and read(source/'EXIT.json')['exit_code']==0 and read(diagnostic/'AUDIT.json')['status']=='PASS'
        records=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json');folds=read(diagnostic/'PRIVATE_FOLDS.json');assert sorted({x['patient_id'] for x in records})==sorted(folds) and len(records)==46
        paths=[source/'RESULTS.json',mean/'RESULTS.json',Path(s['calibration_source'])/'PUBLIC_RESULTS.json',diagnostic/'PRIVATE_FOLDS.json']+[folder/f'CASE{i:03}.pt' for folder in folders.values() for i in range(46)]
        metadata={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in paths};private={};quad={};temperatures={}
        for arm in ('MEAN_READ','Mmu'):
            private[arm]=[];quad[arm]=[]
            for i,row in enumerate(records):
                assert time.time()-s['original_started_unix']<85500
                d=load(folders[arm]/f'CASE{i:03}.pt');p=restore_probability(d['native_probability']);costs['raw_reconstruction_phases']+=1
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                costs['label_phase_reads']+=1;private[arm].append(dict(patient=row['patient_id'],case=row['case_id'],risk=risk(p,y),fold=folds[row['patient_id']]))
                quad[arm].append(dict(patient=row['patient_id'],fold=folds[row['patient_id']],logp=p.permute(0,2,3,1).reshape(-1,4)[::64].double().clamp_min(1e-30).log(),target=torch.from_numpy(np.asarray(y,dtype=np.int64).reshape(-1)[::64].copy())))
                write(root/'STATUS.json',dict(status='RAW_SCORING',arm=arm,phase=i,costs=costs))
        for arm in ('MEAN_READ','Mmu'):
            temperatures[arm]={}
            for fold in range(5):
                assert time.time()-s['original_started_unix']<85500
                selected=[x for x in quad[arm] if x['fold']!=fold];logp=torch.cat([x['logp'] for x in selected]).cuda();target=torch.cat([x['target'] for x in selected]).cuda();weight=torch.cat([torch.full((len(x['target']),),1/(len(selected)*len(x['target'])),dtype=torch.float64) for x in selected]).cuda()
                fit=fit_temperature(logp,target,weight);fit.update(fit_patients=sorted({x['patient'] for x in selected}),heldout_patients=sorted(p for p in folds if folds[p]==fold));assert not set(fit['fit_patients'])&set(fit['heldout_patients'])
                temperatures[arm][str(fold)]=fit;costs['temperature_fits']+=1;costs['temperature_function_evaluations']+=fit['function_evaluations'];del logp,target,weight
        save(root/'PRIVATE_TEMPERATURES.pt',temperatures);write(root/'TEMPERATURES_SEALED.json',dict(time_unix=time.time(),fits=10,costs=dict(costs)))
        for arm in ('MEAN_READ','Mmu'):
            for i,row in enumerate(records):
                assert time.time()-s['original_started_unix']<85500
                d=load(folders[arm]/f'CASE{i:03}.pt');p=restore_probability(d['native_probability']);costs['temperature_reconstruction_phases']+=1
                fit=temperatures[arm][str(folds[row['patient_id']])];assert row['patient_id'] not in fit['fit_patients']
                z=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);changes=int((p.argmax(1)!=z.argmax(1)).sum());costs['argmax_changes']+=changes;assert changes==0
                append(root/'temperature_output_manifest.jsonl',dict(arm=arm,phase=i,T=fit['T'],fold=folds[row['patient_id']],argmax_changes=changes,sealed_before_heldout_label=True,representation='recomputable frozen probability and scalar T'))
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                costs['label_phase_reads']+=1;private[arm][i]['temperature_risk']=risk(z,y)
                write(root/'STATUS.json',dict(status='TEMPERATURE_SCORING',arm=arm,phase=i,costs=costs))
        assert costs['temperature_fits']==10 and costs['temperature_function_evaluations']<=640 and costs['label_phase_reads']==184 and costs['raw_reconstruction_phases']==costs['temperature_reconstruction_phases']==92
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in metadata.items())
        save(root/'PRIVATE_RESULTS.pt',private)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,source_frozen=True,heldout_patient_isolation=True,test_access=False,original_elapsed_seconds=time.time()-s['original_started_unix'],phase_elapsed_seconds=time.time()-start,space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
