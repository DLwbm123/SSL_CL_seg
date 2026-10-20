"""Recompute matched global-mean calibration states without temperature refits."""
import os,json
from pathlib import Path
import torch,numpy as np,h5py
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability
from sc_bayeseg.window_v1.patient_pro_diagnostics import risk
from sc_bayeseg.window_v1.r_start import write


def audit(s):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    c=r['costs'];assert c['network_optimizer']==c['reverse']==c['encoder_images']==c['G_forward']==c['Theta']==c['argmax_changes']==0
    assert c['raw_reconstruction_phases']==c['temperature_reconstruction_phases']==92 and c['label_phase_reads']==184 and c['temperature_fits']==10 and c['temperature_function_evaluations']<=640
    source=Path(s['mean_training_source']);mean=Path(s['mean_source']);folders={'MEAN_READ':mean/'val','Mmu':source/'Mmu/val'};diag=Path(s['diagnostic_source']);records=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json');folds=read(diag/'PRIVATE_FOLDS.json')
    assert sorted({x['patient_id'] for x in records})==sorted(folds) and [folds[x] for x in sorted(folds)]==[i%5 for i in range(23)]
    private=load(root/'PRIVATE_RESULTS.pt');temps=load(root/'PRIVATE_TEMPERATURES.pt');manifest=[json.loads(x) for x in (root/'temperature_output_manifest.jsonl').read_text().splitlines()]
    assert len(manifest)==92 and [(x['arm'],x['phase']) for x in manifest]==[(a,i) for a in ('MEAN_READ','Mmu') for i in range(46)] and all(x['sealed_before_heldout_label'] for x in manifest)
    assert all(x['argmax_changes']==0 for x in manifest) and read(root/'TEMPERATURES_SEALED.json')['fits']==10
    fits=[f for a in temps.values() for f in a.values()];assert len(fits)==10 and sum(f['function_evaluations'] for f in fits)==c['temperature_function_evaluations']
    error=[0.]
    def eq(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:eq(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):eq(x,y)
        elif isinstance(a,float):
            e=abs(a-b);assert e<1e-10;error[0]=max(error[0],e)
        else:assert a==b
    for arm in ('MEAN_READ','Mmu'):
        quad=[]
        for i,row in enumerate(records):
            d=load(folders[arm]/f'CASE{i:03}.pt');p=restore_probability(d['native_probability'])
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            a=private[arm][i];assert a['patient']==row['patient_id'] and a['case']==row['case_id'] and a['fold']==folds[row['patient_id']]
            eq(risk(p,y),a['risk']);fit=temps[arm][str(a['fold'])];assert row['patient_id'] not in fit['fit_patients']
            z=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);assert torch.equal(z.argmax(1),p.argmax(1));eq(risk(z,y),a['temperature_risk'])
            quad.append((a['fold'],p.permute(0,2,3,1).reshape(-1,4)[::64].double().clamp_min(1e-30).log(),torch.from_numpy(np.asarray(y,dtype=np.int64).reshape(-1)[::64].copy())))
        for fold in range(5):
            fit=temps[arm][str(fold)];assert fit['success'] and -3<=fit['logT']<=3 and 0<fit['function_evaluations']<=64 and abs(fit['T']-np.exp(fit['logT']))<1e-12
            assert set(fit['fit_patients'])=={p for p in folds if folds[p]!=fold} and set(fit['heldout_patients'])=={p for p in folds if folds[p]==fold}
            values=[]
            for f,logp,target in quad:
                if f!=fold:
                    z=logp.cuda()/fit['T'];t=target.cuda();values.append(float((torch.logsumexp(z,1)-z.gather(1,t[:,None]).squeeze(1)).mean()))
            assert abs(np.mean(values)-fit['objective'])<1e-10
        print('recomputed',arm,flush=True)
    assert r['source_frozen'] and r['heldout_patient_isolation'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    a=dict(status='PASS',raw_risk_states=92,calibrated_risk_states=92,maximum_metric_error=error[0],temperature_fits=10,original_temperature_function_evaluations=c['temperature_function_evaluations'],fit_objectives_independently_checked=10,fold_isolation=True,all_pixel_argmax_changes=0,output_manifests=92,source_freeze_asserted_by_worker=True,new_network_forwards=0,new_reverse=0,new_network_updates=0,audit_source_interpolations=92,audit_label_phase_reads=92,audit_temperature_transforms=92,no_temperature_refits=True)
    write(root/'AUDIT.json',a);print(json.dumps(a))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
