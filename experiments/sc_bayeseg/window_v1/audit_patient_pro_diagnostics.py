"""Independent saved-artifact audit of RV masks, risk, and heldout temperature scoring."""
import json,os
from pathlib import Path
import torch,numpy as np,h5py
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability
from sc_bayeseg.window_v1.patient_pro_diagnostics import risk,rv_summary,expand_masks
from sc_bayeseg.window_v1.r_start import write


def audit(s):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    c=r['costs'];assert c==dict(network_optimizer=0,reverse=0,encoder_images=2322,theta=4776,g_forward=23880,g_image_forward=46440,rv_reconstruction_batches=23880,label_phase_reads=448,cache_reconstruction_phases=462,temperature_fits=20,temperature_function_evaluations=245,temperature_reconstruction_phases=184,temperature_argmax_changes=0)
    # nfev is verified against all saved optimizer receipts, rather than assumed for future runs.
    maps=dict(train=read(Path(s['train_readout_source'])/'PRIVATE_EVALUATION_MAP.json'),val=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json'))
    folders={'K0':Path(s['r32_source']),'K1':Path(s['kl_source'])/'R32_K1','KQ':Path(s['kl_source'])/'R32_KQ','U_D':Path(s['factor_source'])/'U_D'}
    private=load(root/'PRIVATE_RESULTS.pt');temps=load(root/'PRIVATE_TEMPERATURES.pt');folds=read(root/'PRIVATE_FOLDS.json')
    assert [folds[p] for p in sorted(folds)]==[i%5 for i in range(23)]
    fits=[x for arm in temps.values() for x in arm.values()];assert len(fits)==20 and sum(x['function_evaluations'] for x in fits)==c['temperature_function_evaluations']
    for arm in temps:
        for k,fit in temps[arm].items():
            assert fit['success'] and 0<fit['function_evaluations']<=64 and -3<=fit['logT']<=3 and abs(fit['T']-np.exp(fit['logT']))<1e-12
            assert set(fit['heldout_patients'])=={p for p in folds if folds[p]==int(k)} and set(fit['fit_patients'])=={p for p in folds if folds[p]!=int(k)}
    ledger=[json.loads(x) for x in (root/'prediction_ledger.jsonl').read_text().splitlines()];assert len(ledger)==198 and all(x['label_reads']==0 and x['parity']<=1e-6 for x in ledger)
    assert [(x['arm'],x['cohort'],x['phase']) for x in ledger]==[(a,c,i) for a in ('K0','K1','KQ') for c,n in (('train',20),('val',46)) for i in range(n)]
    assert read(root/'RV_PREDICTIONS_SEALED.json')['costs']['label_phase_reads']==0
    assert read(root/'TEMPERATURES_SEALED.json')['fits']==20
    manifest=[json.loads(x) for x in (root/'temperature_output_manifest.jsonl').read_text().splitlines()];assert len(manifest)==184
    assert all(x['argmax_changes']==0 and x['sealed_before_this_heldout_label_read'] for x in manifest)
    error=[0.];counts=dict(risk_phases=0,RV_phases=0,temperature_phases=0,mask_draws=0)
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
    for arm,folder in folders.items():
        quad=[]
        for cohort,records in maps.items():
            for i,row in enumerate(records):
                d=load(folder/cohort/f'CASE{i:03}.pt');p=restore_probability(d['native_probability'])
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                a=private[arm][cohort][i];assert a['patient']==row['patient_id'] and a['case']==row['case_id']
                rp=risk(p,y);eq(rp,a['risk']);assert abs(sum(x['full_image_contribution'] for x in rp['classes'])-rp['NLL'])<1e-7;counts['risk_phases']+=1
                rv=rv_summary(p.argmax(1).numpy()==3,y);rv.update(soft_volume=float(p[:,3].sum()),soft_volume_mm3=float(p[:,3].sum())*float(np.prod(d['spacing'])),max_RV_probability=float(p[:,3].max()),GT_to_classes=rp['confusion'][3]);eq(rv,a['RV_mixture'])
                yn=np.asarray(y)[:,::3,::3];eq(rv_summary(d['native_probability'].argmax(1).numpy()==3,yn),a['native_RV'])
                assert a['native_RV_present']==bool((d['native_probability'].argmax(1)==3).any())
                if arm!='U_D':
                    e=load(root/arm/cohort/f'CASE{i:03}.pt');n=expand_masks(e['native_masks']);v=expand_masks(e['evaluation_masks'])
                    assert n.shape==(20,len(y),128,16) and v.shape==(20,len(y),384,48) and e['mixture_replay_error']<=1e-6
                    for j in range(20):
                        eq(rv_summary(np.unpackbits(v[j],axis=-1).astype(bool),y),a['RV_draws'][j]);eq(rv_summary(np.unpackbits(n[j],axis=-1).astype(bool),yn),a['RV_native_draws'][j]);counts['mask_draws']+=1
                    mean=restore_probability(e['mean_native_probability']);eq(rv_summary(mean.argmax(1).numpy()==3,y),a['RV_mean'])
                    eq(e['soft_volume'].sum(1).tolist(),a['RV_soft_volume']);eq(e['max_RV_probability'].max(1).tolist(),a['RV_max_probability']);counts['RV_phases']+=1
                if cohort=='val':
                    fit=temps[arm][str(folds[row['patient_id']])];assert row['patient_id'] not in fit['fit_patients']
                    z=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);assert torch.equal(z.argmax(1),p.argmax(1));eq(risk(z,y),a['temperature_risk']);counts['temperature_phases']+=1
                    quad.append((folds[row['patient_id']],p.permute(0,2,3,1).reshape(-1,4)[::64].double().clamp_min(1e-30).log(),torch.from_numpy(np.asarray(y,dtype=np.int64).reshape(-1)[::64].copy())))
            print('recomputed',arm,cohort,flush=True)
        # Verify the stored fit objective on precisely its fit-fold quadrature; no re-optimization.
        for fold in range(5):
            fit=temps[arm][str(fold)];values=[]
            for f,logp,target in quad:
                if f!=fold:
                    z=logp.cuda()/fit['T'];target=target.cuda();values.append(float((torch.logsumexp(z,1)-z.gather(1,target[:,None]).squeeze(1)).mean()))
            assert abs(np.mean(values)-fit['objective'])<1e-10
    assert counts==dict(risk_phases=264,RV_phases=198,temperature_phases=184,mask_draws=3960)
    assert r['source_frozen'] and r['all_RV_predictions_before_labels'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    a=dict(status='PASS',**counts,maximum_metric_error=error[0],temperature_fits=20,original_temperature_function_evaluations=c['temperature_function_evaluations'],audit_fit_objective_evaluations=20,heldout_patient_isolation=True,all_pixel_argmax_changes=0,class_risk_additivity=True,all_masks_losslessly_decoded=True,source_freeze_asserted_by_worker=True,new_network_updates=0,new_reverse=0,new_G=0,audit_reconstruction_phases=462,audit_label_phase_reads=264)
    write(root/'AUDIT.json',a);print(json.dumps(a))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
