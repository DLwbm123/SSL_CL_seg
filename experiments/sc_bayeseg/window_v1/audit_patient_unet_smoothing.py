"""Audit a fixed U-Net label-smoothing control and its matched scalar calibration."""
import json,os
from pathlib import Path
import torch,numpy as np,h5py
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.patient_e2e_validation import EndToEndDirect
from sc_bayeseg.window_v1.patient_validation import phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_train_readout import aggregate_training
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability,paired_summary
from sc_bayeseg.window_v1.patient_pro_diagnostics import risk
from sc_bayeseg.window_v1.r_start import write

def paired(a,b,k):
    assert [x['patient'] for x in a]==[x['patient'] for x in b]
    d=np.array([x[k]-y[k] for x,y in zip(a,b)]);g=np.random.default_rng(163)
    return dict(mean_difference=float(d.mean()),patient_bootstrap95=np.quantile(d[g.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975]).tolist())

def audit(s):
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    assert r['label_smoothing']==.1 and r['lambda_Dice']==1 and r['trainable_parameters']==482500
    c=r['costs'];assert (c['optimizer'],c['reverse'],c['training_encoder_images'],c['training_logit_forwards'],c['evaluation_encoder_images'],c['evaluation_logit_forwards'])==(5000,5000,10000,5000,774,398)
    assert c['G_forward']==c['Theta']==c['local_draws']==c['argmax_changes']==0
    assert (c['reconstruction_phases'],c['temperature_reconstruction_phases'],c['label_phase_reads'],c['temperature_fits'])==(66,46,112,5) and c['temperature_function_evaluations']<=320
    factor=Path(s['factor_source']);initial=load(factor/'U_D/S0000.pt')
    steps=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()];updates=[json.loads(x) for x in (root/'physical_updates.jsonl').read_text().splitlines()];old=[json.loads(x) for x in (factor/'steps.jsonl').read_text().splitlines() if json.loads(x)['stage']=='U_D'];assert len(steps)==len(updates)==len(old)==5000
    for i,(a,b,u) in enumerate(zip(steps,old,updates),1):
        assert a['step']==u['step']==u['index']==i and all(a[k]==b[k] for k in ('step','absolute_schedule_step','indices','flips'))
        assert abs(a['smoothed_CE']-.9*a['CE']-.1*a['uniform_CE'])<1e-6 and abs(a['loss']-a['smoothed_CE']-a['Dice_loss'])<5e-6
    template=EndToEndDirect(build_model());template.backbone.requires_grad_(True);template.backbone.segmentation_head.requires_grad_(False);assert sum(p.numel() for p in template.parameters() if p.requires_grad)==482500
    for step in range(0,5001,1000):
        d=load(root/f'S{step:04}.pt');assert d['step']==step and d['label_smoothing']==.1 and d['lambda_Dice']==1 and d['trainable_parameters']==482500
        states=d['optimizer']['state'];assert (not states) if step==0 else all(int(v['step'])==step for v in states.values())
        if step==0:assert same(d['model'],initial['model']) and same(d['rng'],initial['rng'])
        assert same({k:v for k,v in d['model'].items() if k.startswith('backbone.segmentation_head.')},{k:v for k,v in initial['model'].items() if k.startswith('backbone.segmentation_head.')})
    assert read(root/'PREDICTIONS_SEALED.json')['costs']['label_phase_reads']==0
    private=load(root/'PRIVATE_PHASE_RESULTS.pt');risks=load(root/'PRIVATE_RISKS.pt');fits=load(root/'PRIVATE_TEMPERATURES.pt');folds=read(Path(s['diagnostic_source'])/'PRIVATE_FOLDS.json');assert [folds[p] for p in sorted(folds)]==[i%5 for i in range(23)]
    manifest=[json.loads(x) for x in (root/'temperature_output_manifest.jsonl').read_text().splitlines()];assert len(manifest)==46 and [m['phase'] for m in manifest]==list(range(46)) and all(m['argmax_changes']==0 and m['sealed_before_heldout_label'] for m in manifest)
    assert read(root/'TEMPERATURES_SEALED.json')['fits']==len(fits)==5 and sum(f['function_evaluations'] for f in fits.values())==c['temperature_function_evaluations'];err=[0.];quad=[]
    def eq(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:eq(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):eq(x,y)
        elif isinstance(a,float):
            e=abs(a-b);assert e<1e-10;err[0]=max(err[0],e)
        else:assert a==b
    controls=read(factor/'RESULTS.json')['aggregates']
    for cohort,source in (('train',s['train_readout_source']),('val',s['patient_control'])):
        records=read(Path(source)/'PRIVATE_EVALUATION_MAP.json');rows=[];slices=0
        for i,row in enumerate(records):
            d=load(root/cohort/f'CASE{i:03}.pt');p=d['native_probability'];assert p.dtype==torch.float32 and p.shape[1:]==(4,128,128) and torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6;slices+=len(p);p=restore_probability(p)
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p,y,d['spacing'])))
            if cohort=='val':
                a=risks[i];assert a['patient']==row['patient_id'] and a['case']==row['case_id'] and a['fold']==folds[row['patient_id']];eq(a['risk'],risk(p,y));fit=fits[str(a['fold'])];assert row['patient_id'] not in fit['fit_patients']
                assert manifest[i]['fold']==a['fold'] and manifest[i]['T']==fit['T'];z=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);assert torch.equal(z.argmax(1),p.argmax(1));eq(a['temperature_risk'],risk(z,y))
                quad.append((a['fold'],p.permute(0,2,3,1).reshape(-1,4)[::64].double().clamp_min(1e-30).log(),torch.from_numpy(np.asarray(y,dtype=np.int64).reshape(-1)[::64].copy())))
        assert slices==(230 if cohort=='train' else 544);eq(rows,private[cohort]);a=(aggregate_training if cohort=='train' else aggregate)(rows);eq(a,r['aggregates'][cohort]);eq(paired_summary(a,controls[cohort]['U_D']),r['contrasts'][cohort]);print('recomputed',cohort,flush=True)
    for fold in range(5):
        fit=fits[str(fold)];assert fit['success'] and -3<=fit['logT']<=3 and abs(fit['T']-np.exp(fit['logT']))<1e-12 and 0<fit['function_evaluations']<=64
        assert set(fit['fit_patients'])=={p for p in folds if folds[p]!=fold} and set(fit['heldout_patients'])=={p for p in folds if folds[p]==fold}
        values=[]
        for f,logp,target in quad:
            if f!=fold:
                z=logp.cuda()/fit['T'];t=target.cuda();values.append(float((torch.logsumexp(z,1)-z.gather(1,t[:,None]).squeeze(1)).mean()))
        assert abs(np.mean(values)-fit['objective'])<1e-10
    assert r['source_frozen'] and r['all_predictions_before_labels'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    a=dict(status='PASS',physical_updates=5000,full_states=6,initial_model_RNG_fresh_Adam_and_smoothing_objective_checked=True,schedule_pairs=5000,unused_head_frozen=True,native_predictions_recomputed=66,maximum_metric_error=err[0],fit_objectives_checked=5,patient_fold_isolation=True,argmax_changes=0,audit_source_interpolations=66,audit_label_phase_reads=66,audit_temperature_transforms=46,no_temperature_refits=True,new_network_forward=0,new_reverse=0,new_optimizer=0);write(root/'AUDIT.json',a)
    previous=read(Path(s['mean_calibration_source'])/'PUBLIC_RESULTS.json');models=previous['models'];ids=sorted({x['patient'] for x in risks});patients=[];phases=[]
    for i,p in enumerate(ids):
        rows=[x for x in risks if x['patient']==p];v=dict(patient=f'VAL{i:03}')
        for k in ('NLL','Brier'):v[k]=float(np.mean([x['risk'][k] for x in rows]));v['calibrated_'+k]=float(np.mean([x['temperature_risk'][k] for x in rows]))
        patients.append(v)
    for i,x in enumerate(risks):phases.append(dict(patient=f'VAL{ids.index(x["patient"]):03}',phase=f'VAL_PHASE{i:03}',**{k:v for k,v in x.items() if k not in ('patient','case')}))
    own=dict(summary={k:float(np.mean([p[k] for p in patients])) for k in ('NLL','Brier','calibrated_NLL','calibrated_Brier')},patients=patients,phases=phases);models['U_D_LS']=own
    probability_contrasts={name:{k:paired(patients,models[name]['patients'],k) for k in ('NLL','Brier','calibrated_NLL','calibrated_Brier')} for name in ('U_D','K1','Dmu','MEAN_READ','Mmu')}
    for cohort,value in r['aggregates'].items():
        for i,p in enumerate(value['patients']):p['patient']=f'{cohort.upper()}{i:03}'
    r.pop('space');r.update(status='COMPLETE_AUDIT_PASS',audit=a,models=models,probability_contrasts=probability_contrasts,new_temperatures={f:{k:v for k,v in fit.items() if k not in ('fit_patients','heldout_patients')} for f,fit in fits.items()},old_temperature_fits_reused=40,scope='One fixed non-Bayesian risk control; repeatedly used development patients, no final test');write(root/'PUBLIC_RESULTS.json',r);print(json.dumps(a))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
