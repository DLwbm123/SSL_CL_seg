"""Saved-ensemble/metric audit; no replay of uncached per-view network probabilities."""
import json,os
from pathlib import Path
import torch,numpy as np,h5py
from sc_bayeseg.window_v1.patient_validation import phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability,paired_summary
from sc_bayeseg.window_v1.patient_pro_diagnostics import risk
from sc_bayeseg.window_v1.r_start import write
ARMS=('U_D_TTA','G_MEAN_TTA')

def interval(d):
    d=np.asarray(d);g=np.random.default_rng(163)
    return dict(mean_difference=float(d.mean()),patient_bootstrap95=np.quantile(d[g.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975]).tolist())

def paired(a,b,key):
    assert [p['patient'] for p in a]==[p['patient'] for p in b]
    return interval([x[key]-y[key] for x,y in zip(a,b)])

def audit(s):
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    c=r['costs'];assert c['optimizer']==c['reverse']==c['Theta']==c['local_draws']==c['argmax_changes']==0
    assert (c['encoder_images'],c['direct_logit_forwards'],c['G_forward'],c['G_image_forward'],c['baseline_probability_reads'],c['new_transformed_phase_views'])==(3264,834,834,1632,92,276)
    assert c['raw_reconstruction_phases']==c['temperature_reconstruction_phases']==92 and c['label_phase_reads']==184 and c['temperature_fits']==10 and c['temperature_function_evaluations']<=640
    metadata=read(root/'PRIVATE_SOURCE_METADATA.json');assert len(metadata)==94
    for path,info in metadata.items():assert (Path(path).stat().st_size,Path(path).stat().st_mtime_ns)==(info['size'],info['mtime_ns'])
    seal=read(root/'PREDICTIONS_SEALED.json');assert seal['costs']['label_phase_reads']==0 and seal['ensemble_sets']==92 and seal['views']==[0,1,2,3] and seal['weights']==[.25]*4 and not seal['per_view_probability_cache_saved']
    records=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json');folds=read(Path(s['diagnostic_source'])/'PRIVATE_FOLDS.json');assert len(records)==46 and [folds[p] for p in sorted(folds)]==[i%5 for i in range(23)]
    private=load(root/'PRIVATE_RISKS.pt');hard=load(root/'PRIVATE_PHASE_RESULTS.pt');temps=load(root/'PRIVATE_TEMPERATURES.pt');equiv=read(root/'PRIVATE_EQUIVARIANCE.json');manifest=[json.loads(x) for x in (root/'temperature_output_manifest.jsonl').read_text().splitlines()]
    assert len(manifest)==92 and [(m['arm'],m['phase']) for m in manifest]==[(a,i) for a in ARMS for i in range(46)] and all(m['sealed_before_heldout_label'] and m['argmax_changes']==0 for m in manifest)
    assert read(root/'TEMPERATURES_SEALED.json')['fits']==10 and sum(f['function_evaluations'] for a in temps.values() for f in a.values())==c['temperature_function_evaluations']
    controls={'U_D_TTA':read(Path(s['factor_source'])/'RESULTS.json')['aggregates']['val']['U_D'],'G_MEAN_TTA':read(Path(s['mean_source'])/'RESULTS.json')['aggregates']['val']}
    caches={'U_D_TTA':Path(s['factor_source'])/'U_D/val','G_MEAN_TTA':Path(s['mean_source'])/'val'};err=[0.];mass_error=[0.]
    ledger=[json.loads(x) for x in (root/'forward_phase_ledger.jsonl').read_text().splitlines()];assert len(ledger)==92
    cum=dict(encoder_images=0,direct_logit_forwards=0,G_forward=0,G_image_forward=0,baseline_probability_reads=0,new_transformed_phase_views=0)
    for j,e in enumerate(ledger):
        arm=ARMS[j//46];i=j%46;assert (e['arm'],e['phase'])==(arm,i)
        n=e['slices'];assert isinstance(n,int) and n>0
        cum['encoder_images']+=3*n;cum['baseline_probability_reads']+=1;cum['new_transformed_phase_views']+=3
        if arm=='U_D_TTA':cum['direct_logit_forwards']+=3*((n+1)//2)
        else:cum['G_forward']+=3*((n+1)//2);cum['G_image_forward']+=3*n
        assert all(e['costs'][k]==v for k,v in cum.items()) and e['costs']['label_phase_reads']==0 and e['costs']['temperature_fits']==0

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
    for j,arm in enumerate(ARMS):
        rows=[];quad=[];slices=0
        for i,row in enumerate(records):
            d=load(root/arm/f'CASE{i:03}.pt');native=d['native_probability'];assert len(native)==ledger[j*46+i]['slices'];identity=load(caches[arm]/f'CASE{i:03}.pt')['native_probability'];assert native.shape==identity.shape and native.dtype==torch.float32 and torch.isfinite(native).all() and (native.sum(1)-1).abs().max()<2e-6;slices+=len(native)
            e=equiv[arm][i];assert e['phase']==i and [v['flip'] for v in e['views']]==[1,2,3]
            for v in e['views']:assert 0<=v['mean_pixel_TV']<=1+2e-6 and 0<=v['hard_disagreement']<=1 and abs(sum(v['soft_class_fraction'])-1)<2e-6
            oldfrac=identity.double().mean((0,2,3));newfrac=native.double().mean((0,2,3));expected=(oldfrac+sum(torch.tensor(v['soft_class_fraction'],dtype=torch.float64) for v in e['views']))/4
            assert max(abs(oldfrac[k]-e['identity_soft_class_fraction'][k]) for k in range(4))<1e-10 and max(abs(newfrac[k]-e['ensemble_soft_class_fraction'][k]) for k in range(4))<1e-10
            residual=float((newfrac-expected).abs().max());mass_error[0]=max(mass_error[0],residual);assert residual<2e-6 and abs(residual-e['class_mass_identity_error'])<1e-10
            tv=float((native.double()-identity.double()).abs().sum())/(2*len(native)*128*128);assert abs(tv-e['ensemble_mean_pixel_TV'])<1e-10 and tv<=sum(v['mean_pixel_TV'] for v in e['views'])/4+2e-6
            p=restore_probability(native)
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p,y,d['spacing'])));a=private[arm][i];assert a['patient']==row['patient_id'] and a['case']==row['case_id'] and a['fold']==folds[row['patient_id']];eq(a['risk'],risk(p,y))
            fit=temps[arm][str(a['fold'])];assert row['patient_id'] not in fit['fit_patients'];m=manifest[j*46+i];assert m['T']==fit['T'] and m['fold']==a['fold']
            z=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);assert torch.equal(z.argmax(1),p.argmax(1));eq(a['temperature_risk'],risk(z,y))
            quad.append((a['fold'],p.permute(0,2,3,1).reshape(-1,4)[::64].double().clamp_min(1e-30).log(),torch.from_numpy(np.asarray(y,dtype=np.int64).reshape(-1)[::64].copy())))
        assert slices==544;eq(rows,hard[arm]);a=aggregate(rows);eq(a,r['aggregates'][arm]);eq(paired_summary(a,controls[arm]),r['contrasts'][arm])
        for fold in range(5):
            fit=temps[arm][str(fold)];assert fit['success'] and -3<=fit['logT']<=3 and abs(fit['T']-np.exp(fit['logT']))<1e-12 and 0<fit['function_evaluations']<=64
            assert set(fit['fit_patients'])=={p for p in folds if folds[p]!=fold} and set(fit['heldout_patients'])=={p for p in folds if folds[p]==fold}
            values=[]
            for f,logp,target in quad:
                if f!=fold:
                    z=logp.cuda()/fit['T'];t=target.cuda();values.append(float((torch.logsumexp(z,1)-z.gather(1,t[:,None]).squeeze(1)).mean()))
            assert abs(np.mean(values)-fit['objective'])<1e-10
        print('recomputed',arm,flush=True)
    assert r['source_frozen'] and r['RNG_restored'] and r['all_predictions_before_labels'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    result=dict(status='PASS',ensemble_sets_recomputed=92,forward_phase_ledger_records_checked=92,maximum_metric_error=err[0],maximum_class_mass_error=mass_error[0],fixed_views_and_equivariance_scalar_invariants_checked=True,per_view_network_probabilities_replayed=False,per_view_full_probabilities_saved=False,source_metadata_records_checked=94,fit_objectives_checked=10,fold_isolation=True,argmax_changes=0,audit_ensemble_restores=92,audit_identity_native_reads=92,audit_label_phase_reads=92,audit_temperature_transforms=92,no_temperature_refits=True,new_network_forward=0,new_reverse=0,new_optimizer=0);write(root/'AUDIT.json',result)
    # Derive paired statistics only after all saved outputs have passed the independent checks.
    prior=read(Path(s['smoothing_source'])/'PUBLIC_RESULTS.json')['models'];alias={p:f'VAL{i:03}' for i,p in enumerate(sorted(folds))};models={};probability_contrasts={};equiv_patients={}
    for arm,base in (('U_D_TTA','U_D'),('G_MEAN_TTA','MEAN_READ')):
        patients=[];phase_public=[];equiv_patients[arm]=[]
        for p in sorted(folds):
            rows=[x for x in private[arm] if x['patient']==p];assert len(rows)==2;v=dict(patient=alias[p]);indices=[i for i,row in enumerate(records) if row['patient_id']==p]
            for k in ('NLL','Brier'):v[k]=float(np.mean([x['risk'][k] for x in rows]));v['calibrated_'+k]=float(np.mean([x['temperature_risk'][k] for x in rows]))
            patients.append(v);equiv_patients[arm].append(dict(patient=alias[p],mean_pixel_TV=float(np.mean([v['mean_pixel_TV'] for i in indices for v in equiv[arm][i]['views']])),hard_disagreement=float(np.mean([v['hard_disagreement'] for i in indices for v in equiv[arm][i]['views']]))))
        for i,p in enumerate(private[arm]):phase_public.append(dict(patient=alias[p['patient']],phase=f'VAL_PHASE{i:03}',**{k:v for k,v in p.items() if k not in ('patient','case')}))
        models[arm]=dict(summary={k:float(np.mean([p[k] for p in patients])) for k in ('NLL','Brier','calibrated_NLL','calibrated_Brier')},patients=patients,phases=phase_public)
        probability_contrasts[arm]={k:paired(patients,prior[base]['patients'],k) for k in ('NLL','Brier','calibrated_NLL','calibrated_Brier')}
    def vec(values,key):return np.array([v[key] for v in values])
    gd=vec(r['aggregates']['G_MEAN_TTA']['patients'],'macro_Dice')-vec(controls['G_MEAN_TTA']['patients'],'macro_Dice');ud=vec(r['aggregates']['U_D_TTA']['patients'],'macro_Dice')-vec(controls['U_D_TTA']['patients'],'macro_Dice')
    interactions=dict(primary_Dice_gain_interaction=interval(gd-ud),probability_gain_interactions={k:interval((vec(models['G_MEAN_TTA']['patients'],k)-vec(prior['MEAN_READ']['patients'],k))-(vec(models['U_D_TTA']['patients'],k)-vec(prior['U_D']['patients'],k))) for k in ('NLL','Brier','calibrated_NLL','calibrated_Brier')},equivariance_between_fixed_pipelines={k:paired(equiv_patients['G_MEAN_TTA'],equiv_patients['U_D_TTA'],k) for k in ('mean_pixel_TV','hard_disagreement')})
    for group in (r['aggregates'],controls):
        for a in group.values():
            for p in a['patients']:p['patient']=alias[p['patient']]
    r.pop('space');r.update(status='COMPLETE_AUDIT_PASS',audit=result,models=models,identity_controls=controls,identity_risk_controls={k:prior[k] for k in ('U_D','MEAN_READ')},probability_contrasts=probability_contrasts,interactions=interactions,equivariance_patients=equiv_patients,equivariance_phase_statistics=equiv,new_temperatures={a:{f:{k:v for k,v in fit.items() if k not in ('fit_patients','heldout_patients')} for f,fit in ts.items()} for a,ts in temps.items()},scope='One matched four-flip inference augmentation, fixed pipelines; repeatedly used development set; no official BayeSeg claim');write(root/'PUBLIC_RESULTS.json',r);print(json.dumps(result))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
