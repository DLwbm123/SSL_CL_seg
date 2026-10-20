"""One matched four-flip readout: frozen U_D and Dmu at generator weight means."""
import json,os,time,traceback
from pathlib import Path
import torch,numpy as np,h5py
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.patient_e2e_validation import EndToEndDirect
from sc_bayeseg.window_v1.patient_point_estimate import PointG32
from sc_bayeseg.window_v1.patient_global_mean import predict as predict_mean
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.patient_validation import phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability,paired_summary
from sc_bayeseg.window_v1.patient_pro_diagnostics import risk,fit_temperature
from sc_bayeseg.window_v1.r_start import write,save,append
from sc_bayeseg.window_v1.common import storage

ARMS=('U_D_TTA','G_MEAN_TTA')

@torch.no_grad()
def predict(model,x,is_g):
    return predict_mean(model,x) if is_g else model.logits(x).softmax(1)

def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time();load=lambda p:torch.load(p,map_location='cpu',weights_only=False);read=lambda p:json.loads(Path(p).read_text())
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;rng=capture_rng_state()
    costs=dict(optimizer=0,reverse=0,encoder_images=0,direct_logit_forwards=0,G_forward=0,G_image_forward=0,Theta=0,local_draws=0,baseline_probability_reads=0,new_transformed_phase_views=0,raw_reconstruction_phases=0,temperature_reconstruction_phases=0,label_phase_reads=0,temperature_fits=0,temperature_function_evaluations=0,argmax_changes=0)
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=320_000_000)
        assert read(Path(s['smoothing_source'])/'AUDIT.json')['status']=='PASS'
        for source in (s['factor_source'],s['point_source'],s['mean_source']):assert read(Path(source)/'AUDIT.json')['status']=='PASS'
        assert read(Path(s['mean_source'])/'CONTEXT.json')['point_source']==s['point_source']
        assert read(Path(s['mean_source'])/'CONTEXT.json')['patient_control']==s['patient_control']
        assert read(Path(s['factor_source'])/'CONTEXT.json')['patient_control']==s['patient_control']
        records=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json');assert len(records)==46 and len({r['patient_id'] for r in records})==23
        folds=read(Path(s['diagnostic_source'])/'PRIVATE_FOLDS.json');assert sorted({r['patient_id'] for r in records})==sorted(folds)
        sources={'U_D_TTA':Path(s['factor_source'])/'U_D/S5000.pt','G_MEAN_TTA':Path(s['point_source'])/'Dmu/S5000.pt'}
        caches={'U_D_TTA':Path(s['factor_source'])/'U_D/val','G_MEAN_TTA':Path(s['mean_source'])/'val'}
        paths=list(sources.values())+[p/f'CASE{i:03}.pt' for p in caches.values() for i in range(46)]
        metadata={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in paths};equivariance={}
        write(root/'PRIVATE_SOURCE_METADATA.json',{str(p):dict(size=v[0],mtime_ns=v[1]) for p,v in metadata.items()})
        for arm in ARMS:
            is_g=arm=='G_MEAN_TTA';d=load(sources[arm]);assert d['step']==5000
            if is_g:assert d['residual_grid']==32 and d['mean_penalty_weight']==1
            else:assert d['lambda_Dice']==1 and d['beta_KL'] is None
            model=(PointG32(build_model().cuda(),SpatialResidualModel().cuda()) if is_g else EndToEndDirect(build_model().cuda())).cuda();model.load_state_dict(d['model']);model.eval().requires_grad_(False);folder=root/arm;folder.mkdir(exist_ok=False);equivariance[arm]=[]
            for i,row in enumerate(records):
                assert time.time()-s['original_started_unix']<85500
                source=load(caches[arm]/f'CASE{i:03}.pt');identity=source['native_probability'];costs['baseline_probability_reads']+=1
                with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:volume=f['image'][...];spacing=json.loads(f.attrs['processed_spacing'])
                assert len(identity)==len(volume) and identity.dtype==torch.float32
                total=identity.clone();views=[];pixels=len(volume)*128*128
                for k in (1,2,3):
                    tv=0.;hard=0;soft=torch.zeros(4,dtype=torch.float64)
                    for offset in range(0,len(volume),2):
                        x=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda();p=predict(model,flip(x,k),is_g);inverse=flip(p,k).cpu();reference=identity[offset:offset+len(x)]
                        assert torch.isfinite(inverse).all() and (inverse.sum(1)-1).abs().max()<2e-6
                        total[offset:offset+len(x)]+=inverse
                        tv+=float((inverse.double()-reference.double()).abs().sum())/2;hard+=int((inverse.argmax(1)!=reference.argmax(1)).sum());soft+=inverse.double().sum((0,2,3))
                        costs['encoder_images']+=len(x)
                        if is_g:costs['G_forward']+=1;costs['G_image_forward']+=len(x)
                        else:costs['direct_logit_forwards']+=1
                    views.append(dict(flip=k,mean_pixel_TV=tv/pixels,hard_disagreement=hard/pixels,soft_class_fraction=(soft/pixels).tolist()));costs['new_transformed_phase_views']+=1
                ensemble=total/4;assert (ensemble.sum(1)-1).abs().max()<2e-6
                identity_fraction=identity.double().mean((0,2,3));mean_fraction=ensemble.double().mean((0,2,3));expected_fraction=(identity_fraction+sum(torch.tensor(v['soft_class_fraction'],dtype=torch.float64) for v in views))/4
                residual=float((mean_fraction-expected_fraction).abs().max());assert residual<2e-6
                ensemble_tv=float((ensemble.double()-identity.double()).abs().sum())/(2*pixels);assert ensemble_tv<=sum(v['mean_pixel_TV'] for v in views)/4+2e-6
                equivariance[arm].append(dict(phase=i,views=views,identity_soft_class_fraction=identity_fraction.tolist(),ensemble_soft_class_fraction=mean_fraction.tolist(),class_mass_identity_error=residual,ensemble_mean_pixel_TV=ensemble_tv))
                save(folder/f'CASE{i:03}.pt',dict(native_probability=ensemble,spacing=spacing));storage(s,reserve=50_000_000)
                write(root/'STATUS.json',dict(status='PREDICTING',arm=arm,phase=i,costs=costs))
            assert same(model.state_dict(),d['model']);del model;torch.cuda.empty_cache()
        assert costs['encoder_images']==3264 and costs['G_forward']==costs['direct_logit_forwards']==834 and costs['G_image_forward']==1632 and costs['new_transformed_phase_views']==276 and costs['label_phase_reads']==0
        write(root/'PRIVATE_EQUIVARIANCE.json',equivariance);write(root/'PREDICTIONS_SEALED.json',dict(time_unix=time.time(),costs=dict(costs),ensemble_sets=92,views=(0,1,2,3),weights=(.25,.25,.25,.25),per_view_probability_cache_saved=False))
        private={};quad={};hard={};aggregates={}
        for arm in ARMS:
            private[arm]=[];quad[arm]=[];hard[arm]=[]
            for i,row in enumerate(records):
                assert time.time()-s['original_started_unix']<85500
                d=load(root/arm/f'CASE{i:03}.pt');p=restore_probability(d['native_probability']);costs['raw_reconstruction_phases']+=1
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                costs['label_phase_reads']+=1;hard[arm].append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p,y,d['spacing'])));private[arm].append(dict(patient=row['patient_id'],case=row['case_id'],fold=folds[row['patient_id']],risk=risk(p,y)))
                quad[arm].append(dict(patient=row['patient_id'],fold=folds[row['patient_id']],logp=p.permute(0,2,3,1).reshape(-1,4)[::64].double().clamp_min(1e-30).log(),target=torch.from_numpy(np.asarray(y,dtype=np.int64).reshape(-1)[::64].copy())))
                write(root/'STATUS.json',dict(status='RAW_SCORING',arm=arm,phase=i,costs=costs))
            aggregates[arm]=aggregate(hard[arm])
        temperatures={}
        for arm in ARMS:
            temperatures[arm]={}
            for fold in range(5):
                assert time.time()-s['original_started_unix']<85500
                selected=[x for x in quad[arm] if x['fold']!=fold];logp=torch.cat([x['logp'] for x in selected]).cuda();target=torch.cat([x['target'] for x in selected]).cuda();weight=torch.cat([torch.full((len(x['target']),),1/(len(selected)*len(x['target'])),dtype=torch.float64) for x in selected]).cuda()
                fit=fit_temperature(logp,target,weight);fit.update(fit_patients=sorted({x['patient'] for x in selected}),heldout_patients=sorted(p for p in folds if folds[p]==fold));assert not set(fit['fit_patients'])&set(fit['heldout_patients']);temperatures[arm][str(fold)]=fit;costs['temperature_fits']+=1;costs['temperature_function_evaluations']+=fit['function_evaluations'];del logp,target,weight
        save(root/'PRIVATE_TEMPERATURES.pt',temperatures);write(root/'TEMPERATURES_SEALED.json',dict(time_unix=time.time(),fits=10,costs=dict(costs)))
        for arm in ARMS:
            for i,row in enumerate(records):
                assert time.time()-s['original_started_unix']<85500
                d=load(root/arm/f'CASE{i:03}.pt');p=restore_probability(d['native_probability']);costs['temperature_reconstruction_phases']+=1;fit=temperatures[arm][str(folds[row['patient_id']])];assert row['patient_id'] not in fit['fit_patients']
                z=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);changes=int((p.argmax(1)!=z.argmax(1)).sum());costs['argmax_changes']+=changes;assert changes==0
                append(root/'temperature_output_manifest.jsonl',dict(arm=arm,phase=i,T=fit['T'],fold=folds[row['patient_id']],argmax_changes=changes,sealed_before_heldout_label=True))
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                costs['label_phase_reads']+=1;private[arm][i]['temperature_risk']=risk(z,y);write(root/'STATUS.json',dict(status='TEMPERATURE_SCORING',arm=arm,phase=i,costs=costs))
        assert costs['raw_reconstruction_phases']==costs['temperature_reconstruction_phases']==92 and costs['label_phase_reads']==184 and costs['temperature_fits']==10 and costs['temperature_function_evaluations']<=640
        controls={'U_D_TTA':read(Path(s['factor_source'])/'RESULTS.json')['aggregates']['val']['U_D'],'G_MEAN_TTA':read(Path(s['mean_source'])/'RESULTS.json')['aggregates']['val']};contrasts={a:paired_summary(aggregates[a],controls[a]) for a in ARMS}
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in metadata.items());restore_rng_state(rng);assert same(capture_rng_state(),rng)
        save(root/'PRIVATE_RISKS.pt',private);save(root/'PRIVATE_PHASE_RESULTS.pt',hard)
        write(root/'RESULTS.json',dict(status='COMPLETE',aggregates=aggregates,contrasts=contrasts,costs=costs,source_frozen=True,RNG_restored=True,all_predictions_before_labels=True,test_access=False,phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s,reserve=50_000_000)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        restore_rng_state(rng);write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
