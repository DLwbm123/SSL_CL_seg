"""One fixed label-smoothing U-Net+Dice control, with matched crossfit calibration."""
import json,os,time,traceback
from pathlib import Path
import torch,numpy as np,h5py
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.patient_e2e_validation import EndToEndDirect
from sc_bayeseg.window_v1.patient_validation import Schedule,phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_train_readout import aggregate_training
from sc_bayeseg.window_v1.patient_loss_factor import foreground_dice_loss,restore_probability,paired_summary
from sc_bayeseg.window_v1.patient_pro_diagnostics import risk,fit_temperature
from sc_bayeseg.window_v1.r_start import write,append,save
from sc_bayeseg.window_v1.common import storage

SMOOTHING=.1

def smoothing_loss(logits,y):
    ce=F.cross_entropy(logits,y,reduction='none').mean()
    uniform_ce=-logits.log_softmax(1).mean()
    smooth=(1-SMOOTHING)*ce+SMOOTHING*uniform_ce
    dice=foreground_dice_loss(logits.softmax(1),y)
    return smooth+dice,dict(CE=float(ce.detach()),uniform_CE=float(uniform_ce.detach()),smoothed_CE=float(smooth.detach()),Dice_loss=float(dice.detach()))


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time();active={};load=lambda p:torch.load(p,map_location='cpu',weights_only=False);read=lambda p:json.loads(Path(p).read_text())
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    costs=dict(optimizer=0,reverse=0,training_encoder_images=0,training_logit_forwards=0,evaluation_encoder_images=0,evaluation_logit_forwards=0,G_forward=0,Theta=0,local_draws=0,reconstruction_phases=0,temperature_reconstruction_phases=0,label_phase_reads=0,temperature_fits=0,temperature_function_evaluations=0,argmax_changes=0)
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=500_000_000)
        factor=Path(s['factor_source']);assert read(factor/'AUDIT.json')['status']=='PASS' and read(Path(s['mean_calibration_source'])/'AUDIT.json')['status']=='PASS'
        source=factor/'U_D/S0000.pt';initial=load(source);assert initial['step']==0 and initial['lambda_Dice']==1 and initial['beta_KL'] is None and not initial['optimizer']['state']
        sourcefiles=[source,factor/'RESULTS.json',Path(s['mean_calibration_source'])/'PUBLIC_RESULTS.json'];meta={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in sourcefiles}
        maps=dict(train=read(Path(s['train_readout_source'])/'PRIVATE_EVALUATION_MAP.json'),val=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json'));assert len(maps['train'])==20 and len(maps['val'])==46
        folds=read(Path(s['diagnostic_source'])/'PRIVATE_FOLDS.json');assert sorted({r['patient_id'] for r in maps['val']})==sorted(folds)
        model=EndToEndDirect(build_model().cuda()).cuda();model.load_state_dict(initial['model']);model.backbone.requires_grad_(True);model.backbone.segmentation_head.requires_grad_(False)
        params=tuple(p for p in model.parameters() if p.requires_grad);assert sum(p.numel() for p in params)==482500
        restore_rng_state(initial['rng']);opt=torch.optim.Adam(params,lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        schedule=Schedule(H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None));assert not set(schedule.patients)&set(folds)
        active=dict(model=model,opt=opt,step=0)
        def post(optimizer,args,kwargs):
            costs['optimizer']+=1;active['step']=costs['optimizer'];assert costs['optimizer']<=5000
            append(root/'physical_updates.jsonl',dict(index=costs['optimizer'],step=costs['optimizer']))
        handle=opt.register_step_post_hook(post)
        def snapshot(step):return dict(step=step,model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng_state(),costs=costs,label_smoothing=SMOOTHING,lambda_Dice=1,trainable_parameters=482500)
        save(root/'S0000.pt',snapshot(0))
        for step in range(1,5001):
            assert time.time()-s['original_started_unix']<85500
            x,y,ids,flips=schedule.batch(5000+step);x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
            logits=model.logits(x);loss,parts=smoothing_loss(logits,y);assert torch.isfinite(loss);costs['training_encoder_images']+=len(x);costs['training_logit_forwards']+=1
            loss.backward();costs['reverse']+=1;assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step()
            append(root/'steps.jsonl',dict(step=step,absolute_schedule_step=5000+step,indices=ids,flips=flips,loss=float(loss.detach()),gradient_norm_before_clip=norm,**parts))
            if step%100==0:write(root/'STATUS.json',dict(status='TRAINING',step=step,costs=costs))
            if step%1000==0:storage(s,reserve=500_000_000);save(root/f'S{step:04}.pt',snapshot(step))
        handle.remove();assert costs['optimizer']==costs['reverse']==5000 and all(int(v['step'])==5000 for v in opt.state_dict()['state'].values());active={};model.eval()
        for cohort,records in maps.items():
            folder=root/cohort;folder.mkdir(exist_ok=False)
            for i,row in enumerate(records):
                assert time.time()-s['original_started_unix']<85500
                with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:volume=f['image'][...];spacing=json.loads(f.attrs['processed_spacing'])
                result=[]
                for offset in range(0,len(volume),2):
                    x=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda()
                    with torch.no_grad():p=model.logits(x).softmax(1)
                    assert torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6;result.append(p.cpu());costs['evaluation_encoder_images']+=len(x);costs['evaluation_logit_forwards']+=1
                save(folder/f'CASE{i:03}.pt',dict(native_probability=torch.cat(result),spacing=spacing));storage(s,reserve=50_000_000)
                write(root/'STATUS.json',dict(status='PREDICTING',cohort=cohort,phase=i,costs=costs))
        assert costs['evaluation_encoder_images']==774 and costs['evaluation_logit_forwards']==398 and costs['label_phase_reads']==0
        write(root/'PREDICTIONS_SEALED.json',dict(time_unix=time.time(),costs=dict(costs),phases=66))
        aggregates={};private={};quad=[];risks=[]
        for cohort,records in maps.items():
            rows=[]
            for i,row in enumerate(records):
                d=load(root/cohort/f'CASE{i:03}.pt');p=restore_probability(d['native_probability']);costs['reconstruction_phases']+=1
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                costs['label_phase_reads']+=1;rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p,y,d['spacing'])))
                if cohort=='val':
                    risks.append(dict(patient=row['patient_id'],case=row['case_id'],fold=folds[row['patient_id']],risk=risk(p,y)))
                    quad.append(dict(patient=row['patient_id'],fold=folds[row['patient_id']],logp=p.permute(0,2,3,1).reshape(-1,4)[::64].double().clamp_min(1e-30).log(),target=torch.from_numpy(np.asarray(y,dtype=np.int64).reshape(-1)[::64].copy())))
                write(root/'STATUS.json',dict(status='RAW_SCORING',cohort=cohort,phase=i,costs=costs))
            aggregates[cohort]=(aggregate_training if cohort=='train' else aggregate)(rows);private[cohort]=rows
        fits={}
        for fold in range(5):
            assert time.time()-s['original_started_unix']<85500
            selected=[x for x in quad if x['fold']!=fold];logp=torch.cat([x['logp'] for x in selected]).cuda();target=torch.cat([x['target'] for x in selected]).cuda();weights=torch.cat([torch.full((len(x['target']),),1/(len(selected)*len(x['target'])),dtype=torch.float64) for x in selected]).cuda()
            fit=fit_temperature(logp,target,weights);fit.update(fit_patients=sorted({x['patient'] for x in selected}),heldout_patients=sorted(p for p in folds if folds[p]==fold));assert not set(fit['fit_patients'])&set(fit['heldout_patients']);fits[str(fold)]=fit
            costs['temperature_fits']+=1;costs['temperature_function_evaluations']+=fit['function_evaluations'];del logp,target,weights
        save(root/'PRIVATE_TEMPERATURES.pt',fits);write(root/'TEMPERATURES_SEALED.json',dict(time_unix=time.time(),fits=5,costs=dict(costs)))
        for i,row in enumerate(maps['val']):
            assert time.time()-s['original_started_unix']<85500
            d=load(root/'val'/f'CASE{i:03}.pt');p=restore_probability(d['native_probability']);costs['temperature_reconstruction_phases']+=1;fit=fits[str(folds[row['patient_id']])];assert row['patient_id'] not in fit['fit_patients']
            z=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);changes=int((p.argmax(1)!=z.argmax(1)).sum());costs['argmax_changes']+=changes;assert changes==0
            append(root/'temperature_output_manifest.jsonl',dict(phase=i,fold=folds[row['patient_id']],T=fit['T'],argmax_changes=changes,sealed_before_heldout_label=True))
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            costs['label_phase_reads']+=1;risks[i]['temperature_risk']=risk(z,y)
            write(root/'STATUS.json',dict(status='TEMPERATURE_SCORING',phase=i,costs=costs))
        assert costs['label_phase_reads']==112 and costs['temperature_fits']==5 and costs['temperature_function_evaluations']<=320 and costs['argmax_changes']==0
        controls=read(factor/'RESULTS.json')['aggregates'];contrasts={c:paired_summary(a,controls[c]['U_D']) for c,a in aggregates.items()}
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in meta.items())
        assert same({k:v for k,v in model.state_dict().items() if k.startswith('backbone.segmentation_head.')},{k:v for k,v in initial['model'].items() if k.startswith('backbone.segmentation_head.')})
        save(root/'PRIVATE_PHASE_RESULTS.pt',private);save(root/'PRIVATE_RISKS.pt',risks)
        write(root/'RESULTS.json',dict(status='COMPLETE',aggregates=aggregates,contrasts=contrasts,costs=costs,label_smoothing=SMOOTHING,lambda_Dice=1,trainable_parameters=482500,source_frozen=True,all_predictions_before_labels=True,test_access=False,phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:save(root/'failure_partial.pt',dict(step=active['step'],model=active['model'].state_dict(),optimizer=active['opt'].state_dict(),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
