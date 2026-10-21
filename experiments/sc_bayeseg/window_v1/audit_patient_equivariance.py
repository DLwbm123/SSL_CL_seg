"""Independent ledger/state/saved-probability audit and four-arm patient contrasts."""
import json,os
from pathlib import Path
import numpy as np,torch,h5py
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import write
from sc_bayeseg.window_v1.patient_validation import aggregate,phase_metrics
from sc_bayeseg.window_v1.patient_train_readout import aggregate_training
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability
from sc_bayeseg.window_v1.patient_pro_diagnostics import risk
from sc_bayeseg.window_v1.patient_point_estimate import PointG32,VARIANCE_NAMES
from sc_bayeseg.window_v1.patient_e2e_validation import EndToEndDirect
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.patient_equivariance import ARMS

def interval(v):
    a=np.asarray(v,dtype=float);g=np.random.default_rng(163);return dict(mean_difference=float(a.mean()),patient_bootstrap95=np.quantile(a[g.integers(0,len(a),(10000,len(a)))].mean(1),[.025,.975]).tolist())

def audit(s):
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False);lines=lambda n:[json.loads(v) for v in (root/n).read_text().splitlines()]
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    c=r['costs'];expected=dict(optimizer=20000,training_reverse=20000,diagnostic_reverse_requests=24,train_encoder_images=80000,train_U_logits=20000,train_G_forward=40000,train_G_image_forward=80000,train_Theta=20000,evaluation_encoder_images=9624,evaluation_U_logits=2464,evaluation_G_forward=2464,evaluation_G_image_forward=4812,main_probability_sets=264,aligned_flip_sets=552,raw_restores=264,label_phase_reads=112,temperature_fits=20,calibrated_restores=184,argmax_changes=0)
    assert all(c[k]==v for k,v in expected.items()) and 0<c['temperature_functions']<=1280
    updates=lines('physical_updates.jsonl');steps=lines('steps.jsonl');assert len(updates)==len(steps)==20000
    old_steps=[json.loads(v) for v in (Path(s['point_source'])/'steps.jsonl').read_text().splitlines()][:5000];assert len(old_steps)==5000
    for j,(u,v) in enumerate(zip(updates,steps)):
        arm=ARMS[j//5000];n=j%5000+1;assert u==dict(index=j+1,arm=arm,step=n) and v['arm']==arm and v['step']==n and v['absolute_schedule_step']==5000+n
        baseline=steps[j%5000];assert all(v[k]==old_steps[j%5000][k] for k in ('indices','flips','absolute_schedule_step'));assert all(v[k]==baseline[k] for k in ('indices','flips','relative_flips','absolute_schedule_step')) and len(v['indices'])==len(set(v['indices']))==2 and all(k in (1,2,3) for k in v['relative_flips'])
        g=torch.Generator().manual_seed(930163+n);assert v['relative_flips']==torch.randint(1,4,(2,),generator=g).tolist()
        w=.1*min(n/500,1.) if arm.endswith('EQ') else 0.;assert v['JS_weight']==w and v['JS']>=-2e-6 and abs(v['loss']-(v['CE']+v['Dice_loss']+v['mean_penalty']+w*v['JS']))<2e-6
        assert all(np.isfinite(v[k]) for k in ('loss','CE','Dice_loss','mean_penalty','JS','gradient_norm_before_clip'))
    probes=lines('gradient_probes.jsonl');assert len(probes)==12 and {(p['arm'],p['step']) for p in probes}=={(a,n) for a in ('U_EQ','G_EQ') for n in (1,1000,2000,3000,4000,5000)}
    sources=dict(U=load(Path(s['factor_source'])/'U_D/S0000.pt'),G=load(Path(s['r32_source'])/'S0000.pt'))
    for p,v in read(root/'PRIVATE_SOURCE_METADATA.json').items():assert (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==(v['size'],v['mtime_ns'])
    for arm in ARMS:
        is_g=arm.startswith('G');source=sources['G' if is_g else 'U'];model=(PointG32(build_model(),SpatialResidualModel()) if is_g else EndToEndDirect(build_model()));model.backbone.requires_grad_(True);model.backbone.segmentation_head.requires_grad_(False);names=[n for n,p in model.named_parameters() if p.requires_grad]
        for n in range(0,5001,1000):
            d=load(root/arm/f'S{n:04}.pt');assert d['step']==n and d['arm']==arm and d['JS_enabled']==arm.endswith('EQ')
            if n==0:assert same(d['model'],source['model']) and same(d['rng'],source['rng']) and not d['optimizer']['state']
            else:assert all(int(v['step'])==n for v in d['optimizer']['state'].values())
            for name,v in d['model'].items():
                if name.startswith(('generator.','backbone.segmentation_head.')):assert torch.equal(v,source['model'][name])
                if is_g and name in VARIANCE_NAMES:assert torch.equal(v[16:],source['model'][name][16:])
            if is_g and n:
                ids=d['optimizer']['param_groups'][0]['params'];assert len(ids)==len(names)
                for name,index in zip(names,ids):
                    if name in VARIANCE_NAMES:assert not torch.count_nonzero(d['optimizer']['state'][index]['exp_avg'][16:]) and not torch.count_nonzero(d['optimizer']['state'][index]['exp_avg_sq'][16:])
        assert any(not torch.equal(v,source['model'][k]) for k,v in d['model'].items() if k.startswith('backbone.') and not k.startswith('backbone.segmentation_head.'))
    maps=dict(train=read(Path(s['train_readout_source'])/'PRIVATE_EVALUATION_MAP.json'),val=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json'));folds=read(Path(s['diagnostic_source'])/'PRIVATE_FOLDS.json');assert [folds[p] for p in sorted(folds)]==[i%5 for i in range(23)]
    seal=read(root/'PREDICTIONS_SEALED.json');assert seal['costs']['label_phase_reads']==0 and seal['costs']['main_probability_sets']==264 and seal['costs']['aligned_flip_sets']==552
    ledger=lines('forward_phase_ledger.jsonl');assert len(ledger)==816 and sum(v['slices'] for v in ledger)==9624 and all(v['label_reads']==0 for v in ledger)
    expected_keys=[(a,co,i,k) for a in ARMS for co,rows in maps.items() for i,row in enumerate(rows) for k in range(4 if co=='val' else 1)];assert [(v['arm'],v['cohort'],v['phase'],v['flip']) for v in ledger]==expected_keys
    private=load(root/'PRIVATE_PHASE_RESULTS.pt');risks=load(root/'PRIVATE_RISKS.pt');fits=load(root/'PRIVATE_TEMPERATURES.pt');eq=read(root/'PRIVATE_EQUIVARIANCE.json');trans=read(root/'PRIVATE_TRANSITIONS.json');quad={a:[] for a in ARMS};maximum=[0.]
    def same_value(x,y):
        if isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:same_value(x[k],y[k])
        elif isinstance(x,list):
            assert len(x)==len(y)
            for a,b in zip(x,y):same_value(a,b)
        elif isinstance(x,float):
            e=abs(x-y);assert e<1e-10;maximum[0]=max(maximum[0],e)
        else:assert x==y
    for co,records in maps.items():
        recomputed={a:[] for a in ARMS}
        for i,row in enumerate(records):
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            predictions={}
            for arm in ARMS:
                d=load(root/arm/co/'identity'/f'CASE{i:03}.pt');native=d['native_probability'];assert native.dtype==torch.float32 and torch.isfinite(native).all() and (native.sum(1)-1).abs().max()<2e-6
                p=restore_probability(native);predictions[arm]=p.argmax(1).numpy();recomputed[arm].append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p,y,d['spacing'])))
                if co=='val':
                    rr=risks[arm][i];assert rr['patient']==row['patient_id'] and rr['fold']==folds[row['patient_id']];same_value(rr['risk'],risk(p,y));fit=fits[arm][str(rr['fold'])];assert row['patient_id'] not in fit['fit_patients']
                    z=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);assert torch.equal(z.argmax(1),p.argmax(1));same_value(rr['temperature_risk'],risk(z,y))
                    quad[arm].append((rr['fold'],p.permute(0,2,3,1).reshape(-1,4)[::64].double().clamp_min(1e-30).log(),torch.from_numpy(np.asarray(y,dtype=np.int64).ravel()[::64].copy())))
                    for k in (1,2,3):
                        aligned=load(root/arm/co/f'flip{k}'/f'CASE{i:03}.pt')['native_probability'];assert aligned.shape==native.shape and aligned.dtype==torch.float32 and torch.isfinite(aligned).all() and (aligned.sum(1)-1).abs().max()<2e-6;e=eq[arm][3*i+k-1];assert (e['phase'],e['flip'])==(i,k)
                        tv=float((aligned.double()-native.double()).abs().sum())/(2*len(native)*128**2);hd=float((aligned.argmax(1)!=native.argmax(1)).double().mean());same_value(e['mean_pixel_TV'],tv);same_value(e['hard_disagreement'],hd)
            if co=='val':
                for kind in ('U','G'):
                    a=predictions[kind+'_2V'];b=predictions[kind+'_EQ'];v=trans[kind][i];hist=np.histogramdd(np.stack((y.ravel(),a.ravel(),b.ravel()),1),bins=[np.arange(5)-.5]*3)[0].astype(np.int64);assert hist.tolist()==v['counts'];assert v['new_RV_real_loss']==(int(((y==3)&(a==3)).sum())>0 and int(((y==3)&(b==3)).sum())==0)
                    for k,w in zip((1,2,3),v['classes']):
                        gt=y==k;old=a==k;new=b==k;expected=dict(GT=int(gt.sum()),identity_TP=int((gt&old).sum()),TTA_TP=int((gt&new).sum()),lost_TP=int((gt&old&~new).sum()),gained_TP=int((gt&~old&new).sum()),removed_FP=int((~gt&old&~new).sum()),added_FP=int((~gt&~old&new).sum()))
                        assert all(w[key]==value for key,value in expected.items())
            print('recomputed',co,i,flush=True)
        for arm,rows in recomputed.items():same_value(rows,private[co][arm]);same_value((aggregate_training if co=='train' else aggregate)(rows),r['aggregates'][co][arm])
    manifests=lines('temperature_output_manifest.jsonl');assert len(manifests)==184 and read(root/'TEMPERATURES_SEALED.json')['costs']['temperature_fits']==20
    assert [(m['arm'],m['phase']) for m in manifests]==[(a,i) for i in range(46) for a in ARMS] and all(m['sealed_before_label'] and m['argmax_changes']==0 for m in manifests)
    assert sum(f['function_evaluations'] for af in fits.values() for f in af.values())==c['temperature_functions']
    for arm in ARMS:
        for fold in range(5):
            fit=fits[arm][str(fold)];assert fit['success'] and -3<=fit['logT']<=3 and 0<fit['function_evaluations']<=64 and abs(fit['T']-np.exp(fit['logT']))<1e-12
            assert set(fit['fit_patients'])=={p for p in folds if folds[p]!=fold} and set(fit['heldout_patients'])=={p for p in folds if folds[p]==fold}
            vals=[]
            for f,lp,target in quad[arm]:
                if f!=fold:
                    z=lp.cuda()/fit['T'];t=target.cuda();vals.append(float((torch.logsumexp(z,1)-z.gather(1,t[:,None]).squeeze(1)).mean()))
            assert abs(np.mean(vals)-fit['objective'])<1e-10
    assert r['source_frozen'] and r['predictions_sealed_before_labels'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    audit_result=dict(status='PASS',updates=20000,main_reverse=20000,extra_gradient_requests=24,complete_checkpoints=24,initial_model_RNG_freshAdam=True,paired_schedule=True,G_and_unused_head_and_800_variance_rows_and_Adam_frozen=True,main_prediction_sets=264,aligned_flip_sets=552,integer_transitions_exact=True,maximum_metric_error=maximum[0],temperature_fits_checked=20,argmax_changes=0,audit_main_restores=264,audit_aligned_native_reads=552,audit_shared_GT_reads=66,audit_temperature_transforms=184,new_network_forward=0,new_optimizer=0,refits=0)
    write(root/'AUDIT.json',audit_result)
    aliases={p:f'VAL{i:03}' for i,p in enumerate(sorted(folds))};trainp=sorted({x['patient_id'] for x in maps['train']});aliases.update({p:f'TRAIN{i:03}' for i,p in enumerate(trainp)});cases={v['case_id']:f'{co.upper()}_PHASE{i:03}' for co,rows in maps.items() for i,v in enumerate(rows)}
    def anon(v):
        if isinstance(v,dict):return {k:aliases[x] if k=='patient' else cases[x] if k=='case' else anon(x) for k,x in v.items()}
        if isinstance(v,list):return [anon(x) for x in v]
        return v
    agg=r['aggregates']['val'];vec=lambda a:np.array([p['macro_Dice'] for p in agg[a]['patients']]);dg=vec('G_EQ')-vec('G_2V');du=vec('U_EQ')-vec('U_2V');contrasts=dict(Delta_G=interval(dg),Delta_U=interval(du),interaction_I=interval(dg-du),direct_S=interval(vec('G_EQ')-vec('U_EQ')),baseline_S0=interval(vec('G_2V')-vec('U_2V')))
    patient_risks={};eq_patients={}
    for arm in ARMS:
        patient_risks[arm]=[];eq_patients[arm]=[]
        for patient in sorted(folds):
            rows=[v for v in risks[arm] if v['patient']==patient];indices=[i for i,v in enumerate(maps['val']) if v['patient_id']==patient];record=dict(patient=aliases[patient]);assert len(rows)==len(indices)==2
            for key in ('NLL','Brier'):record[key]=float(np.mean([v['risk'][key] for v in rows]));record['calibrated_'+key]=float(np.mean([v['temperature_risk'][key] for v in rows]))
            patient_risks[arm].append(record);eq_patients[arm].append(dict(patient=aliases[patient],**{key:float(np.mean([eq[arm][3*i+k-1][key] for i in indices for k in (1,2,3)])) for key in ('mean_pixel_TV','hard_disagreement')}))
    probability={};equiv_contrasts={}
    for name,a,b in (('G_EQ_minus_G_2V','G_EQ','G_2V'),('U_EQ_minus_U_2V','U_EQ','U_2V'),('G_EQ_minus_U_EQ','G_EQ','U_EQ')):
        probability[name]={key:interval([x[key]-y[key] for x,y in zip(patient_risks[a],patient_risks[b])]) for key in ('NLL','Brier','calibrated_NLL','calibrated_Brier')};equiv_contrasts[name]={key:interval([x[key]-y[key] for x,y in zip(eq_patients[a],eq_patients[b])]) for key in ('mean_pixel_TV','hard_disagreement')}
    newloss={k:sum(v['new_RV_real_loss'] for v in rows) for k,rows in trans.items()};gate=dg.mean()>=.005 and contrasts['Delta_G']['patient_bootstrap95'][0]>0 and newloss['G']==0 and (dg-du).mean()>=0
    public_trans={k:[dict(patient=aliases[v['patient']],phase=f'VAL_PHASE{v["phase"]:03}',new_RV_real_loss=v['new_RV_real_loss'],counts=v['counts'],classes=[{key.replace('identity_','control_').replace('TTA_','EQ_'):value for key,value in w.items()} for w in v['classes']]) for v in rows] for k,rows in trans.items()}
    r.pop('space');r.update(status='COMPLETE_AUDIT_PASS',aggregates=anon(r['aggregates']),audit=audit_result,contrasts=contrasts,patient_risks=patient_risks,probability_contrasts=probability,equivariance_patients=eq_patients,equivariance_contrasts=equiv_contrasts,transitions=public_trans,new_RV_real_loss_counts=newloss,gradient_probes=probes,resource_gate=dict(eligible_for_separately_registered_repeat=bool(gate),criterion='Delta_G>=.005 and bootstrap lower>0 and no new controlTP>0 to EQTP0 RV and point interaction>=0; not confirmation'),temperatures={a:{f:{k:v for k,v in fit.items() if k not in ('fit_patients','heldout_patients')} for f,fit in af.items()} for a,af in fits.items()},scope='Repeated development; four matched two-view pipelines; mean deployment; no TTA or Bayesian claim');write(root/'PUBLIC_RESULTS.json',r);print(json.dumps(audit_result))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
