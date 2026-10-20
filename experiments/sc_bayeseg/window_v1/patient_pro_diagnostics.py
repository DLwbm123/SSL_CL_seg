"""Frozen RV sample/mean readouts, risk decomposition and patient-crossfit temperature."""
import json,os,time,traceback,zlib
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,h5py
from scipy.optimize import minimize_scalar
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.patient_r32 import EndToEndG32
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability
from sc_bayeseg.window_v1.patient_r32_mc import class_nll
from sc_bayeseg.window_v1.r_start import draw_g,local_sample,write,save,append
from sc_bayeseg.window_v1.common import storage


def compress_masks(a):
    return dict(shape=list(a.shape),packed=zlib.compress(a.tobytes(),1))

def expand_masks(blob):
    return np.frombuffer(zlib.decompress(blob['packed']),dtype=np.uint8).reshape(blob['shape'])

def risk(p,y):
    target=torch.from_numpy(np.asarray(y,dtype=np.int64));q=p.double();pred=q.argmax(1);n=target.numel()
    ell=-q.clamp_min(1e-30).log().gather(1,target[:,None]).squeeze(1);wrong=pred!=target;fgbg=(target>0)&(pred==0)
    cells=class_nll(p,y);value=float(ell.mean());assert abs(sum(c['full_image_contribution'] for c in cells)-value)<1e-7
    conf=q.max(1).values
    confusion=[[int(((target==a)&(pred==b)).sum()) for b in range(4)] for a in range(4)]
    return dict(NLL=value,Brier=float(((q-torch.nn.functional.one_hot(target,4).permute(0,3,1,2)).square().sum(1)).mean()),classes=cells,
        correct_NLL_contribution=float(ell[~wrong].sum())/n,error_NLL_contribution=float(ell[wrong].sum())/n,foreground_to_BG_NLL_contribution=float(ell[fgbg].sum())/n,
        high_confidence_wrong=int((wrong&(conf>=.9)).sum()),correct_pixels=int((~wrong).sum()),error_pixels=int(wrong.sum()),correct_conditional_NLL=float(ell[~wrong].mean()) if (~wrong).any() else None,error_conditional_NLL=float(ell[wrong].mean()) if wrong.any() else None,pixels=n,confusion=confusion)


def rv_summary(mask,y):
    gt=np.asarray(y)==3;mask=np.asarray(mask,dtype=bool);tp=int((mask&gt).sum());pn=int(mask.sum());gn=int(gt.sum())
    return dict(GT=gn,predicted=pn,TP=tp,recall=tp/gn if gn else None,Dice=2*tp/(pn+gn) if pn+gn else None,empty=bool(gn and not pn))


def fit_temperature(logp,target,weight):
    # Quadrature labels are restricted to the fit patients by the caller.
    def fun(t):
        z=logp*float(np.exp(-t));return float(((torch.logsumexp(z,1)-z.gather(1,target[:,None]).squeeze(1))*weight).sum())
    result=minimize_scalar(fun,bounds=(-3,3),method='bounded',options=dict(xatol=1e-5,maxiter=64))
    assert result.success and np.isfinite(result.fun) and result.nfev<=64
    return dict(logT=float(result.x),T=float(np.exp(result.x)),objective=float(result.fun),function_evaluations=int(result.nfev),success=bool(result.success),near_boundary=bool(abs(result.x)>2.99))


def run(s):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(s['output']);root.mkdir(exist_ok=False);start=time.time();load=lambda p:torch.load(p,map_location='cpu',weights_only=False);read=lambda p:json.loads(Path(p).read_text())
    costs=dict(network_optimizer=0,reverse=0,encoder_images=0,theta=0,g_forward=0,g_image_forward=0,rv_reconstruction_batches=0,label_phase_reads=0,cache_reconstruction_phases=0,temperature_fits=0,temperature_function_evaluations=0,temperature_reconstruction_phases=0,temperature_argmax_changes=0)
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=1_500_000_000)
        assert read(Path(s['patient_mc_source'])/'AUDIT.json')['status']=='PASS'
        maps=dict(train=read(Path(s['train_readout_source'])/'PRIVATE_EVALUATION_MAP.json'),val=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json'))
        kroot=Path(s['kl_source']);rroot=Path(s['r32_source']);froot=Path(s['factor_source'])
        def cached(arm,cohort,i):
            folder={'K0':rroot,'K1':kroot/'R32_K1','KQ':kroot/'R32_KQ','U_D':froot/'U_D'}[arm]
            return load(folder/cohort/f'CASE{i:03}.pt')
        sources=dict(K0=rroot/'S5000.pt',K1=kroot/'R32_K1/S5000.pt',KQ=kroot/'R32_KQ/S5000.pt')
        metadata={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in sources.values()}
        for arm,source in sources.items():
            model=EndToEndG32(build_model().cuda(),SpatialResidualModel().cuda()).cuda();state=load(source)['model'];model.load_state_dict(state);model.eval().requires_grad_(False)
            for cohort,records in maps.items():
                folder=root/arm/cohort;folder.mkdir(parents=True,exist_ok=False)
                for i,row in enumerate(records):
                    assert time.time()-s['original_started_unix']<85500
                    with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:volume=f['image'][...]
                    masks128=[[] for _ in range(20)];masks384=[[] for _ in range(20)];soft=[[] for _ in range(20)];maxp=[[] for _ in range(20)];means=[];parity=0.
                    original=cached(arm,cohort,i)['native_probability']
                    with torch.no_grad():
                        for offset in range(0,len(volume),2):
                            x=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda();q=model.posterior(x);costs['encoder_images']+=len(x)
                            seed=620163+1000*i+offset//2;tg=torch.Generator(device=x.device).manual_seed(seed);lg=torch.Generator(device=x.device).manual_seed(seed+1000000)
                            mix=torch.zeros(len(x),4,128,128,device=x.device);mean=torch.zeros_like(mix)
                            for t in range(4):
                                draw_g(model.generator,tg);costs['theta']+=1
                                for local in range(5):
                                    if local<4:c,r=local_sample(q,lg);slot=t*4+local
                                    else:c,r=q[0],q[2];slot=16+t
                                    p=model.generator.spatial_logits(c,r).softmax(1);costs['g_forward']+=1;costs['g_image_forward']+=len(x)
                                    if local<4:mix+=p/16
                                    else:mean+=p/4
                                    native=p.cpu();restored=restore_probability(native);costs['rv_reconstruction_batches']+=1
                                    masks128[slot].append(np.packbits((native.argmax(1).numpy()==3),axis=-1));masks384[slot].append(np.packbits((restored.argmax(1).numpy()==3),axis=-1))
                                    soft[slot].append(np.stack((native[:,3].sum((1,2)).numpy(),restored[:,3].sum((1,2)).numpy()),1));maxp[slot].append(np.stack((native[:,3].amax((1,2)).numpy(),restored[:,3].amax((1,2)).numpy()),1))
                            parity=max(parity,float((mix.cpu()-original[offset:offset+len(x)]).abs().max()));means.append(mean.cpu());model.generator.clear_draw()
                    assert parity<=1e-6
                    artifact=dict(native_masks=compress_masks(np.stack([np.concatenate(x) for x in masks128])),evaluation_masks=compress_masks(np.stack([np.concatenate(x) for x in masks384])),soft_volume=np.stack([np.concatenate(x) for x in soft]),max_RV_probability=np.stack([np.concatenate(x) for x in maxp]),mean_native_probability=torch.cat(means),mixture_replay_error=parity)
                    save(folder/f'CASE{i:03}.pt',artifact);append(root/'prediction_ledger.jsonl',dict(arm=arm,cohort=cohort,phase=i,label_reads=costs['label_phase_reads'],parity=parity))
                    write(root/'STATUS.json',dict(status='RV_PREDICTING',arm=arm,cohort=cohort,phase=i,costs=costs));storage(s,reserve=100_000_000)
            assert same(model.state_dict(),state);del model;torch.cuda.empty_cache()
        assert (costs['encoder_images'],costs['theta'],costs['g_forward'],costs['g_image_forward'],costs['label_phase_reads'])==(2322,4776,23880,46440,0)
        write(root/'RV_PREDICTIONS_SEALED.json',dict(time_unix=time.time(),phases=198,costs=dict(costs)))
        private={};quadrature={};patients=sorted({r['patient_id'] for r in maps['val']});folds={p:i%5 for i,p in enumerate(patients)};write(root/'PRIVATE_FOLDS.json',folds)
        for arm in ('K0','K1','KQ','U_D'):
            private[arm]={};quadrature[arm]=[]
            for cohort,records in maps.items():
                rows=[]
                for i,row in enumerate(records):
                    d=cached(arm,cohort,i);p=restore_probability(d['native_probability']);costs['cache_reconstruction_phases']+=1
                    with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                    costs['label_phase_reads']+=1;rp=risk(p,y);rv=rv_summary(p.argmax(1).numpy()==3,y)
                    rv.update(soft_volume=float(p[:,3].sum()),soft_volume_mm3=float(p[:,3].sum())*float(np.prod(d['spacing'])),max_RV_probability=float(p[:,3].max()),GT_to_classes=rp['confusion'][3])
                    native_y=np.asarray(y)[:,::3,::3];native_rv=rv_summary(d['native_probability'].argmax(1).numpy()==3,native_y)
                    entry=dict(native_RV=native_rv,patient=row['patient_id'],case=row['case_id'],risk=rp,RV_mixture=rv,native_RV_present=bool((d['native_probability'].argmax(1)==3).any()))
                    if arm!='U_D':
                        extra=load(root/arm/cohort/f'CASE{i:03}.pt');draws=[];native_draws=[];native_masks=expand_masks(extra['native_masks']);evaluation_masks=expand_masks(extra['evaluation_masks'])
                        for j in range(20):
                            mask=np.unpackbits(evaluation_masks[j],axis=-1).astype(bool);draws.append(rv_summary(mask,y));native_draws.append(rv_summary(np.unpackbits(native_masks[j],axis=-1).astype(bool),native_y))
                        mean=restore_probability(extra['mean_native_probability']);costs['cache_reconstruction_phases']+=1
                        entry.update(RV_native_draws=native_draws,RV_draws=draws,RV_mean=rv_summary(mean.argmax(1).numpy()==3,y),RV_soft_volume=extra['soft_volume'].sum(1).tolist(),RV_max_probability=extra['max_RV_probability'].max(1).tolist(),mixture_replay_error=extra['mixture_replay_error'])
                    rows.append(entry)
                    if cohort=='val':
                        q=p.permute(0,2,3,1).reshape(-1,4)[::64].double().clamp_min(1e-30).log();t=torch.from_numpy(np.asarray(y,dtype=np.int64).reshape(-1)[::64].copy())
                        quadrature[arm].append(dict(patient=row['patient_id'],fold=folds[row['patient_id']],logp=q,target=t))
                    write(root/'STATUS.json',dict(status='RISK_SCORING',arm=arm,cohort=cohort,phase=i,costs=costs))
                private[arm][cohort]=rows
        temperatures={}
        for arm,rows in quadrature.items():
            temperatures[arm]={}
            for fold in range(5):
                selected=[x for x in rows if x['fold']!=fold];assert all(folds[x['patient']]!=fold for x in selected)
                logp=torch.cat([x['logp'] for x in selected]).cuda();target=torch.cat([x['target'] for x in selected]).cuda()
                weight=torch.cat([torch.full((len(x['target']),),1/(len(selected)*len(x['target'])),dtype=torch.float64) for x in selected]).cuda()
                fit=fit_temperature(logp,target,weight);fit['fit_patients']=sorted({x['patient'] for x in selected});fit['heldout_patients']=[p for p in patients if folds[p]==fold]
                assert not set(fit['fit_patients'])&set(fit['heldout_patients']);temperatures[arm][str(fold)]=fit
                costs['temperature_fits']+=1;costs['temperature_function_evaluations']+=fit['function_evaluations'];del logp,target,weight
        save(root/'PRIVATE_TEMPERATURES.pt',temperatures);write(root/'TEMPERATURES_SEALED.json',dict(time_unix=time.time(),fits=20,costs=dict(costs)))
        for arm in private:
            for i,row in enumerate(maps['val']):
                assert time.time()-s['original_started_unix']<85500
                p=restore_probability(cached(arm,'val',i)['native_probability']);costs['temperature_reconstruction_phases']+=1
                fit=temperatures[arm][str(folds[row['patient_id']])];assert row['patient_id'] not in fit['fit_patients']
                z=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);changes=int((p.argmax(1)!=z.argmax(1)).sum());costs['temperature_argmax_changes']+=changes;assert changes==0
                append(root/'temperature_output_manifest.jsonl',dict(arm=arm,phase=i,fold=folds[row['patient_id']],T=fit['T'],argmax_changes=changes,sealed_before_this_heldout_label_read=True,representation='deterministically recomputable transform of frozen source probability'))
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                costs['label_phase_reads']+=1;private[arm]['val'][i]['temperature_risk']=risk(z,y);private[arm]['val'][i]['temperature_fold']=folds[row['patient_id']]
                write(root/'STATUS.json',dict(status='TEMPERATURE_SCORING',arm=arm,phase=i,costs=costs))
        assert costs['temperature_fits']==20 and costs['temperature_function_evaluations']<=1280 and costs['temperature_argmax_changes']==0
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in metadata.items())
        save(root/'PRIVATE_RESULTS.pt',private)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,source_frozen=True,all_RV_predictions_before_labels=True,test_access=False,phase_elapsed_seconds=time.time()-start,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
