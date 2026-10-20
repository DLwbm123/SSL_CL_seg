"""Fixed-prediction, class-quota selective-risk qualification without training."""
import json,os,time,traceback,zlib
from pathlib import Path
import torch,numpy as np,h5py
from torch.nn import functional as F
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.patient_r32 import EndToEndG32
from sc_bayeseg.window_v1.patient_point_estimate import PointG32
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability
from sc_bayeseg.window_v1.r_start import draw_g,local_sample,write,save,append
from sc_bayeseg.window_v1.common import storage

ARMS=('K1','D0','Dmu','U_D');COVERAGES=(.5,.75,.9)


def entropy(p):
    q=p.double();return -(q*q.clamp_min(1e-30).log()).sum(1)


def masks_for_score(pred,score):
    flat=pred.reshape(-1);score=np.asarray(score).reshape(-1);assert np.isfinite(score).all()
    result=np.zeros((3,len(flat)),dtype=bool);quota=[]
    for k in range(4):
        ids=np.flatnonzero(flat==k);order=ids[np.argsort(score[ids],kind='stable')];counts=[]
        for j,c in enumerate(COVERAGES):
            n=int(np.floor(c*len(ids)));result[j,order[:n]]=True;counts.append(n)
        quota.append(counts)
    top=max(1,int(np.ceil(.01*len(score))));case=float(np.partition(score,len(score)-top)[-top:].mean())
    return dict(shape=list(pred.shape),packed=zlib.compress(np.packbits(result,axis=1).tobytes(),1),quota=quota,case_score=case)


def decode_masks(blob):
    n=int(np.prod(blob['shape']));raw=np.frombuffer(zlib.decompress(blob['packed']),dtype=np.uint8).reshape(3,(n+7)//8)
    return np.unpackbits(raw,axis=1)[:,:n].reshape((3,)+tuple(blob['shape'])).astype(bool)


def selection_counts(pred,y,mask):
    error=pred!=y;counts=[]
    for k in range(4):
        pk=pred==k;gk=y==k
        counts.append(dict(class_index=k,predicted_total=int(pk.sum()),GT_total=int(gk.sum()),accepted_predicted=int((mask&pk).sum()),errors_predicted=int((mask&pk&error).sum()),accepted_GT=int((mask&gk).sum()),errors_GT=int((mask&gk&error).sum())))
    return dict(accepted=int(mask.sum()),errors=int((mask&error).sum()),foreground_to_BG_accepted=int((mask&(y>0)&(pred==0)).sum()),classes=counts)


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time();read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    costs=dict(network_optimizer=0,reverse=0,encoder_images=0,Theta=0,G_forward=0,G_image_forward=0,prediction_phase_sets=0,score_map_interpolations=0,cached_probability_restorations=0,label_phase_reads=0,temperature_fits=0)
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=1_200_000_000)
        calibration=Path(s['calibration_source']);assert read(calibration/'AUDIT.json')['status']=='PASS' and read(calibration/'EXIT.json')['exit_code']==0
        point=Path(s['point_source']);diag=Path(s['diagnostic_source']);kl=Path(s['kl_source']);factor=Path(s['factor_source'])
        records=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json');folds=read(diag/'PRIVATE_FOLDS.json');assert len(records)==46
        temps=dict(load(diag/'PRIVATE_TEMPERATURES.pt'),**load(calibration/'PRIVATE_TEMPERATURES.pt'))
        folders=dict(K1=kl/'R32_K1',D0=point/'D0',Dmu=point/'Dmu',U_D=factor/'U_D');sourcefiles=[folders[a]/'S5000.pt' for a in ARMS if a!='U_D']+[calibration/'PUBLIC_RESULTS.json']
        metadata={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in sourcefiles};negative={}
        for arm_index,arm in enumerate(ARMS):
            folder=root/arm;folder.mkdir(exist_ok=False);model=None
            if arm!='U_D':
                state=load(folders[arm]/'S5000.pt')['model'];cls=EndToEndG32 if arm=='K1' else PointG32
                model=cls(build_model().cuda(),SpatialResidualModel().cuda()).cuda();model.load_state_dict(state);model.eval().requires_grad_(False)
            for i,row in enumerate(records):
                assert time.time()-s['original_started_unix']<85500
                d=load(folders[arm]/'val'/f'CASE{i:03}.pt');native=d['native_probability'];score_native={};parity=0.;neg={}
                if model is not None:
                    parts={}
                    with h5py.File(Path(s['data'])/'h5/v1'/row['image_h5_relpath'],'r') as f:volume=f['image'][...]
                    with torch.no_grad():
                        for offset in range(0,len(volume),2):
                            x=torch.from_numpy(np.ascontiguousarray(volume[offset:offset+2,None])).float().cuda();q=model.posterior(x) if arm=='K1' else model.means(x);costs['encoder_images']+=len(x)
                            seed=620163+1000*i+offset//2;tg=torch.Generator(device=x.device).manual_seed(seed);lg=torch.Generator(device=x.device).manual_seed(seed+1000000)
                            total=torch.zeros(len(x),4,128,128,device=x.device,dtype=torch.float64);eh=torch.zeros(len(x),128,128,device=x.device,dtype=torch.float64);eth=torch.zeros_like(eh);local_count=4 if arm=='K1' else 1
                            for t in range(4):
                                draw_g(model.generator,tg);costs['Theta']+=1;tp=torch.zeros_like(total)
                                for l in range(local_count):
                                    args=local_sample(q,lg) if arm=='K1' else q;p=model.generator.spatial_logits(*args).softmax(1)
                                    tp+=p.double();eh+=entropy(p)/(4*local_count);costs['G_forward']+=1;costs['G_image_forward']+=len(x)
                                total+=tp/(4*local_count);eth+=entropy(tp/local_count)/4
                            h=entropy(total);gaps=dict(global_disagreement=h-eth)
                            if arm=='K1':gaps.update(local_disagreement=eth-eh,total_disagreement=h-eh);assert float((gaps['global_disagreement']+gaps['local_disagreement']-gaps['total_disagreement']).abs().max())<2e-6
                            parity=max(parity,float((total.cpu()-native[offset:offset+len(x)].double()).abs().max()));model.generator.clear_draw()
                            for name,v in gaps.items():
                                minimum=float(v.min());assert minimum>=-2e-6;neg[name]=dict(minimum=min(neg.get(name,dict(minimum=0.))['minimum'],minimum),negative_values=neg.get(name,dict(negative_values=0))['negative_values']+int((v<0).sum()))
                                parts.setdefault(name,[]).append(v.clamp_min(0).float().cpu())
                    assert parity<=1e-6;score_native={k:torch.cat(v) for k,v in parts.items()}
                p=restore_probability(native);costs['cached_probability_restorations']+=1;pred=p.argmax(1).numpy()
                fit=temps[arm][str(folds[row['patient_id']])];assert row['patient_id'] not in fit['fit_patients'];cal=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);assert torch.equal(cal.argmax(1),p.argmax(1))
                scores=dict(confidence=(1-p.max(1).values).numpy(),entropy=entropy(p).numpy(),calibrated_confidence=(1-cal.max(1).values).numpy(),calibrated_entropy=entropy(cal).numpy(),random=np.random.default_rng(920163+1000*arm_index+i).random(pred.shape))
                for name,v in score_native.items():scores[name]=F.interpolate(v[:,None].cuda(),(384,384),mode='bilinear',align_corners=False)[:,0].cpu().numpy();costs['score_map_interpolations']+=1
                selections={name:masks_for_score(pred,score) for name,score in scores.items()};quotas=[v['quota'] for v in selections.values()];assert all(q==quotas[0] for q in quotas)
                save(folder/f'CASE{i:03}.pt',dict(native_scores=score_native,selections=selections,parity=parity,negative_roundoff=neg));append(root/'prediction_ledger.jsonl',dict(arm=arm,phase=i,methods=list(selections),label_phase_reads=costs['label_phase_reads'],parity=parity))
                costs['prediction_phase_sets']+=1;write(root/'STATUS.json',dict(status='RANKING',arm=arm,phase=i,costs=costs));storage(s,reserve=50_000_000)
            if model is not None:assert same(model.state_dict(),state);del model;torch.cuda.empty_cache()
        assert (costs['encoder_images'],costs['Theta'],costs['G_forward'],costs['G_image_forward'],costs['prediction_phase_sets'],costs['label_phase_reads'])==(1632,3336,6672,13056,184,0)
        write(root/'SELECTIONS_SEALED.json',dict(time_unix=time.time(),phase_sets=184,costs=dict(costs)))
        private={}
        for arm in ARMS:
            private[arm]=[]
            for i,row in enumerate(records):
                assert time.time()-s['original_started_unix']<85500
                native=load(folders[arm]/'val'/f'CASE{i:03}.pt')['native_probability'];p=restore_probability(native);costs['cached_probability_restorations']+=1;pred=p.argmax(1).numpy();artifact=load(root/arm/f'CASE{i:03}.pt')
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                costs['label_phase_reads']+=1;methods={};dice=[]
                for k in (1,2,3):
                    a=pred==k;b=y==k;den=int(a.sum()+b.sum());dice.append(2*int((a&b).sum())/den if b.any() and den else None)
                for method,blob in artifact['selections'].items():
                    masks=decode_masks(blob);cells=[]
                    for j,coverage in enumerate(COVERAGES):
                        v=selection_counts(pred,y,masks[j]);assert [x['accepted_predicted'] for x in v['classes']]==[q[j] for q in blob['quota']];cells.append(dict(coverage=coverage,**v))
                    methods[method]=dict(case_score=blob['case_score'],coverage_results=cells)
                private[arm].append(dict(patient=row['patient_id'],phase=i,phase_class_Dice=dice,RV_empty=bool((y==3).any() and not (pred==3).any()),methods=methods))
                write(root/'STATUS.json',dict(status='SCORING',arm=arm,phase=i,costs=costs))
        assert costs['label_phase_reads']==184 and costs['cached_probability_restorations']==368
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in metadata.items());save(root/'PRIVATE_RESULTS.pt',private)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,source_frozen=True,all_selections_before_labels=True,fixed_predictions=True,test_access=False,phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
