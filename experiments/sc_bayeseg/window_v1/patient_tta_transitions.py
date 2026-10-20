"""One read-only diagnostic of TP/FP changes and new empty classes after fixed TTA."""
import json,os,time,traceback
from pathlib import Path
import torch,numpy as np,h5py
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability
from sc_bayeseg.window_v1.r_start import write
from sc_bayeseg.window_v1.common import storage
PAIRS=('U_D','G_MEAN')

def transition_summary(matrix,k):
    a=np.asarray(matrix,dtype=np.int64);gt=int(a[k].sum());oldpred=int(a[:,k,:].sum());newpred=int(a[:,:,k].sum());oldtp=int(a[k,k,:].sum());newtp=int(a[k,:,k].sum())
    losttp=int(a[k,k,:].sum()-a[k,k,k]);gainedtp=int(a[k,:,k].sum()-a[k,k,k]);lostall=int(a[:,k,:].sum()-a[:,k,k].sum());gainedall=int(a[:,:,k].sum()-a[:,k,k].sum())
    result=dict(class_index=k,GT=gt,identity_predicted=oldpred,TTA_predicted=newpred,identity_TP=oldtp,TTA_TP=newtp,identity_FP=oldpred-oldtp,TTA_FP=newpred-newtp,lost_TP=losttp,gained_TP=gainedtp,removed_FP=lostall-losttp,added_FP=gainedall-gainedtp,identity_recall=oldtp/gt if gt else None,TTA_recall=newtp/gt if gt else None,identity_Dice=2*oldtp/(oldpred+gt) if oldpred+gt else None,TTA_Dice=2*newtp/(newpred+gt) if newpred+gt else None,identity_empty_failure=bool(gt and not oldpred),TTA_empty_failure=bool(gt and not newpred),GT_absent=gt==0)
    result['new_empty_type']='lost_GT_TP' if gt and oldpred and not newpred and oldtp else 'removed_only_FP' if gt and oldpred and not newpred else 'not_new_empty'
    assert oldtp+gainedtp-losttp==newtp and result['identity_FP']+result['added_FP']-result['removed_FP']==result['TTA_FP']
    return result

def probability_support(p,y,k):
    q=p.double().numpy();mask=np.asarray(y)==k;gt=int(mask.sum());pk=q[:,k];other=np.max(q[:,[j for j in range(4) if j!=k]],axis=1)
    values=(pk-other)[mask]
    return dict(soft_class_fraction=float(pk.mean()),mean_probability_on_GT=float(pk[mask].mean()) if gt else None,mean_winner_margin_on_GT=float(values.mean()) if gt else None,winner_margin_quantiles_05_50_95=np.quantile(values,[.05,.5,.95]).tolist() if gt else None)

def inputs(s):
    root=Path(s['flip_source']);f=Path(s['factor_source']);m=Path(s['mean_source'])
    caches=dict(U_D=(f/'U_D/val',root/'U_D_TTA'),G_MEAN=(m/'val',root/'G_MEAN_TTA'))
    phase_paths=dict(U_D_identity=f/'PRIVATE_PHASE_RESULTS.pt',G_MEAN_identity=m/'PRIVATE_PHASE_RESULTS.pt',TTA=root/'PRIVATE_PHASE_RESULTS.pt')
    return caches,phase_paths

def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time();read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    costs=dict(network_optimizer=0,reverse=0,encoder_images=0,G_forward=0,Theta=0,temperature_fits=0,cached_probability_restorations=0,label_phase_reads=0,model_phase_pairs=0)
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=50_000_000)
        assert read(Path(s['flip_source'])/'AUDIT.json')['status']=='PASS' and read(Path(s['flip_source'])/'EXIT.json')['exit_code']==0
        records=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json');assert len(records)==46
        caches,phase_paths=inputs(s)
        paths=[folder/f'CASE{i:03}.pt' for folders in caches.values() for folder in folders for i in range(46)]+list(phase_paths.values())+[Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json',Path(s['flip_source'])/'AUDIT.json',Path(s['flip_source'])/'RESULTS.json']+[Path(s['data'])/'h5/v1'/r['label_h5_relpath'] for r in records]
        metadata={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths};write(root/'PRIVATE_INPUT_METADATA.json',metadata);write(root/'FIXED_INPUTS_SEALED.json',dict(time_unix=time.time(),costs=dict(costs),phase_pairs=92,readout_changed=False))
        prior=dict(U_D=load(phase_paths['U_D_identity'])['val']['U_D'],G_MEAN=load(phase_paths['G_MEAN_identity'])['val']);tta=load(phase_paths['TTA']);outputs={a:[] for a in PAIRS}
        for i,row in enumerate(records):
            assert time.time()-s['original_started_unix']<85500
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            costs['label_phase_reads']+=1;assert y.min()>=0 and y.max()<=3
            for arm in PAIRS:
                before=restore_probability(load(caches[arm][0]/f'CASE{i:03}.pt')['native_probability']);after=restore_probability(load(caches[arm][1]/f'CASE{i:03}.pt')['native_probability']);costs['cached_probability_restorations']+=2
                b=before.argmax(1).numpy();a=after.argmax(1).numpy();assert b.shape==a.shape==y.shape
                matrix=np.bincount((np.asarray(y,dtype=np.int64)*16+b*4+a).reshape(-1),minlength=64).reshape(4,4,4);classes=[]
                baseline=prior[arm][i];endpoint=tta['U_D_TTA' if arm=='U_D' else 'G_MEAN_TTA'][i];assert baseline['patient']==endpoint['patient']==row['patient_id'] and baseline['case']==endpoint['case']==row['case_id']
                for k in (1,2,3):
                    v=transition_summary(matrix,k)
                    for old,new in ((baseline['metrics']['classes'][k-1],('identity_predicted','identity_TP','identity_FP','identity_Dice','identity_empty_failure')),(endpoint['metrics']['classes'][k-1],('TTA_predicted','TTA_TP','TTA_FP','TTA_Dice','TTA_empty_failure'))):
                        assert old['GT']==v['GT'] and all(old[key]==v[name] for key,name in zip(('predicted','TP','FP','Dice','empty_prediction_failure'),new))
                    v['identity_probability_support']=probability_support(before,y,k);v['TTA_probability_support']=probability_support(after,y,k);classes.append(v)
                outputs[arm].append(dict(patient=row['patient_id'],case=row['case_id'],phase=i,pixels=int(y.size),GT_identity_TTA_counts=matrix.tolist(),classes=classes));costs['model_phase_pairs']+=1
                del before,after
            write(root/'STATUS.json',dict(status='ANALYZING',phase=i,costs=costs))
        assert costs['cached_probability_restorations']==184 and costs['label_phase_reads']==46 and costs['model_phase_pairs']==92
        assert all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==(v['size'],v['mtime_ns']) for p,v in metadata.items())
        write(root/'PRIVATE_RESULTS.json',outputs);write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,source_frozen=True,readout_changed=False,test_access=False,phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s,reserve=50_000_000)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
