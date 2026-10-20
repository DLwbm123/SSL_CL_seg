"""Independent categorical histogram audit of fixed pre/post TTA predictions."""
import json,os
from pathlib import Path
import h5py,numpy as np,torch
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability
from sc_bayeseg.window_v1.r_start import write
from sc_bayeseg.window_v1.patient_tta_transitions import inputs,PAIRS

def audit(s):
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=read(root/'RESULTS.json');assert read(root/'EXIT.json')['exit_code']==0 and read(root/'STATUS.json')['status']==r['status']=='COMPLETE' and not (root/'FAILURE.json').exists()
    c=r['costs'];assert (c['cached_probability_restorations'],c['label_phase_reads'],c['model_phase_pairs'])==(184,46,92) and all(c[k]==0 for k in ('network_optimizer','reverse','encoder_images','G_forward','Theta','temperature_fits'))
    seal=read(root/'FIXED_INPUTS_SEALED.json');assert seal['costs']['label_phase_reads']==0 and seal['phase_pairs']==92 and not seal['readout_changed']
    for p,v in read(root/'PRIVATE_INPUT_METADATA.json').items():assert (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==(v['size'],v['mtime_ns'])
    records=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json');data=read(root/'PRIVATE_RESULTS.json');caches,_=inputs(s);maximum=0.
    for i,row in enumerate(records):
        with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
        for arm in PAIRS:
            before=restore_probability(load(caches[arm][0]/f'CASE{i:03}.pt')['native_probability']);after=restore_probability(load(caches[arm][1]/f'CASE{i:03}.pt')['native_probability']);b=before.argmax(1).numpy();a=after.argmax(1).numpy();v=data[arm][i]
            hist=np.histogramdd(np.stack((y.ravel(),b.ravel(),a.ravel()),1),bins=[np.arange(5)-.5]*3)[0].astype(np.int64)
            assert hist.tolist()==v['GT_identity_TTA_counts'] and hist.sum()==v['pixels']==y.size and v['patient']==row['patient_id'] and v['case']==row['case_id'] and v['phase']==i
            for k,w in zip((1,2,3),v['classes']):
                gt=y==k;old=b==k;new=a==k;tp0=int((gt&old).sum());tp1=int((gt&new).sum());fp0=int((~gt&old).sum());fp1=int((~gt&new).sum());n=int(gt.sum());p0=int(old.sum());p1=int(new.sum())
                expected=dict(class_index=k,GT=n,identity_predicted=p0,TTA_predicted=p1,identity_TP=tp0,TTA_TP=tp1,identity_FP=fp0,TTA_FP=fp1,lost_TP=int((gt&old&~new).sum()),gained_TP=int((gt&~old&new).sum()),removed_FP=int((~gt&old&~new).sum()),added_FP=int((~gt&~old&new).sum()),identity_recall=tp0/n if n else None,TTA_recall=tp1/n if n else None,identity_Dice=2*tp0/(p0+n) if p0+n else None,TTA_Dice=2*tp1/(p1+n) if p1+n else None,identity_empty_failure=bool(n and not p0),TTA_empty_failure=bool(n and not p1),GT_absent=n==0,new_empty_type='lost_GT_TP' if n and p0 and not p1 and tp0 else 'removed_only_FP' if n and p0 and not p1 else 'not_new_empty')
                assert all(w[key]==value for key,value in expected.items())
                assert tp0+w['gained_TP']-w['lost_TP']==tp1 and fp0+w['added_FP']-w['removed_FP']==fp1
                for name,p in (('identity_probability_support',before),('TTA_probability_support',after)):
                    q=p.numpy().astype(np.float64);pk=q[:,k];margin=(pk-np.max(np.delete(q,k,axis=1),axis=1))[gt]
                    values=dict(soft_class_fraction=float(pk.mean()),mean_probability_on_GT=float(pk[gt].mean()) if n else None,mean_winner_margin_on_GT=float(margin.mean()) if n else None,winner_margin_quantiles_05_50_95=np.quantile(margin,[.05,.5,.95]).tolist() if n else None)
                    for key,val in values.items():
                        if val is None:assert w[name][key] is None
                        else:
                            error=float(np.max(np.abs(np.asarray(val)-np.asarray(w[name][key]))));maximum=max(maximum,error);assert error<1e-10
            del before,after
        print('checked phase',i,flush=True)
    assert r['source_frozen'] and not r['readout_changed'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    result=dict(status='PASS',model_phase_pairs_checked=92,integer_histograms_and_transition_counts_exact=True,maximum_probability_support_error=maximum,audit_cached_probability_restorations=184,audit_label_phase_reads=46,new_network_forward=0,new_reverse=0,new_optimizer=0,refits=0)
    write(root/'AUDIT.json',result)
    alias={p:f'VAL{i:03}' for i,p in enumerate(sorted({v['patient_id'] for v in records}))};public={};summary={}
    for arm,rows in data.items():
        public[arm]=[{**v,'patient':alias[v['patient']],'case':f'VAL_PHASE{v["phase"]:03}'} for v in rows]
        summary[arm]=dict(classes=[])
        for k in (1,2,3):
            cells=[v['classes'][k-1] for v in rows]
            summary[arm]['classes'].append(dict(class_index=k,identity_empty_phases=sum(v['identity_empty_failure'] for v in cells),TTA_empty_phases=sum(v['TTA_empty_failure'] for v in cells),new_empty_lost_GT_TP=sum(v['new_empty_type']=='lost_GT_TP' for v in cells),new_empty_removed_only_FP=sum(v['new_empty_type']=='removed_only_FP' for v in cells),lost_TP=sum(v['lost_TP'] for v in cells),gained_TP=sum(v['gained_TP'] for v in cells),removed_FP=sum(v['removed_FP'] for v in cells),added_FP=sum(v['added_FP'] for v in cells)))
    r.pop('space');r.update(status='COMPLETE_AUDIT_PASS',audit=result,phases=public,summary=summary,scope='One post-hoc fixed-prediction transition diagnostic; no prediction selection or repair; repeated development set');write(root/'PUBLIC_RESULTS.json',r);print(json.dumps(result))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
