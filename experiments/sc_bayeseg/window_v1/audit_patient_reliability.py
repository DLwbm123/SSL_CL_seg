"""Rebuild all class-quota selections and risk counts from frozen predictions and score artifacts."""
import json,os
from pathlib import Path
import torch,numpy as np,h5py
from torch.nn import functional as F
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability
from sc_bayeseg.window_v1.patient_reliability import ARMS,COVERAGES,entropy,masks_for_score,decode_masks,selection_counts
from sc_bayeseg.window_v1.summarize_patient_reliability import summarize,paired,auc
from sc_bayeseg.window_v1.r_start import write


def audit(s):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    assert r['costs']==dict(network_optimizer=0,reverse=0,encoder_images=1632,Theta=3336,G_forward=6672,G_image_forward=13056,prediction_phase_sets=184,score_map_interpolations=230,cached_probability_restorations=368,label_phase_reads=184,temperature_fits=0)
    ledger=[json.loads(x) for x in (root/'prediction_ledger.jsonl').read_text().splitlines()];assert [(x['arm'],x['phase']) for x in ledger]==[(a,i) for a in ARMS for i in range(46)] and all(x['label_phase_reads']==0 and x['parity']<=1e-6 for x in ledger)
    assert read(root/'SELECTIONS_SEALED.json')['costs']['label_phase_reads']==0
    point=Path(s['point_source']);kl=Path(s['kl_source']);diag=Path(s['diagnostic_source']);calibration=Path(s['calibration_source']);factor=Path(s['factor_source'])
    folders=dict(K1=kl/'R32_K1',D0=point/'D0',Dmu=point/'Dmu',U_D=factor/'U_D');records=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json');folds=read(diag/'PRIVATE_FOLDS.json');temps=dict(load(diag/'PRIVATE_TEMPERATURES.pt'),**load(calibration/'PRIVATE_TEMPERATURES.pt'));private=load(root/'PRIVATE_RESULTS.pt');maxerr=[0.];sets=0;masks=0
    def eq(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:eq(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):eq(x,y)
        elif isinstance(a,float):
            e=abs(a-b);assert e<1e-10;maxerr[0]=max(maxerr[0],e)
        else:assert a==b
    for arm_index,arm in enumerate(ARMS):
        for i,row in enumerate(records):
            d=load(folders[arm]/'val'/f'CASE{i:03}.pt');p=restore_probability(d['native_probability']);pred=p.argmax(1).numpy();artifact=load(root/arm/f'CASE{i:03}.pt');assert artifact['parity']<=1e-6
            fit=temps[arm][str(folds[row['patient_id']])];assert row['patient_id'] not in fit['fit_patients'];cal=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);assert torch.equal(cal.argmax(1),p.argmax(1))
            scores=dict(confidence=(1-p.max(1).values).numpy(),entropy=entropy(p).numpy(),calibrated_confidence=(1-cal.max(1).values).numpy(),calibrated_entropy=entropy(cal).numpy(),random=np.random.default_rng(920163+1000*arm_index+i).random(pred.shape))
            expected_native={'global_disagreement','local_disagreement','total_disagreement'} if arm=='K1' else {'global_disagreement'} if arm!='U_D' else set();assert set(artifact['native_scores'])==expected_native
            if arm=='K1':assert float((artifact['native_scores']['global_disagreement']+artifact['native_scores']['local_disagreement']-artifact['native_scores']['total_disagreement']).abs().max())<=2e-6
            for name,v in artifact['native_scores'].items():
                assert v.dtype==torch.float32 and v.shape==(len(p),128,128) and torch.isfinite(v).all() and (v>=0).all()
                assert artifact['negative_roundoff'][name]['minimum']>=-2e-6 and artifact['negative_roundoff'][name]['negative_values']>=0
                scores[name]=F.interpolate(v[:,None].cuda(),(384,384),mode='bilinear',align_corners=False)[:,0].cpu().numpy()
            assert set(scores)==set(artifact['selections'])
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            methods={};class_dice=[]
            for k in (1,2,3):
                a=pred==k;b=y==k;den=int(a.sum()+b.sum());class_dice.append(2*int((a&b).sum())/den if b.any() and den else None)
            quota=None
            for method,score in scores.items():
                expected=masks_for_score(pred,score);actual=artifact['selections'][method];a=decode_masks(actual);b=decode_masks(expected)
                assert np.array_equal(a,b);eq(actual['quota'],expected['quota']);eq(actual['case_score'],expected['case_score'])
                if quota is None:quota=actual['quota']
                else:eq(quota,actual['quota'])
                cells=[]
                for j,coverage in enumerate(COVERAGES):
                    counts=selection_counts(pred,y,a[j]);assert [x['accepted_predicted'] for x in counts['classes']]==[q[j] for q in quota]
                    assert sum(x['accepted_predicted'] for x in counts['classes'])==counts['accepted'] and sum(x['errors_predicted'] for x in counts['classes'])==counts['errors']
                    cells.append(dict(coverage=coverage,**counts));masks+=1
                methods[method]=dict(case_score=actual['case_score'],coverage_results=cells)
            reconstructed=dict(patient=row['patient_id'],phase=i,phase_class_Dice=class_dice,RV_empty=bool((y==3).any() and not (pred==3).any()),methods=methods);eq(reconstructed,private[arm][i]);sets+=1
        print('recomputed selections',arm,flush=True)
    assert sets==184 and masks==3450
    stats=summarize(s,require_audit=False)
    # Independent arithmetic for the primary paired risk, based on separately verified patient counts.
    a=[];b=[]
    for p in stats['models']['K1']['patients']:
        values=[]
        for method in ('local_disagreement','calibrated_confidence'):
            cells=p['methods'][method]['coverage_results'][1]['classes'][1:]
            values.append(sum(v['errors_predicted']/v['accepted_predicted'] for v in cells)/3 if all(v['accepted_predicted'] for v in cells) else None)
        a.append(values[0]);b.append(values[1])
    eq(paired(a,b),stats['primary'])
    for model in stats['models'].values():
        for method,v in model['summaries'].items():
            for label,key in (('macro_failure','macro_failure_AUROC'),('RV_failure','RV_failure_AUROC')):
                pos=[p['methods'][method]['case_score'] for p in model['patients'] if p[label]];neg=[p['methods'][method]['case_score'] for p in model['patients'] if not p[label]]
                manual=float(np.mean([float(a>b)+.5*float(a==b) for a in pos for b in neg])) if pos and neg else None
                eq(manual,v[key]['AUROC'])
    assert auc([.1,.9],[False,True])['AUROC']==1 and auc([.9,.1],[False,True])['AUROC']==0
    assert r['source_frozen'] and r['all_selections_before_labels'] and r['fixed_predictions'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    out=dict(status='PASS',model_phase_sets=184,selection_masks_rebuilt=3450,maximum_count_metric_error=maxerr[0],quotas_equal_across_methods=True,all_masks_losslessly_recovered=True,sealed_before_labels=True,fixed_predictions_and_original_patient_Dice_checked=True,primary_recomputed_from_patient_counts=True,all_case_AUROC_pairwise_direction_checked=True,native_score_additivity_and_roundoff_bounds=True,source_freeze_asserted_by_worker=True,audit_cached_probability_restorations=184,audit_score_map_interpolations=230,audit_label_phases=184,new_network_forward=0,new_reverse=0,new_network_updates=0)
    write(root/'AUDIT.json',out);print(json.dumps(out))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
