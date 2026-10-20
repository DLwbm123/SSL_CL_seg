"""Independent score reconstruction and paired-eligible patient aggregation audit."""
import json,os
from pathlib import Path
import torch,numpy as np
from torch.nn import functional as F
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability
from sc_bayeseg.window_v1.patient_reliability import ARMS,entropy
from sc_bayeseg.window_v1.patient_case_aggregation import top_mean
from sc_bayeseg.window_v1.summarize_patient_case_aggregation import summarize,auc
from sc_bayeseg.window_v1.r_start import write


def audit(s):
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    assert r['costs']==dict(network_optimizer=0,reverse=0,encoder_images=0,G_forward=0,Theta=0,new_GT_label_reads=0,temperature_fits=0,cached_probability_restorations=184,score_map_interpolations=230,model_phase_sets=184,prior_patient_label_table_reads=1)
    assert read(root/'CASE_SCORES_SEALED.json')['time_unix']<read(root/'PRIOR_LABEL_JOIN.json')['time_unix'] and read(root/'CASE_SCORES_SEALED.json')['costs']['prior_patient_label_table_reads']==0
    source=Path(s['reliability_source']);point=Path(s['point_source']);kl=Path(s['kl_source']);diag=Path(s['diagnostic_source']);calibration=Path(s['calibration_source']);factor=Path(s['factor_source'])
    folders=dict(K1=kl/'R32_K1',D0=point/'D0',Dmu=point/'Dmu',U_D=factor/'U_D');records=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json');folds=read(diag/'PRIVATE_FOLDS.json');temps=dict(load(diag/'PRIVATE_TEMPERATURES.pt'),**load(calibration/'PRIVATE_TEMPERATURES.pt'));private=load(root/'PRIVATE_CASE_SCORES.pt');error=[0.]
    def eq(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:eq(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):eq(x,y)
        elif isinstance(a,float):
            e=abs(a-b);assert e<1e-10;error[0]=max(error[0],e)
        else:assert a==b
    for arm_index,arm in enumerate(ARMS):
        for i,row in enumerate(records):
            native=load(folders[arm]/'val'/f'CASE{i:03}.pt')['native_probability'];p=restore_probability(native);pred=p.argmax(1).numpy();artifact=load(source/arm/f'CASE{i:03}.pt')
            fit=temps[arm][str(folds[row['patient_id']])];assert row['patient_id'] not in fit['fit_patients'];cal=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);assert torch.equal(cal.argmax(1),p.argmax(1))
            scores=dict(confidence=(1-p.max(1).values).numpy(),entropy=entropy(p).numpy(),calibrated_confidence=(1-cal.max(1).values).numpy(),calibrated_entropy=entropy(cal).numpy(),random=np.random.default_rng(920163+1000*arm_index+i).random(pred.shape))
            for name,v in artifact['native_scores'].items():scores[name]=F.interpolate(v[:,None].cuda(),(384,384),mode='bilinear',align_corners=False)[:,0].cpu().numpy()
            methods={}
            for name,score in scores.items():
                whole=top_mean(score.reshape(-1),.01);eq(whole,artifact['selections'][name]['case_score']);methods[name]=dict(whole_top01=whole,class_top10=[top_mean(score[pred==k],.1) for k in range(4)])
            eq(dict(patient=row['patient_id'],phase=i,foreground_fraction=float((pred>0).mean()),predicted_class_pixels=[int((pred==k).sum()) for k in range(4)],methods=methods),private[arm][i])
        print('recomputed',arm,flush=True)
    stats=summarize(s,require_audit=False)
    for arm,model in stats['models'].items():
        ids=sorted({x['patient'] for x in private[arm]})
        for i,p in enumerate(model['patients']):
            rows=[x for x in private[arm] if x['patient']==ids[i]]
            for name,v in p['methods'].items():
                means=[]
                for k in (1,2,3):
                    x=[row['methods'][name]['class_top10'][k] for row in rows if row['methods'][name]['class_top10'][k] is not None];means.append(sum(x)/len(x) if x else None)
                eq(sum(means)/3 if all(x is not None for x in means) else None,v['foreground_conditional_score'])
        for name,a in model['summaries'].items():
            valid=[p for p in model['patients'] if p['methods'][name]['foreground_conditional_score'] is not None]
            assert a['valid_patients']==len(valid)
            for label in ('macro_failure','RV_failure'):
                assert a[label]['eligible_positive']==sum(p[label] for p in valid)
                eq(auc([p['methods'][name]['foreground_conditional_score'] for p in valid],[p[label] for p in valid]),a[label]['foreground_conditional_AUROC'])
    assert r['source_frozen'] and r['features_sealed_before_prior_failure_label_join'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    a=dict(status='PASS',model_phase_sets=184,maximum_score_error=error[0],original_whole_score_unchanged=True,new_fixed_conditional_rule_checked=True,NA_and_common_patient_pairing_checked=True,feature_sealing_before_label_join=True,AUROC_direction_fixed=True,audit_probability_restorations=184,audit_score_map_interpolations=230,new_GT_label_reads=0,new_network_forward=0,new_reverse=0,new_optimizer=0,source_freeze_asserted_by_worker=True)
    write(root/'AUDIT.json',a);print(json.dumps(a))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
