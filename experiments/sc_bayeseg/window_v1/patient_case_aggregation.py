"""One bounded diagnostic of fixed uncertainty aggregation versus predicted foreground volume."""
import os,json,time,traceback
from pathlib import Path
import torch,numpy as np
from torch.nn import functional as F
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability
from sc_bayeseg.window_v1.patient_reliability import ARMS,entropy
from sc_bayeseg.window_v1.r_start import write,save,append
from sc_bayeseg.window_v1.common import storage


def top_mean(x,fraction):
    x=np.asarray(x);n=max(1,int(np.ceil(len(x)*fraction)))
    return float(np.partition(x,len(x)-n)[-n:].mean()) if len(x) else None


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);start=time.time();read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    costs=dict(network_optimizer=0,reverse=0,encoder_images=0,G_forward=0,Theta=0,new_GT_label_reads=0,temperature_fits=0,cached_probability_restorations=0,score_map_interpolations=0,model_phase_sets=0,prior_patient_label_table_reads=0)
    try:
        write(root/'CONTEXT.json',s);storage(s,reserve=50_000_000)
        source=Path(s['reliability_source']);assert read(source/'AUDIT.json')['status']=='PASS' and read(source/'EXIT.json')['exit_code']==0
        point=Path(s['point_source']);kl=Path(s['kl_source']);diag=Path(s['diagnostic_source']);calibration=Path(s['calibration_source']);factor=Path(s['factor_source'])
        folders=dict(K1=kl/'R32_K1',D0=point/'D0',Dmu=point/'Dmu',U_D=factor/'U_D');records=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json');folds=read(diag/'PRIVATE_FOLDS.json');temps=dict(load(diag/'PRIVATE_TEMPERATURES.pt'),**load(calibration/'PRIVATE_TEMPERATURES.pt'))
        metadata={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in [source/'PUBLIC_RESULTS.json',diag/'PRIVATE_TEMPERATURES.pt',calibration/'PRIVATE_TEMPERATURES.pt']};private={}
        for arm_index,arm in enumerate(ARMS):
            private[arm]=[]
            for i,row in enumerate(records):
                assert time.time()-s['original_started_unix']<85500
                native=load(folders[arm]/'val'/f'CASE{i:03}.pt')['native_probability'];p=restore_probability(native);pred=p.argmax(1).numpy();costs['cached_probability_restorations']+=1
                artifact=load(source/arm/f'CASE{i:03}.pt');fit=temps[arm][str(folds[row['patient_id']])];assert row['patient_id'] not in fit['fit_patients'];cal=(p.double().clamp_min(1e-30).log()/fit['T']).softmax(1);assert torch.equal(cal.argmax(1),p.argmax(1))
                scores=dict(confidence=(1-p.max(1).values).numpy(),entropy=entropy(p).numpy(),calibrated_confidence=(1-cal.max(1).values).numpy(),calibrated_entropy=entropy(cal).numpy(),random=np.random.default_rng(920163+1000*arm_index+i).random(pred.shape))
                for name,v in artifact['native_scores'].items():scores[name]=F.interpolate(v[:,None].cuda(),(384,384),mode='bilinear',align_corners=False)[:,0].cpu().numpy();costs['score_map_interpolations']+=1
                methods={}
                for method,score in scores.items():
                    whole=top_mean(score.reshape(-1),.01);assert abs(whole-artifact['selections'][method]['case_score'])<1e-10
                    methods[method]=dict(whole_top01=whole,class_top10=[top_mean(score[pred==k],.1) for k in range(4)])
                private[arm].append(dict(patient=row['patient_id'],phase=i,foreground_fraction=float((pred>0).mean()),predicted_class_pixels=[int((pred==k).sum()) for k in range(4)],methods=methods));costs['model_phase_sets']+=1
                write(root/'STATUS.json',dict(status='AGGREGATING',arm=arm,phase=i,costs=costs))
        assert costs['model_phase_sets']==184 and costs['cached_probability_restorations']==184 and costs['score_map_interpolations']==230 and costs['new_GT_label_reads']==0
        save(root/'PRIVATE_CASE_SCORES.pt',private);write(root/'CASE_SCORES_SEALED.json',dict(time_unix=time.time(),costs=dict(costs)))
        # Only after sealing score features, join the already audited prior patient failure table.
        old=read(source/'PUBLIC_RESULTS.json');costs['prior_patient_label_table_reads']+=1
        write(root/'PRIOR_LABEL_JOIN.json',dict(time_unix=time.time(),prior_status=old['status'],source='previously audited patient failure table',no_new_raw_GT=True))
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in metadata.items())
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,source_frozen=True,features_sealed_before_prior_failure_label_join=True,test_access=False,phase_elapsed_seconds=time.time()-start,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
