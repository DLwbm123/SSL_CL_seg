"""Recompute saved MC patient metrics and class-risk additivity without network calls."""
import json,os
from pathlib import Path
import torch,numpy as np,h5py
from sc_bayeseg.window_v1.patient_validation import phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability,paired_summary
from sc_bayeseg.window_v1.patient_r32_mc import class_nll
from sc_bayeseg.window_v1.r_start import write


def audit(s):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    assert r['costs']==dict(optimizer=0,reverse=0,encoder_images=4352,theta=8896,g_forward=35584,g_image_forward=69632,label_phases=368,reconstruction_phases=368)
    ledger=[json.loads(x) for x in (root/'predictions.jsonl').read_text().splitlines()];assert len(ledger)==368
    assert [(x['stream'],x['arm'],x['phase']) for x in ledger]==[(f,a,i) for f in range(4) for a in ('K1','K0') for i in range(46)]
    assert all(x['costs']['optimizer']==x['costs']['reverse']==x['costs']['label_phases']==0 for x in ledger)
    seal=read(root/'PREDICTIONS_SEALED.json');assert seal['sets']==8 and seal['phases']==368 and seal['costs']['label_phases']==0
    records=read(Path(s['patient_control'])/'PRIVATE_EVALUATION_MAP.json');private=load(root/'PRIVATE_PHASE_RESULTS.pt');control=read(Path(s['factor_source'])/'RESULTS.json')['aggregates']['val']['U_D'];error=[0.]
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,float):
            e=abs(a-b);assert e<1e-10;error[0]=max(error[0],e)
        else:assert a==b
    for stream in range(4):
        key=f'FLOW{stream}';ag={}
        for arm in ('K1','K0'):
            rows=[];count=0
            for i,row in enumerate(records):
                d=load(root/key/arm/f'CASE{i:03}.pt');p=d['native_probability'];assert p.dtype==torch.float32 and p.shape[1:]==(4,128,128) and torch.isfinite(p).all();count+=len(p)
                p=restore_probability(p)
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                cells=class_nll(p,y);metrics=phase_metrics(p,y,d['spacing']);assert abs(sum(c['full_image_contribution'] for c in cells)-metrics['NLL'])<1e-7
                rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=metrics,class_NLL=cells))
            assert count==544;compare(rows,private[key][arm]);ag[arm]=aggregate(rows);compare(ag[arm],r['aggregates'][key][arm]);compare(paired_summary(ag[arm],control),r['contrasts'][key][arm+'_minus_U_D'])
            print('recomputed',key,arm,flush=True)
        compare(paired_summary(ag['K1'],ag['K0']),r['contrasts'][key]['K1_minus_K0'])
    assert r['source_frozen'] and r['all_predictions_before_labels'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    a=dict(status='PASS',prediction_ledger=368,native_sets=8,recomputed_phases=368,maximum_metric_error=error[0],class_NLL_additivity_checked=True,complete_patient_metrics_and_contrasts=True,source_freeze_and_zero_updates_asserted_by_worker=True,new_optimizer=0,new_reverse=0,new_network_forward=0,audit_reconstruction_phases=368)
    write(root/'AUDIT.json',a);print(json.dumps(a))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
