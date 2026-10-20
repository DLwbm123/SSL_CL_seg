"""Independent saved-native metrics audit for frozen global-mean readout."""
import json,os
from pathlib import Path
import torch,numpy as np,h5py
from sc_bayeseg.window_v1.patient_validation import phase_metrics,aggregate
from sc_bayeseg.window_v1.patient_train_readout import aggregate_training
from sc_bayeseg.window_v1.patient_loss_factor import restore_probability,paired_summary
from sc_bayeseg.window_v1.r_start import write
from sc_bayeseg.window_v1.summarize_patient_point_estimate import paired

def audit(s):
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    assert r['costs']==dict(optimizer=0,reverse=0,encoder_images=774,G_forward=398,G_image_forward=774,Theta_draws=0,local_draws=0,label_phase_reads=66,reconstruction_phases=66)
    assert read(root/'PREDICTIONS_SEALED.json')['costs']['label_phase_reads']==0
    private=load(root/'PRIVATE_PHASE_RESULTS.pt');prior=read(Path(s['point_source'])/'RESULTS.json')['aggregates'];matched=read(Path(s['factor_source'])/'RESULTS.json')['aggregates'];err=[0.]
    def eq(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:eq(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):eq(x,y)
        elif isinstance(a,float):
            e=abs(a-b);assert e<1e-10;err[0]=max(err[0],e)
        else:assert a==b
    risk={}
    for cohort,source in (('train',s['train_readout_source']),('val',s['patient_control'])):
        records=read(Path(source)/'PRIVATE_EVALUATION_MAP.json');rows=[];slices=0
        for i,row in enumerate(records):
            d=load(root/cohort/f'CASE{i:03}.pt');p=d['native_probability'];assert p.dtype==torch.float32 and p.shape[1:]==(4,128,128) and torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6;slices+=len(p);p=restore_probability(p)
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
            rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=phase_metrics(p,y,d['spacing'])))
        assert slices==(230 if cohort=='train' else 544);eq(rows,private[cohort]);a=(aggregate_training if cohort=='train' else aggregate)(rows);eq(a,r['aggregates'][cohort])
        risk[cohort]={}
        for label,b in (('sampled_Dmu',prior[cohort]['Dmu']),('U_D',matched[cohort]['U_D'])):
            eq(paired_summary(a,b),r['contrasts'][cohort]['mean_minus_'+label]);risk[cohort]['mean_minus_'+label]={m:paired(a['patients'],b['patients'],m) for m in ('macro_Dice','NLL','Brier')}
        print('recomputed',cohort,flush=True)
    assert r['source_frozen'] and r['RNG_restored'] and r['predictions_sealed_before_labels'] and not r['test_access'] and r['original_elapsed_seconds']<86400
    result=dict(status='PASS',native_prediction_sets_recomputed=66,maximum_metric_error=err[0],all_patient_contrasts_recomputed=True,source_freeze_and_RNG_asserted_by_worker=True,audit_reconstruction_phases=66,audit_label_reads=66,new_network_forward=0,new_reverse=0,new_optimizer=0)
    write(root/'AUDIT.json',result)
    r.pop('space');r.update(status='COMPLETE_AUDIT_PASS',audit=result,probability_contrasts=risk,scope='Inference-only weight-mean ablation of fixed Dmu; repeatedly used development val; not a trained deterministic comparator')
    for cohort,a in r['aggregates'].items():
        for i,p in enumerate(a['patients']):p['patient']=f'{cohort.upper()}{i:03}'
    write(root/'PUBLIC_RESULTS.json',r);print(json.dumps(result))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
