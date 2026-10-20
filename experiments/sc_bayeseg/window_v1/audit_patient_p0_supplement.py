"""Recompute P0 diagnostic tables from source predictions and new sealed arrays."""
import json,os
from pathlib import Path
import torch,h5py,numpy as np
from torch.nn import functional as F
from sc_bayeseg.window_v1.patient_p0_supplement import error_metrics,aggregate,posterior_stats
from sc_bayeseg.window_v1.r_start import write


def audit(s):
    torch.set_num_threads(4);root=Path(s['output']);read=lambda p:json.loads(p.read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    c=r['costs'];assert (c['optimizer'],c['reverse'],c['encoder_images'],c['G_forward'],c['Theta'],c['G_image_forward'],c['sealed_phase_reads'],c['native_label_phase_reads'],c['new_validation_forwards'])==(0,0,460,3840,960,7360,264,40,0)
    assert read(root/'NATIVE_SEALED.json')['costs']['native_label_phase_reads']==0
    train=Path(s['train_readout_source']);parent=Path(s['patient_control']);e2e=Path(s['patient_e2e_source']);err=[0.]
    specs=dict(FROZEN_DIRECT=parent/'DIRECT',FROZEN_G0=parent/'CONDITIONAL_G',E2E_DIRECT=e2e/'DIRECT_E2E',E2E_G0=e2e/'CONDITIONAL_G_E2E')
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,float):
            d=abs(a-b);assert d<1e-10;err[0]=max(err[0],d)
        else:assert a==b
    for cohort,mapname in (('train','PRIVATE_TRAIN_MAP.json'),('val','PRIVATE_VAL_MAP.json')):
        records=read(root/mapname)
        for name,folder in specs.items():
            rows=[]
            for i,row in enumerate(records):
                p=load((train/name if cohort=='train' else folder)/'predictions'/f'CASE{i:03}.pt')['probability']
                with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=f['label'][...]
                m=error_metrics(p,y);assert sum(sum(v) for v in m['confusion_GT_rows'])==np.asarray(y).size
                assert abs(sum(v['NLL_contribution'] for v in m['classes'])-m['NLL'])<1e-10
                rows.append(dict(patient=row['patient_id'],case=row['case_id'],metrics=m))
            compare(r['errors'][cohort][name],aggregate(rows));print('recomputed',cohort,name,flush=True)
    records=read(root/'PRIVATE_TRAIN_MAP.json');jmin=1.
    for name in ('FROZEN_G0','E2E_G0'):
        risks=[];posts=[]
        for i,row in enumerate(records):
            d=load(root/name/f'CASE{i:03}.pt');p=d['probability'];lp=d['meanlog'];q=d['posterior']
            assert torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6
            with h5py.File(Path(s['data'])/'h5/v1'/row['label_h5_relpath'],'r') as f:y=torch.from_numpy(f['label'][...].astype(np.int64))
            y=F.interpolate(y[:,None].float(),(128,128),mode='nearest')[:,0].long()
            ec=-lp.double().gather(1,y[:,None])[:,0];mn=-p.double().clamp_min(1e-30).log().gather(1,y[:,None])[:,0];jmin=min(jmin,float((ec-mn).min()));assert jmin>=-2e-6
            risks.append(dict(patient=row['patient_id'],case=row['case_id'],expected_CE128=float(ec.mean()),mixture_NLL128=float(mn.mean()),Jensen_gap128=float((ec-mn).mean())))
            posts.append(dict(patient=row['patient_id'],case=row['case_id'],statistics=posterior_stats(q)))
            for k,mu,ls in (('c',q[0],q[1]),('R',q[2],q[3])):
                direct=.5*(mu.double().square()+(2*ls.double()).exp()-1-2*ls.double()).flatten(1).sum(1).mean()/16384
                assert abs(float(direct)-posts[-1]['statistics'][k]['mean_KL_over_N']-posts[-1]['statistics'][k]['scale_KL_over_N'])<1e-10
        compare(r['native_risk'][name],risks);compare(r['posterior'][name],posts)
    assert r['source_frozen'] and r['RNG_restored'] and r['original_elapsed_seconds']<86400
    out=dict(status='PASS',source_prediction_tables_recomputed=264,native_phase_risks_and_posteriors_recomputed=40,confusion_counts_and_NLL_contributions_checked=True,KL_formula_direct_check=True,minimum_Jensen_gap=jmin,maximum_metric_error=err[0],new_network_forwards=0,new_parameter_updates=0,metric_formula='same diagnostic definitions, independent sealed-array recomputation')
    write(root/'AUDIT.json',out);print(json.dumps(out))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
