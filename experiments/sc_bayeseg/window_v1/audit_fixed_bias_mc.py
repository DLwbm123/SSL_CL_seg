"""Independent sealed-prediction recomputation, no new generator draws."""
import json
import os
from pathlib import Path
import math
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from sc_bayeseg.window_v1.r_start import metrics,write
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.fixed_bias_mc import SEEDS,BIASES


def audit(s):
    torch.set_num_threads(4);root=Path(s['output']);read=lambda p:json.loads(p.read_text())
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE'
    assert read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    assert r['costs']==dict(generator_forward=4096,generator_image_forward=32768,theta=512,reverse=0,optimizer=0,parameter_applications=0)
    sel=read(Path(s['origin'])/'PRIVATE_SELECTION.json')
    ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
    lb=collate_labeled([ds[i] for i in sel['train_indices']]);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long();y=torch.cat([flip(y,k) for k in range(4)])
    proof=read(Path(s['bias_source'])/'RESULTS.json');error=[0.]
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,z in zip(a,b):compare(x,z)
        elif isinstance(a,float):
            assert abs(a-b)<2e-6;error[0]=max(error[0],abs(a-b))
        else:assert a==b
    def calibrated(p,b):
        # Independent log-score definition of post-mixture calibration.
        lp=p.double().log();lp[:,3]+=b;return lp.softmax(1).float()
    for arm,bias in BIASES.items():
        source=next(v for v in proof['rows'] if v['candidate']==arm+'_PIXEL_CE' and v['mode']=='MIXTURE')
        interval=sorted(source['feasible_intervals'],key=lambda v:(-(v[1]-v[0]),v[0]))[0]
        assert abs(sum(interval)/2-bias)<1e-14
        p=torch.load(root/(arm+'_PRED.pt'),map_location='cpu',weights_only=False)
        assert p['seeds']==list(SEEDS) and p['fixed_bias']==bias and len(p['flows'])==4
        for mode,b in (('UNCALIBRATED',0.),('FIXED_BIAS',bias)):
            row=next(v for v in r['rows'] if v['arm']==arm and v['mode']==mode)
            main=metrics(calibrated(torch.stack(p['flows']).mean(0),b),y)
            per=[metrics(calibrated(v,b),y) for v in p['flows']]
            compare(row['metrics'],main);compare(row['flow_metrics'],per)
            assert row['robust_qualified24']==(main['qualified_24'] and all(v['qualified_24'] for v in per))
    assert r['source_frozen'] and r['biases_frozen_before_new_draws'] and r['original_elapsed_seconds']<86400
    out=dict(status='PASS',sealed_prediction_sets=2,readout_rows=4,all_mixture_and_flow_metrics_recomputed=True,
      independent_logscore_calibration_checked=True,maximum_metric_error=error[0],bias_source_and_gate_checked=True,
      source_freeze_worker_checks=True,generator_draws_in_audit=0)
    write(root/'AUDIT.json',out);print(json.dumps(out))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
