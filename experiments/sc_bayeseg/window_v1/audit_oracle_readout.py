"""Recompute sealed four-cell metrics without generator evaluation."""
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset, collate_labeled
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import metrics, write
from sc_bayeseg.window_v1.fullview import flip


def audit(s):
    torch.set_num_threads(4);root=Path(s['output'])
    read=lambda p:json.loads(p.read_text())
    pt=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    r=read(root/'RESULTS.json')
    assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE'
    assert read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    assert r['costs']==dict(optimizer=0,reverse=0,generator_forward=578,generator_image_forward=4624,theta=64,reused_predictions=2)
    sel=read(Path(s['origin'])/'PRIVATE_SELECTION.json')
    ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
    lb=collate_labeled([ds[i] for i in sel['train_indices']])
    y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
    y=torch.cat([flip(y,k) for k in range(4)])
    error=[0.]
    def check(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:check(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,z in zip(a,b):check(x,z)
        elif isinstance(a,float):
            error[0]=max(error[0],abs(a-b));assert abs(a-b)<2e-6
        else:assert a==b
    assert [(z['arm'],z['mode']) for z in r['rows']]==[(a,m) for a in ('ZERO','R1_WARM') for m in ('MM','SM','MS','SS')]
    for row in r['rows']:
        p=pt(root/f"{row['arm']}_{row['mode']}.pt")['flows']
        assert len(p)==(1 if row['mode']=='MM' else 4)
        assert all(v.shape==(8,4,128,128) and torch.isfinite(v).all() for v in p)
        assert max(float((v.sum(1)-1).abs().max()) for v in p)<2e-6
        main=metrics(torch.stack(p).mean(0),y)
        check(row['metrics'],main);check(row['flow_metrics'],[metrics(v,y) for v in p])
        d=[z for v in main['per_view_image'] for z in v['dice'][1:]]
        assert row['passed24']==sum(v>=.8 for v in d) and row['minimum_foreground_dice']==min(d)
        if row['mode']=='SS':
            assert same(p,pt(Path(s['oracle_source'])/row['arm']/'PRED_1000.pt')['generator_flows'])
    assert r['source_frozen'] and r['original_elapsed_seconds']<86400
    out=dict(status='PASS',sealed_prediction_sets=8,all_main_and_flow_metrics_recomputed=True,
      reused_SS_predictions_exact=2,maximum_metric_error=error[0],optimizer=0,reverse=0,
      qualification_counts_checked=True,source_freeze_in_worker_assertions=True)
    write(root/'AUDIT.json',out);print(json.dumps(out))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
