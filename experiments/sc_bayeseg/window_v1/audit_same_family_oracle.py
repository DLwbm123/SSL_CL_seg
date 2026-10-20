"""Independent ledger, checkpoint and sealed-prediction audit; no training."""
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset, collate_labeled
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import metrics, write, local_kl
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.same_family_oracle import posterior, ARMS


def audit(s):
    torch.set_num_threads(4)
    root=Path(s['output'])
    read=lambda name:json.loads((root/name).read_text())
    pt=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    result=read('RESULTS.json')
    assert result['status']=='COMPLETE' and read('STATUS.json')['status']=='COMPLETE'
    assert read('EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    expected=dict(local_optimizer=2000,local_reverse=2000,global_optimizer=0,train_theta=4000,
      train_g_forward=8000,train_g_image_forward=64000,evaluation_theta=224,
      evaluation_g_forward=1792,evaluation_g_image_forward=14336,
      initialization_encoder_image_forward=8,initialization_posterior_batch_forward=1,reused_prediction_records=1)
    assert result['costs']==expected
    ledger=[json.loads(x) for x in (root/'physical_updates.jsonl').read_text().splitlines()]
    steps=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()]
    pairs=[(a,k) for a in ARMS for k in range(1,1001)]
    assert [(r['arm'],r['step']) for r in ledger]==pairs==[(r['arm'],r['step']) for r in steps]
    assert [r['index'] for r in ledger]==list(range(1,2001))
    assert all(torch.isfinite(torch.tensor(r['loss'])) for r in steps)
    initial=pt(root/'INITIAL_RAW.pt')
    assert all(torch.count_nonzero(v)==0 for v in initial['ZERO'])
    for arm in ARMS:
        assert json.loads((root/arm/'EXIT.json').read_text())['exit_code']==0
        for step in range(0,1001,100):
            state=pt(root/arm/f'S{step:04}.pt')
            assert state['step']==step and state['arm']==arm and 'rng' in state
            assert state['source_generator_cursor']==8675 and len(state['raw'])==4
            assert [tuple(v.shape) for v in state['raw']]==[(8,16),(8,16),(8,16,16,16),(8,16,16,16)]
            assert all(torch.isfinite(v).all() for v in state['raw'])
            if step==0:
                assert same(state['raw'],initial[arm]) and not state['optimizer']['state']
            else:
                assert len(state['optimizer']['state'])==4
                assert all(int(v['step'])==step for v in state['optimizer']['state'].values())
    sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
    ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
    lb=collate_labeled([ds[i] for i in sel['train_indices']])
    y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
    y=torch.cat([flip(y,k) for k in range(4)])
    rows=[json.loads(x) for x in (root/'metrics.jsonl').read_text().splitlines()]
    assert [(r['arm'],r['step']) for r in rows]==[(a,k) for a in ARMS for k in (0,250,500,1000)]
    error=[0.]
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,z in zip(a,b):compare(x,z)
        elif isinstance(a,float):
            error[0]=max(error[0],abs(a-b));assert abs(a-b)<2e-6
        else:assert a==b
    for row in rows:
        reused=row['arm']=='R1_WARM' and row['step']==0
        path=Path(s['r_start'])/'R1/PRED_0600.pt' if reused else root/row['arm']/f"PRED_{row['step']:04}.pt"
        pred=pt(path);p=torch.stack(pred['generator_flows'])
        assert p.shape==(4,8,4,128,128) and torch.isfinite(p).all()
        assert torch.max(torch.abs(p.sum(2)-1))<2e-6
        compare(row['oracle_metrics'],metrics(p.mean(0),y))
        compare(row['flow_metrics'],[metrics(v,y) for v in p])
        assert row['robust_qualified24']==(row['oracle_metrics']['qualified_24'] and all(v['qualified_24'] for v in row['flow_metrics']))
        if reused:
            assert row['semantic_A'] is None and row['expected_category_NLL_flows'] is None
        else:
            ex=[float(-v.gather(1,y[:,None]).mean()) for v in pred['mean_logprob_flows']]
            compare(row['expected_category_NLL_flows'],ex)
            assert abs(row['semantic_A']-(sum(ex)/4+row['KLc_per_pixel']+row['KLR_per_pixel']))<2e-6
        if row['step'] in (0,500,1000):
            q=posterior(pt(root/row['arm']/f"S{row['step']:04}.pt")['raw'])
            kc,kr=local_kl(q)
            assert abs(float(kc.mean()/16384)-row['KLc_per_pixel'])<2e-6
            assert abs(float(kr.mean()/16384)-row['KLR_per_pixel'])<2e-6
    assert result['any_robust_oracle_qualified']==any(r['robust_qualified24'] for r in rows if r['step']==1000)
    assert result['source_generator_frozen'] and result['old_history_unchanged']
    assert result['original_elapsed_seconds']<86400
    assert s['original_started_unix']==json.loads((Path(s['r_start'])/'CONTEXT.json').read_text())['original_started_unix']
    out=dict(status='PASS',local_optimizer=2000,local_reverse=2000,global_optimizer=0,
      ordered_complete_ledger=True,full_raw_Adam_RNG_checkpoints=22,sealed_new_predictions=7,
      reused_predictions=1,all_eight_metrics_recomputed=True,maximum_metric_error=error[0],
      semantic_A_closure=True,checkpoint_KL_checks=6,original_clock_preserved=True,
      qualification_gate_recomputed=True,source_freeze_assertions_present=True,
      limitation='Freeze relies on in-worker state comparisons; audit performs no generator or encoder forwards.')
    write(root/'AUDIT.json',out);print(json.dumps(out))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
