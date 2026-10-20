"""Independent changed-risk oracle audit with matched completed controls."""
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import metrics,local_kl,write
from sc_bayeseg.window_v1.same_family_oracle import ARMS,posterior
from sc_bayeseg.window_v1.fullview import flip


def audit(s):
    torch.set_num_threads(4);root=Path(s['output']);old=Path(s['oracle_source'])
    read=lambda p:json.loads(p.read_text())
    pt=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    r=read(root/'RESULTS.json')
    assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE'
    assert read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    expected=dict(local_optimizer=1000,local_reverse=1000,global_optimizer=0,train_theta=2000,train_g_forward=4000,train_g_image_forward=32000,evaluation_theta=64,evaluation_g_forward=512,evaluation_g_image_forward=4096,reused_baseline_records=2)
    assert r['costs']==expected
    ledger=[json.loads(v) for v in (root/'physical_updates.jsonl').read_text().splitlines()]
    steps=[json.loads(v) for v in (root/'steps.jsonl').read_text().splitlines()]
    pairs=[(a,m,k) for a in ARMS for m in ('CLASS_BALANCED',) for k in range(1,501)]
    assert [(v['arm'],v['mode'],v['step']) for v in ledger]==pairs==[(v['arm'],v['mode'],v['step']) for v in steps]
    assert [v['index'] for v in ledger]==list(range(1,1001))
    assert all(v['source_step']==1000+v['step'] for v in ledger)
    sel=read(Path(s['origin'])/'PRIVATE_SELECTION.json')
    ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
    lb=collate_labeled([ds[i] for i in sel['train_indices']])
    y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long();y=torch.cat([flip(y,k) for k in range(4)])
    counts=torch.stack([(y==k).sum((1,2)) for k in range(4)],1);assert (counts>0).all()
    assert counts.tolist()==read(root/'DESIGN.json')['class_counts']
    weights=counts.float().reciprocal().gather(1,y.flatten(1)).reshape_as(y)/4
    controls=read(Path(s['allocation_source'])/'RESULTS.json')['arms']
    error=[0.]
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
    for arm in ARMS:
        origin=pt(old/arm/'S1000.pt')
        for mode in ('CLASS_BALANCED',):
            out=root/(arm+'_'+mode)
            assert read(out/'EXIT.json')['exit_code']==0
            for k in range(0,501,100):
                state=pt(out/f'S{k:04}.pt')
                assert state['step']==k and state['source_step']==1000+k
                assert state['source_generator_cursor']==8675 and 'rng' in state
                assert all(torch.isfinite(v).all() for v in state['raw'])
                assert all(int(v['step'])==1000+k for v in state['optimizer']['state'].values())
                if k==0:
                    assert same(state['raw'],origin['raw']) and same(state['optimizer'],origin['optimizer']) and same(state['rng'],origin['rng'])
            assert r['arms'][arm+'_'+mode]['matched_control']==controls[arm+'_T2L2']
            matched=pt(Path(s['allocation_source'])/(arm+'_T2L2')/'S0000.pt')
            assert same(matched['raw'],origin['raw']) and same(matched['optimizer'],origin['optimizer']) and same(matched['rng'],origin['rng'])
            final=r['arms'][arm+'_'+mode]['final'];p=pt(out/'PRED_0500.pt')
            flows=p['generator_flows'];assert len(flows)==4
            compare(final['oracle_metrics'],metrics(torch.stack(flows).mean(0),y))
            compare(final['flow_metrics'],[metrics(v,y) for v in flows])
            ex=[float(-v.gather(1,y[:,None]).mean()) for v in p['mean_logprob_flows']]
            compare(ex,final['expected_category_NLL_flows'])
            balanced=[float((-v.gather(1,y[:,None])[:,0]*weights).sum((1,2)).mean()) for v in p['mean_logprob_flows']]
            compare(balanced,final['expected_class_balanced_NLL_flows'])
            assert abs(sum(balanced)/4+final['KLc_per_pixel']+final['KLR_per_pixel']-final['class_balanced_A'])<2e-6
            q=posterior(pt(out/'S0500.pt')['raw']);kc,kr=local_kl(q)
            assert abs(float(kc.mean()/16384)-final['KLc_per_pixel'])<2e-6
            assert abs(float(kr.mean()/16384)-final['KLR_per_pixel'])<2e-6
            assert abs(sum(ex)/4+final['KLc_per_pixel']+final['KLR_per_pixel']-final['semantic_A'])<2e-6
            assert final['robust_qualified24']==(final['oracle_metrics']['qualified_24'] and all(v['qualified_24'] for v in final['flow_metrics']))
    assert r['any_robust_oracle_qualified']==any(v['final']['robust_qualified24'] for v in r['arms'].values())
    assert r['source_frozen'] and r['source_Adam_not_aliased'] and r['original_elapsed_seconds']<86400
    out=dict(status='PASS',optimizer=1000,reverse=1000,complete_ledger=True,full_raw_Adam_RNG_checkpoints=12,
      source_and_matched_control_raw_Adam_RNG_exact=True,Adam_step_range=[1000,1500],sealed_predictions_recomputed=2,
      all_four_flow_metrics_checked=True,maximum_metric_error=error[0],KL_and_original_A_and_class_balanced_A_closed=True,
      qualification_gate_checked=True,source_freeze_worker_checks=True)
    write(root/'AUDIT.json',out);print(json.dumps(out))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
