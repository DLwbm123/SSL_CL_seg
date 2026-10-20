"""Recompute direction diagnostics from saved vectors; no new gradients."""
import json
import os
from pathlib import Path
import torch
from sc_bayeseg.window_v1.class_kl_gradient import TRAIN,EVAL,BLOCKS,summarize
from sc_bayeseg.window_v1.r_start import write


def audit(s):
    torch.set_num_threads(4);root=Path(s['output'])
    load=lambda p:json.loads(p.read_text())
    r=load(root/'RESULTS.json')
    assert r['status']==load(root/'STATUS.json')['status']=='COMPLETE'
    assert load(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    assert r['costs']==dict(reverse=34,optimizer=0,parameter_applications=0,generator_forward=64,generator_image_forward=512,theta=32)
    ledger=[json.loads(x) for x in (root/'reverse.jsonl').read_text().splitlines()]
    expected=[]
    for arm in ('ZERO','R1_WARM'):
        expected.append((arm,None,'K'))
        for seed,es in zip(TRAIN,EVAL):
            expected.extend([(arm,seed,k) for k in ('D','Q','A')]+[(arm,es,'Q_eval')])
    assert [(v['arm'],v['seed'],v['term']) for v in ledger]==expected
    assert [v['index'] for v in ledger]==list(range(1,35))
    vectors=torch.load(root/'GRADIENTS.pt',map_location='cpu',weights_only=False)
    assert [(v['arm'],v['seed'],v['eval_seed']) for v in r['rows']]==[(a,t,e) for a in ('ZERO','R1_WARM') for t,e in zip(TRAIN,EVAL)]
    maxerr=[0.]
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        else:
            assert abs(a-b)<1e-12;maxerr[0]=max(maxerr[0],abs(a-b))
    for i,row in enumerate(r['rows']):
        arm=vectors[row['arm']];g=arm['flows'][i%4]
        values=(g['D'],arm['K'],g['A'],g['Q'],g['Q_eval'])
        assert all(torch.isfinite(v).all() for v in values)
        assert torch.allclose(g['A'],g['D']+arm['K'],atol=2e-9,rtol=2e-5)
        compare(row['summary'],summarize(*values))
        sizes=[int(torch.tensor(v).prod()) for v in arm['raw_shapes']];offset=0
        assert sum(sizes)==len(arm['K'])
        for name,n in zip(BLOCKS,sizes):
            compare(row['blocks'][name],summarize(*(v[offset:offset+n] for v in values)));offset+=n
        dots=row['summary']['independent_unnormalized']
        assert abs(dots['A']-dots['D']-dots['K'])<1e-9
    assert max(r['closed_form_KL_max_errors'])<1e-9
    assert r['source_frozen'] and r['raw_unchanged'] and r['original_elapsed_seconds']<86400
    out=dict(status='PASS',ordered_reverse_ledger=34,optimizer=0,parameter_applications=0,
      gradient_records=8,all_norms_dots_and_blocks_recomputed=True,maximum_scalar_error=maxerr[0],
      A_gradient_sum_closed=True,independent_direction_additivity_checked=True,
      analytic_KL_worker_checks=True,source_freeze_worker_checks=True)
    write(root/'AUDIT.json',out);print(json.dumps(out))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
