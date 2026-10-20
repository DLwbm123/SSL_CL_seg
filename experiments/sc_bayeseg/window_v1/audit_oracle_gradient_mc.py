"""Audit crossed-grid projections and descriptive variance decomposition."""
import json
import os
from pathlib import Path
import torch
from sc_bayeseg.window_v1.oracle_gradient_mc import decompose
from sc_bayeseg.window_v1.r_start import write


def audit(s):
    root=Path(s['output']);read=lambda p:json.loads(p.read_text());r=read(root/'RESULTS.json')
    assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE'
    assert read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    assert r['costs']==dict(reverse=128,optimizer=0,parameter_applications=0,generator_forward=128,generator_image_forward=1024,theta=16)
    ledger=[json.loads(x) for x in (root/'reverse.jsonl').read_text().splitlines()]
    assert [(v['arm'],v['theta'],v['local']) for v in ledger]==[(a,t,l) for a in ('ZERO','R1_WARM') for t in range(8) for l in range(8)]
    assert [v['index'] for v in ledger]==list(range(1,129))
    p=torch.load(root/'SAMPLES.pt',map_location='cpu',weights_only=False)
    errors=[]
    for arm in ('ZERO','R1_WARM'):
        z=p[arm];gd=z['data_gradients'];assert gd.shape[:2]==(8,8) and torch.isfinite(gd).all()
        grid=-(gd.double()+z['KL'])@z['reference']
        target=torch.tensor(r['arms'][arm]['projection_grid'],dtype=torch.float64)
        errors.append(float((grid-target).abs().max()));assert errors[-1]<1e-12
        assert decompose(target.tolist())==r['arms'][arm]['statistics']
    gate=all(v['statistics']['ratio_4x1_to_2x2']<=.8 for v in r['arms'].values())
    assert gate==r['allocation_control_gate']
    assert r['source_frozen'] and r['raw_unchanged'] and r['original_elapsed_seconds']<86400
    out=dict(status='PASS',ordered_reverse_ledger=128,projection_cells_checked=128,
      maximum_projection_error=max(errors),ANOVA_recomputed=True,allocation_gate_recomputed=gate,
      optimizer=0,parameter_applications=0,source_freeze_worker_checks=True)
    write(root/'AUDIT.json',out);print(json.dumps(out))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
