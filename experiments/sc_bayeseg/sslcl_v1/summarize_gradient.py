"""Collect only formal image-only audit tasks, excluding smoke checks."""
import json,os
from pathlib import Path
root=Path(os.environ['AUDIT_ROOT']);state=json.loads((root/'CONTROLLER.json').read_text());assert state['status']=='COMPLETE' and len(state['finished'])==12
rows={}
for name,exit in state['finished'].items():
    assert exit['exit_code']==0
    rows[name]=json.loads((root/name/'METRICS.json').read_text())
    assert rows[name]['optimizer_updates']==0 and rows[name]['batches']==32
r=dict(status='COMPLETE',optimizer_updates=0,tasks=rows,runtime=state,caveat='Eval-mode local gradients, not validation accuracy or training trajectory causality. Twelve checkpoints,32 batches each; smoke excluded.')
(root/'PUBLIC_GRADIENT_RESULTS.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
for name,row in sorted(rows.items()):
 print(name,{k:dict(cos=round(v['cosine']['mean'],3),ratio=round(v['norm_ratio']['mean'],3),negative=v['negative_dot_batches']) for k,v in row['metrics'].items()})
