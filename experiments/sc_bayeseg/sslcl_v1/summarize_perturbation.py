"""Collect only formal image-only audit tasks, excluding smoke checks."""
import json,os
from pathlib import Path
root=Path(os.environ['AUDIT_ROOT']);state=json.loads((root/'CONTROLLER.json').read_text());assert state['status']=='COMPLETE' and len(state['finished'])==16
rows={}
for name,exit in state['finished'].items():
    assert exit['exit_code']==0
    rows[name]=json.loads((root/name/'METRICS.json').read_text())
    assert rows[name]['optimizer_updates']==0 and rows[name]['patients']==14
r=dict(status='COMPLETE',optimizer_updates=0,tasks=rows,runtime=state,caveat='Image-only sensitivity, not accuracy; fixed14 target training U patients and42 selected slices; paired latent draws. Smoke checks excluded.')
(root/'PUBLIC_PERTURBATION_RESULTS.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
for name,row in sorted(rows.items()):
 print(name,{mode:{k:round(v['mean'],6) for k,v in values.items()} for mode,values in row['metrics'].items()})
