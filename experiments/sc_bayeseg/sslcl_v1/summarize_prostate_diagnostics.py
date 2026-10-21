"""Fixed trajectory evaluation aggregates; never select a replacement endpoint."""
import json,os
from pathlib import Path
base=Path(os.environ['RUNS_ROOT']);rep=base/'sslcl_prostate_replication_20261022';root=rep/'diagnostics';rows={}
for seed in (163,164,165):
    rows[str(seed)]={}
    for arm in ('source_B','source_U','U10','U11'):
        rows[str(seed)][arm]={}
        for step in (1000,3000,5000):
            folder=(root/f'{arm}_s{seed}_S{step:04}') if step!=5000 else (base/'sslcl_prostate_pilot_20261022'/arm if seed==163 else rep/f'{arm}_s{seed}')
            assert json.loads((folder/'EXIT.json').read_text())['exit_code']==0
            metrics=json.loads((folder/'METRICS.json').read_text());rows[str(seed)][arm][str(step)]=metrics
r=dict(status='COMPLETE',optimizer_updates=0,evaluations=24,trajectories=rows,runtime=json.loads((root/'CONTROLLER.json').read_text()),caveat='All frozen5000 endpoints retained;4 validation patients per domain; no checkpoint selection.')
(root/'PUBLIC_DIAGNOSTIC_RESULTS.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
for seed,s in rows.items():
 for arm,steps in s.items():print(seed,arm,{k:{d:round(100*v['mean_Dice'],3) for d,v in m.items()} for k,m in steps.items()})
