"""Combine three independent complete prostate pipelines without patient export."""
import json,os
from pathlib import Path
import numpy as np
from sc_bayeseg.sslcl_v1.summarize_prostate import run
base=Path(os.environ['RUNS_ROOT']);root=base/'sslcl_prostate_replication_20261022'
seeds={str(s):run(base/'sslcl_prostate_pilot_20261022' if s==163 else root,'' if s==163 else f'_s{s}') for s in (163,164,165)}
effects={}
for c in seeds['163']['effects']:
    effects[c]={}
    for d in ('RUNMC','BMC'):
        v=[seeds[str(s)]['effects'][c][d]['mean_pp'] for s in (163,164,165)]
        effects[c][d]=dict(per_seed_pp=v,mean_pp=float(np.mean(v)),sample_sd_pp=float(np.std(v,ddof=1)))
r=dict(status='COMPLETE',seeds=seeds,pooled_effects=effects,runtime=json.loads((root/'CONTROLLER.json').read_text()),caveat='163 exploratory;164/165 subsequent full-pipeline confirmation. Same four validation patients per domain, not twelve independent patients. Patient intervals are descriptive.')
(root/'PUBLIC_PROSTATE_REPLICATION.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(endpoints={s:v['endpoints'] for s,v in seeds.items()},effects=effects),indent=2))
