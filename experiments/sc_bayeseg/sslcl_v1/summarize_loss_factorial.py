"""Fixed-JL loss factorial summary; export aggregates only."""
import json, os
from pathlib import Path
import numpy as np
import torch
from sc_bayeseg.sslcl_v1.summarize_replication import patient_diagnostic

base=Path(os.environ['RUNS_ROOT'])
old=base/'sslcl_v1_20261021'; rep=base/'sslcl_replication_20261021'
current=base/'sslcl_loss_factorial_20261022'
arms=('J0C0','J1C0','J0C1','J1C1'); domains=('Siemens','Philips')
source=json.loads((old/'source_B/METRICS.json').read_text())
result=dict(status='COMPLETE',seeds={},pooled_effects={},caveat='Fixed source, target-seed variation only; Patient intervals descriptive, not seed uncertainty.')
for seed in (164,165):
    folders={a:current/f'{a}_s{seed}' for a in arms}
    metrics={}; patients={}; endpoints={}; effects={}
    for a,p in folders.items():
        assert json.loads((p/'EXIT.json').read_text())['exit_code']==0
        assert json.loads((p/'RESULTS.json').read_text())['updates']==5000
        metrics[a]=json.loads((p/'METRICS.json').read_text())
        patients[a]=torch.load(p/'PRIVATE_PATIENT_METRICS.pt',map_location='cpu',weights_only=False)
        o,n=[metrics[a][d]['mean_Dice']*100 for d in domains]
        f=source['Siemens']['mean_Dice']*100-o
        endpoints[a]=dict(old_Dice_percent=o,new_Dice_percent=n,forgetting_pp=f,BWT_pp=-f)
    contrasts={'J1C0-J0C0':{'J1C0':1,'J0C0':-1},'J1C1-J0C1':{'J1C1':1,'J0C1':-1},'J0C1-J0C0':{'J0C1':1,'J0C0':-1},'J1C1-J1C0':{'J1C1':1,'J1C0':-1},'interaction':{'J1C1':1,'J1C0':-1,'J0C1':-1,'J0C0':1}}
    for c,weights in contrasts.items():
        effects[c]={}
        for d in domains:
            ids=None; values=[]
            for a,w in weights.items():
                x={v['patient']:v['Dice'] for v in patients[a][d]['patients']}
                if ids is None: ids=sorted(x)
                assert ids==sorted(x)
                values.append(w*np.array([100*x[k] for k in ids]))
            effects[c][d]=patient_diagnostic(sum(values))
    result['seeds'][str(seed)]=dict(metrics=metrics,endpoints=endpoints,effects=effects)
for c in effects:
    result['pooled_effects'][c]={}
    for d in domains:
        v=[result['seeds'][str(s)]['effects'][c][d]['mean_pp'] for s in (164,165)]
        result['pooled_effects'][c][d]=dict(per_seed_pp=v,mean_pp=float(np.mean(v)),sample_sd_pp=float(np.std(v,ddof=1)))
result['runtime']=json.loads((current/'CONTROLLER.json').read_text())
(current/'PUBLIC_LOSS_FACTORIAL.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({s:r['endpoints'] for s,r in result['seeds'].items()},indent=2))
