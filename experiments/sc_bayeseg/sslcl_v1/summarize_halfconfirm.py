"""Paired half-strength contrasts, retaining all seeds and patient uncertainty."""
import json,os
from pathlib import Path
import numpy as np
import torch
from sc_bayeseg.sslcl_v1.summarize_replication import patient_diagnostic
base=Path(os.environ['RUNS_ROOT']);root=base/'sslcl_prostate_halfconfirm_20261022';seeds={}
for seed in (166,167):
    old=root;suffix=f'_s{seed}'
    metrics={};patients={};endpoints={};effects={}
    for family in ('B',):
        for arm in ('source_'+family,family+'10',family+'HALF'):
            folder=root/(arm+f'_s{seed}') if arm.endswith('HALF') else old/(arm+suffix)
            assert json.loads((folder/'EXIT.json').read_text())['exit_code']==0
            assert json.loads((folder/'RESULTS.json').read_text())['updates']==5000
            metrics[arm]=json.loads((folder/'METRICS.json').read_text())
            patients[arm]=torch.load(folder/'PRIVATE_PATIENT_METRICS.pt',map_location='cpu',weights_only=False)
            m=metrics[arm];o,n=[100*m[d]['mean_Dice'] for d in ('RUNMC','BMC')]
            endpoints[arm]=dict(old_Dice_percent=o,new_Dice_percent=n)
            if not arm.startswith('source'):
                so=metrics['source_'+family];bwt=o-100*so['RUNMC']['mean_Dice']
                endpoints[arm].update(BWT_pp=bwt,forgetting_pp=-bwt,arrival_gain_pp=n-100*so['BMC']['mean_Dice'])
        for control in (family+'10','source_'+family):
            a=family+'HALF';effects[a+'-'+control]={}
            for domain in ('RUNMC','BMC'):
                x={v['patient']:v['Dice'] for v in patients[a][domain]['patients']};y={v['patient']:v['Dice'] for v in patients[control][domain]['patients']}
                assert x.keys()==y.keys() and len(x)==4
                effects[a+'-'+control][domain]=patient_diagnostic(np.array([100*(x[k]-y[k]) for k in sorted(x)]))
    seeds[str(seed)]=dict(metrics=metrics,endpoints=endpoints,effects=effects)
pooled={c:{d:dict(per_seed_pp=(v:=[seeds[str(s)]['effects'][c][d]['mean_pp'] for s in (166,167)]),mean_pp=float(np.mean(v)),sample_sd_pp=float(np.std(v,ddof=1))) for d in ('RUNMC','BMC')} for c in seeds['166']['effects']}
r=dict(status='COMPLETE',formal_updates=30000,seeds=seeds,pooled_effects=pooled,runtime=json.loads((root/'CONTROLLER.json').read_text()),caveat='New training seeds166/167; family and alpha selected after prior validation. Same four patients per domain, not external patient confirmation. No extra seed grid.')
r['direction_gate_passed']=all(seeds[str(seed)]['effects']['BHALF-B10'][domain]['mean_pp']>0 for seed in (166,167) for domain in ('RUNMC','BMC'))
(root/'PUBLIC_HALFCONFIRM_RESULTS.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(endpoints={s:v['endpoints'] for s,v in seeds.items()},effects=pooled),indent=2))
