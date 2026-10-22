"""Pair all existing case keys privately; export aggregate diagnostics only."""
import json
from pathlib import Path
import numpy as np
import torch


def run(s):
    out=Path(s['output']);out.mkdir(exist_ok=False)
    public={};private={}
    for seed in (163,164,165):
        data={}
        for arm in ('source_U','F0','FG','FH','HL','HU'):
            root=s['placement'] if arm in ('HL','HU') else s['baseline']
            obj=torch.load(Path(root)/f'{arm}_s{seed}'/'PRIVATE_PATIENT_METRICS.pt',map_location='cpu',weights_only=False)
            data[arm]={d:{p['patient']:p for p in v['patients']} for d,v in obj.items()}
        public[str(seed)]={};private[str(seed)]={}
        for domain,n in [('REFUGE',100),('RIM_ONE_r3',40)]:
            keys=sorted(data['F0'][domain]);assert len(keys)==n
            assert all(set(v[domain])==set(keys) for v in data.values())
            metrics=['mean_Dice','cup_Dice','disc_Dice']
            arr={a:{k:np.array([v[domain][i][k] if k=='mean_Dice' else v[domain][i][k.split('_')[0]]['Dice'] for i in keys]) for k in metrics} for a,v in data.items()}
            rows={}
            for base in ('source_U','F0'):
                rows[base]={}
                for a in data:
                    if a==base:continue
                    rows[base][a]={}
                    for k in metrics:
                        delta=100*(arr[a][k]-arr[base][k])
                        rows[base][a][k]=dict(mean_delta_pp=float(delta.mean()),improved=int((delta>0).sum()),drop_over_2pp=int((delta < -2).sum()),drop_over_5pp=int((delta < -5).sum()),paired_delta_q10_pp=float(np.quantile(delta,.1)),difference_of_q10_pp=float(100*(np.quantile(arr[a][k],.1)-np.quantile(arr[base][k],.1))),worst_delta_pp=float(delta.min()))
            count=int(np.ceil(.1*n));low={a:set(np.argsort(v['mean_Dice'],kind='stable')[:count].tolist()) for a,v in arr.items()}
            public[str(seed)][domain]=dict(keys=n,comparisons=rows,bottom_decile_count=count,FG_FH_bottom_overlap=len(low['FG']&low['FH']),F0_FH_bottom_overlap=len(low['F0']&low['FH']))
            private[str(seed)][domain]=dict(keys=keys,values={a:{k:v.tolist() for k,v in vals.items()} for a,vals in arr.items()},bottom_indices={a:sorted(v) for a,v in low.items()})
    torch.save(private,out/'PRIVATE_CASE_PAIRS.pt')
    result=dict(status='PASS',model_forwards=0,optimizer_updates=0,all_keys_paired=True,domains=public)
    (out/'CASE_PAIRING.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    import os
    run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
