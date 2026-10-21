"""Fixed-source target-seed summary and zero-update patient influence diagnostics."""
import json
import os
from pathlib import Path
import numpy as np
import torch

ARMS = ('B00','B10','B01','B11','U10','U11')
DOMAINS = ('Siemens','Philips')
CONTRASTS = {
    'B11-B10': {'B11':1,'B10':-1},
    'B11-B01': {'B11':1,'B01':-1},
    'B10-B00': {'B10':1,'B00':-1},
    'B01-B00': {'B01':1,'B00':-1},
    'interaction': {'B11':1,'B10':-1,'B01':-1,'B00':1},
    'U11-U10': {'U11':1,'U10':-1},
    'B11-U11': {'B11':1,'U11':-1},
}


def patient_diagnostic(delta):
    n=len(delta);rng=np.random.default_rng(163)
    boot=delta[rng.integers(0,n,(10000,n))].mean(1)
    loo=(delta.sum()-delta)/(n-1)
    return dict(mean_pp=float(delta.mean()),median_pp=float(np.median(delta)),improved_fraction=float((delta>0).mean()),
                worst_quartile_mean_pp=float(np.sort(delta)[:int(np.ceil(n/4))].mean()),
                leave_one_out_mean_range_pp=[float(loo.min()),float(loo.max())],
                descriptive_patient_CI95_pp=np.quantile(boot,[.025,.975]).tolist(),patient_count=n)


def run(old,rep):
    source={f:json.loads((old/('source_'+f)/'METRICS.json').read_text()) for f in ('B','U')}
    source_pat={f:torch.load(old/('source_'+f)/'PRIVATE_PATIENT_METRICS.pt',map_location='cpu',weights_only=False) for f in ('B','U')}
    result=dict(status='PARTIAL',source_metrics=source,seeds={},pooled_effects={},caveat='Fixed source checkpoints; target seed variation only. Patient identities never exported. Patient CIs are descriptive, not seed uncertainty.')
    for seed in (163,164,165):
        metrics={};pat={};endpoints={}
        for arm in ARMS:
            folder=old/arm if seed==163 else rep/(arm+'_s'+str(seed))
            if not (folder/'EXIT.json').exists():continue
            assert json.loads((folder/'EXIT.json').read_text())['exit_code']==0, folder.name
            assert json.loads((folder/'RESULTS.json').read_text())['updates']==5000
            metrics[arm]=json.loads((folder/'METRICS.json').read_text())
            pat[arm]=torch.load(folder/'PRIVATE_PATIENT_METRICS.pt',map_location='cpu',weights_only=False)
            o,n=(metrics[arm][d]['mean_Dice']*100 for d in DOMAINS)
            endpoints[arm]=dict(old_Dice_percent=o,new_Dice_percent=n,average_Dice_percent=(o+n)/2,
                  forgetting_pp=source[arm[0]]['Siemens']['mean_Dice']*100-o,
                  arrival_gain_pp=n-source[arm[0]]['Philips']['mean_Dice']*100)
        effects={};diagnostics={}
        def compare(weights,domain):
            vec=[];ids=None
            for name,weight in weights.items():
                data=source_pat[name[-1]] if name.startswith('source_') else pat[name]
                values={v['patient']:v['Dice'] for v in data[domain]['patients']}
                current=sorted(values)
                if ids is not None:assert current==ids
                ids=current;vec.append(weight*np.array([values[k] for k in ids])*100)
            return patient_diagnostic(sum(vec))
        for name,weights in CONTRASTS.items():
            if all(a in pat for a in weights):effects[name]={d:compare(weights,d) for d in DOMAINS}
        if 'B11' in pat:diagnostics['B11-source_B']={d:compare({'B11':1,'source_B':-1},d) for d in DOMAINS}
        result['seeds'][str(seed)]=dict(complete=len(endpoints)==6,endpoints=endpoints,metrics=metrics,effects=effects,patient_diagnostics=diagnostics)
    if all(v['complete'] for v in result['seeds'].values()):
        result['status']='COMPLETE'
        for c in CONTRASTS:
            result['pooled_effects'][c]={}
            for d in DOMAINS:
                v=[result['seeds'][str(seed)]['effects'][c][d]['mean_pp'] for seed in (163,164,165)]
                result['pooled_effects'][c][d]=dict(per_seed_pp=v,mean_pp=float(np.mean(v)),sample_sd_pp=float(np.std(v,ddof=1)),positive_seeds=sum(x>0 for x in v))
        result['mean_endpoints']={}
        for a in ARMS:
            result['mean_endpoints'][a]={k:dict(mean=float(np.mean([result['seeds'][str(seed)]['endpoints'][a][k] for seed in (163,164,165)])),sample_sd=float(np.std([result['seeds'][str(seed)]['endpoints'][a][k] for seed in (163,164,165)],ddof=1))) for k in result['seeds']['163']['endpoints'][a]}
    return result


if __name__=='__main__':
    old=Path(os.environ['SOURCE_RUN']);rep=Path(os.environ['REPLICA_RUN']);r=run(old,rep)
    (rep/'PUBLIC_REPLICATION_SUMMARY.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],endpoints={s:v['endpoints'] for s,v in r['seeds'].items()},pooled_effects=r['pooled_effects']),indent=2))
