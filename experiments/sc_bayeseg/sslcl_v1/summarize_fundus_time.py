"""Aggregate bounded trajectories and fixed endpoint-selected cohorts without inference."""
import json,os
from pathlib import Path
from statistics import mean
import torch


def run(s):
    root=Path(s['root']);baseline=Path(s['baseline']);public={};total=0
    for seed in (163,164,165):
        r=json.loads((root/f'T_s{seed}'/'RESULTS.json').read_text());assert r['status']=='COMPLETE' and r['image_forwards']==840 and r['optimizer_updates']==0
        total+=r['image_forwards'];public[str(seed)]=r['metrics']
    assert total==2520
    endpoint={a:{str(seed):json.loads((baseline/f'{a}_s{seed}'/'RESULTS.json').read_text())['metrics'] for seed in (163,164,165)} for a in ('source_U','F0','FG','FH')}
    def metrics(arm,seed,step):return endpoint['source_U'][seed] if step==0 else endpoint[arm][seed] if step==5000 else public[seed][f'{arm}_S{step:04}']
    means={}
    for arm in ('F0','FG','FH'):
        means[arm]={}
        for step in (0,1000,3000,5000):
            means[arm][str(step)]={}
            for d in ('REFUGE','RIM_ONE_r3'):
                rows=[metrics(arm,str(seed),step)[d] for seed in (163,164,165)]
                v={k:mean(r[k] for r in rows) for k in ('mean_Dice','lower_decile_Dice','NLL','Brier')}
                for c in ('cup','disc'):v[c]={k:mean(r[c][k] for r in rows) for k in ('Dice','HD95_penalized_pixels','empty_prediction_failures')}
                means[arm][str(step)][d]=v
    # Full key alignment, then fixed seed165 endpoint-low cohorts; no image access.
    pairing=torch.load(Path(s['pairing'])/'PRIVATE_CASE_PAIRS.pt',map_location='cpu',weights_only=False)
    cohorts={};seed='165'
    for d in ('REFUGE','RIM_ONE_r3'):
        keys=pairing[seed][d]['keys'];cohorts[d]={}
        for selected_by in ('FG','FH'):
            selected=[keys[i] for i in pairing[seed][d]['bottom_indices'][selected_by]];values={}
            for arm in ('F0','FG','FH'):
                values[arm]={}
                for step in (0,1000,3000,5000):
                    file=(baseline/f'source_U_s{seed}'/'PRIVATE_PATIENT_METRICS.pt') if step==0 else (baseline/f'{arm}_s{seed}'/'PRIVATE_PATIENT_METRICS.pt') if step==5000 else root/f'T_s{seed}'/f'{arm}_S{step:04}'/'PRIVATE_PATIENT_METRICS.pt'
                    patients=torch.load(file,map_location='cpu',weights_only=False)[d]['patients'];lookup={v['patient']:v for v in patients};assert set(lookup)==set(keys)
                    values[arm][str(step)]=dict(mean_Dice=mean(lookup[k]['mean_Dice'] for k in selected),cup_Dice=mean(lookup[k]['cup']['Dice'] for k in selected),disc_Dice=mean(lookup[k]['disc']['Dice'] for k in selected))
            cohorts[d][selected_by]=dict(count=len(selected),selection='seed165 endpoint lowest ceil(0.1*N) main Dice; post-hoc descriptive cohort',values=values)
    out=dict(status='COMPLETE',image_forwards=total,optimizer_updates=0,raw_intermediate_metrics=public,seed_mean_trajectories=means,seed165_endpoint_low_cohort_trajectories=cohorts)
    (root/'TIME_RESULTS.json').write_text(json.dumps(out,indent=2));return out

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
