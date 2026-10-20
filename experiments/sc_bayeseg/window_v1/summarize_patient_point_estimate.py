"""Anonymous complete point-estimate results and matched patient risk contrasts."""
import json,os
from pathlib import Path
import numpy as np
from sc_bayeseg.window_v1.r_start import write


def paired(a,b,metric):
    assert [p['patient'] for p in a]==[p['patient'] for p in b]
    d=np.array([x[metric]-y[metric] for x,y in zip(a,b)]);g=np.random.default_rng(163)
    return dict(mean_difference=float(d.mean()),patient_bootstrap95=np.quantile(d[g.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975]).tolist())


def summarize(s):
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());r=read(root/'RESULTS.json');audit=read(root/'AUDIT.json');assert audit['status']=='PASS'
    controls={c:dict(K0=read(Path(s['r32_source'])/'RESULTS.json')['aggregates'][c],K1=read(Path(s['kl_source'])/'RESULTS.json')['aggregates'][c]['R32_K1'],U_D=read(Path(s['factor_source'])/'RESULTS.json')['aggregates'][c]['U_D']) for c in ('train','val')}
    contrasts={};classes={}
    for cohort,arms in r['aggregates'].items():
        contrasts[cohort]={};classes[cohort]={}
        combined=dict(arms,**controls[cohort])
        for name,a in combined.items():
            a['mean_NLL']=float(np.mean([p['NLL'] for p in a['patients']]));a['mean_Brier']=float(np.mean([p['Brier'] for p in a['patients']]))
        for new,old in (('D0','K0'),('D0','U_D'),('Dmu','K1'),('Dmu','U_D'),('Dmu','D0'),('K1','Dmu')):
            key=new+'_minus_'+old;a=combined[new]['patients'];b=combined[old]['patients']
            contrasts[cohort][key]={m:paired(a,b,m) for m in ('macro_Dice','NLL','Brier')};contrasts[cohort][key]['positive_patient_macro_Dice']=sum(x['macro_Dice']>y['macro_Dice'] for x,y in zip(a,b))
            classes[cohort][key]=[]
            for k in range(3):
                d=np.array([x['classes'][k]['Dice']-y['classes'][k]['Dice'] for x,y in zip(a,b) if x['classes'][k]['Dice'] is not None and y['classes'][k]['Dice'] is not None]);g=np.random.default_rng(163)
                classes[cohort][key].append(dict(class_index=k+1,valid_pairs=len(d),mean_Dice_difference=float(d.mean()) if len(d) else None,patient_bootstrap95=np.quantile(d[g.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975]).tolist() if len(d) else None))
        for a in combined.values():
            for i,p in enumerate(a['patients']):p['patient']=f'{cohort.upper()}{i:03}'
    r.pop('space');r.update(status='COMPLETE_AUDIT_PASS',audit=audit,matched_controls=controls,probability_and_Dice_contrasts=contrasts,class_contrasts=classes,effective_update_scalars=483776,optimizer_scalar_slots=484576,frozen_variance_scalars=800,scope='Matched local point estimates; G weights remain sampled; repeatedly used development validation; no full official BayeSeg claim')
    write(root/'PUBLIC_RESULTS.json',r);return r

if __name__=='__main__':summarize(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
