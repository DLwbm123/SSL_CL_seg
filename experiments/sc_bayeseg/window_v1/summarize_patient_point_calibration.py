"""Merge new point-temperature results with unchanged earlier crossfit controls."""
import json,os
from pathlib import Path
import torch,numpy as np
from sc_bayeseg.window_v1.r_start import write


def paired(a,b,k):
    assert [x['patient'] for x in a]==[x['patient'] for x in b]
    d=np.array([x[k]-y[k] for x,y in zip(a,b)]);g=np.random.default_rng(163)
    return dict(mean_difference=float(d.mean()),patient_bootstrap95=np.quantile(d[g.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975]).tolist())


def summarize(s):
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());r=read(root/'RESULTS.json');audit=read(root/'AUDIT.json');assert audit['status']=='PASS'
    private=torch.load(root/'PRIVATE_RESULTS.pt',map_location='cpu',weights_only=False);temps=torch.load(root/'PRIVATE_TEMPERATURES.pt',map_location='cpu',weights_only=False)
    old=read(Path(s['diagnostic_source'])/'PUBLIC_RESULTS.json');models={a:v['val'] for a,v in old['models'].items()};contrasts={}
    for arm,phases in private.items():
        ids=sorted({x['patient'] for x in phases});aliases={p:f'VAL{i:03}' for i,p in enumerate(ids)};patients=[];public_phases=[]
        for p in ids:
            rows=[x for x in phases if x['patient']==p];assert len(rows)==2
            v=dict(patient=aliases[p])
            for k in ('NLL','Brier'):
                v[k]=float(np.mean([x['risk'][k] for x in rows]));v['calibrated_'+k]=float(np.mean([x['temperature_risk'][k] for x in rows]))
            patients.append(v)
        for i,p in enumerate(phases):public_phases.append(dict(patient=aliases[p['patient']],phase=f'VAL_PHASE{i:03}',**{k:v for k,v in p.items() if k not in ('patient','case')}))
        models[arm]=dict(summary={k:float(np.mean([p[k] for p in patients])) for k in ('NLL','Brier','calibrated_NLL','calibrated_Brier')},patients=patients,phases=public_phases)
    for a,b in (('K1','Dmu'),('D0','K0'),('Dmu','D0'),('D0','U_D'),('Dmu','U_D')):
        contrasts[a+'_minus_'+b]={k:paired(models[a]['patients'],models[b]['patients'],k) for k in ('NLL','Brier','calibrated_NLL','calibrated_Brier')}
    for arm in ('D0','Dmu'):
        contrasts[arm+'_temperature_minus_raw']={}
        for k in ('NLL','Brier'):
            a=[dict(patient=p['patient'],value=p['calibrated_'+k]) for p in models[arm]['patients']];b=[dict(patient=p['patient'],value=p[k]) for p in models[arm]['patients']]
            contrasts[arm+'_temperature_minus_raw'][k]=paired(a,b,'value')
    publictemps={a:{f:{k:v for k,v in fit.items() if k not in ('fit_patients','heldout_patients')} for f,fit in ts.items()} for a,ts in temps.items()}
    out=dict(status='COMPLETE_AUDIT_PASS',scope='Separate point-estimate calibration closure; earlier four controls reused unchanged; repeatedly developed 23 patients, no final test',models=models,contrasts=contrasts,new_temperatures=publictemps,reused_control_temperature_fits=20,new_temperature_fits=10,costs=r['costs'],audit=audit,source_frozen=True,test_access=False,phase_elapsed_seconds=r['phase_elapsed_seconds'],original_elapsed_seconds=r['original_elapsed_seconds'])
    write(root/'PUBLIC_RESULTS.json',out);return out

if __name__=='__main__':summarize(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
