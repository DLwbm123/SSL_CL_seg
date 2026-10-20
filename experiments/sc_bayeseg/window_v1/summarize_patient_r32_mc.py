"""Publish anonymous patient metrics and risk contrasts from completed fixed MC streams."""
import json,os
from pathlib import Path
import numpy as np,torch
from sc_bayeseg.window_v1.r_start import write


def paired(a,b,metric):
    assert [p['patient'] for p in a]==[p['patient'] for p in b]
    d=np.array([x[metric]-y[metric] for x,y in zip(a,b)]);g=np.random.default_rng(163)
    ci=np.quantile(d[g.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975])
    return dict(mean_difference=float(d.mean()),patient_bootstrap95=ci.tolist())


def summarize(s):
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());r=read(root/'RESULTS.json');audit=read(root/'AUDIT.json');assert audit['status']=='PASS'
    control=read(Path(s['factor_source'])/'RESULTS.json')['aggregates']['val']['U_D']
    initial=dict(K1=read(Path(s['kl_source'])/'RESULTS.json')['aggregates']['val']['R32_K1'],K0=read(Path(s['r32_source'])/'RESULTS.json')['aggregates']['val'])
    flows=dict(ORIGINAL=initial,**r['aggregates']);private=torch.load(root/'PRIVATE_PHASE_RESULTS.pt',map_location='cpu',weights_only=False);contrasts={};risks={}
    for flow,arms in flows.items():
        contrasts[flow]={}
        for arm,a in arms.items():
            a['mean_NLL']=float(np.mean([p['NLL'] for p in a['patients']]));a['mean_Brier']=float(np.mean([p['Brier'] for p in a['patients']]))
            contrasts[flow][arm+'_minus_U_D']={m:paired(a['patients'],control['patients'],m) for m in ('macro_Dice','NLL','Brier')}
        contrasts[flow]['K1_minus_K0']={m:paired(arms['K1']['patients'],arms['K0']['patients'],m) for m in ('macro_Dice','NLL','Brier')}
        if flow!='ORIGINAL':
            risks[flow]={}
            for arm,rows in private[flow].items():
                patients={}
                for row in rows:patients.setdefault(row['patient'],[]).append(row)
                patientrows=[]
                for alias,(patient,phases) in enumerate(sorted(patients.items())):
                    assert len(phases)==2
                    cells=[]
                    for k in range(4):
                        cols=[p['class_NLL'][k] for p in phases];valid=[x['conditional_NLL'] for x in cols if x['conditional_NLL'] is not None]
                        cells.append(dict(class_index=k,conditional_NLL=float(np.mean(valid)) if valid else None,full_image_contribution=float(np.mean([x['full_image_contribution'] for x in cols])),GT_absent_phases=sum(x['pixels']==0 for x in cols)))
                    patientrows.append(dict(patient=f'VAL{alias:03}',classes=cells))
                means=[float(np.mean([p['classes'][k]['full_image_contribution'] for p in patientrows])) for k in range(4)]
                assert abs(sum(means)-flows[flow][arm]['mean_NLL'])<1e-7
                risks[flow][arm]=dict(mean_NLL_contribution=means,mean_conditional_NLL=[float(np.mean([p['classes'][k]['conditional_NLL'] for p in patientrows if p['classes'][k]['conditional_NLL'] is not None])) for k in range(4)],patients=patientrows)
    # Stable source ordering is verified before replacing identities.
    for arms in flows.values():
        for a in arms.values():
            assert [p['patient'] for p in a['patients']]==[p['patient'] for p in control['patients']]
            for i,p in enumerate(a['patients']):p['patient']=f'VAL{i:03}'
    for i,p in enumerate(control['patients']):p['patient']=f'VAL{i:03}'
    out=dict(status='COMPLETE_AUDIT_PASS',scope='23 repeatedly used development patients, fixed training states, original and four additional MC streams; conditional G without F, auxiliary Dice',aggregates=flows,U_D_control=control,contrasts=contrasts,class_risk_new_streams=risks,audit=audit,costs=r['costs'],phase_elapsed_seconds=r['phase_elapsed_seconds'],original_elapsed_seconds=r['original_elapsed_seconds'],test_access=False,new_optimizer=0,new_reverse=0)
    write(root/'PUBLIC_RESULTS.json',out);return out

if __name__=='__main__':summarize(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
