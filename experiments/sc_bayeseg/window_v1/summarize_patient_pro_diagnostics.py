"""Anonymous patient risk, crossfit calibration and RV event summaries."""
import json,os
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
from sc_bayeseg.window_v1.r_start import write


def contrast(a,b):
    a=np.asarray(a);b=np.asarray(b);assert len(a)==len(b);d=a-b;g=np.random.default_rng(163)
    return dict(mean_difference=float(d.mean()),patient_bootstrap95=np.quantile(d[g.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975]).tolist())


def summarize(s):
    root=Path(s['output']);r=json.loads((root/'RESULTS.json').read_text());audit=json.loads((root/'AUDIT.json').read_text());assert audit['status']=='PASS'
    data=torch.load(root/'PRIVATE_RESULTS.pt',map_location='cpu',weights_only=False);temps=torch.load(root/'PRIVATE_TEMPERATURES.pt',map_location='cpu',weights_only=False);out={};contrasts={}
    for arm,cohorts in data.items():
        out[arm]={}
        for cohort,phases in cohorts.items():
            ids=sorted({x['patient'] for x in phases});aliases={p:f'{cohort.upper()}{i:03}' for i,p in enumerate(ids)};patients=[]
            for p in ids:
                rows=[x for x in phases if x['patient']==p];assert len(rows)==2
                risk={k:float(np.mean([x['risk'][k] for x in rows])) for k in ('NLL','Brier','correct_NLL_contribution','error_NLL_contribution','foreground_to_BG_NLL_contribution')}
                risk['class_NLL_contribution']=[float(np.mean([x['risk']['classes'][k]['full_image_contribution'] for x in rows])) for k in range(4)]
                risk['class_conditional_NLL']=[float(np.mean([x['risk']['classes'][k]['conditional_NLL'] for x in rows if x['risk']['classes'][k]['conditional_NLL'] is not None])) if any(x['risk']['classes'][k]['conditional_NLL'] is not None for x in rows) else None for k in range(4)]
                patient=dict(patient=aliases[p],**risk)
                if cohort=='val':patient.update(calibrated_NLL=float(np.mean([x['temperature_risk']['NLL'] for x in rows])),calibrated_Brier=float(np.mean([x['temperature_risk']['Brier'] for x in rows])))
                patients.append(patient)
            phase_rows=[]
            for i,x in enumerate(phases):
                row={k:v for k,v in x.items() if k not in ('patient','case')};row.update(patient=aliases[x['patient']],phase=f'{cohort.upper()}_PHASE{i:03}')
                phase_rows.append(row)
            summary={k:float(np.mean([x[k] for x in patients])) for k in ('NLL','Brier','correct_NLL_contribution','error_NLL_contribution','foreground_to_BG_NLL_contribution')}
            if cohort=='val':summary.update(calibrated_NLL=float(np.mean([x['calibrated_NLL'] for x in patients])),calibrated_Brier=float(np.mean([x['calibrated_Brier'] for x in patients])))
            summary['empty_RV_mixture']=sum(x['RV_mixture']['empty'] for x in phases)
            if arm!='U_D':
                summary.update(empty_RV_mean=sum(x['RV_mean']['empty'] for x in phases),empty_mixture_with_any_RV_draw=sum(x['RV_mixture']['empty'] and any(d['predicted']>0 for d in x['RV_draws'][:16]) for x in phases))
            out[arm][cohort]=dict(summary=summary,patients=patients,phases=phase_rows)
    for arm in out:
        a=out[arm]['val']['patients'];b=out['U_D']['val']['patients'];assert [x['patient'] for x in a]==[x['patient'] for x in b]
        contrasts[arm+'_minus_U_D']={k:contrast([x[k] for x in a],[x[k] for x in b]) for k in ('NLL','Brier','calibrated_NLL','calibrated_Brier')}
        contrasts[arm+'_temperature_minus_raw']={k:contrast([x['calibrated_'+k] for x in a],[x[k] for x in a]) for k in ('NLL','Brier')}
    anonymous_temps={arm:{fold:{k:v for k,v in f.items() if k not in ('fit_patients','heldout_patients')} for fold,f in fs.items()} for arm,fs in temps.items()}
    result=dict(status='COMPLETE_AUDIT_PASS',scope='Repeatedly used development validation; scalar-temperature optimization with network zero updates; G omits F, not full official BayeSeg',models=out,contrasts=contrasts,temperatures=anonymous_temps,costs=r['costs'],audit=audit,phase_elapsed_seconds=r['phase_elapsed_seconds'],original_elapsed_seconds=r['original_elapsed_seconds'],test_access=False)
    write(root/'PUBLIC_RESULTS.json',result);return result

if __name__=='__main__':summarize(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
