"""One-shot case aggregation control; no score direction reversal or feature selection."""
import json,os
from pathlib import Path
import numpy as np,torch
from scipy.stats import spearmanr
from sc_bayeseg.window_v1.r_start import write


def auc(scores,labels):
    scores=np.asarray(scores);labels=np.asarray(labels,dtype=bool);a=scores[labels];b=scores[~labels]
    return float(((a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])).mean()) if len(a) and len(b) else None


def auc_difference(a,b,labels):
    a=np.asarray(a);b=np.asarray(b);labels=np.asarray(labels,dtype=bool);g=np.random.default_rng(163);values=[]
    for ids in g.integers(0,len(a),(10000,len(a))):
        x=auc(a[ids],labels[ids]);y=auc(b[ids],labels[ids])
        if x is not None:values.append(x-y)
    x=auc(a,labels);y=auc(b,labels)
    return dict(mean_difference=x-y if x is not None else None,patient_bootstrap95=np.quantile(values,[.025,.975]).tolist() if values else None,valid_bootstrap_repeats=len(values),requested_repeats=10000)


def correlation(a,b):
    cells=[(x,y) for x,y in zip(a,b) if x is not None and y is not None]
    if len(cells)<3:return None
    a,b=np.asarray(cells).T
    return float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None


def summarize(s,require_audit=True):
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());r=read(root/'RESULTS.json');assert r['status']=='COMPLETE'
    prior=read(Path(s['reliability_source'])/'PUBLIC_RESULTS.json');private=torch.load(root/'PRIVATE_CASE_SCORES.pt',map_location='cpu',weights_only=False);models={}
    for arm,phases in private.items():
        ids=sorted({x['patient'] for x in phases});patients=[]
        for i,patient in enumerate(ids):
            rows=[x for x in phases if x['patient']==patient];assert len(rows)==2
            old=prior['models'][arm]['patients'][i];assert old['patient']==f'VAL{i:03}'
            out={k:old[k] for k in ('patient','macro_Dice','macro_failure','RV_failure')};out['foreground_fraction']=float(np.mean([x['foreground_fraction'] for x in rows]));out['methods']={}
            for method in rows[0]['methods']:
                classes=[float(np.mean([x['methods'][method]['class_top10'][k] for x in rows if x['methods'][method]['class_top10'][k] is not None])) if any(x['methods'][method]['class_top10'][k] is not None for x in rows) else None for k in range(4)]
                whole=float(np.mean([x['methods'][method]['whole_top01'] for x in rows]));assert abs(whole-old['methods'][method]['case_score'])<1e-10
                out['methods'][method]=dict(whole_score=whole,foreground_conditional_score=float(np.mean(classes[1:])) if all(x is not None for x in classes[1:]) else None,background_conditional_score=classes[0],class_scores=classes)
            patients.append(out)
        summaries={}
        for method in patients[0]['methods']:
            valid=[i for i,p in enumerate(patients) if p['methods'][method]['foreground_conditional_score'] is not None];a=[patients[i]['methods'][method]['foreground_conditional_score'] for i in valid];b=[patients[i]['methods'][method]['whole_score'] for i in valid];summary=dict(valid_patients=len(valid),excluded_patients=len(patients)-len(valid))
            for label in ('macro_failure','RV_failure'):
                labels=[patients[i][label] for i in valid];alllabels=[p[label] for p in patients];allold=[p['methods'][method]['whole_score'] for p in patients]
                summary[label]=dict(eligible_positive=sum(labels),eligible_negative=len(labels)-sum(labels),excluded_positive=sum(p[label] for j,p in enumerate(patients) if j not in valid),whole_AUROC_all=auc(allold,alllabels),whole_AUROC_same_patients=auc(b,labels),foreground_conditional_AUROC=auc(a,labels),paired_AUROC_difference=auc_difference(a,b,labels) if valid else None)
            area=[p['foreground_fraction'] for p in patients];whole=[p['methods'][method]['whole_score'] for p in patients];cond=[p['methods'][method]['foreground_conditional_score'] for p in patients];dice=[p['macro_Dice'] for p in patients]
            summary['Spearman']=dict(whole_vs_foreground_fraction=correlation(whole,area),conditional_vs_foreground_fraction=correlation(cond,area),whole_vs_Dice=correlation(whole,dice),conditional_vs_Dice=correlation(cond,dice))
            summaries[method]=summary
        area_risk=[1-p['foreground_fraction'] for p in patients];baseline={label:dict(positive=sum(p[label] for p in patients),negative=sum(not p[label] for p in patients),AUROC=auc(area_risk,[p[label] for p in patients])) for label in ('macro_failure','RV_failure')}
        models[arm]=dict(patients=patients,summaries=summaries,geometry_reference=baseline)
    out=dict(status='COMPLETE_AUDIT_PASS' if require_audit else 'PENDING_AUDIT',scope='One post-hoc preregistered aggregation control; fixed prior failure labels; not confirmation, new model, or SSL qualification',models=models,costs=r['costs'],phase_elapsed_seconds=r['phase_elapsed_seconds'],original_elapsed_seconds=r['original_elapsed_seconds'])
    if require_audit:out['audit']=read(root/'AUDIT.json');assert out['audit']['status']=='PASS'
    write(root/('PUBLIC_RESULTS.json' if require_audit else 'STATISTICS_PENDING_AUDIT.json'),out);return out

if __name__=='__main__':summarize(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()),require_audit=os.environ.get('PENDING_AUDIT')!='1')
