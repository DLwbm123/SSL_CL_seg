"""Patient ED/ES aggregation of fixed-prediction selective-risk counts."""
import json,os
from pathlib import Path
import torch,numpy as np
from scipy.stats import rankdata
from sc_bayeseg.window_v1.r_start import write


def paired(a,b):
    valid=[(x,y) for x,y in zip(a,b) if x is not None and y is not None]
    if not valid:return dict(valid_pairs=0,mean_difference=None,patient_bootstrap95=None)
    d=np.array([x-y for x,y in valid]);g=np.random.default_rng(163)
    return dict(valid_pairs=len(d),mean_difference=float(d.mean()),patient_bootstrap95=np.quantile(d[g.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975]).tolist())


def auc(scores,labels):
    labels=np.array(labels,dtype=bool);scores=np.array(scores);p=int(labels.sum());n=len(labels)-p
    return dict(positive=p,negative=n,AUROC=float((rankdata(scores)[labels].sum()-p*(p+1)/2)/(p*n)) if p and n else None)


def aggregate_private(data,references):
    models={}
    for arm,phases in data.items():
        ids=sorted({x['patient'] for x in phases});patients=[]
        assert ids==[p['patient'] for p in references[arm]['patients']]
        for i,patient in enumerate(ids):
            rows=[x for x in phases if x['patient']==patient];assert len(rows)==2
            classes=[float(np.mean([x['phase_class_Dice'][k] for x in rows if x['phase_class_Dice'][k] is not None])) if any(x['phase_class_Dice'][k] is not None for x in rows) else None for k in range(3)]
            dice=float(np.mean([x for x in classes if x is not None]));assert abs(dice-references[arm]['patients'][i]['macro_Dice'])<1e-10
            out=dict(patient=f'VAL{i:03}',macro_Dice=dice,macro_failure=dice<.5,RV_failure=any(x['RV_empty'] for x in rows),methods={})
            for method in rows[0]['methods']:
                cells=[]
                for j,coverage in enumerate((.5,.75,.9)):
                    pieces=[x['methods'][method]['coverage_results'][j] for x in rows];c=[]
                    for k in range(4):
                        v={name:sum(x['classes'][k][name] for x in pieces) for name in ('predicted_total','GT_total','accepted_predicted','errors_predicted','accepted_GT','errors_GT')}
                        v.update(class_index=k,predicted_risk=v['errors_predicted']/v['accepted_predicted'] if v['accepted_predicted'] else None,GT_coverage=v['accepted_GT']/v['GT_total'] if v['GT_total'] else None,GT_risk=v['errors_GT']/v['accepted_GT'] if v['accepted_GT'] else None);c.append(v)
                    accepted=sum(x['accepted'] for x in pieces);errors=sum(x['errors'] for x in pieces);fg=sum(x['GT_total'] for x in c[1:]);fgbg=sum(x['foreground_to_BG_accepted'] for x in pieces)
                    valid=[v['predicted_risk'] for v in c[1:] if v['predicted_risk'] is not None]
                    cells.append(dict(coverage=coverage,accepted=accepted,errors=errors,overall_risk=errors/accepted if accepted else None,foreground_macro_risk=float(np.mean(valid)) if len(valid)==3 else None,foreground_valid_predicted_classes=len(valid),foreground_to_BG_accepted=fgbg,foreground_to_BG_accepted_GT_fraction=fgbg/fg if fg else None,classes=c))
                out['methods'][method]=dict(case_score=float(np.mean([x['methods'][method]['case_score'] for x in rows])),coverage_results=cells)
            patients.append(out)
        summaries={}
        for method in patients[0]['methods']:
            coverages=[]
            for j,coverage in enumerate((.5,.75,.9)):
                vals=[p['methods'][method]['coverage_results'][j] for p in patients]
                def mean(xs):
                    xs=[x for x in xs if x is not None];return float(np.mean(xs)) if xs else None
                coverages.append(dict(coverage=coverage,mean_overall_risk=mean([v['overall_risk'] for v in vals]),mean_foreground_macro_risk=mean([v['foreground_macro_risk'] for v in vals]),foreground_valid_patients=sum(v['foreground_macro_risk'] is not None for v in vals),mean_foreground_to_BG_accepted_GT_fraction=mean([v['foreground_to_BG_accepted_GT_fraction'] for v in vals]),class_predicted_risk=[mean([v['classes'][k]['predicted_risk'] for v in vals]) for k in range(4)],class_GT_coverage=[mean([v['classes'][k]['GT_coverage'] for v in vals]) for k in range(4)],class_GT_risk=[mean([v['classes'][k]['GT_risk'] for v in vals]) for k in range(4)]))
            scores=[p['methods'][method]['case_score'] for p in patients]
            summaries[method]=dict(coverage_results=coverages,macro_failure_AUROC=auc(scores,[p['macro_failure'] for p in patients]),RV_failure_AUROC=auc(scores,[p['RV_failure'] for p in patients]))
        contrasts={}
        for method in patients[0]['methods']:
            contrasts[method+'_minus_calibrated_confidence']={}
            for j,coverage in enumerate((.5,.75,.9)):
                a=[p['methods'][method]['coverage_results'][j] for p in patients];b=[p['methods']['calibrated_confidence']['coverage_results'][j] for p in patients]
                contrasts[method+'_minus_calibrated_confidence'][str(coverage)]={name:paired([x[name] for x in a],[x[name] for x in b]) for name in ('foreground_macro_risk','overall_risk','foreground_to_BG_accepted_GT_fraction')}
        models[arm]=dict(patients=patients,summaries=summaries,contrasts=contrasts)
    return dict(models=models,primary=models['K1']['contrasts']['local_disagreement_minus_calibrated_confidence']['0.75']['foreground_macro_risk'],primary_definition='K1 local disagreement minus same-prediction calibrated confidence,75% each phase/predicted-class quota; ED/ES counts merged before per-class risk; FG macro requires all3 classes, missing is NA')


def summarize(s,require_audit=True):
    root=Path(s['output']);read=lambda p:json.loads(Path(p).read_text());r=read(root/'RESULTS.json');assert r['status']=='COMPLETE'
    data=torch.load(root/'PRIVATE_RESULTS.pt',map_location='cpu',weights_only=False)
    references=dict(K1=read(Path(s['kl_source'])/'RESULTS.json')['aggregates']['val']['R32_K1'],D0=read(Path(s['point_source'])/'RESULTS.json')['aggregates']['val']['D0'],Dmu=read(Path(s['point_source'])/'RESULTS.json')['aggregates']['val']['Dmu'],U_D=read(Path(s['factor_source'])/'RESULTS.json')['aggregates']['val']['U_D'])
    out=aggregate_private(data,references);out.update(status='COMPLETE_AUDIT_PASS' if require_audit else 'PENDING_AUDIT',scope='23 repeatedly used development patients; frozen predictions/quotas; no SSL or finaltest; native disagreement upsampled, not 384 perdraw MI',costs=r['costs'],phase_elapsed_seconds=r['phase_elapsed_seconds'],original_elapsed_seconds=r['original_elapsed_seconds'])
    if require_audit:out['audit']=read(root/'AUDIT.json');assert out['audit']['status']=='PASS'
    write(root/('PUBLIC_RESULTS.json' if require_audit else 'STATISTICS_PENDING_AUDIT.json'),out);return out

if __name__=='__main__':summarize(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()),require_audit=os.environ.get('PENDING_AUDIT')!='1')
