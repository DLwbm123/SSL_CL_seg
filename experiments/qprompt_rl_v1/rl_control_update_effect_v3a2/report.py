"""Independent post-seal scoring of all registered outputs; public aggregates only."""
import csv,json,statistics
from collections import Counter
import numpy as np
from .runtime import *
from .pipeline import feature_table,values
from . import policy

def table(name,rows):
    p=ROOT/'reports'/name;p.parent.mkdir(exist_ok=True)
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row}),lineterminator='\n');w.writeheader();w.writerows(rows)

def report():
    manifest=read(ROOT/'CONTEXT_MANIFEST.private.json');lock=receipt(ROOT/'TRANSFER_PREDICTION_LOCK.json');assert all(digest(ROOT/p)==d for p,d in lock['files'].items());methods=PLAN['learners']['methods'];rows=[];repeats=[];contrasts=[];features=[];correlations=[];horizons=[];extrap=[];primary={};stage={};state_rows=[]
    for cell,spec in manifest.items():
        phase='development' if spec['seed']==261 else 'transfer';x=feature_table(cell);panels=list((ROOT/'panels'/cell).glob('*.private.json'));assert len(panels)==144
        br=[receipt(p) for p in panels];assert all(v['root_restored'] and v['retained_after_discard']==0 and len(v['steps'])==5 for v in br);assert len({v['root_fingerprint'] for v in br})==1
        for v in br:assert [z['adam'] for z in v['steps']]==[2001,2002,2003,2004,2005] and [z['scheduler'] for z in v['steps']]==[1,2,3,4,5] and all(z['teacher_updated'] for z in v['steps'])
        fs=[receipt(p) for p in (ROOT/'features'/cell).glob('*.private.json')];assert len(fs)==48 and all(v['invariant'] for v in fs);state_rows.append(dict(cell=cell,branches=144,feature_extractions=48,root_invariance=True,teacher_scheduler_adam=True,root_fingerprint=br[0]['root_fingerprint']))
        for col in range(40):
            repeat_values=np.array([[receipt(ROOT/'features'/cell/f'{j}_{r}.private.json')['features'][col] for r in range(3)] for j in range(16)])
            features.append(dict(cell=cell,column=col,mean=float(x[:,col].mean()),variance=float(x[:,col].var()),distinct_values=int(len(np.unique(x[:,col]))),mean_within_scene_repeat_variance=float(repeat_values.var(1).mean()),max_repeat_span=float(np.ptp(repeat_values,axis=1).max())))
        predictions={}
        for method in methods:
            if phase=='transfer':pred=receipt(ROOT/'predictions'/cell/f'{method}.private.json')['prediction']
            else:
                pred=dict(actions=[None]*16,probabilities=[None]*16)
                for k in range(4):
                    fit=receipt(ROOT/'fits'/cell/f'{k}_{method}.private.json')
                    for pos,j in enumerate(fit['test_indices']):
                        pred['actions'][j]=fit['prediction']['actions'][pos];pred['probabilities'][j]=fit['prediction']['probabilities'][pos]
            predictions[method]=pred
        for h in (1,5):
            audit=values(cell,'audit',h);online=values(cell,'online',h);v_by_method={};worst=float(np.ptp(audit,axis=0).max());horizons.append(dict(cell=cell,h=h,oracle=float(audit.max(2).mean()),online_selector_value=float(np.take_along_axis(audit,online.argmax(2)[...,None],2).mean()),worst_scene_action_repeat_span=worst))
            for col in range(40):
                for a in (0,1):
                    target=audit[:,:,a].mean(0);corr=float(np.corrcoef(x[:,col],target)[0,1]) if x[:,col].std()>0 and target.std()>0 else None;correlations.append(dict(cell=cell,h=h,column=col,action=a,descriptive_correlation=corr))
            for method,pred in predictions.items():
                act=np.array(pred['actions']);prob=np.array(pred['probabilities']);v=np.take_along_axis(audit,np.broadcast_to(act,(3,16))[...,None],2)[...,0];mean_r=v.mean(1);v_by_method[method]=mean_r;band=max(1e-6,5*float(np.ptp(mean_r)));intervention=act!=2
                row=dict(cell=cell,phase=phase,seed=spec['seed'],backbone=spec['backbone'],domain=spec['domain'],h=h,method=method,value=float(mean_r.mean()),softmax_value=float((audit*prob[None]).sum(2).mean()),repeat_value_span=float(np.ptp(mean_r)),numeric_band=band,numeric_status='INCONCLUSIVE_NUMERICS' if abs(float(mean_r.mean()))<=band else 'RESOLVED_DIRECTION',intervention_rate=float(intervention.mean()),fine_rate=float((act==2).mean()),negative_intervention_rate=float(((v<0)&intervention).mean()),value_per_intervention=float(v[:,intervention].mean()) if intervention.any() else None,scenes=16)
                for a in (0,1,2):row['action_'+str(a)+'_count']=int((act==a).sum());row['action_'+str(a)+'_value']=float(v[:,act==a].mean()) if (act==a).any() else None
                rows.append(row);repeats.extend(dict(cell=cell,h=h,method=method,repeat=r,value=float(mean_r[r])) for r in range(3))
            for comparator in methods:
                if comparator=='EFFECT_POLICY':continue
                difference=v_by_method['EFFECT_POLICY']-v_by_method[comparator];contrasts.append(dict(cell=cell,phase=phase,h=h,contrast='EFFECT_POLICY-'+comparator,value=float(difference.mean()),numeric_band=max(1e-6,5*float(np.ptp(difference)))))
            if h==5:stage[cell]=v_by_method
            if phase=='transfer' and h==5:primary[cell]=v_by_method
        # Scaler diagnostics use frozen train-only scalers; no fitting on transfer.
        dev=cell if phase=='development' else 'S261__'+spec['backbone']+'__'+spec['domain']
        for method in ('BASE_POLICY','EFFECT_POLICY','EFFECT_SHUFFLE','BASE_RIDGE','EFFECT_RIDGE'):
            fit=receipt(ROOT/'fits'/dev/f'all_{method}.private.json')['fit'];sc=fit['scaler'];z=policy.normalize(x,sc);tr=feature_table(dev);zt=policy.normalize(tr,sc)
            for col in range(40):extrap.append(dict(cell=cell,method=method,column=col,train_raw_variance=float(tr[:,col].var()),train_standardized_variance=float(zt[:,col].var()),evaluation_standardized_variance=float(z[:,col].var()),evaluation_max_abs_standardized=float(abs(z[:,col]).max()),outside_train_range_fraction=float(((x[:,col]<sc['train_min'][col])|(x[:,col]>sc['train_max'][col])).mean()),forced_zero=bool(sc['constant'][col])))
    assert len(primary)==8
    overall=[];criteria={}
    for method in methods:
        if method=='EFFECT_POLICY':continue
        v=np.stack([d['EFFECT_POLICY']-d[method] for d in primary.values()]).mean(0);band=max(1e-6,5*float(np.ptp(v)));criteria[method]=(float(v.mean()),band);overall.append(dict(scope='overall_transfer_h5',contrast='EFFECT_POLICY-'+method,value=float(v.mean()),numeric_band=band,repeat_span=float(np.ptp(v))))
    cells_positive=sum(x['value']>x['numeric_band'] for x in contrasts if x['phase']=='transfer' and x['h']==5 and x['contrast']=='EFFECT_POLICY-FINE');bb={b:float(np.mean([d['EFFECT_POLICY'] for k,d in primary.items() if b in k])) for b in PLAN['backbones']};passed=all(criteria[m][0]>=1e-5 and criteria[m][0]>criteria[m][1] for m in ('BASE_POLICY','FINE')) and cells_positive>=6 and all(v>0 for v in bb.values()) and all(criteria[m][0]>criteria[m][1] for m in ('NC_OPT','EFFECT_SHUFFLE'))
    table('TRANSFER_POLICY_VALUES.csv',[x for x in rows if x['phase']=='transfer']);table('DEVELOPMENT_OOF.csv',[x for x in rows if x['phase']=='development']);table('MAIN_CONTRASTS.csv',contrasts+overall);table('REPEAT_ERROR.csv',repeats);table('HORIZON_TRANSFER.csv',horizons);table('FEATURE_VARIATION.csv',features);table('FEATURE_REWARD_CORRELATIONS.csv',correlations);table('SCALER_VARIATION.csv',extrap);save(ROOT/'reports/STATE_INVARIANCE_AUDIT.json',cells=state_rows)
    events=c.events(ROOT/'PHYSICAL_LEDGER.jsonl');attempt=Counter((v['key'],v['category']) for v in events if v['event']=='attempt');success=Counter((v['key'],v['category']) for v in events if v['event']=='success');failed=Counter((v['key'],v['category']) for v in events if v['event']=='failed');assert attempt==success+failed;counts=Counter(v['category'] for v in events if v['event']=='attempt')
    from .runtime import Ledger
    assert all(counts[k]<=Ledger.caps[k] for k in counts);assert counts['student']==8640 and counts['controller']==15360 and counts['vjp']==2304 and counts['preview']==1728 and counts['extraction']==576
    all_features=[receipt(p) for p in (ROOT/'features').glob('*/*.private.json')];cost_events=[v for p in (ROOT/'panels').glob('*/COST_EVENTS.jsonl') for v in c.events(p)];save(ROOT/'reports/ALL_COSTS.json',physical_attempts=dict(counts),failed_attempts=sum(failed.values()),unresolved=0,student_retained=0,medical_endpoints=0,feature_student_forwards=sum(v['student_forwards'] for v in all_features),feature_teacher_forwards=sum(v['teacher_forwards'] for v in all_features),feature_seconds=sum(v['seconds'] for v in all_features),peak_gpu_bytes=max(v['peak_gpu_bytes'] for v in all_features),panel_student_forwards=sum(v.get('student_forwards',0) for v in cost_events),panel_teacher_forwards=sum(v.get('teacher_forwards',0) for v in cost_events),neural_fits=60,analytic_fits=60,branches=1728)
    for name in ('FINAL_POLICY_LOCK.json','TRANSFER_PREDICTION_LOCK.json'):
        v=receipt(ROOT/name);save(ROOT/'reports'/name,source_sha=digest(ROOT/name),files_sealed=len(v['files']),sealed_at=v.get('sealed_at'),code_at_seal=v['code_commit'])
    save(ROOT/'reports/QUALIFICATION.json',**{k:v for k,v in receipt(ROOT/'QUALIFICATION.json').items() if k not in ('code_commit','config_sha')});save(ROOT/'reports/PRIMARY_RESULT.json',contrasts={k:dict(mean=v[0],numeric_band=v[1]) for k,v in criteria.items()},cells_above_band=cells_positive,backbone_values=bb,proposal_criterion=bool(passed),horizon=5,units='feedback quality, not Dice',complete_cells=8)
    text=['# V3A2 final interpretation','','All 12 cells and all registered methods completed. Primary evaluation is sealed deterministic transfer on seeds262/263, h5. Softmax and h1 are secondary only.','',f'Primary A EFFECT−BASE: {criteria["BASE_POLICY"]}. Primary B EFFECT−FINE: {criteria["FINE"]}. Each pair is (mean quality difference, numerical repeat band).',f'Cells above their net-value band: {cells_positive}/8. Backbone net values: {bb}. Frozen proposal criterion: {passed}.','', '## Feature variation and predictability','FEATURE_VARIATION gives the actual variation of all 40 entry-state dimensions and within-scene repeat variation. FEATURE_REWARD_CORRELATIONS gives descriptive h1/h5 correlations without independent-sample significance claims. SCALER_VARIATION uses frozen training means/stds; constant training columns remain zero at transfer.','', '## Matched learner and transfer','DEVELOPMENT_OOF and MAIN_CONTRASTS separate local grouped OOF from sealed cross-prefix transfer. EFFECT−BASE is the state-information comparison within this protocol; changes from V3A also include common pool, repetition and execution-rule changes.',f'Transfer EFFECT−NC_OPT: {criteria["NC_OPT"]}; EFFECT−SHUFFLE: {criteria["EFFECT_SHUFFLE"]}. Context utility cannot be inferred solely from improving a harmful baseline.','', '## Non-RL controls and numerical resolution',f'EFFECT−EFFECT_RIDGE: {criteria["EFFECT_RIDGE"]}; EFFECT−EFFECT_RULE: {criteria["EFFECT_RULE"]}. These controls remain mandatory; do not claim RL-specific advantage if they explain the gain.','REPEAT_ERROR and HORIZON_TRANSFER retain raw values and worst-scene spans. Results within a band are numerically inconclusive, not set to zero or excluded. Three repeats are not a confidence interval.','', '## Returning to FINE and future work','TRANSFER_POLICY_VALUES reports FINE frequency, intervention rate and per-action utility. A return to FINE without positive net value is not intervention success. No closed-loop pilot, new medical endpoint, retained student update, original KI substitution or automatic V3B was executed.','', '## Limits','U development and transfer pools are disjoint including continuation images. Optimization seeds were used previously; L cohorts and feedback identities are shared. This is not patient-independent confirmation, external validation, or an unbiased long-term return estimate. Feedback quality is not Dice percentage points. Publication requires separate current permission.']
    (ROOT/'reports/FINAL_INTERPRETATION.md').write_text('\n'.join(text)+'\n')
