"""Independent post-seal outer evaluator. No fit or candidate-selection calls."""
import time
import numpy as np
from care_hr_v0_7_1.io_r1 import read_json,write_json,digest,verify_files
from care_hr_v0_7_1.execute_r1 import write_csv,require,ROOT,FORMAL
from care_hr_v0_7_1.scoring_r1 import gains
from care_hr_v0_7_1.summary_r1 import summarize,groups_for,balanced
from .learning import patient_weights
from .pipeline import (verify,probabilities,DOC,PARENT,R2,POLICIES,CONTROL_NAMES,BASE_POLICIES,source_state)


def tail(rows,reference=None):
    result=[]
    for level,key,subset in groups_for(rows):
        for policy in dict.fromkeys(r['policy'] for r in rows):
            rr=[r for r in subset if r['policy']==policy]
            if not rr:continue
            delta=np.array([r['gain_macro_fg'] for r in rr]); harm=np.array([r['harm'] for r in rr])
            pd={}
            for r in rr:pd.setdefault(r['patient_id'],[]).append(r['gain_macro_fg'])
            patient_means=np.array([np.mean(v) for v in pd.values()])
            accepted=[r for r in rr if r['route']<2]
            vetoed=[r for r in rr if r['shor_route']<2 and r['route']==2]
            veto_gains=[gains(r['expert_metrics'][2],r['expert_metrics'][r['shor_route']]) for r in vetoed]
            result.append(dict(level=level,key=key,policy=policy,rows=len(rr),patients=len(pd),
                accepted=len(accepted),vetoed=sum(r['shor_route']<2 and r['route']==2 for r in rr),
                current_domain_historical_overrides=sum(r['domain_index']==2 and r['route']<2 for r in rr),
                source_mismatches=sum(r['route']<2 and r['route']!=r['domain_index'] for r in rr),
                beneficial_accepted=sum(r['gain_macro_fg']>0 for r in accepted),
                harmful_accepted=sum(r['harm']>0 for r in accepted),
                beneficial_vetoed=sum(g['gain_macro_fg']>0 for g in veto_gains),
                class_harm_vetoed=sum(g['harm']>0 for g in veto_gains),
                zero_macro_vetoed=sum(g['gain_macro_fg']==0 for g in veto_gains),
                class_nonharm_accepted=sum(r['harm']==0 for r in accepted),
                correction_precision=None if not accepted else sum(r['gain_macro_fg']>0 for r in accepted)/len(accepted),
                class_nonharm_precision=None if not accepted else sum(r['harm']==0 for r in accepted)/len(accepted),
                worst_case_delta=float(delta.min()),worst_class_harm=float(harm.max()),
                case_delta_below_minus_010=int((delta<-.10).sum()),
                patients_with_case_below_minus_010=len({r['patient_id'] for r in rr if r['gain_macro_fg']<-.10}),
                worst_patient_mean_delta=float(patient_means.min()),
                patient_mean_below_minus_010=int((patient_means<-.10).sum())))
    return result


def gates(table,rows):
    def get(policy,level,key,field='gain_macro_fg'):return table[policy,level,key][field]
    p='SHOR_UV_CF'; ref='SHOR_FROZEN'
    current=[r for r in rows if r['policy']==p and r['domain_index']==2]
    current_drop=max(0.,-get(p,'domain','Drishti_GS'))
    ref_drop=max(0.,-get(ref,'domain','Drishti_GS'))
    checks=[
      ('VALUE','overall_gain',get(p,'overall','all'),'>=',.17),
      ('VALUE','historical_gain',get(p,'historical','history'),'>=',.27),
      *[('VALUE','positive_'+d,get(p,'domain',d),'>',0.) for d in ('REFUGE','RIM_ONE_r3')],
      *[('VALUE','positive_seed_'+str(s),get(p,'seed',str(s)),'>',0.) for s in range(3)],
      ('SAFETY','current_mean_drop',current_drop,'<=',.01),
      ('SAFETY','current_maximum_class_drop',max(get(p,'domain','Drishti_GS','rim_drop'),get(p,'domain','Drishti_GS','cup_drop')),'<=',.015),
      ('SAFETY','maximum_seed_domain_mean_drop',max([0.]+[-r['gain_macro_fg'] for r in table.values() if r['policy']==p and r['level']=='seed_domain']),'<=',.02),
      ('SAFETY','current_case_delta_below_minus_010',sum(r['gain_macro_fg']<-.10 for r in current),'==',0),
      ('BASELINE','overall_difference_from_SHOR',get(p,'overall','all')-get(ref,'overall','all'),'>=',-.01),
      ('BASELINE','historical_difference_from_SHOR',get(p,'historical','history')-get(ref,'historical','history'),'>=',-.02),
      ('INCREMENTAL','current_drop_reduction_from_SHOR',ref_drop-current_drop,'>=',.005)]
    functions={'>=':lambda a,b:a>=b,'>':lambda a,b:a>b,'<=':lambda a,b:a<=b,'==':lambda a,b:a==b}
    return [dict(category=c,gate=n,observed=float(v),comparison=op,threshold=t,
                 passed=bool(functions[op](v,t))) for c,n,v,op,t in checks]


def pair_audit(records,predictions):
    out=[];private=[]
    for name,preds in predictions.items():
        if preds is None:continue
        true=np.array([r['targets'] for r in records]); good=np.array([r['has_evaluable_gt'] for r in records])
        available=np.isfinite(preds).all((1,2)) & good
        yt=true[available].reshape(-1,3); yp=preds[available].reshape(-1,3)
        pats=np.repeat([r['patient_id'] for i,r in enumerate(records) if available[i]],2)
        if not len(yt):
            out.append(dict(policy=name,pairs=0,status='NO_HEAD_PREDICTIONS'));continue
        w=patient_weights(pats);g=yt[:,:2].mean(1);gh=yp[:,:2].mean(1)
        nonzero=g!=0; dangerous=yt[:,2]>0
        def wavg(v,mask):
            return float(np.average(v[mask],weights=w[mask])) if mask.any() else None
        out.append(dict(policy=name,pairs=len(yt),patients=len(set(pats)),excluded_pairs=int((~available).sum()*2),
            status='DEFINED',MAE_d_rim=wavg(abs(yp[:,0]-yt[:,0]),np.ones(len(w),bool)),
            MAE_d_cup=wavg(abs(yp[:,1]-yt[:,1]),np.ones(len(w),bool)),
            MAE_H=wavg(abs(yp[:,2]-yt[:,2]),np.ones(len(w),bool)),
            gain_sign_accuracy_nonzero=wavg((np.sign(g)==np.sign(gh)).astype(float),nonzero),
            zero_gain_pairs=int((~nonzero).sum()),zero_target_pairs=int((yt==0).all(1).sum()),
            harmful_pairs=int(dangerous.sum()),harmful_pair_recall_Hhat_gt_zero=wavg((yp[:,2]>0).astype(float),dangerous),
            worst_true_H=float(yt[:,2].max()),
            Hhat_at_worst_H=float(yp[int(np.argmax(yt[:,2])),2]),
            diagnostic_threshold='H>0 and Hhat>0; not a calibrated bound'))
        for i,r in enumerate(records):
            if not available[i]:continue
            for h in (0,1):private.append(dict(policy=name,row_index=i,patient_id=r['patient_id'],seed=r['seed'],
                domain=r['domain'],expert=h,target=true[i,h].tolist(),predicted=preds[i,h].tolist()))
    return out,private


def bootstrap(rows):
    policies=list(dict.fromkeys(r['policy'] for r in rows))
    base=sorted([r for r in rows if r['policy']=='CURRENT'],key=lambda r:r['row_index'])
    patients=sorted({r['patient_id'] for r in base})
    pi=np.array([patients.index(r['patient_id']) for r in base])
    rng=np.random.default_rng(2026090801)
    multiplicity=rng.multinomial(len(patients),np.ones(len(patients))/len(patients),size=2000)
    w=multiplicity[:,pi]
    groups=[np.array([r['seed']==s and r['domain_index']==d for r in base]) for s in range(3) for d in range(3)]
    support=np.stack([w[:,g].sum(1) for g in groups],axis=1)
    valid=(support>0).all(1)
    vals={}
    for policy in policies:
        rr=sorted([r for r in rows if r['policy']==policy],key=lambda r:r['row_index'])
        delta=np.array([r['gain_macro_fg'] for r in rr])
        means=np.stack([np.divide(np.einsum('bi,i->b',w[:,g],delta[g],optimize=False),support[:,j],out=np.full(2000,np.nan),where=support[:,j]>0)
                        for j,g in enumerate(groups)],axis=1)
        vals[policy]={'overall_gain':means.mean(1),'historical_gain':means[:,[0,1,3,4,6,7]].mean(1),
                      'current_drop':np.maximum(0,-means[:,[2,5,8]].mean(1))}
    result=[]
    for policy in policies:
        for measure,values in vals[policy].items():
            for paired in (False,True):
                a=values-vals['SHOR_FROZEN'][measure] if paired else values
                q=np.quantile(a[valid],[.025,.975]) if valid.any() else [None,None]
                result.append(dict(policy=policy,measure=measure,paired_difference_from_SHOR=paired,
                    requested=2000,valid=int(valid.sum()),empty_group_invalid=int((~valid).sum()),
                    lower_025=q[0],upper_975=q[1],seed=2026090801,
                    interpretation='fixed outer predictions, paired patient cluster resampling; no refit'))
    return result


def evaluate(output):
    verify(output,'INPUT_SEAL.json');verify(output,'TARGET_SEAL.json')
    verify(output,'ALL_OUTER_PREDICTIONS_SEALED.json')
    for f in range(5):verify(output,f'fold{f}/PREDICTION_SEAL.json')
    write_json(output/'OUTER_EVALUATION_RESERVATION.json',dict(source_commit=source_state(),
        all_predictions_seal_sha256=digest(output/'ALL_OUTER_PREDICTIONS_SEALED.json'),created_unix=time.time()))
    records=read_json(output/'evaluation_targets.json');baseline=read_json(output/'baseline_records.json')
    meta=read_json(output/'meta.json');shor=np.load(output/'SHOR_routes.npy',allow_pickle=False)
    prob=probabilities(meta)
    rows=[];fits=[];selections=[];fullprivate=[]
    c6_routes=np.load(PARENT/'routes.npy',allow_pickle=False)
    predictions={p:np.full((198,2,3),np.nan) for p in POLICIES if p!='SHOR_CONFIDENCE_VETO'}
    for r in baseline:
        oldkey=next(k for k,v in BASE_POLICIES.items() if v==r['policy'])
        # Baseline hard masks were checked against exact expert argmax in target service.
        route=2 if oldkey=='C0' else int(np.argmax(r['alpha'])) if oldkey=='C1' else int(shor[r['row_index']]) if oldkey=='C3' else int(c6_routes[r['row_index']])
        rows.append({**r,'policy':CONTROL_NAMES[oldkey],'route':route,'shor_route':int(shor[r['row_index']])})
    for f in range(5):
        ix=meta['fold_indices'][str(f)]; d=output/f'fold{f}'
        selection=read_json(d/'INNER_SELECTION.json')
        for variant in ('cf','routed_only'):
            for fit in selection[variant]['fits']:
                fits.append(dict(outer_fold=f,variant=variant,**{k:v for k,v in fit.items() if k!='patient_ids'},
                                  patient_ids_sha256=__import__('hashlib').sha256(__import__('json').dumps(fit['patient_ids'],sort_keys=True).encode()).hexdigest()))
        bundles=read_json(d/'models.json');decision=read_json(d/'decisions.json');pr=read_json(d/'pair_predictions.json')
        for policy,bundle in bundles.items():
            candidate=bundle['candidate']
            selections.append(dict(outer_fold=f,policy=policy,rows=len(ix),
                lambda_selected=None if bundle['model'] is None else bundle['model']['regularization'],
                candidate_id=None if candidate is None else candidate['id'],
                status='CURRENT_FALLBACK_NO_INNER_SAFE_CANDIDATE' if candidate is None else 'INNER_SAFE_CANDIDATE_SELECTED',
                candidate=candidate,model_seal_sha256=digest(d/'MODEL_SEAL.json'),
                prediction_seal_sha256=digest(d/'PREDICTION_SEAL.json')))
            if policy in predictions and pr[policy] is not None:predictions[policy][ix]=np.array(pr[policy])
            masks=np.load(d/(policy+'.npy'),mmap_mode='r',allow_pickle=False)
            require(masks.shape==(len(ix),384,384),'incomplete deployment masks')
            for j,i in enumerate(ix):
                r=records[i];h=decision[policy][j]
                require(h in (2,int(shor[i])),'invalid deployed expert')
                expected=prob[r['seed'],h][meta['probability_positions'][str(i)]].argmax(0)
                require(np.array_equal(masks[j],expected),'deployed mask not exact selected cache')
                after=r['expert_metrics'][h];before=r['expert_metrics'][2]
                rows.append({**r,**after,**gains(before,after),'policy':policy,'route':h,'shor_route':int(shor[i])})
    require(len(rows)==198*8 and len({r['row_index'] for r in records})==198,'incomplete outer rows')
    require(sum(f['actual_design_solves'] for f in fits)>0,'BLOCKED_NO_REAL_FITS')
    require(balanced([r for r in rows if r['policy']=='CURRENT'],'foreground_dice',evaluable_only=True) is not None,'BLOCKED_NO_EVALUABLE_SUPPORT')
    metrics=summarize(rows);table={(r['policy'],r['level'],r['key']):r for r in metrics}
    paired=[]
    for r in metrics:
        ref=table['SHOR_FROZEN',r['level'],r['key']]
        paired.append(dict(level=r['level'],key=r['key'],policy=r['policy'],
            macro_Dice_difference_from_SHOR=r['foreground_dice']-ref['foreground_dice'],
            gain_difference_from_SHOR=r['gain_macro_fg']-ref['gain_macro_fg'],
            rim_difference_from_SHOR=r['rim_dice']-ref['rim_dice'],cup_difference_from_SHOR=r['cup_dice']-ref['cup_dice'],
            gain_retention_relative_SHOR=None if ref['gain_macro_fg']==0 else r['gain_macro_fg']/ref['gain_macro_fg']))
    tails=tail(rows);checks=gates(table,rows)
    passed=all(r['passed'] for r in checks)
    status='PASS_DEVELOPMENT_UTILITY_SIGNAL' if passed else 'DEVELOPMENT_UTILITY_SIGNAL_NOT_ESTABLISHED'
    def value(p,l,k,f='gain_macro_fg'):return table[p,l,k][f]
    primary='SHOR_UV_CF';simple='SHOR_CONFIDENCE_VETO'
    def safety_vector(p):
        return [max(0.,-value(p,'domain','Drishti_GS')),
                max(value(p,'domain','Drishti_GS','rim_drop'),value(p,'domain','Drishti_GS','cup_drop')),
                max([0.]+[-r['gain_macro_fg'] for r in metrics if r['policy']==p and r['level']=='seed_domain']),
                sum(r['domain_index']==2 and r['gain_macro_fg']<-.10 for r in rows if r['policy']==p)]
    no_added=(value(simple,'overall','all')>=value(primary,'overall','all')
              and value(simple,'historical','history')>=value(primary,'historical','history')
              and all(a<=b for a,b in zip(safety_vector(simple),safety_vector(primary))))
    incremental='NO_ADDED_VALUE_OVER_SIMPLE_VETO' if no_added else 'MIXED_TRADEOFF_OR_POSSIBLE_INCREMENTAL_VALUE'
    predictability,pair_private=pair_audit(records,predictions)
    audit=[]
    for domain in range(3):
        for h in (0,1):
            rr=[r for r in records if r['domain_index']==domain and r['has_evaluable_gt']]
            y=np.array([r['targets'][h] for r in rr])
            audit.append(dict(domain=records[next(i for i,r in enumerate(records) if r['domain_index']==domain)]['domain'],
                expert=h,pairs=len(rr),patients=len({r['patient_id'] for r in rr}),
                beneficial_macro=int((y[:,:2].mean(1)>0).sum()),harmful_class=int((y[:,2]>0).sum()),
                harmful_fraction=float((y[:,2]>0).mean()),rim_harm=int((y[:,0]<0).sum()),
                cup_harm=int((y[:,1]<0).sum()),zero_target=int((y==0).all(1).sum()),
                original_SHOR_selected_pairs=int(sum(shor[r['row_index']]==h for r in rr))))
    bad=[r for r in rows if r['harm']>0 or (r['route']<2 and r['route']!=r['domain_index'])]
    write_json(output/'private_failure_rows.json',[{k:r[k] for k in ('policy','row_index','patient_id','case_id','seed','domain','route','shor_route','gain_macro_fg','gain_rim','gain_cup','harm')} for r in bad])
    write_json(output/'private_pair_predictions_and_targets.json',pair_private)
    write_json(output/'private_outer_records.json',rows)
    pub=output/'public'
    write_csv(pub/'OUTER_POLICY_METRICS.csv',metrics)
    write_csv(pub/'PAIRED_COMPARISONS.csv',paired)
    write_csv(pub/'SAFETY_AND_TAIL.csv',tails)
    ablation=[]
    for p in list(CONTROL_NAMES.values())+list(POLICIES):
        o=table[p,'overall','all'];hh=table[p,'historical','history'];curr=table[p,'domain','Drishti_GS']
        t=next(r for r in tails if r['policy']==p and r['level']=='domain' and r['key']=='Drishti_GS')
        ablation.append(dict(policy=p,overall_Dice=o['foreground_dice'],overall_gain=o['gain_macro_fg'],
            historical_gain=hh['gain_macro_fg'],historical_retention=None if value('SHOR_FROZEN','historical','history')==0 else hh['gain_macro_fg']/value('SHOR_FROZEN','historical','history'),
            current_Dice=curr['foreground_dice'],current_drop=max(0.,-curr['gain_macro_fg']),
            current_max_class_drop=max(curr['rim_drop'],curr['cup_drop']),
            current_large_harm_rows=t['case_delta_below_minus_010'],
            current_historical_overrides=t['current_domain_historical_overrides'],
            current_worst_case_delta=t['worst_case_delta'],current_patients=t['patients'],
            fallback_folds=sum(s['policy']==p and s['candidate'] is None for s in selections)))
    write_csv(pub/'ABLATION.csv',ablation)
    write_csv(pub/'GATE_ACCOUNTING.csv',checks)
    write_json(pub/'FIT_ACCOUNTING.json',dict(actual_shared_design_solves=sum(f['actual_design_solves'] for f in fits),
        actual_scalar_head_fits=sum(f['actual_scalar_heads'] for f in fits),fits=fits,
        gain_only_refits=0,gain_only_shares_CF_heads=True,weighting='each participating patient total weight 1 per fit',
        inherited_Ridge_and_SHOR_refits=0,network_training=0))
    write_json(pub/'INNER_SELECTION_SUMMARY.json',selections)
    write_json(pub/'HEAD_PREDICTABILITY.json',predictability)
    write_json(pub/'COUNTERFACTUAL_TARGET_AUDIT.json',dict(rows=198,patients=177,potential_pairs=396,
        supervised_pairs=sum(r['has_evaluable_gt'] for r in records)*2,all_ignore_rows=sum(not r['has_evaluable_gt'] for r in records),
        valid_pixels=sum(r['expert_metrics'][2]['n_valid_pixels'] for r in records),
        ignored_pixels=sum(r['expert_metrics'][2]['n_ignored_pixels'] for r in records),
        group_expert_counts=audit,independent_sample_count=177,grouping='all seed/h duplicates stay with patient'))
    write_csv(pub/'PATIENT_BOOTSTRAP.csv',bootstrap(rows))
    write_json(pub/'STATUS.json',dict(status=status,incremental_comparison=incremental,gates=checks,
        source_commit=source_state(),all_outer_rows=198,policies=8,outer_new_policy_rows=792,
        prediction_seal_sha256=digest(output/'ALL_OUTER_PREDICTIONS_SEALED.json'),
        interpretation='development-exposed cohort; only new heads and selection exclude outer patients',
        old_R2_terminal_unchanged='FAIL_FROZEN_ACTION_SPACE_CAPACITY',
        old_router_refit_p90_p10='NOT_EVALUATED',next_stage_started=False))
    history=read_json(output/'history_private.json')
    verify_files(PARENT,history['parent_files']);verify_files(R2,history['R2_files'])
    original=read_json(output/'INPUT_SEAL.json').get('original_files',{})
    verify_files(FORMAL,original)
    if (output/'parent_terminal_hashes.json').exists():verify_files(PARENT,read_json(output/'parent_terminal_hashes.json'))
    verify_files(ROOT,read_json(DOC/'HISTORY_PROTECTION.json'))
    write_json(pub/'HISTORY_CLOSEOUT.json',dict(protected_source_files=len(read_json(DOC/'HISTORY_PROTECTION.json')),
        parent_files_verified=len(history['parent_files']),R2_artifacts_verified=len(history['R2_files']),
        changed_files=0,old_locks_changed=False,main_merged=False,network_forwards=0,network_updates=0,
        image_reads=0,old_formal_03_reads=0,forbidden_GT_reads=0))
    write_reports(pub,ablation,checks,incremental,status,selections,tails,audit,records,rows)
    write_json(pub/'PRIVATE_ARTIFACT_MANIFEST.json',{'files':{str(p.relative_to(output)):{'sha256':digest(p),'bytes':p.stat().st_size}
        for p in sorted(output.rglob('*')) if p.is_file() and 'public' not in p.relative_to(output).parts},
        'privacy':'generic filenames/hashes only; no patient IDs, masks, targets or individual predictions published'})


def write_reports(pub,ablation,checks,incremental,status,selections,tails,audit,records,rows):
    lines=['# SHOR-UV V0.8 development utility pilot','',
        '**Terminal: '+status+'**','',
        'This is a new development experiment. CARe-HR R2 remains FAIL_FROZEN_ACTION_SPACE_CAPACITY.',
        'The deployed gate sees only 18 prediction/alpha features and the frozen SHOR route; it returns the frozen historical whole-image prediction or current. No oracle action or GT enters deployment.','',
        'Tested source: '+source_state()+'. All five outer patient folds were deployed and sealed before this evaluation. Only the new heads, scaler, lambda and threshold exclude outer patients. The upstream segmenters, descriptor alpha and SHOR used these development patients previously. This is neither independent confirmation nor a strict historical-label-free online protocol.','',
        '| Policy | Overall Dice | Overall gain | Historical gain | History retention | Current drop | Current worst delta | Current delta < -0.10 | Fallback folds |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in ablation:
        lines.append('| '+r['policy']+' | '+' | '.join(('UNDEFINED' if r[k] is None else f'{r[k]:.9f}') for k in ('overall_Dice','overall_gain','historical_gain','historical_retention','current_drop','current_worst_case_delta'))+f" | {r['current_large_harm_rows']} | {r['fallback_folds']} |")
    lines+=['','## All primary numerical gates','',
        '| Category | Gate | Observed | Condition | Pass |','| --- | --- | ---: | --- | --- |']
    for r in checks:lines.append(f"| {r['category']} | {r['gate']} | {r['observed']:.12g} | {r['comparison']} {r['threshold']} | {r['passed']} |")
    lines+=['','Simple-veto comparison: **'+incremental+'**. The flag requires confidence veto to weakly dominate primary overall/history gains and all four safety coordinates; otherwise the tradeoff is reported without claiming superiority.',
        '','## Fit and evidence boundaries','',
        'FIT_ACCOUNTING.json reports every actual weighted Ridge design solve and three scalar outputs. CF and GAIN_ONLY share identical heads; ROUTED_ONLY is fitted separately. Each participating patient has total weight 1 across seed/expert duplicates. Standardization is weighted and fit-split-only. Lambda MSE uses clipped inner OOF predictions; exact ties select larger lambda.',
        'INNER_SELECTION_SUMMARY.json exposes every fold choice/fallback. Full private inner candidate tables and models are sealed on NAS. A fallback emits current and remains in all denominators.',
        'BASELINE_PARITY.json checks 792 case-policy / 3168 scalar comparisons and inherited grouped controls at rtol=0, atol=1e-12. New deployed masks are checked byte-for-byte against the selected cached expert argmax.',
        'The target service read only the 198 authorized own-seed train_labeled labels using the frozen R2 original-bytes/hash/decode loader. Fit services received only their own outer-training package, and fit calls only inner-training pairs. Deployment read no GT/domain fields. Outer evaluation reused target-service metrics after verifying all prediction seals.',
        'SAFETY_AND_TAIL.csv separates seed rows from unique patients, source mismatches from segmentation harm, and macro-beneficial from class-nonharm acceptance. HEAD_PREDICTABILITY.json reports patient-weighted MAE, nonzero gain-sign accuracy and harmful-pair recall. Zero targets are separate.',
        'PATIENT_BOOTSTRAP.csv contains 2000 fixed-policy paired patient-cluster draws with seed 2026090801, including invalid empty-group counts and no redraws. These intervals do not measure retraining stability; old router-refit p90/p10 remain NOT_EVALUATED.',
        'Current-domain support is only 30 seed rows / 28 patients. Zero observed extreme losses cannot establish a strong tail-safety guarantee.',
        'No new image reads, segmentation forward, checkpoint tensor loading, optimizer/backward, EMA/GAS/prototype update, conformal fitting, old formal_03 or forbidden GT access occurred. Old protected source and R1/R2 artifacts were verified unchanged.',
        '','## Stop','',
        'All fixed controls and ablations are complete. No outer-result-driven recovery, tuning, new feature, threshold change, new experiment or external test was started. Any further work requires a separately frozen protocol and independent confirmation.']
    (pub/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    fl=['# Failure and mechanism attribution','',
        'All statements below describe this fixed development cohort and sealed predictions. No known bad-case identity was available to the feature, fitting or deployment APIs.','']
    for p in list(CONTROL_NAMES.values())+list(POLICIES):
        rr=[r for r in rows if r['policy']==p]
        source=[r for r in rr if r['route']<2 and r['route']!=r['domain_index']]
        cur=[r for r in source if r['domain_index']==2]
        severe=[r for r in rr if r['gain_macro_fg']<-.10]
        fl.append(f"- {p}: historical-route source mismatches {len(source)} seed rows / {len({r['patient_id'] for r in source})} patients; current-domain subset {len(cur)} rows / {len({r['patient_id'] for r in cur})} patients; all-domain macro delta < -0.10 in {len(severe)} rows / {len({r['patient_id'] for r in severe})} patients.")
    fl+=['','The full private failure table includes every class-harm or source-mismatch row in every policy, across both historical and current domains. Identity checks use actual frozen route/domain records; they are not inferred from 154/155. Source error does not itself imply segmentation harm.',
        '','Target composition (two counterfactual experts, not independent patients):']
    for a in audit:fl.append(f"- {a['domain']} expert {a['expert']}: {a['harmful_class']}/{a['pairs']} class-harm pairs; rim {a['rim_harm']}, cup {a['cup_harm']}; macro-beneficial {a['beneficial_macro']}; frozen SHOR selected {a['original_SHOR_selected_pairs']}.")
    fl+=['','Actual incremental comparison: '+incremental+'. Consult ablation and paired tables rather than treating a better ablation as a main-method win.',
         '','Failed gates: '+', '.join(r['category']+':'+r['gate'] for r in checks if not r['passed'])+'.',
         '','The H head predicts a point value, not an upper confidence bound. Correcting a rare source mistake is distinct from reliably predicting its class loss. Small current-domain support and prior development exposure limit every safety/utility conclusion.']
    (pub/'FAILURE_ATTRIBUTION.md').write_text('\n'.join(fl)+'\n')
