"""Complete-evidence B1--B7 compiler; comparators never rescue the primary map."""
import csv
import math
from pathlib import Path

import numpy as np

from di_dmpa_gate1.gate1a_reporting import write_csv
from .binding import DOMAINS, IncompleteEvidence, InvalidTransportOutput, require, sha256, check_hash, read_json, write_json, write_text


def complete(condition, message):
    if not condition:raise IncompleteEvidence(message)


def all_finite(value):
    if isinstance(value,dict):return all(all_finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(all_finite(v) for v in value)
    return math.isfinite(value) if isinstance(value,(float,np.floating)) else True


def relative(reference, candidate, threshold, *, improvement):
    require(reference>=0 and candidate>=0 and all_finite([reference,candidate]), 'invalid relative metric')
    defined=reference>0 or candidate==0
    change=((reference-candidate)/reference if improvement else (candidate-reference)/reference) if reference>0 else (0. if candidate==0 else None)
    bound=reference*((1-threshold) if improvement else (1+threshold))
    # Compare the equivalent raw-error inequality: no rounding or tolerance at a gate.
    passed=defined and candidate<=bound and (reference>0 or not improvement or threshold<=0)
    return dict(reference=reference,candidate=candidate,relative_change=change,relative_change_defined=defined,
        threshold=threshold,comparison_lhs=candidate,comparison_rhs=bound,pass_=bool(passed))


def admission(transports, oracles):
    complete(len(transports)==6 and {(t['seed'],t['stage_index']) for t in transports}=={(s,t) for s in range(3) for t in (1,2)}, 'six unique transitions required')
    complete(len(oracles)==9 and {(o['seed'],o['source_stage'],o['target_stage']) for o in oracles}=={(s,a,b) for s in range(3) for a,b in ((0,1),(1,2),(0,2))}, 'nine unique oracle units required')
    for o in oracles:
        for method in ('T0','T1','T2'):
            complete([r['class_id'] for r in o['metrics'][method]['class_angles']]==[0,1,2], 'oracle class coverage incomplete')
    if not all_finite([transports,oracles]):raise InvalidTransportOutput('B7 nonfinite reported value')
    B1=[];B2=[];B4=[];B5=[];B6=[];angular={m:[] for m in ('T0','T2')}
    for stage in (1,2):
        ts=sorted((t for t in transports if t['stage_index']==stage),key=lambda x:x['seed'])
        errors={m:[t['feature_errors']['holdout'][m]['full_null_aware_support_error'] for t in ts] for m in ('T0','T2')}
        B1.append(dict(stage_index=stage,domain=DOMAINS[stage],seed_errors=errors,
            **relative(float(np.mean(errors['T0'])),float(np.mean(errors['T2'])),.15,improvement=True)))
        strict=[s for s,(x,y) in enumerate(zip(errors['T0'],errors['T2'])) if y<x]
        B2.append(dict(stage_index=stage,improving_seeds=strict,improving_seed_count=len(strict),threshold=2,pass_=len(strict)>=2))
    for o in sorted(oracles,key=lambda x:(x['source_stage'],x['target_stage'],x['seed'])):
        ids={k:o[k] for k in ('seed','source_stage','target_stage')};immediate=o['target_stage']==o['source_stage']+1
        for c in (1,2):
            values={m:o['metrics'][m]['class_angles'][c]['mean_angular_error'] for m in ('T0','T2')}
            row=dict(ids,class_id=c,**relative(values['T0'],values['T2'],.05,improvement=False))
            if immediate:
                B4.append(row)
                for m in angular:angular[m].append(values[m])
            else:B6.append(row)
        ref=o['metrics']['T0']['accuracy']['macro_accuracy'];candidate=o['metrics']['T2']['accuracy']['macro_accuracy']
        require(0<=ref<=1 and 0<=candidate<=1, 'invalid macro accuracy')
        B5.append(dict(ids,T0_macro_accuracy=ref,T2_macro_accuracy=candidate,absolute_accuracy_drop=ref-candidate,
            maximum_drop=.005,comparison_lhs=candidate,comparison_rhs=ref-.005,pass_=candidate>=ref-.005))
    complete(len(angular['T0'])==len(B4)==12 and len(B5)==9 and len(B6)==6, 'gate unit coverage changed')
    B3=dict(unit_count=12,unit_weight='equal seed/transition/foreground class',
        **relative(float(np.mean(angular['T0'])),float(np.mean(angular['T2'])),.10,improvement=True))
    gates={name:dict(pass_=all(r['pass_'] for r in rows),units=rows) for name,rows in [('B1',B1),('B2',B2),('B4',B4),('B5',B5),('B6',B6)]}
    gates['B3']=B3;gates['B7']=dict(pass_=True,all_defined_values_finite=True)
    gates={k:gates[k] for k in sorted(gates)}
    passed=all(g['pass_'] for g in gates.values())
    return dict(B1_B7=gates,transport_status='PASS_TRANSPORT_SUPPORTED' if passed else 'FAIL_TRANSPORT_NOT_SUPPORTED',
        selected_transport='T2_residual_full_linear' if passed else 'T0_identity',
        selected_transport_role='PRIMARY_ADMITTED_TRANSPORT' if passed else 'DOWNSTREAM_FALLBACK_ONLY_NOT_EXECUTED',
        T1_can_rescue=False,admission_metric='full null-aware support error; never AA-conditional',threshold_tolerance=0.)


def validate_evidence(output, metadata, transports, oracles, census):
    output=Path(output)
    complete(census['paired_units_completed']==12 and census['registered_count']==638976, 'paired census incomplete')
    require(census['metadata']==metadata and census['all_finite'] and census['all_rows_preserved'] and not census['labels_read'], 'invalid census')
    barrier=read_json(output/'TRANSFORM_START_BARRIER.json');oracle_barrier=read_json(output/'ORACLE_START_BARRIER.json')
    require(barrier['status']=='PASS' and barrier['transport_optimizer_steps']==0 and oracle_barrier['transport_optimizer_steps']==6000, 'barrier violation')
    for rel,digest in barrier['evidence_sha256'].items():check_hash(output/rel,digest)
    audit=read_json(output/'GATE1B_V2_MODEL_IMMUTABILITY_AUDIT.json')
    require(audit['status']=='PASS' and audit['all9_B0_disk_hashes_unchanged'] and audit['all_model_states_unchanged'], 'model mutation')
    for row in transports:
        check_hash(output/'transport_models'/f'seed{row["seed"]}_stage{row["stage_index"]}.json',
            oracle_barrier['model_sha256'][f'seed{row["seed"]}_stage{row["stage_index"]}.json'])
        require(row['metadata']==metadata and row['role']=='train_unlabeled' and not row['oracle_or_GT_access'], 'transport provenance/leakage')
        complete(row['models']['T2']['optimizer_steps']==1000 and row['trace_rows']==1001, 'incomplete transport steps')
        require(row['models']['T0']['optimizer_steps']==row['models']['T1']['optimizer_steps']==row['model_optimizer_steps']==0, 'model/comparator optimizer used')
        check_hash(output/row['trace_path'],row['trace_sha256'])
        with (output/row['trace_path']).open(newline='') as handle:trace=list(csv.DictReader(handle))
        complete([int(r['step']) for r in trace]==list(range(1001)), 'missing or duplicate trace step')
        for r in trace:
            require(r['finite']=='True' and r['gradient_finite']=='True', 'nonfinite trace flag')
            if not all(math.isfinite(float(v)) for k,v in r.items() if k not in ('finite','gradient_finite','gradient_evaluated')):
                raise InvalidTransportOutput('nonfinite trace value')
        for partition in ('fit','holdout'):
            errors=row['feature_errors'][partition]
            for m in ('T0','T1','T2'):
                require(errors[m]['counts']==errors['T0']['counts'] and errors[m]['mass']==errors['T0']['mass'], 'map-dependent support')
                require(errors[m]['all_original_rows_used']==errors[m]['registered_count'], 'dropped feature rows')
    for o in oracles:
        require(o['metadata']==metadata and o['role']=='val' and o['gt_consumer']=='diagnostic_evaluator_only' and
            not o['transform_fit_called'] and not o['operational_refit'] and not o['oracle_GT_used_for_transform_fit'], 'oracle isolation failure')
        require(o['model_optimizer_steps']==o['transport_optimizer_steps_in_evaluator']==0 and o['test_gt_usage']=='none', 'forbidden oracle action')
        complete(len(o['oracle_fits'])==len(o['class_caches'])==3, 'incomplete oracle classes')
        require(all(f['K']==2 and len(f['restarts'])==5 and f['source_stage_for_clustering_seeds']==o['source_stage'] for f in o['oracle_fits']), 'oracle source-stage/K changed')
        from di_dmpa_gate1_v2.features import load_cache
        for cache in o['class_caches']:load_cache(output,cache)
    complete(sum(t['models']['T2']['optimizer_steps'] for t in transports)==6000, 'total steps must be 6000')
    require(all_finite([transports,oracles,census]), 'nonfinite evidence')


def report(output, metadata, transports, oracles, census):
    output=Path(output);validate_evidence(output,metadata,transports,oracles,census);decision=admission(transports,oracles)
    feature=[];prototype=[];chain=[];accuracy=[];spectra=[];trace=[]
    for t in sorted(transports,key=lambda t:(t['seed'],t['stage_index'])):
        ids={k:t[k] for k in ('seed','stage_index','domain')}
        for partition,errors in t['feature_errors'].items():
            for method,values in errors.items():feature.append(dict(ids,partition=partition,method=method,**values))
        for method,values in t['spectra'].items():spectra.append(dict(ids,method=method,**values))
        with (output/t['trace_path']).open(newline='') as handle:
            trace.extend(dict(ids,**r) for r in csv.DictReader(handle))
    warnings=[]
    for o in sorted(oracles,key=lambda o:(o['seed'],o['source_stage'],o['target_stage'])):
        ids={k:o[k] for k in ('seed','source_stage','target_stage','source_domain','kind')}
        for method,values in o['metrics'].items():
            for a in values['class_angles']:
                (chain if o['kind']=='chain' else prototype).append(dict(ids,method=method,foreground_macro_angular_error=values['foreground_macro_angular_error'],**a))
            accuracy.append(dict(ids,method=method,**values['accuracy']))
        for c,f in enumerate(o['oracle_fits']):
            for r in f['restarts']:
                if not r['converged']:warnings.append(dict(ids,class_id=c,restart=r['restart'],kind='oracle_restart_iteration_cap',selected=r['restart']==f['selected_restart']))
            if not all(f['active']):warnings.append(dict(ids,class_id=c,kind='inactive_oracle_slot',active=f['active']))
    for name,rows in [('transport_feature_error_v2.csv',feature),('transport_prototype_error_v2.csv',prototype),
        ('transport_chain_error_v2.csv',chain),('transport_prototype_accuracy_v2.csv',accuracy),('transport_spectrum_v2.csv',spectra),('transport_fit_trace.csv',trace)]:
        write_csv(output/name,rows)
    detail=dict(metadata=metadata,**decision,transports=transports,oracle_units=oracles,paired_support=census,
        warnings=warnings,transport_optimizer_steps=6000,model_optimizer_steps=0,report_commit_resolution='Git commit first adding these exact report bytes; recorded in external publication receipt')
    write_json(output/'TRANSPORT_FEASIBILITY_DIAGNOSTIC_V2.json',detail)
    status=dict(metadata,**decision,transport_optimizer_steps=6000,model_optimizer_steps=0,six_transition_seed_completion=[{k:t[k] for k in ('seed','stage_index','domain')} for t in transports],
        paired_units_completed=12,oracle_units_completed=9,paired_support_census=dict(registered_count=census['registered_count'],counts=census['counts'],units=census['units']),
        model_checkpoint_immutability='PASS',report_commit=None,report_commit_resolution=detail['report_commit_resolution'],
        diagnostic_report_sha256=sha256(output/'TRANSPORT_FEASIBILITY_DIAGNOSTIC_V2.json'),warnings=warnings,
        method_registered=False,di_dmpa_training_launched=False,Gate1C=False,next_action='STOP_FOR_INDEPENDENT_REVIEW')
    write_json(output/'GATE1B_V2_STATUS.json',status)
    lines=['# Gate 1B v2 transport diagnostic','',f"Result: **{decision['transport_status']}**.",'',
        'Primary: B0 previous/current EMA, frozen UNet decoder.dec1 post-ReLU16-D; K=2. Gate1A v2 was not rerun.',
        '',f"Selected transport: `{decision['selected_transport']}` ({decision['selected_transport_role']}). T1 is only a comparator.",'',
        '| Gate | Result |','| --- | --- |']
    lines += [f"| {k} | {'PASS' if g['pass_'] else 'FAIL'} |" for k,g in decision['B1_B7'].items()]
    lines += ['', 'B1 held-out full-support error (unrounded):','']
    for b in decision['B1_B7']['B1']['units']:
        lines.append(f"- {b['domain']}: T0={b['reference']!r}, T2={b['candidate']!r}, relative reduction={b['relative_change']!r}.")
    b=decision['B1_B7']['B3'];lines+=['',f"B3 immediate foreground angular mean: T0={b['reference']!r}, T2={b['candidate']!r}, relative reduction={b['relative_change']!r}.",'',
        'All B4/B5/B6 per-unit values, T0/T1/T2 metrics, support masses, oracle convergence and full optimizer traces are in the JSON/CSVs.',
        '',f"Coverage:12 paired units /638976 registered pairs;6/6 maps;9/9 historical-val evaluator units;6000 transport updates;0 segmentation model updates. Pair counts: {census['counts']}.",
        '',f"Oracle warning records: {len(warnings)}; no restart/iteration extension or replacement. Finite nulls remain in all applicable metrics.",
        '', 'GT usage: current-domain train_unlabeled fit has no label access; historical-val membership is diagnostic_evaluator_only; hidden-GT training=none; final-test-GT=none.',
        '', 'Frozen inputs, complete model/classifier/GAS/buffer state and all9 B0 checkpoint files are unchanged. Exact-code tests and every cache/artifact hash accompany this report.',
        '', 'This is mechanism admission only, not evidence of segmentation performance improvement. No method registration, Gate1C, reliability, gradient conflict, teacher noise, theory final, training, other benchmarks or main merge.',
        '',f"Freeze commit `{metadata['gate1a_v2_freeze_commit']}`; preregistration `{metadata['preregistration_commit']}`; authorization `{metadata['authorization_commit']}`; exact code `{metadata['diagnostic_code_commit']}`.",
        '', 'Report commit: resolve the commit first adding these exact bytes; the publication receipt records it separately to avoid a self-referential hash.',
        '', '**STOP_FOR_INDEPENDENT_REVIEW**.']
    text='\n'.join(lines)+'\n'
    write_text(output/'TRANSPORT_FEASIBILITY_DIAGNOSTIC_V2.md',text)
    write_text(output/'GATE1B_V2_FINAL_REPORT.md',text)
    return warnings


def artifact_manifest(output):
    output=Path(output);excluded={'GATE1B_V2_ARTIFACT_MANIFEST.json','GATE1B_V2_ARTIFACT_MANIFEST.sha256'}
    artifacts=[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=sha256(p))
        for p in sorted(output.rglob('*')) if p.is_file() and p.name not in excluded]
    manifest=dict(scope='GATE1B_V2_ONLY',artifacts=artifacts,file_count=len(artifacts),total_bytes=sum(a['bytes'] for a in artifacts),
        raw_feature_tensors='remote_only; public copies retain descriptors and hashes')
    digest=write_json(output/'GATE1B_V2_ARTIFACT_MANIFEST.json',manifest)
    write_text(output/'GATE1B_V2_ARTIFACT_MANIFEST.sha256',digest+'  GATE1B_V2_ARTIFACT_MANIFEST.json\n')
    return manifest
