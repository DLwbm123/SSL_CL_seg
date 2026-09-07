"""R2 engineering recovery. R1 scientific orchestration below is copied unchanged except I/O/counters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback
import numpy as np

from . import recovery_r2 as recovery
from .execute_r1 import (ROOT, DOC, DATA, FORMAL, BASE_POLICIES, REFERENCE, require,
                         source_state, original_inputs, check_original, write_csv, load_prepared)
from .io_r1 import digest, read_json, write_json, verify_files
from .io_r2 import counters, dataset_asset, load_label, projected_domains


def evaluate(output):
    from .scoring_r1 import case_metrics, gains
    from .oracle_r1 import score_actions, select_oracles, pareto_actions
    from .summary_r1 import FIELDS, DOMAINS, legacy_aggregate, summarize, terminal, groups_for
    start=time.monotonic(); seal=recovery.verify_recovery(output); source=source_state()
    admission=read_json(output/'ASSET_BINDING_ADMISSION.json')
    require(admission['status']=='PASS_ALL_198_STAT_ONLY' and admission['source_commit']==source
            and admission['R2_seal_sha256']==digest(output/'ACTION_SPACE_SEAL.json')
            and admission['allowlist_sha256']==digest(output/'RESOLVED_ASSET_ALLOWLIST.json')
            and admission['preflight_sha256']==digest(output/'public'/'ASSET_BINDING_PREFLIGHT.json'),'asset admission mismatch')
    preflight=read_json(output/'public'/'ASSET_BINDING_PREFLIGHT.json')
    require(preflight['rows_admitted']==198 and preflight['errors']==0,'incomplete asset admission')
    expected,entries=original_inputs();meta=read_json(recovery.PARENT/'input_metadata.json');rows=read_json(recovery.PARENT/'blind_rows.json')
    allowlist=read_json(output/'RESOLVED_ASSET_ALLOWLIST.json')
    require([r['row_index'] for r in allowlist]==list(range(198)),'asset allowlist order mismatch')
    write_json(output/'EVALUATOR_ACCESS_RESERVATION.json',dict(schema_version='R2',source_commit=source,
      action_generation_source_commit=recovery.ACTION_SOURCE,parent_seal_sha256=recovery.PARENT_HASH,
      seal_sha256=digest(output/'ACTION_SPACE_SEAL.json'),asset_admission_sha256=digest(output/'ASSET_BINDING_ADMISSION.json'),
      created_unix=time.time(),new_GT_payload_reads_before_reservation=0,new_true_domain_records_before_reservation=0,
      prior_exposure=seal['prior_exposure'],status='RESERVED_SINGLE_R2_EVALUATOR'))
    (output/'private_evaluation').mkdir()
    count=counters();count['asset_metadata_rows_checked']=preflight['rows_checked'];count['asset_unique_files_stat_checked']=preflight['unique_physical_assets']
    status='INCOMPLETE_EVALUATION';all_rows=[];labels={};revealed={};percase=[];baseline_report={};decoded_files=set()
    try:
        for seed in range(3):
            wanted={r['case_id']:r for r in rows if r['seed']==seed}
            columns=('case_id','patient_id','primary_20pct_split','dataset','site_or_vendor','label_h5_relpath','label_sha256','image_h5_relpath','image_sha256')
            projected=projected_domains(DATA/'manifests/training'/f'lcrseg_v1_seed{seed}.csv',columns,wanted,count)
            require(len(projected)==66,'domain projection population mismatch')
            for row in projected:
                require(row['case_id'] in wanted,'unexpected metadata case')
                blindrow=wanted[row['case_id']];i=blindrow['row_index'];allowed=allowlist[i]
                require(row['primary_20pct_split']=='train_labeled' and row['dataset']=='fundus' and row['label_h5_relpath'] and row['site_or_vendor'] in DOMAINS,'GT role/domain invalid')
                require(row['patient_id']==blindrow['patient_id'] and i not in revealed,'evaluator lineage/duplicate mismatch')
                require(all(row[k]==allowed[k] for k in ('case_id','patient_id','primary_20pct_split','label_h5_relpath','label_sha256'))
                        and seed==allowed['seed'] and str(dataset_asset(DATA,row['label_h5_relpath']))==allowed['canonical_path'],'metadata differs from asset admission')
                count['true_domain_rows_processed']+=1
                revealed[i]=dict(seed=seed,case_id=row['case_id'],patient_id=row['patient_id'],domain=row['site_or_vendor'],domain_index=DOMAINS.index(row['site_or_vendor']),row_index=i)
                labels[i],_=load_label(DATA,row,count,decoded_files)
                case_metrics(np.zeros((384,384),dtype=np.uint8),labels[i])
        require(set(revealed)==set(range(198)) and count['GT_rows_bound_to_validated_labels']==198,'incomplete evaluator population')
        # This mixed historical score file is opened only inside the admitted evaluator.
        check_original(entries,'case_metrics.jsonl')
        with (FORMAL/'case_metrics.jsonl').open() as f: original=[json.loads(line) for line in f if line.strip()]
        lookup={(r['seed'],r['case_id'],r['policy']):r for r in original if r['policy'] in BASE_POLICIES}
        originals=[]; baseline=[]; max_error=0.0
        # Preserve the original fold, case, policy append sequence and full grouping function.
        for fold in range(5):
            arrays={p:np.load(FORMAL/meta['control_paths'][f'{fold}:{p}'],mmap_mode='r',allow_pickle=False) for p in BASE_POLICIES}
            for pos,i in enumerate(meta['fold_indices'][str(fold)]):
                before=case_metrics(arrays['C0'][pos],labels[i])
                for p,name in BASE_POLICIES.items():
                    metrics=case_metrics(arrays[p][pos],labels[i]); old=lookup[rows[i]['seed'],rows[i]['case_id'],p]
                    require(old['domain']==revealed[i]['domain'] and old['domain_index']==revealed[i]['domain_index'],'baseline domain identity mismatch')
                    error=max(abs(metrics[k]-old[k]) for k in FIELDS); max_error=max(max_error,error)
                    require(error<=1e-12,'BLOCKED_BASELINE_MISMATCH: per-case metrics')
                    baseline.append({**revealed[i],**metrics,**gains(before,metrics),'policy':name})
                    originals.append({**old,'policy':name})
                    count['baseline_case_policy_comparisons_completed']+=1
                    count['baseline_scalar_comparisons_completed']+=4
        oldagg=legacy_aggregate(originals); newagg=legacy_aggregate(baseline)
        require(len(oldagg)==len(newagg),'baseline aggregation shape')
        for old,new in zip(oldagg,newagg):
            require(all(old[k]==new[k] for k in ('level','key','policy','cases')),'baseline aggregation keys')
            error=max(abs(old[k]-new[k]) for k in FIELDS); max_error=max(max_error,error)
            require(error<=1e-12,'BLOCKED_BASELINE_MISMATCH: grouped metric')
        for r in newagg:
            if r['level']=='overall': require(abs(r['foreground_dice']-REFERENCE[r['policy']])<=1e-12,'BLOCKED_BASELINE_MISMATCH: frozen control reference')
        baseline_report=dict(status='PASS_BASELINE_PARITY',case_policy_comparisons=len(baseline),scalar_comparisons=len(baseline)*4,grouped_rows=len(newagg),maximum_absolute_difference=max_error,atol=1e-12,rtol=0,original_aggregation='AST exact ppc_shor_v0_6a.aggregate_case_metrics; original fold/case/policy order')
        write_json(output/'public'/'BASELINE_PARITY.json',baseline_report)
        all_rows=baseline
        with (output/'private_evaluation'/'pareto.jsonl').open('x') as pareto_file:
            for fold in range(5):
                for i in meta['fold_indices'][str(fold)]:
                    prepared=load_prepared(recovery.PARENT,i); actions=score_actions(prepared,labels[i]); selected=select_oracles(actions)
                    frontier=pareto_actions(actions)
                    pareto_file.write(json.dumps({'row_index':i,'actions':frontier},allow_nan=False)+'\n')
                    for policy,score in selected.items():
                        all_rows.append({**revealed[i],**score,'policy':policy})
                    percase.append({**revealed[i],'pareto_actions':len(frontier),'evaluated_actions':len(actions),
                                    'maximum_free_gain':selected['O_FREE_SUBSET_envelope']['gain_macro_fg'],
                                    'cap_to_no_area':selected['O_NO_AREA_envelope']['gain_macro_fg']-selected['O_CAP_envelope']['gain_macro_fg'],
                                    'no_area_to_free':selected['O_FREE_SUBSET_envelope']['gain_macro_fg']-selected['O_NO_AREA_envelope']['gain_macro_fg']})
                    count['oracle_case_rows_completed']+=1
                    if len(percase)%25==0: print(json.dumps({'phase':'oracle','completed_rows':len(percase)}),flush=True)
        metrics=summarize(all_rows)
        # Cross-check new main grouping with the literal inherited full aggregator.
        inherited=legacy_aggregate(all_rows); m={(r['level'],r['key'],r['policy']):r for r in metrics}
        require(all(abs(r[k]-m[r['level'],r['key'],r['policy']][k])<=1e-12 for r in inherited for k in FIELDS),'BLOCKED_BASELINE_MISMATCH: summary call chain')
        write_csv(output/'public'/'CAPACITY_METRICS.csv',metrics)
        # Preserve every original draft gate; values are descriptive oracle diagnostics.
        gates=read_json(DOC/'R1_PREREGISTRATION.json')['original_draft_gates_preserved']
        index={(r['policy'],r['level'],r['key']):r for r in metrics}
        domain_oracle=legacy_aggregate([r for r in original if r['policy']=='C7'])
        domain_oracle_score=next(r['foreground_dice'] for r in domain_oracle if r['level']=='overall')
        def gate_values(policy):
            overall=index[policy,'overall','all']; history=index[policy,'historical','history']; current=index[policy,'domain',DOMAINS[2]]
            current_cases=[r for r in all_rows if r['policy']==policy and r['domain_index']==2 and r['has_evaluable_gt']]
            seed_domain=[r for r in metrics if r['policy']==policy and r['level']=='seed_domain']
            return {'three_domain_gain_min':overall['gain_macro_fg'],'historical_gain_min':history['gain_macro_fg'],
               'REFUGE_gain':index[policy,'domain',DOMAINS[0]]['gain_macro_fg'],'RIM_ONE_r3_gain':index[policy,'domain',DOMAINS[1]]['gain_macro_fg'],
               'positive_seed_count':sum(index[policy,'seed',str(s)]['gain_macro_fg']>0 for s in range(3)),
               'domain_oracle_gap_max':domain_oracle_score-overall['foreground_dice'],
               'current_domain_drop_max':max(0.,-current['gain_macro_fg']),
               'maximum_current_class_drop_max':max(current['rim_drop'],current['cup_drop']),
               'maximum_seed_domain_drop_max':max([0.]+[-r['gain_macro_fg'] for r in seed_domain]),
               'current_domain_cases_with_delta_below_minus_0_10':sum(r['gain_macro_fg']<-.1 for r in current_cases),
               'worst_current_domain_case_delta_min':min((r['gain_macro_fg'] for r in current_cases),default=None),
               'overall_gain_difference_min':overall['gain_minus_frozen_PPC'],'historical_gain_difference_min':history['gain_minus_frozen_PPC']}
        gate_rows=[]; ppc_values=gate_values('frozen_PPC_C6')
        for policy in ('O_CAP_envelope','O_SAFE0_envelope'):
            values=gate_values(policy)
            values['current_drop_reduction_min']=ppc_values['current_domain_drop_max']-values['current_domain_drop_max']
            values['maximum_seed_domain_drop_reduction_min']=ppc_values['maximum_seed_domain_drop_max']-values['maximum_seed_domain_drop_max']
            for group,definitions in gates.items():
                if not isinstance(definitions,dict):continue
                for name,threshold in definitions.items():
                    observed=values.get(name)
                    met=None if observed is None else (observed>0 if threshold=='greater_than_zero' else
                        observed==threshold if name in ('positive_seed_count','current_domain_cases_with_delta_below_minus_0_10') else
                        observed<=threshold if name.endswith('_max') else observed>=threshold)
                    gate_rows.append(dict(policy=policy,gate=group+'.'+name,original_threshold=threshold,observed=observed,
                         status='NOT_EVALUATED' if met is None else 'DESCRIPTIVE_ORACLE_MEETS' if met else 'DESCRIPTIVE_ORACLE_DOES_NOT_MEET',
                         method_pass=False,refit_or_calibration_guarantee=False))
        write_csv(output/'public'/'DRAFT_GATE_ACCOUNTING.csv',gate_rows)
        write_csv(output/'public'/'EVALUABLE_ONLY_SENSITIVITY.csv',summarize(all_rows,evaluable_only=True))
        write_csv(output/'public'/'PATIENT_EQUAL_SENSITIVITY.csv',summarize(all_rows,patient_equal=True))
        # Independent patient-equal sensitivity collapses repeated cross-seed observations first.
        global_patient=[]
        for level,key,subset in groups_for(all_rows):
            for policy in dict.fromkeys(r['policy'] for r in subset):
                patients={}
                for r in subset:
                    if r['policy']==policy: patients.setdefault(r['patient_id'],[]).append(r)
                global_patient.append(dict(level=level,key=key,policy=policy,patients=len(patients),
                    foreground_dice=float(np.mean([np.mean([r['foreground_dice'] for r in rr]) for rr in patients.values()])),
                    gain_macro_fg=float(np.mean([np.mean([r['gain_macro_fg'] for r in rr]) for rr in patients.values()])),
                    weighting='equal_patient_after_mean_across_seed_case_observations; auxiliary_not_primary'))
        write_csv(output/'public'/'PATIENT_CROSS_SEED_EQUAL_SENSITIVITY.csv',global_patient)
        coverage=[]
        for level,key,subset in groups_for([r for r in baseline if r['policy']=='current']):
            invalid=[r for r in subset if not r['has_evaluable_gt']]
            coverage.append(dict(level=level,key=key,rows=len(subset),patients=len({r['patient_id'] for r in subset}),
                  evaluable_rows=len(subset)-len(invalid),all_ignore_rows=len(invalid),all_ignore_patients=len({r['patient_id'] for r in invalid}),
                  valid_pixels=sum(r['n_valid_pixels'] for r in subset),ignored_pixels=sum(r['n_ignored_pixels'] for r in subset),
                  compatibility='all-ignore score1/gain0 retained; no safety evidence'))
        write_csv(output/'public'/'EVALUATION_COVERAGE.csv',coverage)
        diagnostics=read_json(recovery.PARENT/'blind_diagnostics.json'); budgets=[]
        for policy in dict.fromkeys(r['policy'] for r in all_rows if r['policy'].startswith('O_')):
            selected=[r for r in all_rows if r['policy']==policy]; observed=[r for r in selected if r['has_evaluable_gt'] and r['indices']]
            budgets.append(dict(policy=policy,rows=len(selected),no_op_rows=sum(not r['indices'] for r in selected),
                observed_selected_rows=len(observed),observed_nonharm_rows=sum(r['harm']==0 for r in observed),
                correction_precision_numerator=sum(r['gain_macro_fg']>0 for r in observed),correction_precision_denominator=len(observed),
                original_mask_pixels=sum(r['mask_union_pixels'] for r in selected),hard_changed_pixels=sum(r['hard_label_changed_pixels'] for r in selected),
                probability_changed_pixels=sum(r['probability_changed_pixels'] for r in selected),GT_valid_hard_changed_pixels=sum(r['GT_valid_hard_changed_pixels'] for r in selected),
                proposals_total=sum(r['proposals'] for r in diagnostics),strict_actions_total=sum(r['strict_actions'] for r in diagnostics),
                no_area_actions_total=sum(r['no_area_actions'] for r in diagnostics),free_actions_total=sum(r['free_actions'] for r in diagnostics),
                quantity_rejected_total=sum(r['quantity_rejected'] for r in diagnostics),foreground_area_rejected_total=sum(r['foreground_area_rejected'] for r in diagnostics),
                image_area_rejected_total=sum(r['image_area_rejected'] for r in diagnostics),zero_current_foreground_rows=sum(r['zero_current_foreground'] for r in diagnostics)))
        write_csv(output/'public'/'BUDGET_ATTRIBUTION.csv',budgets)
        attribution=[]
        for level,key,subset in groups_for(percase):
            from .summary_r1 import balanced
            attribution.append(dict(level=level,key=key,rows=len(subset),cap_to_no_area_gain=balanced(subset,'cap_to_no_area'),
                 no_area_to_free_gain=balanced(subset,'no_area_to_free'),pareto_actions_total=sum(r['pareto_actions'] for r in subset),
                 exhaustive_actions_total=sum(r['evaluated_actions'] for r in subset)))
        write_csv(output/'public'/'ORACLE_CAPACITY_ATTRIBUTION.csv',attribution)
        write_json(output/'private_evaluation'/'selected_case_metrics.json',all_rows)
        status=terminal(all_rows)
    except Exception as exc:
        text=str(exc)
        status='BLOCKED_BASELINE_MISMATCH' if 'BASELINE' in text or 'baseline' in text else 'INCOMPLETE_EVALUATION'
        trace=output/'private_evaluation'/'traceback.txt';trace.write_text(traceback.format_exc())
        write_json(output/'public'/'EXECUTION_ERROR.json',{'exception':type(exc).__name__,'stage':'R2_evaluator',
            'reason':'baseline mismatch' if status=='BLOCKED_BASELINE_MISMATCH' else 'see private traceback for exact I/O/integrity failure',
            'private_traceback_sha256':digest(trace),'post_GT_code_changes_allowed':False})
        raise
    finally:
        runtime={**count,'elapsed_seconds':time.monotonic()-start,'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'bootstrap_p90':'NOT_EVALUATED','feasible_refits':'NOT_EVALUATED','prior_R1_exposure':seal['prior_exposure']}
        write_json(output/'public'/'RUNTIME_COUNTERS.json',runtime)
        write_json(output/'public'/'CAPACITY_STATUS.json',dict(status=status,evaluator_source_commit=source,
            action_generation_source_commit=recovery.ACTION_SOURCE,cohort_expected={'rows':198,'patients':177},
            metadata_revealed=count['true_domain_records_materialized'],metadata_rows_processed=count['true_domain_rows_processed'],
            GT_validated=count['GT_rows_bound_to_validated_labels'],baseline_completed=count['baseline_case_policy_comparisons_completed'],
            oracle_completed=count['oracle_case_rows_completed'],baseline_parity=baseline_report,all_terminal_states_stop=True,
            next_stage_started=False,action_seal_sha256=digest(output/'ACTION_SPACE_SEAL.json'),
            evaluator_reservation_sha256=digest(output/'EVALUATOR_ACCESS_RESERVATION.json')))
    print(json.dumps({'phase':'evaluate_R2','status':status,'GT_successful_decodes':count['GT_label_decodes_completed'],'oracle_rows':len(percase)}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('qualify','preflight','evaluate'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--qualification',type=Path)
    args=parser.parse_args()
    require(args.output.resolve().is_relative_to(Path('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg').resolve()),'output must be NAS')
    require(Path(os.environ.get('TMPDIR','')).resolve().is_relative_to(Path('/data_nas').resolve()),'launch through NAS wrapper')
    if args.phase=='qualify':recovery.qualify(args.output,args.qualification)
    elif args.phase=='preflight':recovery.preflight(args.output)
    else:evaluate(args.output)


if __name__=='__main__':main()
