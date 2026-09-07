"""Two separately launched phases: blind seal, then isolated evaluator.

Run only through with_nas_storage.sh. No optimizer, fitter or model forward is
imported or called. Original cache files are immutable, hash-bound seal inputs.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import numpy as np

from .io_r1 import digest, project_training, read_json, safe_path, validate_population, verify_files, write_json

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "experiments/lcrseg/docs/care_hr_v0_7_1/continuation_r1"
OLD = ROOT / "experiments/lcrseg/docs/care_hr_v0_7_1"
FORMAL = Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/ppc_shor_v0_6b_0b3f490/formal_02")
DATA = Path("/home/jiangsuiyang/SSL_CL")
BASE_POLICIES = {"C0":"current", "C1":"frozen_Ridge_hard", "C3":"frozen_SHOR", "C6":"frozen_PPC_C6"}
REFERENCE = {"current":.6016731401004847, "frozen_Ridge_hard":.8156293208569301,
             "frozen_SHOR":.8052579174291238, "frozen_PPC_C6":.802187012880861}


def require(condition, message):
    if not condition: raise ValueError(message)


def source_state():
    require(not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(), "dirty source")
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()


def write_csv(path, rows):
    if not rows: return
    with Path(path).open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def original_inputs():
    expected=read_json(DOC/'INPUT_EXPECTATIONS.json')
    require(digest(FORMAL.parent/'FORMAL_GT_ACCESS_RESERVATION.json')==expected['original_reservation_sha256'],'original reservation hash mismatch')
    receipt=read_json(FORMAL.parent/'FORMAL_GT_ACCESS_RESERVATION.json')
    require(Path(receipt['formal_output']).resolve()==FORMAL.resolve(),'original receipt does not select formal_02')
    require(digest(FORMAL/'PPC_SHOR_V0_6B_PRIVATE_MANIFEST.json')==expected['private_manifest_sha256'],'private manifest hash mismatch')
    manifest=read_json(FORMAL/'PPC_SHOR_V0_6B_PRIVATE_MANIFEST.json')
    require(manifest['content_sha256']==expected['private_manifest_content_sha256'],'private content hash mismatch')
    entries={e['path']:e for e in manifest['entries']}
    return expected, entries


def check_original(entries, name):
    path=safe_path(FORMAL,name)
    require(name in entries and path.stat().st_size==entries[name]['bytes'] and digest(path)==entries[name]['sha256'],'input integrity mismatch: '+name)
    return path


def blind(output, test_report, admission_path):
    from .blind_r1 import prepare_case
    from .actions import probability_pair
    start=time.monotonic(); source=source_state()
    evidence=read_json(test_report)
    require(evidence['source_commit']==source and evidence['pytest_exit_code']==0 and evidence['failures']==0 and evidence['errors']==0 and evidence['skipped']==0,'exact-source A1 evidence missing')
    admission=read_json(admission_path)
    require(admission['source_commit']==source and admission['A1_complete'] and admission['server_test_sha256']==digest(test_report),'published A1 admission mismatch')
    output.mkdir(exist_ok=False,parents=True); (output/'cases').mkdir()
    write_json(output/'A1_RELEASE.json',admission)
    expected, entries=original_inputs(); files={}
    def load_path(name):
        path=check_original(entries,name); files[name]=entries[name]['sha256']; return path
    rows=read_json(load_path('outer_blind_cache.json'))
    # Fixed usecols projection: never decode GT/domain fields into Python records.
    projected={}; manifest_hashes={}
    for spec in expected['seed_manifests']:
        seed=int(spec['seed']); path=DATA/'manifests/training'/f'lcrseg_v1_seed{seed}.csv'
        require(digest(path)==spec['sha256'],'seed manifest mismatch')
        manifest_hashes[str(seed)]=spec['sha256']; projected[seed]=project_training(path)
    population=validate_population(rows,projected)
    write_json(output/'blind_rows.json',rows)
    routes=np.full(198,-1,dtype=np.int64); fold_indices={}; control_paths={}
    for fold in range(5):
        seal=read_json(load_path(f'candidate_seals/fold{fold}.json'))
        require(seal['status']=='PASS_OUTER_CANDIDATES_SEALED_BEFORE_GT','old blind seal missing')
        candidate=seal['selected_candidate']['candidate_id']
        require(candidate==('k010_tau095' if fold<3 else 'k100_tau090'),'frozen candidate changed')
        indices=[i for i,r in enumerate(rows) if r['fold']==fold]; fold_indices[str(fold)]=indices
        order=read_json(load_path(f'case_order/fold{fold}.json'))
        require(order==[{k:rows[i][k] for k in ('case_id','row_index','seed')} for i in indices],'fold order mismatch')
        name=f'candidate_routes/fold{fold}.npz'
        with np.load(load_path(name),allow_pickle=False) as z: route=z[candidate+'_C6']
        require(route.shape==(len(indices),) and np.issubdtype(route.dtype,np.integer) and np.isin(route,(0,1,2)).all(),'route invalid')
        require(files[name]==seal['sealed_files'][name],'route original seal mismatch')
        routes[indices]=route
        for policy in BASE_POLICIES:
            name=f'candidate_predictions/fold{fold}_{policy}.npy'; path=load_path(name)
            require(files[name]==seal['sealed_files'][name],'control original seal mismatch')
            arr=np.load(path,mmap_mode='r',allow_pickle=False)
            require(arr.shape==(len(indices),384,384) and np.issubdtype(arr.dtype,np.integer) and np.isin(arr,(0,1,2)).all(),'control shape/labels invalid')
            control_paths[f'{fold}:{policy}']=name
    require(hashlib.sha256(routes.tobytes()).hexdigest()==expected['C6_route']['route_sha256'] and int(np.sum(routes<2))==155,'stitched C6 route mismatch')
    np.save(output/'routes.npy',routes,allow_pickle=False)
    probabilities={}
    for seed in range(3):
        for expert in range(3):
            stem=f'seed{seed}_expert{expert}'; name=f'expert_probability_cache/{stem}.npy'
            require(entries[name]['sha256']==expected['probability_cache_sha256'][stem],'probability lineage mismatch')
            array=np.load(load_path(name),mmap_mode='r',allow_pickle=False)
            require(array.shape==(66,3,384,384) and array.dtype==np.float32,'frozen probability shape/dtype mismatch')
            probabilities[seed,expert]=array
    baseline_manifest=expected['baseline_snapshot_manifest']
    require(digest(ROOT/baseline_manifest['path'])==baseline_manifest['sha256'],'baseline snapshot changed')
    checkpoints=[]
    for checkpoint in read_json(ROOT/baseline_manifest['path'])['checkpoints']:
        require(digest(checkpoint['path'])==checkpoint['sha256'],'checkpoint integrity mismatch')
        checkpoints.append({k:checkpoint[k] for k in ('checkpoint_id','sha256','bytes')})
    # State tensors only for cost accounting, no architecture construction/forward.
    import torch
    ck=read_json(ROOT/baseline_manifest['path'])['checkpoints'][0]
    state=torch.load(ck['path'],map_location='cpu')['ema_teacher']
    parameter_keys=[k for k in state if k.endswith(('weight','bias','weight_mu','weight_rho','bias_mu','bias_rho','grad_update'))]
    other={k:list(v.shape) for k,v in state.items() if k not in parameter_keys}
    require(not other,'unclassified checkpoint state tensors; cannot claim parameter count')
    parameters=sum(state[k].numel() for k in parameter_keys); del state
    positions={}; counts={s:0 for s in range(3)}; diagnostics=[]
    for index,row in enumerate(rows):
        seed=row['seed']; pos=counts[seed]; counts[seed]+=1; positions[str(index)]=pos
        c=probabilities[seed,2][pos]
        for expert in range(3): probability_pair(c,probabilities[seed,expert][pos])
        history=probabilities[seed,int(routes[index])][pos]
        prepared=prepare_case(c,history,int(routes[index]))
        fold=row['fold']; fold_pos=fold_indices[str(fold)].index(index)
        current_control=np.load(FORMAL/control_paths[f'{fold}:C0'],mmap_mode='r',allow_pickle=False)[fold_pos]
        require(np.array_equal(prepared['current_hard'],current_control),'frozen current probability/hard mismatch')
        ridge=np.load(FORMAL/control_paths[f'{fold}:C1'],mmap_mode='r',allow_pickle=False)[fold_pos]
        require(np.array_equal(ridge,probabilities[seed,int(np.argmax(row['alpha']))][pos].argmax(axis=0)),'Ridge cache/top1 mismatch')
        c6=np.load(FORMAL/control_paths[f'{fold}:C6'],mmap_mode='r',allow_pickle=False)[fold_pos]
        require(np.array_equal(c6,history.argmax(axis=0)),'C6 route/prediction mismatch')
        arrays={k:prepared.pop(k) for k in ('current_hard','blended_hard','masks','probability_changed')}
        np.savez_compressed(output/'cases'/f'{index:03d}.npz',**arrays)
        write_json(output/'cases'/f'{index:03d}.json',prepared)
        acts=prepared['actions']; diagnostics.append(dict(row_index=index,proposals=len(prepared['proposals']),
             strict_actions=sum(a['O_CAP'] for a in acts),no_area_actions=sum(a['O_NO_AREA'] for a in acts),free_actions=len(acts),
             quantity_rejected=sum(a['quantity_rejected'] for a in acts),foreground_area_rejected=sum(a['foreground_area_rejected'] for a in acts),
             image_area_rejected=sum(a['image_area_rejected'] for a in acts),zero_current_foreground=prepared['current_foreground_pixels']==0))
        if (index+1)%25==0: print(json.dumps({'phase':'blind','sealed_case_progress':index+1,'GT_reads':0}),flush=True)
    write_json(output/'blind_diagnostics.json',diagnostics)
    write_json(output/'input_metadata.json',dict(original_files=files,manifest_hashes=manifest_hashes,checkpoints=checkpoints,
               fold_indices=fold_indices,control_paths=control_paths,probability_positions=positions,population=population,
               original_private_manifest_sha256=expected['private_manifest_sha256'],original_case_metrics_sha256=entries['case_metrics.jsonl']['sha256'],
               reused_sample_expert_forwards=594,new_sample_expert_forwards=0,new_batch_forwards=0,checkpoint_loads=1,
               parameters_per_expert=parameters,parameters_nine_snapshots=9*parameters,probability_cache_bytes=sum(entries[n]['bytes'] for n in files if n.startswith('expert_probability_cache/'))))
    write_json(output/'BLIND_RUNTIME.json',{'source_commit':source,'elapsed_seconds':time.monotonic()-start,'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'GT_reads':0,'domain_reads':0,'real_fits':0})
    sealed={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'ACTION_SPACE_SEAL.json',dict(status='PASS_ACTION_SPACE_SEALED_BEFORE_GT',source_commit=source,
               files=sealed,original_files=files,protocol_sha256=digest(DOC/'R1_PREREGISTRATION.json'),scoring_contract_sha256=digest(DOC/'SCORING_CONTRACT_R1.json'),
               A1_release_sha256=digest(output/'A1_RELEASE.json'),server_test_sha256=digest(test_report),GT_reads=0,domain_reads=0))
    verify_files(output,sealed)
    print(json.dumps({'phase':'blind','status':'PASS_ACTION_SPACE_SEALED_BEFORE_GT','seal_sha256':digest(output/'ACTION_SPACE_SEAL.json'),'rows':198,'GT_reads':0}),flush=True)


def load_prepared(output,index):
    prepared=read_json(output/'cases'/f'{index:03d}.json')
    with np.load(output/'cases'/f'{index:03d}.npz',allow_pickle=False) as arrays:
        for k in arrays.files: prepared[k]=arrays[k]
    return prepared


def evaluate(output):
    # This function is launched in a fresh process after the blind process exits.
    from .scoring_r1 import case_metrics, gains
    from .oracle_r1 import score_actions, select_oracles, pareto_actions
    from .summary_r1 import FIELDS, DOMAINS, legacy_aggregate, summarize, terminal, groups_for
    import h5py
    start=time.monotonic(); source=source_state(); seal=read_json(output/'ACTION_SPACE_SEAL.json')
    require(seal['source_commit']==source and seal['status']=='PASS_ACTION_SPACE_SEALED_BEFORE_GT','source/seal gate mismatch')
    require(digest(DOC/'R1_PREREGISTRATION.json')==seal['protocol_sha256'] and digest(DOC/'SCORING_CONTRACT_R1.json')==seal['scoring_contract_sha256'],'protocol changed')
    require(digest(output/'A1_RELEASE.json')==seal['A1_release_sha256'],'A1 publication changed')
    verify_files(output,seal['files']); verify_files(FORMAL,seal['original_files'])
    expected,entries=original_inputs(); meta=read_json(output/'input_metadata.json'); rows=read_json(output/'blind_rows.json')
    for seed,h in meta['manifest_hashes'].items(): require(digest(DATA/'manifests/training'/f'lcrseg_v1_seed{seed}.csv')==h,'seed manifest changed')
    write_json(output/'EVALUATOR_ACCESS_RESERVATION.json',{'source_commit':source,'seal_sha256':digest(output/'ACTION_SPACE_SEAL.json'),'created_unix':time.time(),'GT_reads_before_reservation':0,'domain_reads_before_reservation':0,'status':'RESERVED_SINGLE_EVALUATOR'})
    (output/'public').mkdir(); (output/'private_evaluation').mkdir()
    counters=dict(GT_reads=0,domain_reads=0,new_sample_expert_forwards=0,reused_sample_expert_forwards=594,new_batch_forwards=0,
                  real_Ridge_fits=0,real_PAV_fits=0,real_temperature_fits=0,real_risk_head_fits=0,real_router_fits=0,real_conformal_calibrations=0,
                  optimizer_updates=0,segmentation_updates=0,EMA_updates=0,GAS_updates=0,prototype_updates=0,old_formal_03_reads=0,own_seed_forbidden_GT_reads=0)
    status='INCOMPLETE_EVALUATION'; all_rows=[]; labels={}; revealed={}; percase=[]; baseline_report={}
    try:
        for seed in range(3):
            wanted={r['case_id']:r for r in rows if r['seed']==seed}
            columns=('case_id','patient_id','primary_20pct_split','dataset','site_or_vendor','label_h5_relpath','label_sha256','image_h5_relpath','image_sha256')
            for row in project_training(DATA/'manifests/training'/f'lcrseg_v1_seed{seed}.csv',columns,selected_ids=wanted):
                if row['case_id'] not in wanted: continue
                blindrow=wanted[row['case_id']]; i=blindrow['row_index']
                require(row['primary_20pct_split']=='train_labeled' and row['dataset']=='fundus' and row['label_h5_relpath'] and row['site_or_vendor'] in DOMAINS,'GT role/domain invalid')
                require(row['patient_id']==blindrow['patient_id'] and i not in revealed,'evaluator lineage/duplicate mismatch')
                counters['domain_reads']+=1
                revealed[i]=dict(seed=seed,case_id=row['case_id'],patient_id=row['patient_id'],domain=row['site_or_vendor'],domain_index=DOMAINS.index(row['site_or_vendor']),row_index=i)
                counters['GT_reads']+=1
                payload=safe_path(DATA,row['label_h5_relpath']).read_bytes()
                require(hashlib.sha256(payload).hexdigest()==row['label_sha256'],'GT file hash mismatch')
                with h5py.File(io.BytesIO(payload),'r') as f: labels[i]=np.asarray(f['label'][...])
                require(labels[i].shape==(384,384),'GT shape mismatch')
                case_metrics(np.zeros((384,384),dtype=np.uint8),labels[i])
        require(set(revealed)==set(range(198)),'incomplete evaluator population')
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
        oldagg=legacy_aggregate(originals); newagg=legacy_aggregate(baseline)
        require(len(oldagg)==len(newagg),'baseline aggregation shape')
        for old,new in zip(oldagg,newagg):
            require(all(old[k]==new[k] for k in ('level','key','policy','cases')),'baseline aggregation keys')
            error=max(abs(old[k]-new[k]) for k in FIELDS); max_error=max(max_error,error)
            require(error<=1e-12,'BLOCKED_BASELINE_MISMATCH: grouped metric')
        for r in newagg:
            if r['level']=='overall': require(abs(r['foreground_dice']-REFERENCE[r['policy']])<=1e-12,'BLOCKED_BASELINE_MISMATCH: frozen control reference')
        baseline_report=dict(status='PASS_BASELINE_PARITY',per_case_comparisons=len(baseline)*4,grouped_rows=len(newagg),maximum_absolute_difference=max_error,atol=1e-12,rtol=0,original_aggregation='AST exact ppc_shor_v0_6a.aggregate_case_metrics; original fold/case/policy order')
        write_json(output/'public'/'BASELINE_PARITY.json',baseline_report)
        all_rows=baseline
        with (output/'private_evaluation'/'pareto.jsonl').open('x') as pareto_file:
            for fold in range(5):
                for i in meta['fold_indices'][str(fold)]:
                    prepared=load_prepared(output,i); actions=score_actions(prepared,labels[i]); selected=select_oracles(actions)
                    frontier=pareto_actions(actions)
                    pareto_file.write(json.dumps({'row_index':i,'actions':frontier},allow_nan=False)+'\n')
                    for policy,score in selected.items():
                        all_rows.append({**revealed[i],**score,'policy':policy})
                    percase.append({**revealed[i],'pareto_actions':len(frontier),'evaluated_actions':len(actions),
                                    'maximum_free_gain':selected['O_FREE_SUBSET_envelope']['gain_macro_fg'],
                                    'cap_to_no_area':selected['O_NO_AREA_envelope']['gain_macro_fg']-selected['O_CAP_envelope']['gain_macro_fg'],
                                    'no_area_to_free':selected['O_FREE_SUBSET_envelope']['gain_macro_fg']-selected['O_NO_AREA_envelope']['gain_macro_fg']})
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
        diagnostics=read_json(output/'blind_diagnostics.json'); budgets=[]
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
        write_json(output/'public'/'EXECUTION_ERROR.json',{'exception':type(exc).__name__,'message':text,'post_GT_code_changes_allowed':False})
        raise
    finally:
        runtime={**counters,'elapsed_seconds':time.monotonic()-start,'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'bootstrap_p90':'NOT_EVALUATED','feasible_refits':'NOT_EVALUATED','scientific_rows_completed':len(percase)}
        write_json(output/'public'/'RUNTIME_COUNTERS.json',runtime)
        write_json(output/'public'/'CAPACITY_STATUS.json',{'status':status,'source_commit':source,'rows':len(revealed),'patients':len({r['patient_id'] for r in revealed.values()}),
                   'baseline_parity':baseline_report,'all_terminal_states_stop':True,'next_stage_started':False,
                   'action_seal_sha256':digest(output/'ACTION_SPACE_SEAL.json'),'evaluator_reservation_sha256':digest(output/'EVALUATOR_ACCESS_RESERVATION.json')})
    print(json.dumps({'phase':'evaluate','status':status,'GT_reads':counters['GT_reads'],'oracle_rows':len(percase)}),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=('blind','evaluate')); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--test-report',type=Path); parser.add_argument('--admission',type=Path)
    args=parser.parse_args()
    require(args.output.resolve().is_relative_to(Path('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg').resolve()),'output must be NAS')
    require(Path(os.environ.get('TMPDIR','')).resolve().is_relative_to(Path('/data_nas').resolve()),'launch through NAS wrapper')
    if args.phase=='blind': blind(args.output,args.test_report,args.admission)
    else: evaluate(args.output)


if __name__=='__main__': main()
