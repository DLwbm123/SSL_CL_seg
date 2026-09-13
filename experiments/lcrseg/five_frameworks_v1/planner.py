#!/usr/bin/env python3
"""Emit metadata-only plans. This script cannot launch training or access data."""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,math
from pathlib import Path


def expand(plan: dict) -> list[dict]:
    rows=[]
    for family in plan['priority_order']:
        spec=plan['families'][family];keys=list(spec['grid'])
        for i, values in enumerate(itertools.product(*(spec['grid'][k] for k in keys)),1):
            rows.append({'family':family,'candidate_id':f'{family}_C{i:02d}',
                         'hyperparameters':dict(zip(keys,values)),
                         'fixed':{k:v for k,v in spec.items() if k not in ('grid','name')}})
    return rows


def expand_baselines(plan:dict)->list[dict]:
    result=[]
    for name,spec in plan['baseline_search'].items():
        keys=list(spec['grid'])
        for i,values in enumerate(itertools.product(*(spec['grid'][k] for k in keys)),1):
            result.append({'baseline':name,'candidate_id':name+f'_C{i:02d}','hyperparameters':dict(zip(keys,values))})
    return result


def budget(plan:dict, stage_updates:list[int], source_updates:int|None=None)->dict:
    if len(stage_updates)!=2 or any(x<=0 for x in stage_updates):
        raise ValueError('two positive incremental-domain budgets required')
    n_orders=len(plan['orders']);n_dev=len(plan['development_seeds_proposed']);n_rep=len(plan['replication_seeds_proposed'])
    parent=len(plan['parent_configs'])*n_orders*n_dev
    shared=len(expand_baselines(plan))*n_orders*n_dev
    search=len(expand(plan))*n_orders*n_dev
    rep=(len(plan['families'])+len(plan['baselines']))*n_orders*n_rep
    n=parent+shared+search+rep;s=sum(stage_updates)
    e=plan['phase_E']['max_new_ablation_configs_per_finalist']*plan['phase_E']['max_finalists']*n_orders*n_rep
    source_seeds=n_dev+n_rep
    return {'not_an_execution_receipt':True, 'input_stage_updates':stage_updates,
            'parent_tuning_sequences':parent,'shared_new_sequences':shared,
            'framework_search_sequences':search,'phase_C_sequences':parent+shared+search,
            'phase_D_sequences':rep,'approved_candidate_phase_C_D_sequences':n,
            'incremental_training_stages':2*n,'incremental_optimizer_updates':n*s,
            'new_source_models_if_no_legal_reuse':source_seeds,
            'source_updates_if_no_legal_reuse':None if source_updates is None else source_seeds*source_updates,
            'optional_phase_E_sequences_not_authorized':e,'optional_phase_E_updates_not_authorized':e*s,
            'extra_VJP_readout_costs':'separate exact accounting required; optimizer budget alone does not bound these'}


def dag(plan):
    """Selection nodes are unresolved future events, not invented candidate wins."""
    nodes=[];sources={};groups={}
    for seed in plan['development_seeds_proposed']+plan['replication_seeds_proposed']:
        sid=f'SOURCE_S{seed}';sources[seed]=sid
        nodes.append({'id':sid,'kind':'source','seed':seed,'domain':'REFUGE','dependencies':[],
                      'status':'PENDING_PARENT_BINDING_AND_SOURCE_PROVENANCE','conditional_new_training':True})
    def trajectories(phase,candidates,seeds):
        for family,cid in candidates:
            for seed in seeds:
                for oi,order in enumerate(plan['orders'],1):
                    seq=f'{phase}__{cid}__S{seed}__O{oi}'
                    previous=sources[seed]
                    for stage,domain in enumerate(order[1:],1):
                        ident=f'{seq}__STAGE{stage}'
                        deps=[previous]
                        if phase=='C' and family!='PARENT':deps.append('SELECT_PARENT')
                        if phase=='D':deps += ['SELECT_PARENT',f'SELECT_{family}','FREEZE_C_REFERENCES']
                        nodes.append({'id':ident,'kind':'target_stage','phase':phase,'family':family,
                            'candidate_id':cid,'seed':seed,'order':oi,'sequence_id':seq,'stage':stage,
                            'domain':domain,'dependencies':deps,'parent_checkpoint':previous,
                            'resolved':False,'status':'PLANNED_NOT_LAUNCHED','updates':None})
                        previous=ident
                    groups.setdefault(family,[]).append(previous)
    trajectories('C',[('PARENT',p['id']) for p in plan['parent_configs']],plan['development_seeds_proposed'])
    nodes.append({'id':'SELECT_PARENT','kind':'selection','dependencies':groups['PARENT'].copy(),
                  'resolved_candidate':None,'status':'PENDING_C0_SCORES'})
    candidates=[(r['baseline'],r['candidate_id']) for r in expand_baselines(plan)]+[(r['family'],r['candidate_id']) for r in expand(plan)]
    trajectories('C',candidates,plan['development_seeds_proposed'])
    for family in plan['baselines']+plan['priority_order']:
        deps=['SELECT_PARENT'] if family.startswith('B0') else groups[family].copy()
        nodes.append({'id':f'SELECT_{family}','kind':'selection','dependencies':deps,
                      'resolved_candidate':None,'status':'PENDING_C_SCORES','reuse_parent':family.startswith('B0')})
    nodes.append({'id':'FREEZE_C_REFERENCES','kind':'selection','dependencies':[f'SELECT_{x}' for x in plan['baselines']],
                  'status':'PENDING_C_SCORES','strong_L_only':None,'strong_SSL':None})
    trajectories('D',[(f,f+'__SELECTED_AFTER_C') for f in plan['baselines']+plan['priority_order']],plan['replication_seeds_proposed'])
    validate_dag(nodes)
    return nodes


def validate_dag(nodes):
    byid={n['id']:n for n in nodes}
    if len(byid)!=len(nodes):raise ValueError('duplicate DAG node')
    visiting=set();done=set()
    def visit(key):
        if key in visiting:raise ValueError('DAG cycle')
        if key in done:return
        if key not in byid:raise ValueError('missing DAG dependency '+key)
        visiting.add(key)
        for d in byid[key]['dependencies']:visit(d)
        visiting.remove(key);done.add(key)
    for key in byid:visit(key)
    for n in nodes:
        if n['kind']=='target_stage' and n['stage']==2:
            predecessor=byid[n['parent_checkpoint']]
            for key in ('family','candidate_id','seed','order','sequence_id'):
                if n[key]!=predecessor[key]:raise ValueError('cross-trajectory stage predecessor')
    return True


def effective_aliases(candidates,feature_width=None):
    if feature_width is None:return {'status':'PENDING_PARENT_FEATURE_WIDTH','aliases':{}}
    if feature_width<2:raise ValueError('incompatible feature width')
    seen={};aliases={}
    for row in candidates:
        options={**row.get('fixed',{}),**row['hyperparameters']}
        if 'rank_ratio' in options:
            options['rank']=max(1,min(feature_width-1,round(feature_width*options.pop('rank_ratio'))))
        key=json.dumps([row.get('family',row.get('baseline')),options],sort_keys=True)
        if key in seen:aliases[row['candidate_id']]=seen[key]
        else:seen[key]=row['candidate_id']
    return {'status':'RESOLVED_METADATA_ONLY','aliases':aliases,'savings_policy':'freeze deduplicated matrix before approval; no replacement candidates'}


def cost_plan(plan,stage_updates=None,source_updates=None):
    # Counts follow the unpruned search and all-five replication DAG.
    result={'status':'PENDING_PARENT_BUDGET','target_trajectories':212,'target_stages':424,
            'source_nodes_conditional':4,'formal_updates':None,'source_updates':None,
            'formal_formula':'212 * (S_RIM + S_DRISHTI)','source_formula':'4 * S_REFUGE if no legal source reuse',
            'F2_probe_VJPs':1056,'F2_probe_full_forwards':704,'F4_calibration_VJPs':704,
            'F4_calibration_full_forwards':352,'F4_inner_VJPs':None,'F4_readout_only_upper_bound':None,
            'F5_extra_clean_U_forwards':None,'synthetic_qualification_cap':256,'real_L_smoke_cap':24,
            'smoke_allocation':{'F1':4,'F2':8,'F3':4,'F4':4,'F5':4},
            'phase_E':'PLAN_ONLY_SEPARATE_REVIEW','phase_E_max_trajectories':48,
            'common_forwards_per_active_step':{'student_L':2,'student_UL':2,'teacher_U':1,'teacher_L_prototype':1},
            'F3_extra_donor_readout_per_active_step':1,'Q_eigensolve_per_subspace_stage':1,
            'compute_comparability':'same updates does not imply same FLOPs/time; profile after approval'}
    if stage_updates is not None:
        basic=budget(plan,stage_updates,source_updates)
        active=sum(s-math.ceil(.2*s) for s in stage_updates)
        result.update(status='BOUND_METADATA',formal_updates=basic['incremental_optimizer_updates'],
                      source_updates=basic['source_updates_if_no_legal_reuse'],F4_inner_VJPs=22*active*3,
                      F4_readout_only_upper_bound=22*active*5,F5_extra_clean_U_forwards=22*active)
    return result


def write(plan,out):
    from .gate import digest
    from .registry import ABLATIONS
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    candidates=expand(plan);baselines=expand_baselines(plan);nodes=dag(plan)
    payload={'study':plan,'nodes':nodes,'effective_aliases':effective_aliases(candidates+baselines),
             'seed_status':'PROPOSED_PENDING_BOUND_PARENT_METADATA_COLLISION_CHECK',
             'parent_binding':'PARENT_BINDING_REQUIRED','actual_steps':None}
    files={'SEARCH_CANDIDATES.json':candidates,'BASELINE_CANDIDATES.json':baselines,
           'RESOLVED_PROTOCOL.json':payload,'BUDGET_PLAN.json':cost_plan(plan),
           'ABLATIONS_PLAN_ONLY.json':ABLATIONS,
           'PLAN_DIGEST.json':{'expanded_plan_sha256':digest(payload),'target_stage_count':424},
           'ILLUSTRATIVE_BUDGET.json':{'not_actual_budget':True,**cost_plan(plan,[3200,2100])}}
    for name,value in files.items():(out/name).write_text(json.dumps(value,indent=2)+'\n')
    rows=[n for n in nodes if n['kind']=='target_stage']
    with (out/'TASK_MATRIX_PREVIEW.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for n in rows:w.writerow({**n,'dependencies':json.dumps(n['dependencies'])})
    return payload
