#!/usr/bin/env python3
"""Emit metadata-only plans. This script cannot launch training or access data."""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json
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


def write(plan:dict,out:Path)->None:
    out.mkdir(parents=True,exist_ok=True); rows=expand(plan)
    content=json.dumps(rows,ensure_ascii=False,indent=2)+'\n'
    (out/'SEARCH_CANDIDATES.json').write_text(content,encoding='utf-8')
    with (out/'SEARCH_CANDIDATES.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['family','candidate_id','hyperparameters','fixed']);w.writeheader()
        for row in rows:w.writerow({**row,'hyperparameters':json.dumps(row['hyperparameters'],ensure_ascii=False,sort_keys=True),'fixed':json.dumps(row['fixed'],ensure_ascii=False,sort_keys=True)})
    (out/'BASELINE_CANDIDATES.json').write_text(json.dumps(expand_baselines(plan),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tasks=[]
    for phase, candidates,seeds in [
        ('C_PARENT',[('PARENT',p['id']) for p in plan['parent_configs']],plan['development_seeds_proposed']),
        ('C_SHARED',[(r['baseline'],r['candidate_id']) for r in expand_baselines(plan)],plan['development_seeds_proposed']),
        ('C_SEARCH',[(r['family'],r['candidate_id']) for r in rows],plan['development_seeds_proposed']),
        ('D_REPLICATION',[(f,f+'__SELECTED_AFTER_C') for f in plan['priority_order']]+[('BASELINE',b+'__SELECTED_AFTER_C') for b in plan['baselines']],plan['replication_seeds_proposed'])]:
        for family,cid in candidates:
            for seed in seeds:
                for oi,order in enumerate(plan['orders'],1):
                    sid=f'{phase}__{cid}__S{seed}__O{oi}'
                    for stage,domain in enumerate(order[1:],1):
                        tasks.append({'phase':phase,'family':family,'candidate_id':cid,'seed':seed,'order':oi,
                                      'sequence_id':sid,'stage':stage,'domain':domain,
                                      'parent_checkpoint':f'COMMON_SOURCE_S{seed}' if stage==1 else sid+'__STAGE1_OWN_CHECKPOINT',
                                      'status':'PLANNED_NOT_LAUNCHED','resolved':False})
    with (out/'TASK_SLOTS.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(tasks[0]));w.writeheader();w.writerows(tasks)
    (out/'PLAN_DIGEST.json').write_text(json.dumps({'search_candidates_sha256':hashlib.sha256(content.encode()).hexdigest(),
                  'candidate_count':len(rows),'task_slots_excluding_source':len(tasks),'phase_D_requires_selected_candidate_binding':True},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=Path(__file__).resolve().parents[1]/'configs/study_plan.json')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'manifests')
    parser.add_argument('--example-stage-updates',nargs=2,type=int,default=[3200,2100])
    parser.add_argument('--source-updates',type=int)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());write(plan,args.out)
    result=budget(plan,args.example_stage_updates,args.source_updates)
    (args.out/'ILLUSTRATIVE_BUDGET.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
