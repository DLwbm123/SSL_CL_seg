"""Historical-val GT is consumed here only, after every transform is frozen."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from di_dmpa_gate1.sampling import sample_layout
from di_dmpa_gate1_v2.features import extract_unit, ImmutableModels, weight_hash, load_cache
from di_dmpa_gate1_v2.geometry import fit, validate_centers, validate_directions
from di_dmpa_gate1_v2.binding import InvalidCenter
from .binding import (DOMAINS, H, PLAN_SHA, ProtocolError, InvalidTransportOutput, IncompleteEvidence,
    require, checkpoint, load_b0, check_hash, write_json, read_json)
from .pairs import save_arrays
from .transport import apply_model, finite


def operational(frozen, seed, source_stage):
    records=sorted((r for r in frozen['prototype_records'] if r['seed']==seed and r['stage_index']==source_stage),key=lambda r:r['class_id'])
    require([r['class_id'] for r in records]==[0,1,2], 'incomplete frozen operational bank')
    require(all(r['panel']=='B0-EMA' and r['K']==2 and r['active_mask']==[True,True] and r['converged']
        and r['training_source']=='train_labeled' and not r['operational_refit_allowed'] for r in records), 'invalid operational bank')
    values=np.asarray([r['centers'] for r in records],dtype=np.float64)
    validate_directions(values.reshape(-1,16),np.ones(6,dtype=bool))
    return values.copy()


def transported(prototypes, models):
    original=np.asarray(prototypes,dtype=np.float64);result=original.copy().reshape(-1,16)
    require(original.shape==(3,2,16) and bool(models), 'wrong prototype/chain input')
    for model in models:result,_=apply_model(result,np.ones(6,dtype=bool),model)
    require(np.array_equal(original,prototypes), 'operational prototype mutated')
    return result.reshape(3,2,16)


def angular_match(prototypes, oracle, oracle_active):
    p,pa=validate_centers(prototypes,np.ones(2,dtype=bool));q,qa=validate_centers(oracle,oracle_active)
    require(p.shape==q.shape==(2,16) and pa.all(), 'fixed K2 angular matching required')
    costs=np.full((2,2),np.pi,dtype=np.float64)
    costs[:,qa]=np.arccos(np.clip(p@q[qa].T,-1,1))
    row,col=linear_sum_assignment(costs)
    values=costs[row,col];finite(values,'angular matching')
    return dict(mean_angular_error=float(values.mean()),matched_angles=values.tolist(),
        matching=list(zip(row.tolist(),col.tolist())),inactive_oracle_slots=int((~qa).sum()),units='radians')


def prototype_accuracy(queries, weights, prototypes):
    p=np.asarray(prototypes,dtype=np.float64);validate_directions(p.reshape(-1,16),np.ones(6,dtype=bool))
    require(p.shape==(3,2,16) and set(queries)==set(weights)=={0,1,2}, 'incomplete accuracy classes')
    classes=[]
    for c in range(3):
        x,active=validate_directions(queries[c]['directions'],queries[c]['active_mask']);w=np.asarray(weights[c],dtype=np.float64)
        require(w.shape==(len(x),) and np.isfinite(w).all() and (w>=0).all() and w.sum()>0, 'invalid accuracy weights')
        correct=np.zeros(len(x),dtype=bool)
        if active.any():
            scores=(x[active]@p.reshape(6,16).T).reshape(-1,3,2).max(axis=2)
            correct[active]=scores.argmax(axis=1)==c
        numerator=float(w[correct].sum());mass=float(w.sum());am=float(w[active].sum())
        classes.append(dict(class_id=c,accuracy=numerator/mass,registered_count=len(x),null_count=int((~active).sum()),
            active_mass=am/mass,null_mass=float(w[~active].sum()/mass),correct_original_mass=numerator/mass,
            directional_conditional_accuracy=None if am==0 else numerator/am,directional_conditional_defined=am>0))
    conditional=[r['directional_conditional_accuracy'] for r in classes]
    return dict(classes=classes,macro_accuracy=float(np.mean([r['accuracy'] for r in classes])),
        foreground_macro_accuracy=float(np.mean([r['accuracy'] for r in classes[1:]])),
        directional_conditional_macro_accuracy=None if any(x is None for x in conditional) else float(np.mean(conditional)),
        directional_conditional_macro_defined=all(x is not None for x in conditional),
        foreground_directional_conditional_accuracy=None if any(x is None for x in conditional[1:]) else float(np.mean(conditional[1:])),
        foreground_directional_conditional_defined=all(x is not None for x in conditional[1:]),
        tie_rule='lowest class ID',null_query_is_incorrect=True,weighting='case equal within true class; class equal macro')


def oracle_fit(arrays, weights, uid_rank, *, seed, source_stage, class_id):
    try:result=fit(arrays['directions'],arrays['active_mask'],weights,2,seed=seed,stage=source_stage,class_id=class_id,replicate=-1,uid_rank=uid_rank)
    except InvalidCenter as error:raise InvalidTransportOutput('oracle center: '+str(error)) from error
    require(len(result['restarts'])==5 and result['K']==2, 'oracle solver contract changed')
    return {**result,'centers':result['centers'].tolist(),'active':result['active'].tolist(),
            'source_stage_for_clustering_seeds':source_stage,'gt_consumer':'diagnostic_evaluator_only'}


def evaluate_unit(root, data_root, p, frozen, output, metadata, seed, source_stage, target_stage, *, device='cuda:0'):
    require((source_stage,target_stage) in ((0,1),(1,2),(0,2)), 'unregistered oracle unit')
    output=Path(output);check_hash(output/'FROZEN_GEOMETRY_SAMPLING_PLAN.json',PLAN_SHA)
    barrier=read_json(output/'ORACLE_START_BARRIER.json')
    if barrier['transport_optimizer_steps']!=6000 or barrier['six_transports_complete'] is not True:
        raise IncompleteEvidence('all six maps must be frozen before evaluator')
    key=f'seed{seed}_source{source_stage}_target{target_stage}'
    plan=read_json(output/'FROZEN_GEOMETRY_SAMPLING_PLAN.json')
    unit=next(u for u in plan['units'] if u['seed']==seed and u['stage_index']==source_stage and u['role']=='val')
    require(unit['domain']==DOMAINS[source_stage] and unit['gt_consumer']=='diagnostic_evaluator_only', 'oracle role violation')
    context=dict(seed=seed,source_stage=source_stage,target_stage=target_stage,source_domain=DOMAINS[source_stage],role='val',
        sampling_plan_sha256=PLAN_SHA,sampling_unit_sha256=H(unit),gt_consumer='diagnostic_evaluator_only')
    cp=checkpoint(p,seed,target_stage)
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=ProtocolError('evaluator optimizer construction forbidden')):
        models=load_b0(root,p,seed,target_stage,device)
        with ImmutableModels(models,cp,output/'oracle_audits'/key,dict(metadata,phase='oracle_evaluator',transport_optimizer_steps_before_phase=6000)):
            arrays,cases=extract_unit(models['ema_teacher'],unit,data_root,context,device=device,batch_size=8)
    del models
    if torch.cuda.is_available():torch.cuda.empty_cache()
    caches=[];weights={};oracles=[]
    for c in range(3):
        lay=sample_layout(unit,c);a=arrays[c];weights[c]=lay['weights']
        cache=dict(class_id=c,registered_count=len(lay['uids']),uid_order_sha256=H(lay['uids']),original_weight_order_sha256=weight_hash(lay['weights']),
            active_count=int(a['active_mask'].sum()),null_count=int((~a['active_mask']).sum()),
            arrays=save_arrays(output,Path('oracle_features')/key/f'class{c}',a))
        # Verify persisted bytes too, before fitting a diagnostic-only oracle.
        a=load_cache(output,cache)
        oracles.append(oracle_fit(a,lay['weights'],lay['uid_rank'],seed=seed,source_stage=source_stage,class_id=c))
        caches.append(cache)
    prototypes=operational(frozen,seed,source_stage);original=prototypes.copy();metrics={}
    stages=[target_stage] if target_stage==source_stage+1 else [1,2]
    maps=[]
    for stage in stages:
        path=output/'transport_models'/f'seed{seed}_stage{stage}.json'
        check_hash(path,barrier['model_sha256'][path.name]);record=read_json(path)
        require(record['metadata']==metadata and record['seed']==seed and record['stage_index']==stage and
            record['models']['T2']['optimizer_steps']==1000, 'mixed/incomplete evaluator map')
        maps.append(record['models'])
    for method in ('T0','T1','T2'):
        candidate=transported(prototypes,[m[method] for m in maps])
        angles=[dict(class_id=c,**angular_match(candidate[c],oracles[c]['centers'],oracles[c]['active'])) for c in range(3)]
        metrics[method]=dict(class_angles=angles,foreground_macro_angular_error=float(np.mean([a['mean_angular_error'] for a in angles[1:]])),
            accuracy=prototype_accuracy(arrays,weights,candidate),transported_centers=candidate.tolist())
    require(np.array_equal(prototypes,original), 'historical prototype bank changed during evaluator')
    entry=dict(context,metadata=metadata,kind='chain' if target_stage-source_stage==2 else 'immediate',checkpoint=cp,
        class_caches=caches,case_support=cases,oracle_fits=oracles,metrics=metrics,all_finite=True,operational_refit=False,
        immutable_chain_input=True,transform_fit_called=False,model_optimizer_steps=0,transport_optimizer_steps_in_evaluator=0,
        oracle_GT_used_for_transform_fit=False,test_gt_usage='none')
    write_json(output/'oracle_units'/f'{key}.json',entry)
    print(f'oracle complete {key}; all three maps evaluated',flush=True)
    return entry
