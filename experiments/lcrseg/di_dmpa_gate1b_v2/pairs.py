"""Read-only paired features, complete support census and immutable caches."""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from di_dmpa_gate1.feature_extraction import _images, seed_after_load
from di_dmpa_gate1.gate1a_reporting import write_csv
from di_dmpa_gate1_v2.features import split_support, validate_cache, ImmutableModels, weight_hash
from .binding import (DOMAINS, H, ProtocolError, NonfiniteFeature, IncompleteEvidence,
    NoDirectionalPairs, require, checkpoint, load_b0, sha256, check_hash, write_json, read_json)
from .plan import layout

STATES = ('AA','A_NULL','NULL_A','NULL_NULL')


def pair_states(source_active, target_active):
    return 2*(~source_active).astype(np.uint8)+(~target_active).astype(np.uint8)


def support(states, weights):
    require(states.ndim==weights.ndim==1 and len(states)==len(weights)>0, 'wrong support shape')
    require(np.isin(states,range(4)).all() and np.isfinite(weights).all() and (weights>=0).all() and weights.sum()>0, 'invalid paired support/weight')
    mass={name:float(weights[states==i].sum()/weights.sum()) for i,name in enumerate(STATES)}
    return dict(registered_count=len(states),counts={name:int((states==i).sum()) for i,name in enumerate(STATES)},mass=mass,
        source_null_mass=mass['NULL_A']+mass['NULL_NULL'],target_null_mass=mass['A_NULL']+mass['NULL_NULL'])


def extract_arrays(source, target, unit, data_root, device):
    lay=layout(unit)
    require(not source.training and not target.training, 'paired encoders must be eval')
    seed_after_load(unit['split_seed'])
    groups={side:{name:[] for name in ('directions','active_mask','raw_norms')} for side in ('source','target')}
    cases=[]
    with torch.no_grad():
        for start in range(0,len(unit['cases']),8):
            batch=unit['cases'][start:start+8];images=_images(batch,data_root).to(device)
            features={side:model(images,stochastic_classifier=False)[1] for side,model in [('source',source),('target',target)]}
            for side,f in features.items():
                require(tuple(f.shape)==(len(batch),16,384,384) and f.dtype==torch.float32, 'wrong paired feature tensor')
                if not torch.isfinite(f).all():raise NonfiniteFeature(f'{side} full-map NaN/Inf; cases={[c["case_id"] for c in batch]}')
            for bi,case in enumerate(batch):
                coords=np.asarray(case['coordinates']);arrays={};stats={}
                for side,f in features.items():
                    raw=f[bi].detach().cpu().numpy();a=split_support(raw[:,coords[:,0],coords[:,1]].T);arrays[side]=a
                    for name,value in a.items():groups[side][name].append(value)
                    positive=a['raw_norms'][a['raw_norms']>0]
                    stats.update({side+'_minimum_positive_norm':float(positive.min()) if len(positive) else None,
                        side+'_full_map_exact_zero_count':int(np.all(raw==0,axis=0).sum()),
                        side+'_full_map_nonfinite_count':0,side+'_registered_nonfinite_count':0,
                        side+'_null_count':int((~a['active_mask']).sum())})
                states=pair_states(arrays['source']['active_mask'],arrays['target']['active_mask'])
                stats.update(support(states,np.ones(len(states),dtype=np.float64)))
                nullcoords=coords[states!=0].tolist()
                cases.append(dict(seed=unit['seed'],stage_index=unit['stage_index'],domain=unit['domain'],partition=unit['partition'],
                    role='train_unlabeled',case_id=case['case_id'],coordinate_uid_hash=case['coordinate_uid_sha256'],
                    null_coordinate_hash=H([[case['case_id'],y,x] for y,x in nullcoords]),first_null_coordinates=nullcoords[:32],
                    mismatch_fraction=float(np.isin(states,(1,2)).mean()),null_normalized=False,**stats))
    arrays={side+'_'+name:np.concatenate(parts) for side,group in groups.items() for name,parts in group.items()}
    for side in groups:validate_cache({name:arrays[side+'_'+name] for name in groups[side]},len(lay['uids']))
    arrays['pair_state']=pair_states(arrays['source_active_mask'],arrays['target_active_mask'])
    require(all(p.grad is None for model in (source,target) for p in model.parameters()), 'model gradient appeared')
    return arrays,cases


def save_arrays(output, relative, arrays):
    descriptors={}
    for name,array in arrays.items():
        rel=Path(relative)/(name+'.npy');path=Path(output)/rel;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:np.save(stream,array,allow_pickle=False)
        descriptors[name]=dict(path=str(rel),shape=list(array.shape),dtype=str(array.dtype),sha256=sha256(path))
    return descriptors


def extract_transition(root, data_root, p, units, output, metadata, *, device='cuda:0'):
    first=units[0];seed,stage=first['seed'],first['stage_index'];key=f'seed{seed}_stage{stage}'
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=ProtocolError('model optimizer construction forbidden')):
        with ExitStack() as stack:
            models={}
            for side,index in [('source',stage-1),('target',stage)]:
                cp=checkpoint(p,seed,index);loaded=load_b0(root,p,seed,index,device)
                stack.enter_context(ImmutableModels(loaded,cp,Path(output)/'paired_audits'/key/side,dict(metadata,phase='paired_extraction',transport_optimizer_steps_before_phase=0)))
                models[side]=loaded['ema_teacher']
            for unit in units:
                arrays,cases=extract_arrays(models['source'],models['target'],unit,data_root,device)
                lay=layout(unit);summary=support(arrays['pair_state'],lay['weights'])
                entry={k:unit[k] for k in ('seed','stage_index','domain','role','partition','case_count','registered_count','coordinate_uid_hash','original_weight_hash','split_hash')}
                entry.update(metadata=metadata,source_checkpoint=checkpoint(p,seed,stage-1),target_checkpoint=checkpoint(p,seed,stage),
                    arrays=save_arrays(output,Path('paired_features')/f'{key}_{unit["partition"]}',arrays),support=summary,
                    case_support=cases,all_finite=True,all_rows_preserved=True,labels_read=False,old_raw_cache_reused=False)
                write_json(Path(output)/'paired_units'/f'{key}_{unit["partition"]}.json',entry)
                print(f'paired complete {key} {unit["partition"]} {summary["counts"]}',flush=True)
    if torch.cuda.is_available():torch.cuda.empty_cache()


def load_pair(output, entry, unit):
    lay=layout(unit);expected=set(['pair_state']+[s+'_'+n for s in ('source','target') for n in ('directions','active_mask','raw_norms')])
    require(set(entry['arrays'])==expected and entry['all_rows_preserved'] and not entry['labels_read'], 'invalid paired cache schema')
    for field in ('seed','stage_index','domain','role','partition','registered_count','coordinate_uid_hash','original_weight_hash','split_hash'):
        require(entry[field]==unit[field], 'paired cache/plan metadata differs: '+field)
    arrays={}
    for name,desc in entry['arrays'].items():
        path=Path(output)/desc['path'];check_hash(path,desc['sha256']);array=np.load(path,mmap_mode='r',allow_pickle=False)
        require(list(array.shape)==desc['shape'] and str(array.dtype)==desc['dtype'], 'paired array descriptor changed')
        arrays[name]=array
    for side in ('source','target'):validate_cache({n:arrays[side+'_'+n] for n in ('directions','active_mask','raw_norms')},len(lay['uids']))
    require(arrays['pair_state'].dtype==np.uint8 and np.array_equal(arrays['pair_state'],pair_states(arrays['source_active_mask'],arrays['target_active_mask'])), 'pair-state mismatch')
    require(H(lay['uids'])==entry['coordinate_uid_hash'] and weight_hash(lay['weights'])==entry['original_weight_hash'], 'original UID/weight mismatch')
    require(support(arrays['pair_state'],lay['weights'])==entry['support'], 'support census/cache mismatch')
    return dict(x=arrays['source_directions'],y=arrays['target_directions'],source_active=arrays['source_active_mask'],
        target_active=arrays['target_active_mask'],pair_state=arrays['pair_state'],weights=lay['weights'],seed=unit['seed'],
        stage_index=unit['stage_index'],domain=unit['domain'],role='train_unlabeled',partition=unit['partition'])


def census(output, entries, plan, metadata):
    keys=[(e['seed'],e['stage_index'],e['partition']) for e in entries]
    if len(keys)!=12 or set(keys)!={(s,t,r) for s in range(3) for t in (1,2) for r in ('fit','holdout')}:
        raise IncompleteEvidence('all12 paired units required before any transport fit')
    units=[];case_rows=[]
    for e in entries:
        u=next(u for u in plan['units'] if (u['seed'],u['stage_index'],u['partition'])==(e['seed'],e['stage_index'],e['partition']))
        load_pair(output,e,u)
        require(e['metadata']==metadata and e['all_finite'] and not e['old_raw_cache_reused'], 'mixed/invalid paired cache')
        require([c['case_id'] for c in e['case_support']]==[c['case_id'] for c in u['cases']], 'case census coverage changed')
        for state in STATES:require(sum(c['counts'][state] for c in e['case_support'])==e['support']['counts'][state], 'case support counts mismatch')
        units.append(dict(seed=e['seed'],stage_index=e['stage_index'],domain=e['domain'],partition=e['partition'],
            **e['support'],coordinate_uid_hash=e['coordinate_uid_hash'],original_weight_hash=e['original_weight_hash'],
            maximum_case_mismatch_fraction=max(c['mismatch_fraction'] for c in e['case_support'])))
        case_rows.extend(e['case_support'])
    def aggregate(selected, **ids):
        return dict(ids,units=len(selected),registered_count=sum(u['registered_count'] for u in selected),
            counts={k:sum(u['counts'][k] for u in selected) for k in STATES},mass={k:float(np.mean([u['mass'][k] for u in selected])) for k in STATES},
            source_null_mass=float(np.mean([u['source_null_mass'] for u in selected])),target_null_mass=float(np.mean([u['target_null_mass'] for u in selected])),
            maximum_case_mismatch_fraction=max(u['maximum_case_mismatch_fraction'] for u in selected))
    transitions=[aggregate([u for u in units if u['stage_index']==t and u['partition']==r],stage_index=t,domain=DOMAINS[t],partition=r) for t in (1,2) for r in ('fit','holdout')]
    seeds=[aggregate([u for u in units if u['seed']==s],seed=s) for s in range(3)]
    result=dict(status='PASS',metadata=metadata,paired_units_completed=12,registered_count=sum(u['registered_count'] for u in units),
        counts={k:sum(u['counts'][k] for u in units) for k in STATES},units=units,transitions=transitions,seeds=seeds,
        aggregation='macro equal paired-unit masses for census summaries; each admission transition averages its three seeds only',
        transport_optimizer_steps=0,labels_read=False,all_rows_preserved=True,all_finite=True)
    write_json(Path(output)/'PAIRED_FEATURE_SUPPORT_CENSUS.json',result)
    for name,rows in [('paired_support_by_case.csv',case_rows),('paired_support_by_transition.csv',transitions),('paired_support_by_seed.csv',seeds)]:write_csv(Path(output)/name,rows)
    if any(u['mass']['AA']==0 for u in units):raise NoDirectionalPairs('at least one fit/holdout unit has zero AA mass; no transforms fitted')
    return result
