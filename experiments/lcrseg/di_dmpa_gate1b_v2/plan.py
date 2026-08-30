"""Label-free exact inherited case splits and coordinate hash ranking."""
import csv
import heapq
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from di_dmpa_gate1_v2.features import weight_hash
from .binding import H, DOMAINS, require, check_hash, safe_asset, write_json, write_text


def validate_split(split):
    stage,seed=split['stage_index'],split['seed']
    require(stage in (1,2) and seed in (0,1,2), 'wrong transition/seed')
    require(split['source_domain']==DOMAINS[stage-1] and split['current_domain']==DOMAINS[stage], 'wrong transition domains')
    fit,hold=split['fit_case_ids'],split['holdout_case_ids']
    require(len(fit)==(50 if stage==1 else 32) and len(hold)==(13 if stage==1 else 9), 'wrong split counts')
    require(len(set(fit+hold))==len(fit)+len(hold), 'fit/holdout overlap or duplicates')
    require(split['split_seed']==20261830+100*seed+stage, 'split seed changed')
    ordered=sorted(fit+hold,key=lambda c:(H(['transport-split-v1',split['split_seed'],c]),c))
    require(ordered==fit+hold, 'registered split hash rank changed')


def image_records(data_root, p, split):
    validate_split(split)
    asset=next(a for a in p['benchmark']['manifest_assets'] if a['seed']==split['seed'])
    path=Path(data_root)/f'manifests/training/lcrseg_v1_seed{split["seed"]}.csv'
    check_hash(path,asset['sha256'])
    with path.open(newline='') as stream:
        rows=[r for r in csv.DictReader(stream) if r['dataset']=='fundus' and r['site_or_vendor']==split['current_domain'] and r['primary_20pct_split']=='train_unlabeled']
    expected=split['fit_case_ids']+split['holdout_case_ids']
    require(len(rows)==len(expected) and {r['case_id'] for r in rows}==set(expected), 'current-unlabeled case mismatch')
    records={}
    for r in rows:
        require(not r['label_h5_relpath'] and not r['label_sha256'], 'hidden GT in unlabeled record')
        safe_asset(data_root,r['image_h5_relpath'])
        records[r['case_id']]={k:r[k] for k in ('case_id','image_h5_relpath','image_sha256')}
    return records


def coordinates(seed, stage, case_id):
    # heapq.nsmallest gives the exact sorted prefix without retaining all hashes.
    ranked=((H(['transport-pixel-v1',seed,stage,case_id,y,x]),y,x) for y in range(384) for x in range(384))
    return [[y,x] for _,y,x in heapq.nsmallest(2048,ranked)]


def case_plan(task):
    seed,stage,record=task
    require(set(record)=={'case_id','image_h5_relpath','image_sha256'}, 'labels or extra role data in plan API')
    coords=coordinates(seed,stage,record['case_id'])
    return dict(record,coordinates=coords,coordinate_uid_sha256=H([[record['case_id'],y,x] for y,x in coords]))


def materialize(data_root, p, output, metadata):
    splits=p['transport']['split_plans'];tasks=[]
    for split in splits:
        records=image_records(data_root,p,split)
        tasks.extend((split['seed'],split['stage_index'],records[c]) for c in split['fit_case_ids']+split['holdout_case_ids'])
    with ProcessPoolExecutor(max_workers=16) as pool:cases=list(pool.map(case_plan,tasks,chunksize=1))
    units=[];cursor=0
    for split in splits:
        for partition in ('fit','holdout'):
            count=len(split[f'{partition}_case_ids']);selected=cases[cursor:cursor+count];cursor+=count
            require([r['case_id'] for r in selected]==split[f'{partition}_case_ids'], 'coordinate task order changed')
            unit=dict(seed=split['seed'],stage_index=split['stage_index'],domain=split['current_domain'],role='train_unlabeled',partition=partition,
                split_seed=split['split_seed'],split_hash=split['split_hash'],cases=selected,case_count=count,registered_count=count*2048)
            lay=layout(unit);unit.update(coordinate_uid_hash=H(lay['uids']),original_weight_hash=weight_hash(lay['weights']))
            units.append(unit)
    require(len(units)==12 and cursor==312, 'incomplete coordinate plan')
    plan=dict(schema_version=1,scope='GATE1B_V2_ONLY',metadata=metadata,units=units,labels_read=False,all_methods_share_coordinates=True)
    digest=write_json(Path(output)/'SHARED_TRANSPORT_COORDINATE_PLAN.json',plan)
    write_text(Path(output)/'SHARED_TRANSPORT_COORDINATE_PLAN.sha256',digest+'  SHARED_TRANSPORT_COORDINATE_PLAN.json\n')
    write_json(Path(output)/'TRANSPORT_SPLIT_AND_COORDINATE_AUDIT.json',dict(status='PASS',metadata=metadata,coordinate_plan_sha256=digest,
        units=12,cases=312,registered_rows=312*2048,fit_holdout_disjoint=True,labels_read=False,transport_optimizer_steps=0,
        split_plans=splits,coordinate_hashes=[{k:u[k] for k in ('seed','stage_index','partition','registered_count','coordinate_uid_hash','original_weight_hash')} for u in units]))
    return plan,digest


def layout(unit):
    require(unit['seed'] in (0,1,2) and unit['stage_index'] in (1,2), 'unregistered paired seed/stage')
    require(unit['role']=='train_unlabeled' and unit['partition'] in ('fit','holdout') and unit['domain']==DOMAINS[unit['stage_index']], 'non-current/unlabeled unit')
    uids=[];case_ids=[]
    for case in unit['cases']:
        require(set(case)=={'case_id','image_h5_relpath','image_sha256','coordinates','coordinate_uid_sha256'}, 'label/historical data in paired plan')
        coords=case['coordinates']
        require(len(coords)==2048 and len({tuple(x) for x in coords})==2048, 'wrong coordinate multiplicity')
        require(all(len(x)==2 and all(isinstance(v,int) and 0<=v<384 for v in x) for x in coords), 'invalid coordinates')
        current=[[case['case_id'],y,x] for y,x in coords]
        require(H(current)==case['coordinate_uid_sha256'], 'coordinate UID order changed')
        uids.extend(current);case_ids.extend([case['case_id']]*len(coords))
    require(len(uids)==unit['registered_count']==unit['case_count']*2048 and unit['case_count']>0, 'missing paired coordinates')
    require(len({c['case_id'] for c in unit['cases']})==unit['case_count'], 'duplicate case in paired unit')
    return dict(uids=uids,case_ids=np.asarray(case_ids),weights=np.full(len(uids),1/len(uids),dtype=np.float64))
