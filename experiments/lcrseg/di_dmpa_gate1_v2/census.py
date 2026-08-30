"""Complete72 support census, immutable caches, and before-clustering barrier."""
from pathlib import Path
import numpy as np

from di_dmpa_gate1.sampling import sample_layout
from di_dmpa_gate1.gate1a_reporting import write_csv
from .binding import H, PANELS, IncompletePanel, check_hash, require, write_json
from .features import load_cache, weight_hash


def complete_feature_keys(entries):
    keys=[(e['panel_id'],e['seed'],e['stage_index'],e['role']) for e in entries]
    expected={(p,s,t,r) for p in PANELS for s in range(3) for t in range(3) for r in ('train_labeled','val')}
    if len(keys)!=72 or len(set(keys))!=72 or set(keys)!=expected:
        raise IncompletePanel(f'72 unique feature units required before first geometry job; observed={len(keys)}')


def validate_features(output,entries,plan,metadata):
    complete_feature_keys(entries)
    for e in entries:
        require(e['sampling_plan_sha256']==metadata['sampling_plan_sha256'],'plan hash changed')
        require(e['metadata']['diagnostic_code_git_commit']==metadata['diagnostic_code_git_commit'],'mixed code caches')
        require(e['all_finite'] and e['null_rows_preserved'] and not e['old_raw_cache_reused'],'invalid support flags')
        unit=next(u for u in plan['units'] if u['seed']==e['seed'] and u['stage_index']==e['stage_index'] and u['role']==e['role'])
        require(e['sampling_unit_sha256']==H(unit),'unit coordinate/multiplicity changed')
        require(len(e['class_caches'])==3 and {x['class_id'] for x in e['class_caches']}=={0,1,2},'missing class cache')
        for cache in e['class_caches']:
            c=cache['class_id'];layout=sample_layout(unit,c)
            require(cache['registered_count']==len(layout['uids']),'missing registered/null cache rows')
            require(cache['uid_order_sha256']==H(layout['uids']),'UID order changed')
            require(cache['original_weight_order_sha256']==weight_hash(layout['weights']),'weights changed')
            arrays=load_cache(output,cache)
            rows=[x for x in e['case_support'] if x['class_id']==c]
            require([x['case_id'] for x in rows]==[case['case_id'] for case in unit['cases']],'missing/duplicate census case')
            require(sum(x['registered_count'] for x in rows)==cache['registered_count'],'census row count mismatch')
            require(sum(x['null_count'] for x in rows)==cache['null_count'],'census hidden null drop')
            actual_mass=float(layout['weights'][~arrays['active_mask']].sum()/layout['weights'].sum())
            require(np.isclose(sum(x['weighted_null_mass'] for x in rows),actual_mass,atol=1e-12,rtol=1e-10),'null weight mismatch')


def compile_census(output,entries,metadata):
    complete_feature_keys(entries)
    rows=[row for e in entries for row in e['case_support']]
    units=[]
    for e in entries:
        selected=e['case_support']
        units.append(dict(panel_id=e['panel_id'],seed=e['seed'],stage_index=e['stage_index'],domain=e['domain'],role=e['role'],
            total_registered=sum(r['registered_count'] for r in selected),total_active=sum(r['active_count'] for r in selected),
            total_null=sum(r['null_count'] for r in selected),null_mass_by_class={str(c):sum(r['weighted_null_mass'] for r in selected if r['class_id']==c) for c in range(3)},
            maximum_case_class_null_fraction=max(r['null_fraction'] for r in selected),
            cases_with_no_active_directions=[dict(case_id=r['case_id'],class_id=r['class_id']) for r in selected if r['registered_count']>0 and r['active_count']==0],
            all_finite=all(r['registered_nonfinite_count']==r['full_map_nonfinite_count']==0 for r in selected)))
    panels=[]
    for p in PANELS:
        selected=[u for u in units if u['panel_id']==p]
        panels.append(dict(panel_id=p,feature_units=len(selected),**{f:sum(u[f] for u in selected) for f in ('total_registered','total_active','total_null')},
            null_mass_by_class={str(c):float(np.mean([u['null_mass_by_class'][str(c)] for u in selected])) for c in range(3)},
            maximum_case_class_null_fraction=max(u['maximum_case_class_null_fraction'] for u in selected),
            cases_with_no_active_directions=sum(len(u['cases_with_no_active_directions']) for u in selected),all_finite=all(u['all_finite'] for u in selected)))
    census=dict(status='PASS',metadata=metadata,feature_units_completed=72,unique_keys_verified=True,
        clustering_jobs_started=0,registered_count=sum(p['total_registered'] for p in panels),
        active_count=sum(p['total_active'] for p in panels),null_count=sum(p['total_null'] for p in panels),
        panel_aggregation='class null mass is equal mean over18 feature units, not across panels',panels=panels,units=units)
    write_json(Path(output)/'GATE1A_V2_FEATURE_SUPPORT_CENSUS.json',census)
    write_json(Path(output)/'FEATURE_SUPPORT_CENSUS.json',census)
    write_csv(Path(output)/'feature_support_by_case.csv',rows)
    write_csv(Path(output)/'feature_support_by_unit.csv',units)
    write_csv(Path(output)/'feature_support_by_panel.csv',panels)
    return census
