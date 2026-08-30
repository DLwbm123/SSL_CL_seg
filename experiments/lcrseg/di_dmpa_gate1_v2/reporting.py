"""Worst-case-radius-only admission, complete controls and immutable reports."""
from pathlib import Path
import numpy as np

from di_dmpa_gate1.gate1a_reporting import primary_conditions, relative_reduction, write_csv
from .binding import PANELS, IncompletePanel, require, write_json, write_text
from .geometry import verify_null_identity


def validate_rows(rows):
    expected={(p,s,t,c,k) for p in PANELS for s in range(3) for t in range(3) for c in range(3) for k in (1,2,3,5)}
    keys=[(r['panel_id'],r['seed'],r['stage_index'],r['class_id'],r['K']) for r in rows]
    if len(keys)!=432 or len(set(keys))!=432 or set(keys)!=expected:raise IncompletePanel('432 complete unique geometry jobs required')
    for r in rows:
        require(r['admission_radius_field']=='R95_null_worst_case','conditional R95 forbidden for admission')
        require(set(r['metrics'])=={'train_labeled','val'} and len(r['bootstrap'])==5,'incomplete role/bootstrap metrics')
        for role,m in r['metrics'].items():
            require(m['metric_schema']=='NULL_AWARE_SPHERE_V2' and m['admission_radius_field']=='R95_null_worst_case','wrong metric schema/radius')
            require(m['registered_count']==r['expected_registered_counts'][role]==m['full_uid_count_used'],'missing UID/null rows')
            require(m['null_count']==r['expected_null_counts'][role]==m['null_rows_retained'],'hidden null drop')
            require(m['active_count']+m['null_count']==m['registered_count'],'support counts do not sum')
            require(all(np.isfinite(m[k]) for k in ('Q_null_worst_case','R95_null_worst_case','null_mass')),'nonfinite metric')
        require(all(len(b['matched_cosines'])==r['K'] for b in r['bootstrap']),'bootstrap slots lost')
    lookup={(r['panel_id'],r['seed'],r['stage_index'],r['class_id'],r['K']):r for r in rows}
    identities=[]
    for key,r in lookup.items():
        if r['K']==1:continue
        reference=lookup[(*key[:4],1)]
        for role in ('train_labeled','val'):
            identities.append(dict(panel_id=r['panel_id'],seed=r['seed'],stage_index=r['stage_index'],class_id=r['class_id'],K=r['K'],role=role,
                **verify_null_identity(reference['metrics'][role],r['metrics'][role])))
    return identities


def statistics(rows,panel,K):
    lookup={(r['seed'],r['stage_index'],r['class_id'],r['K']):r for r in rows if r['panel_id']==panel}
    radius=lambda s,t,c,k:lookup[(s,t,c,k)]['metrics']['val']['R95_null_worst_case']
    reductions=[relative_reduction(np.mean([radius(s,t,c,1) for c in (1,2)]),np.mean([radius(s,t,c,K) for c in (1,2)])) for s in range(3) for t in range(3)]
    occ=[o for s in range(3) for t in range(3) for c in (1,2) for o in lookup[(s,t,c,K)]['metrics']['train_labeled']['occupancy'] if o>0]
    stability=[v for s in range(3) for t in range(3) for c in (1,2) for b in lookup[(s,t,c,K)]['bootstrap'] for v in b['matched_cosines']]
    deltas=[float(np.mean([radius(s,t,c,1)-radius(s,t,c,K) for s in range(3) for c in (1,2)])) for t in range(3)]
    return dict(A1_improving_units=sum(radius(s,t,c,K)<radius(s,t,c,1) for s in range(3) for t in range(3) for c in (1,2)),foreground_unit_count=18,
        A2_median_relative_fg_macro_R95_reduction=None if any(v is None for v in reductions) else float(np.median(reductions)),
        A2_nine_pair_relative_reductions=reductions,A3_active_cluster_fraction_occupancy_at_least_005=float(np.mean(np.asarray(occ)>=.05)) if occ else 0.,
        A3_active_cluster_count=len(occ),A4_matched_cosine_median=float(np.median(stability)),A4_matched_slot_count=len(stability),
        A5_improving_domain_count=sum(d>0 for d in deltas),A5_domain_mean_radius_decreases=deltas,A6_background_excluded=True,
        admission_radius_field='R95_null_worst_case')


def adjudicate(rows):
    identities=validate_rows(rows)
    stats={p:{str(k):statistics(rows,p,k) for k in (2,3,5)} for p in PANELS}
    primary={k:dict(statistics=s,conditions=primary_conditions(s)) for k,s in stats['B0-EMA'].items()}
    passing=[k for k in (2,3,5) if all(primary[str(k)]['conditions'].values())]
    unsupported=all(r['fit']['directional_support']=='NONE' for r in rows if r['panel_id']=='B0-EMA' and r['class_id'] in (1,2) and r['K']==1)
    verdict='PASS_MULTI_MODALITY_SUPPORTED' if passing else 'FAIL_DIRECTIONAL_SUPPORT_NOT_SUPPORTED' if unsupported else 'FAIL_MULTI_MODALITY_NOT_SUPPORTED'
    return dict(prototype_geometry_status=verdict,primary_panel='B0-EMA',units_per_panel=18,primary_A1_A6=primary,
        passing_K=passing,selected_K=min(passing) if passing else 1,
        selected_K_role='PRIMARY_ADMITTED_K' if passing else 'EXPLICIT_DOWNSTREAM_FALLBACK_ONLY',
        panel_summaries=stats,control_thresholds_applied=False,admission_radius_field='R95_null_worst_case',null_identity_checks=identities)


def report(output,metadata,rows,census):
    output=Path(output);verdict=adjudicate(rows)
    status={**metadata,**{k:v for k,v in verdict.items() if k!='null_identity_checks'},
        'v1_closure_status':'CLOSED_V1_FEATURE_SUPPORT_ASSUMPTION_FALSIFIED','feature_units_completed':72,
        'geometry_jobs_completed':432,'support_census':census,'gate1_overall_status':'INCOMPLETE_GATE1B_GATE1C_NOT_RUN',
        'errors':[],'next_action':'STOP_FOR_INDEPENDENT_REVIEW'}
    write_json(output/'GATE1A_V2_STATUS.json',status)
    quant=[];occ=[];stable=[];boundary=[]
    for r in rows:
        ids={k:r[k] for k in ('panel_id','seed','stage_index','domain','class_id','K')}
        for role,m in r['metrics'].items():
            quant.append({**ids,'role':role,**{k:v for k,v in m.items() if k not in ('occupancy','active_assignment_slots')}})
            for slot,value in enumerate(m['occupancy']):occ.append({**ids,'role':role,'slot':slot,'occupancy_active_conditional':value,'null_mass':m['null_mass']})
        for b in r['bootstrap']:
            for slot,value in enumerate(b['matched_cosines']):stable.append({**ids,'replicate':b['replicate'],'slot':slot,'matched_cosine':value,'directional_support':b['fit']['directional_support']})
        for role,strata in r['boundary_interior'].items():
            for name,m in strata.items():
                boundary.append({**ids,'role':role,'stratum':name,'null_mass':None if m is None else m['null_mass'],
                    'registered_count':0 if m is None else m['registered_count'],'null_count':0 if m is None else m['null_count'],
                    'Q_null_worst_case':None if m is None else m['Q_null_worst_case'],'R95_null_worst_case':None if m is None else m['R95_null_worst_case'],
                    'Q_directional_conditional':None if m is None else m['Q_directional_conditional'],'R95_directional':None if m is None else m['R95_directional']})
    macro=[]
    for p in PANELS:
        for s in range(3):
            for t in range(3):
                for k in (1,2,3,5):
                    fg=[r for r in rows if (r['panel_id'],r['seed'],r['stage_index'],r['K'])==(p,s,t,k) and r['class_id'] in (1,2)]
                    for role in ('train_labeled','val'):
                        value=dict(panel_id=p,seed=s,stage_index=t,domain=fg[0]['domain'],class_id='foreground_macro',K=k,role=role)
                        for field in ('Q_null_worst_case','R95_null_worst_case','cosine_distance_p95_null_worst_case','null_mass','active_direction_mass',
                                      'Q_directional_conditional','R95_directional','cosine_distance_p95_directional'):
                            vals=[r['metrics'][role][field] for r in fg]
                            value[field]=None if any(v is None for v in vals) else float(np.mean(vals))
                        macro.append(value)
                        quant.append({key:value.get(key) for key in quant[0]})
    write_json(output/'PROTOTYPE_GEOMETRY_DIAGNOSTIC_V2.json',dict(metadata=metadata,verdict=verdict,units=rows,foreground_macro_units=macro))
    for name,data in [('prototype_quantization_v2.csv',quant),('prototype_occupancy_v2.csv',occ),('prototype_stability_v2.csv',stable),('prototype_boundary_interior_v2.csv',boundary)]:write_csv(output/name,data)
    text=f"# Gate1A v2 null-aware geometry\n\n{verdict['prototype_geometry_status']}; selected_K={verdict['selected_K']}.\n\n72 feature units and432 geometry jobs complete. All UID/null rows retained. Only B0-EMA admits; radius=R95_null_worst_case.\n\n"
    for p in PANELS:
        text+=f'## {p}\n\n'
        for k in (2,3,5):text+=f"- K={k}: {verdict['panel_summaries'][p][str(k)]}\n"
        text+='\n'
    text+='All inherited v1 evidence remains immutable. Optimizer steps=0, no hidden/test GT, no method registration, training or downstream Gate1B/C. STOP_FOR_INDEPENDENT_REVIEW.\n'
    write_text(output/'GATE1A_V2_FINAL_REPORT.md',text)
    write_text(output/'PROTOTYPE_GEOMETRY_DIAGNOSTIC_V2.md',text)
    return status
