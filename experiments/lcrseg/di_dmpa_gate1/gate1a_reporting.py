"""Complete-panel adjudication and provenance-bearing, append-only reports."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .binding import PANELS, NumericalError, ProtocolError, read_json, require, sha256, write_json, write_text


def expected_keys():
    return {(p,s,t,c,k) for p in PANELS for s in range(3) for t in range(3) for c in range(3) for k in (1,2,3,5)}


def complete_results(results):
    keys=[(r['panel_id'],r['seed'],r['stage_index'],r['class_id'],r['K']) for r in results]
    require(len(keys)==len(set(keys)),"duplicate geometry unit")
    if set(keys)!=expected_keys():
        raise NumericalError(f"incomplete four-panel geometry: {len(keys)}/432")
    hashes={r['metadata']['sampling_plan_sha256'] for r in results}
    require(len(hashes)==1,"panel sampling hash differs")
    for row in results:
        require(set(row['metrics'])=={'train_labeled','val'},"train/val metric missing")
        require(len(row['bootstrap'])==5,"bootstrap replica missing")
        require(all(len(b['matched_cosines'])==row['K'] for b in row['bootstrap']),"bootstrap slot missing")
        for role in ('train_labeled','val'):
            require(set(row['boundary_interior'][role])=={'boundary','interior'},"boundary stratum missing")
        for metrics in row['metrics'].values():
            for field in ('Q_K','R95','cosine_distance_p95'):
                if not np.isfinite(metrics[field]):
                    raise NumericalError('nonfinite final geometry metric')


def relative_reduction(reference,candidate):
    if reference==0:
        return 0.0 if candidate==0 else None
    return (reference-candidate)/reference


def panel_statistics(rows,panel,K):
    lookup={(r['seed'],r['stage_index'],r['class_id'],r['K']):r for r in rows if r['panel_id']==panel}
    r95=lambda s,t,c,k:lookup[(s,t,c,k)]['metrics']['val']['R95']
    improvements=sum(r95(s,t,c,K)<r95(s,t,c,1) for s in range(3) for t in range(3) for c in (1,2))
    reductions=[relative_reduction(np.mean([r95(s,t,c,1) for c in (1,2)]),np.mean([r95(s,t,c,K) for c in (1,2)])) for s in range(3) for t in range(3)]
    median=None if any(v is None for v in reductions) else float(np.median(reductions))
    occupancies=[o for s in range(3) for t in range(3) for c in (1,2) for o in lookup[(s,t,c,K)]['metrics']['train_labeled']['occupancy'] if o>0]
    stable=[v for s in range(3) for t in range(3) for c in (1,2) for b in lookup[(s,t,c,K)]['bootstrap'] for v in b['matched_cosines']]
    domain_deltas=[float(np.mean([r95(s,t,c,1)-r95(s,t,c,K) for s in range(3) for c in (1,2)])) for t in range(3)]
    require(bool(occupancies) and len(stable)==18*5*K,"foreground cluster statistics incomplete")
    return dict(A1_improving_units=int(improvements),foreground_unit_count=18,
                A2_median_relative_fg_macro_R95_reduction=median,A2_nine_pair_relative_reductions=reductions,
                A3_active_cluster_fraction_occupancy_at_least_005=float(np.mean(np.asarray(occupancies)>=.05)),
                A3_active_cluster_count=len(occupancies),A4_matched_cosine_median=float(np.median(stable)),
                A4_matched_slot_count=len(stable),A5_improving_domain_count=sum(x>0 for x in domain_deltas),
                A5_domain_mean_radius_decreases=domain_deltas,A6_background_excluded=True)


def primary_conditions(stats):
    return dict(A1=stats['A1_improving_units']>=12,
                A2=stats['A2_median_relative_fg_macro_R95_reduction'] is not None and stats['A2_median_relative_fg_macro_R95_reduction']>=.10,
                A3=stats['A3_active_cluster_fraction_occupancy_at_least_005']>=.90,
                A4=stats['A4_matched_cosine_median']>=.85,
                A5=stats['A5_improving_domain_count']>=2,A6=stats['A6_background_excluded'])


def adjudicate(results):
    complete_results(results)
    summaries={p:{str(k):panel_statistics(results,p,k) for k in (2,3,5)} for p in PANELS}
    primary={str(k):{'statistics':summaries['B0-EMA'][str(k)],'conditions':primary_conditions(summaries['B0-EMA'][str(k)])} for k in (2,3,5)}
    passing=[k for k in (2,3,5) if all(primary[str(k)]['conditions'].values())]
    return dict(primary_panel='B0-EMA',units_per_panel=18,K_reference=1,K_candidates=[2,3,5],
                primary_A1_A6=primary,passing_K=passing,selected_K=min(passing) if passing else 1,
                selected_K_role='PRIMARY_ADMITTED_K' if passing else 'EXPLICIT_DOWNSTREAM_FALLBACK_ONLY',
                prototype_geometry_status='PASS_MULTI_MODALITY_SUPPORTED' if passing else 'FAIL_MULTI_MODALITY_NOT_SUPPORTED',
                panel_summaries=summaries,control_thresholds_applied=False,
                control_note='Statistics are separate/descriptive; no control selects K, gets admission thresholds, or rescues primary.')


def write_csv(path,rows):
    require(bool(rows),'cannot fabricate an empty completed table')
    fields=list(rows[0])
    with Path(path).open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def foreground_macro_rows(results):
    grouped={}
    for row in results:
        if row['class_id'] not in (1,2):
            continue
        key=(row['panel_id'],row['seed'],row['stage_index'],row['K'])
        grouped.setdefault(key,[]).append(row)
    output=[]
    for key,classes in sorted(grouped.items()):
        require(len(classes)==2,'foreground macro requires rim and cup')
        for role in ('train_labeled','val'):
            output.append(dict(panel_id=key[0],seed=key[1],stage_index=key[2],domain=classes[0]['domain'],K=key[3],role=role,
                class_id='foreground_macro',aggregation='equal rim/cup mean',
                **{metric:float(np.mean([c['metrics'][role][metric] for c in classes])) for metric in ('Q_K','R95','cosine_distance_p95')}))
    return output


def artifact_manifest(output):
    output=Path(output)
    excluded={'GATE1A_ARTIFACT_MANIFEST.json','GATE1A_ARTIFACT_MANIFEST.sha256'}
    files=[dict(path=str(p.relative_to(output)),size_bytes=p.stat().st_size,sha256=sha256(p))
           for p in sorted(output.rglob('*')) if p.is_file() and p.name not in excluded
           and not p.name.endswith(('.active.log','.pid','.exit'))]
    result=dict(schema_version=1,self_hash_policy='manifest excludes itself and its sidecar; sidecar gives raw manifest hash',files=files)
    digest=write_json(output/'GATE1A_ARTIFACT_MANIFEST.json',result)
    write_text(output/'GATE1A_ARTIFACT_MANIFEST.sha256',digest+'  GATE1A_ARTIFACT_MANIFEST.json\n')
    return result


def report(output,metadata,results):
    output=Path(output)
    verdict=adjudicate(results)
    status={**metadata,**verdict,'preregistration_commit':metadata['preregistration_git_commit'],
            'baseline_freeze_hash':metadata['baseline_freeze_sha256'],
            'four_panel_status':{p:dict(status='COMPLETE',foreground_units=18,total_class_units=27,K_count=4) for p in PANELS},
            'gate1_overall_status':'INCOMPLETE_GATE1B_GATE1C_NOT_RUN','errors':[],
            'next_action':'STOP_FOR_INDEPENDENT_REVIEW','report_git_commit_resolution':'Commit first adding these exact report bytes; distinct from diagnostic_code_git_commit'}
    write_json(output/'GATE1A_STATUS.json',status)
    macro_rows=foreground_macro_rows(results)
    write_json(output/'PROTOTYPE_GEOMETRY_DIAGNOSTIC.json',dict(metadata=metadata,verdict=verdict,units=results,foreground_macro_units=macro_rows))
    quantization=[];occupancy=[];stability=[];boundary=[]
    binding={k:metadata[k] for k in ('preregistration_git_commit','preregistration_json_sha256','preregistration_md_sha256','authorization_git_commit','diagnostic_code_git_commit','sampling_plan_sha256')}
    for row in results:
        ids={k:row[k] for k in ('panel_id','seed','stage_index','domain','class_id','K')}
        for role,metrics in row['metrics'].items():
            quantization.append({**ids,'role':role,**{k:v for k,v in metrics.items() if k not in ('occupancy','active_assignment_slots')},**binding})
            for slot,value in enumerate(metrics['occupancy']):
                occupancy.append({**ids,'role':role,'slot':slot,'occupancy':value,'active_assignment':value>0,**binding})
        for replicate in row['bootstrap']:
            for slot,value in enumerate(replicate['matched_cosines']):
                stability.append({**ids,'replicate':replicate['replicate'],'slot':slot,'matched_cosine':value,**binding})
        for role,strata in row['boundary_interior'].items():
            for name,metrics in strata.items():
                boundary.append({**ids,'role':role,'stratum':name,'Q_K':None if metrics is None else metrics['Q_K'],
                    'R95':None if metrics is None else metrics['R95'],'cosine_distance_p95':None if metrics is None else metrics['cosine_distance_p95'],
                    'null_reason':'empty sampled own-class stratum' if metrics is None else '',**binding})
    for macro in macro_rows:
        template={k:None for k in quantization[0]}
        template.update({k:v for k,v in macro.items() if k in template})
        template.update(binding)
        quantization.append(template)
    for name,rows in [('prototype_quantization.csv',quantization),('prototype_occupancy.csv',occupancy),('prototype_stability.csv',stability),('prototype_boundary_interior.csv',boundary)]:
        write_csv(output/name,rows)
    table='\n'.join('| '+str(k)+' | '+' | '.join(str(verdict['primary_A1_A6'][str(k)]['statistics'][f]) for f in (
        'A1_improving_units','A2_median_relative_fg_macro_R95_reduction','A3_active_cluster_fraction_occupancy_at_least_005',
        'A4_matched_cosine_median','A5_improving_domain_count'))+' | '+str(verdict['primary_A1_A6'][str(k)]['conditions'])+' |' for k in (2,3,5))
    text=f"# Gate 1A geometry report\n\nStatus: {verdict['prototype_geometry_status']}; selected K={verdict['selected_K']} ({verdict['selected_K_role']}).\n\nOnly B0-EMA controls admission. Four panels complete independently; 18 foreground units each.\n\n| K | A1 count | A2 median relative R95 reduction | A3 occupancy fraction | A4 stability median | A5 domains | Conditions |\n|---|---|---|---|---|---|---|\n{table}\n\nA6 excludes background from every admission statistic. No rounding precedes decisions.\n\n"
    for panel in PANELS[1:]:
        text+=f"## {panel} — descriptive control only\n\n"
        for k in (2,3,5):
            text+=f"- K={k}: {verdict['panel_summaries'][panel][str(k)]}\n"
        text+='\n'
    text+='## Provenance and stop\n\n'+ '\n'.join(f'- {k}: `{v}`' for k,v in binding.items())
    text+='\n\nBoth optimizer step counts are zero; hidden/test GT usage is none. Gate 1B/1C and overall Gate 1 remain incomplete. All method flags remain false. No training or main merge. Next action: STOP_FOR_INDEPENDENT_REVIEW.\n'
    write_text(output/'PROTOTYPE_GEOMETRY_DIAGNOSTIC.md',text)
    write_text(output/'GATE1A_FINAL_REPORT.md',text+'\nExact commands, tests, warnings, input/model/sampling audits and artifact hashes accompany this report. Report commit is the immutable Git commit first adding these report bytes, not the execution-code commit.\n')
    return status


def blocked_report(output,metadata,error):
    output=Path(output)
    state=getattr(error,'status','BLOCKED_NUMERICAL_FAILURE')
    status={**metadata,'primary_panel':'B0-EMA','prototype_geometry_status':state,'passing_K':None,'selected_K':None,
            'units_per_panel':18,'K_reference':1,'K_candidates':[2,3,5],
            'four_panel_status':{p:{'status':'INCOMPLETE_OR_BLOCKED'} for p in PANELS},
            'gate1_overall_status':'INCOMPLETE','errors':[f'{type(error).__name__}: {error}'],
            'next_action':'STOP_FOR_INDEPENDENT_REVIEW'}
    write_json(output/'GATE1A_STATUS.json',status)
    text=f"# Gate 1A blocked\n\n{state}: {type(error).__name__}: {error}\n\nNo candidate PASS or selected K is published. Partial outputs and this attempt remain preserved. No downstream diagnostics or training.\n"
    write_text(output/'GATE1A_FINAL_REPORT.md',text)
    write_json(output/'PROTOTYPE_GEOMETRY_DIAGNOSTIC.json',dict(status=state,metadata=metadata,errors=status['errors'],complete=False))
    write_text(output/'PROTOTYPE_GEOMETRY_DIAGNOSTIC.md',text)
    return status
