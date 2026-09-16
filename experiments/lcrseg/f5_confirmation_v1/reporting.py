"""Aggregate-only exports; no tensor deserialization, patient IO or launch path."""
import csv
from pathlib import Path
from .protocol import DOC,OLD,B0,B2,F5,METRICS,read,write,validate_plan,metrics,paired,decisions

FILES=('PUBLIC_RESULTS.json','FINAL_REPORT.md','FINAL_METRICS.csv','STAGE_METRICS.csv',
       'PAIRED_COMPARISONS.csv','COST_AND_COMPLETION.json')
FINAL_FIELDS=['family','seed','order','origin',*METRICS]
STAGE_FIELDS=['node_id','family','seed','order','stage','domain','origin','rim','cup','disc_union','macro_Dice']
PAIR_FIELDS=['cohort','reference','aggregation','seed','order',*METRICS]


def csv_write(path,fields,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def export_reports(root,result):
    """Renderer also testable on generated aggregates; not an integrity gate."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    finals=[];stages=[];pairs=[]
    for r in result['rows']:
        i=r['identity'];origin=r['origin']
        if i['stage']==2:finals.append(dict(family=i['family'],seed=i['seed'],order=i['order'],origin=origin,**metrics(r)))
        for domain,s in r['scores'].items():
            stages.append(dict(node_id=r['node_id'],family=i['family'],seed=i['seed'],order=i['order'],stage=i['stage'],
                               domain=domain,origin=origin,**{k:s[k] for k in ('rim','cup','disc_union','macro_Dice')}))
    for cohort,data in [('primary_163_164',result['primary']),('supplementary_162_164',result['supplementary_descriptive'])]:
        for reference,a in data.items():
            for key,values in a['per_order'].items():
                seed,order=key.split('/O');pairs.append(dict(cohort=cohort,reference=reference,aggregation='seed_order',seed=seed,order=order,**values))
            for seed,values in a['per_seed'].items():
                pairs.append(dict(cohort=cohort,reference=reference,aggregation='within_seed_two_orders',seed=seed,order='mean',**values))
            pairs.append(dict(cohort=cohort,reference=reference,aggregation='across_seeds',seed='mean',order='mean',**a['mean']))
    csv_write(root/'FINAL_METRICS.csv',FINAL_FIELDS,finals)
    csv_write(root/'STAGE_METRICS.csv',STAGE_FIELDS,stages)
    csv_write(root/'PAIRED_COMPARISONS.csv',PAIR_FIELDS,pairs)
    write(root/'PUBLIC_RESULTS.json',result)
    write(root/'COST_AND_COMPLETION.json',dict(status=result['status'],publication_status='LOCAL_REPORT_READY',
        new_target_stages=28,imported_target_stages=8,reused_sources=3,new_source_updates=0,
        execution_commit=result['execution_commit'],plan_sha256=result['plan_sha256'],
        costs=result['costs'],integrity=result['integrity'],qualification=result['qualification'],
        environment=result['environment'],historical_import_cost='historical only; excluded from new execution totals'))
    text=['# P1 confirmation — awaiting scientific review','',
          'Local reports only. Public GitHub publication has not been verified by this exporter.',
          'Primary: seeds 163/164; supplementary descriptive: 162–164. Two orders average within seed first.',
          'These are development patients, not independent-patient generalization, original KI, or SOTA evidence.','',
          '| Method | Seed | Order | Origin | Final | Old | Incoming | Forget |',
          '|---|---:|---:|---|---:|---:|---:|---:|']
    for r in finals:text.append('| '+' | '.join(str(r[k]) if k not in METRICS else f'{r[k]:.6f}' for k in FINAL_FIELDS)+' |')
    text+=['','## Frozen budget decisions (not significance tests)','',str(result['gates']),
           '', 'All positive and negative seed/order outcomes are retained in PAIRED_COMPARISONS.csv. Passing only recommends P2 review; no P2 runs.',
           '', '## Cost and integrity','',
           '28 new targets, 8 historical target imports, 3 reused sources, zero new source updates.',
           'Worker session seconds include evaluation and shared-GPU waiting; they are not exclusive GPU compute or evidence of speed advantage.',
           'Trainer telemetry and operation_counts overlap; never sum them as independent operations.',
           'Formal, source verification, model integrity, CUDA qualification, smoke, failed and resumed sessions are separate in COST_AND_COMPLETION.json.',
           'All 28 target files have code/plan/node-bound tensor and file integrity evidence; metadata freshness was checked for this export.']
    (root/'FINAL_REPORT.md').write_text('\n'.join(text)+'\n')
    validate_exports(root,result)


def validate_exports(root,result):
    root=Path(root)
    if any(not (root/f).is_file() or not (root/f).stat().st_size for f in FILES):raise ValueError('missing report artifact')
    def table(name,fields):
        with (root/name).open() as f:
            r=csv.DictReader(f)
            if r.fieldnames!=fields:raise ValueError('report schema '+name)
            return list(r)
    final=table('FINAL_METRICS.csv',FINAL_FIELDS);stage=table('STAGE_METRICS.csv',STAGE_FIELDS)
    pairs=table('PAIRED_COMPARISONS.csv',PAIR_FIELDS)
    expected={(f,str(s),str(o)) for f in (B0,B2,F5) for s in (162,163,164) for o in (1,2)}
    if len(final)!=18 or {(r['family'],r['seed'],r['order']) for r in final}!=expected:raise ValueError('18 trajectory coverage')
    if sum(r['origin']=='historical_import' for r in final)!=4:raise ValueError('final import coverage')
    truth={(r['identity']['family'],str(r['identity']['seed']),str(r['identity']['order'])):r for r in result['rows'] if r['identity']['stage']==2}
    for row in final:
        source=truth[row['family'],row['seed'],row['order']]
        if row['origin']!=source['origin'] or any(float(row[m])!=metrics(source)[m] for m in METRICS):raise ValueError('final metric/origin mismatch')
    if result['primary']!=paired(result['rows'],[163,164]) or result['supplementary_descriptive']!=paired(result['rows'],[162,163,164]):raise ValueError('paired metric mismatch')
    if result['gates']!=decisions(result['primary']):raise ValueError('gate mismatch')
    if len(stage)!=90 or len({r['node_id'] for r in stage})!=36:raise ValueError('stage/domain coverage')
    if len(pairs)!=34 or {r['cohort'] for r in pairs}!={'primary_163_164','supplementary_162_164'}:raise ValueError('paired report coverage')
    if read(root/'PUBLIC_RESULTS.json')!=result:raise ValueError('result schema/write mismatch')
    costs=read(root/'COST_AND_COMPLETION.json')
    if costs['new_target_stages']!=28 or costs['imported_target_stages']!=8 or costs['new_source_updates']!=0:raise ValueError('cost coverage')


def complete_report(root,execution_commit):
    from .execution import stage_state,check_costs,integrity_bindings,require_qualification
    from .assurance import integrity_current,session_totals
    root=Path(root);plan=validate_plan(read(DOC/'PLAN.json'));config={'execution_commit':execution_commit}
    rows=[];proofs={};sessions={};acceptance_cost={}
    for node in plan['nodes']:
        nr=root/node['id']
        if stage_state(node,root,execution_commit)!='METADATA_SEALED':return dict(status='PENDING_FULL_MATRIX',gates='NOT_EVALUATED')
        r=read(nr/'receipt.json')
        if r['resolved_options']!=plan['options'][node['family']][node['domain']]:raise ValueError('CONFIG_BINDING_MISMATCH '+node['id'])
        proof=read(nr/'INTEGRITY.json') if (nr/'INTEGRITY.json').exists() else {}
        if not integrity_current(nr/'student.pt',r,proof,integrity_bindings(node,config,plan)):
            return dict(status='PENDING_INTEGRITY',node_id=node['id'],gates='NOT_EVALUATED')
        try:
            sessions[node['id']]=session_totals(nr/'costs');acceptance_cost[node['id']]=session_totals(nr/'integrity_costs')
        except ValueError:return dict(status='PENDING_COST_EVIDENCE',node_id=node['id'],gates='NOT_EVALUATED')
        proofs[node['id']]=proof;rows.append(r)
    try:require_qualification(root,config,plan)
    except (OSError,ValueError,KeyError,PermissionError):return dict(status='PENDING_QUALIFICATION',gates='NOT_EVALUATED')
    physical=check_costs(plan,root)
    if physical['formal']!=74200 or sum(r['step'] for r in rows)!=74200:raise ValueError('formal completion budget mismatch')
    historic=read(OLD/'FINAL_RESULTS_REDUCED.json')['index']
    imported=[r for r in historic if r['node_id'] in plan['historical_target_imports']]
    primary=paired(rows,[163,164]);supplementary=paired(rows+imported,[162,163,164])
    fields=('node_id','identity','step','scores','timeline','resolved_options','spectral','physical_optimizer_calls','cost',
            'seconds','operation_counts','peak_cuda_allocated','peak_cuda_reserved','session_cost')
    result=dict(status='REPORTS_PENDING_VALIDATION',publication_status='LOCAL_REPORT_READY',
        execution_commit=execution_commit,plan_sha256=plan['plan_sha256'],primary=primary,supplementary_descriptive=supplementary,
        gates=decisions(primary),rows=[{**{k:r.get(k,'NA') for k in fields},'origin':'new_execution' if origin else 'historical_import'}
                                    for origin,items in [(True,rows),(False,imported)] for r in items],
        costs=dict(physical=physical,formal_scientific=74200,new_source_updates=0,formal_sessions=sessions,
                   integrity_sessions=acceptance_cost,formal_uncommitted_physical_calls=physical['formal']-sum(r['step'] for r in rows),
                   formal_restarted_sessions=sum(max(0,v['sessions']-1) for v in sessions.values()),source_preflight=session_totals(root/'costs'/'SOURCE_PREFLIGHT'),
                   cuda=session_totals(root/'costs'/'CUDA_QUALIFICATION'),smoke=session_totals(root/'costs'/'SMOKE'),
                   CPU=read(DOC/'TEST_REPORT.json'),intentional_cuda_failed_updates=3,
                   recovery='all sessions retained; incomplete hard-kill cost evidence blocks completion'),
        integrity=proofs,environment=read(root/'RUNTIME_ENVIRONMENT.json'),
        qualification={name:read(root/(name+'.json')) for name in ('CUDA_QUALIFICATION','SMOKE')},P2_started=False)
    export_reports(root,result)
    # Only promote completion after all six artifacts have passed schema validation.
    result['status']='COMPLETE_P1_AWAITING_SCIENTIFIC_REVIEW'
    write(root/'PUBLIC_RESULTS.json',result)
    completion=read(root/'COST_AND_COMPLETION.json');completion['status']=result['status']
    write(root/'COST_AND_COMPLETION.json',completion)
    validate_exports(root,result)
    write(root/'STATUS.json',dict(status=result['status'],publication_status='LOCAL_REPORT_READY',P2_started=False))
    return result
