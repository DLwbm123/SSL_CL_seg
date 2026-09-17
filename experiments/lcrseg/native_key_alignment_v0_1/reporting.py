"""All-16 aggregate report; no patient or model-tensor reads."""
import csv
from pathlib import Path
from .protocol import canonical_plan,read,write,digest,STUDY,DOC
from .authority import execution_plan


def metrics(r):
    # Reuse the frozen macro validation and early-domain forgetting definition.
    from ..f5_confirmation_v1.protocol import metrics as native_metrics
    return native_metrics({'identity':r['identity'],'scores':r['scores'],
        'timeline':{'0':r['prefix_timeline']['0'],'1':r['prefix_scores']}})


def summarize(rows):
    expected={(a,s,o) for a in ('C0','C1','C2','C3') for s in (163,164) for o in (1,2)}
    indexed={(r['identity']['arm'],r['identity']['seed'],r['identity']['order']):r for r in rows}
    if len(rows)!=16 or set(indexed)!=expected:raise ValueError('complete 16-stage diagnostic required')
    for r in rows:
        required={str(n) for n in execution_plan()['diagnostics'][r['identity']['domain']]}
        if set(r['diagnostics'])!=required:raise ValueError('incomplete diagnostic records')
        for step,d in r['diagnostics'].items():
            if set(d['norms'])!={'supervised','KL','alignment'} or d['extra_vjps'] not in (2,3) or 'alignment' not in d:raise ValueError('invalid diagnostic schema')
    pairs={}
    for reference in ('C0','C2','C1'):
        order={};seeds={}
        for seed in (163,164):
            for o in (1,2):
                a=indexed['C3',seed,o];b=indexed[reference,seed,o]
                if a['identity']['prefix_binding_sha256']!=b['identity']['prefix_binding_sha256'] or a['prefix_scores']!=b['prefix_scores'] or a['prefix_timeline']!=b['prefix_timeline']:
                    raise ValueError('not the same frozen prefix')
                ma,mb=metrics(a),metrics(b);delta={k:ma[k]-mb[k] for k in ma}
                if abs(delta['Forget']+delta['Old'])>1e-10:raise ValueError('common-prefix Forget/Old identity violated')
                order[f'{seed}/O{o}']=delta
            seeds[str(seed)]={k:sum(order[f'{seed}/O{o}'][k] for o in (1,2))/2 for k in ma}
        mean={k:sum(v[k] for v in seeds.values())/2 for k in ma}
        pairs[reference]=dict(per_seed_order=order,per_seed=seeds,mean=mean)
    p=pairs['C0'];old_o2=sum(p['per_seed_order'][f'{s}/O2']['Old'] for s in (163,164))/2
    incoming={str(o):sum(p['per_seed_order'][f'{s}/O{o}']['Incoming'] for s in (163,164))/2 for o in (1,2)}
    gate=all(v['Final']>0 for v in p['per_seed'].values()) and p['mean']['Final']>=.005 and old_o2>=0 and min(incoming.values())>=-.005
    mechanism=all(v['Final']>0 for v in pairs['C2']['per_seed'].values()) and pairs['C2']['mean']['Final']>0
    strongest=all(pairs[c]['mean']['Final']>0 for c in ('C1','C2'))
    return dict(pairs=pairs,gate=dict(D1_performance=gate,key_vs_random_direction_consistent=mechanism,
        exceeds_both_controls=strongest,O2_Old_delta=old_o2,Incoming_by_order=incoming,
        recommendation='CONDITIONAL_REVIEW_ONLY' if gate and strongest else 'STOP_NO_AUTOMATIC_FOLLOWUP'))


def export(root,result):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    write(root/'PUBLIC_RESULTS.json',result)
    def csv_file(name,fields,rows):
        with (root/name).open('w') as f:
            w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    final=[];domains=[];pairs=[]
    for r in result['rows']:
        i=r['identity'];ident={k:i[k] for k in ('arm','seed','order')}
        final.append({**ident,**metrics(r)})
        first='RIM_ONE_r3' if i['order']==1 else 'Drishti_GS'
        for d,s in r['scores'].items():domains.append({**ident,'domain':d,**s,**{'first_target_after_prefix_'+k:r['prefix_scores'][d][k] if d==first else 'NA' for k in ('rim','cup','macro_Dice')}})
    for ref,p in result['pairs'].items():
        for level,key in [('seed_order','per_seed_order'),('seed','per_seed')]:
            for ident,values in p[key].items():pairs.append(dict(reference=ref,aggregation=level,id=ident,**values))
        pairs.append(dict(reference=ref,aggregation='overall',id='mean',**p['mean']))
    csv_file('FINAL_METRICS.csv',['arm','seed','order','Final','Old','Incoming','Forget'],final)
    csv_file('DOMAIN_METRICS.csv',['arm','seed','order','domain','rim','cup','disc_union','macro_Dice','first_target_after_prefix_rim','first_target_after_prefix_cup','first_target_after_prefix_macro_Dice'],domains)
    csv_file('PAIRED_COMPARISONS.csv',['reference','aggregation','id','Final','Old','Incoming','Forget'],pairs)
    write(root/'DIAGNOSTICS.json',{r['node_id']:dict(gradients=r['diagnostics'],support=r['support'],alignment_cost=r['alignment_cost']) for r in result['rows']})
    write(root/'COST_AND_COMPLETION.json',dict(status=result['status'],costs=result['costs']))
    (root/'FINAL_REPORT.md').write_text('# D1 common-prefix diagnostic\n\nDevelopment seeds 163/164; 16 second-target stages. Not complete new-method trajectories or independent-patient validation.\n\n'+
        str(result['gate'])+'\n\nAll C3-C0/C2/C1 seed/order outcomes, domain rim/cup and first-target early/final scores are retained. DeltaForget=-DeltaOld is not independent evidence. No D2/D3 was started. Costs include extra forwards and diagnostic VJPs; equal steps do not imply equal compute.\n')
    for name,count in [('FINAL_METRICS.csv',16),('DOMAIN_METRICS.csv',48),('PAIRED_COMPARISONS.csv',21)]:
        with (root/name).open() as f:
            if len(list(csv.DictReader(f)))!=count:raise ValueError('report coverage')


def report(root,config):
    from .execution import stage_state,proof_binding,check_costs,require_qualification
    from ..f5_confirmation_v1.assurance import integrity_current,session_totals
    root=Path(root);plan=canonical_plan();rows=[];costs={'formal':{},'integrity':{}}
    for n in plan['nodes']:
        if stage_state(n,root,config)!='METADATA_SEALED':return dict(status='PENDING_FULL_MATRIX')
        nr=root/n['id'];r=read(nr/'receipt.json');proof=read(nr/'INTEGRITY.json')
        if r['resolved_options']!=plan['options'][n['domain']] or not integrity_current(nr/'student.pt',r,proof,proof_binding(n,config,plan)):
            return dict(status='PENDING_INTEGRITY',node=n['id'])
        rows.append(r);costs['formal'][n['id']]=session_totals(nr/'costs');costs['integrity'][n['id']]=session_totals(nr/'integrity_costs')
    require_qualification(root,config,plan);costs['physical']=check_costs(root,plan)
    if costs['physical']['formal']!=42400:raise ValueError('formal count mismatch')
    costs['qualifications']={n:session_totals(root/'costs'/n) for n in ('CUDA_QUALIFICATION','SMOKE')}
    costs['prefix']={p.name:session_totals(p) for p in (root/'costs').glob('PREFIX_*')}
    costs['CPU_original']=read(DOC/'COST_SUMMARY.json');costs['CPU_INTEGRATION_R1']=read(DOC/'CPU_INTEGRATION_R1/TEST_REPORT.json')['cost']
    result=dict(study_id=STUDY,status='COMPLETE_D1_AWAITING_SCIENTIFIC_REVIEW',execution_commit=config['execution_commit'],
        plan_sha256=plan['plan_sha256'],rows=rows,costs=costs,**summarize(rows),D2_D3_started=False)
    export(root,result);write(root/'STATUS.json',dict(status=result['status'],publication_status='LOCAL_REPORT_READY'))
    return result
