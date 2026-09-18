"""28 trajectories, 84 domains, 63 contrasts and 64 new diagnostic points."""
import csv
from pathlib import Path
from .protocol import canonical_plan,anchored,read,write,digest,DOC,STUDY,DOSES
from .authority import execution_plan
from ..native_key_alignment_v0_1.reporting import metrics


def historical(plan):
    source=anchored('PUBLIC_RESULTS.json');rows=[]
    for bound in plan['imports']:
        row=next(r for r in source['rows'] if r['node_id']==bound['node_id'])
        if digest(row)!=bound['row_sha256']:raise ValueError('BASELINE_REUSE_BLOCKED')
        rows.append({**row,'origin':'historical_import','lambda_align':bound['lambda_align'],'source_commit':bound['source_commit'],
            'source_identity':row['identity'],'adam_diagnostics':'NOT_MEASURED_HISTORICAL'})
    return rows


def summarize(rows):
    expected={(a,w,s,o) for a in ('C2','C3') for w in (.05,.5,2.) for s in (163,164) for o in (1,2)}|{('C0',0.,s,o) for s in (163,164) for o in (1,2)}
    index={(r['identity']['arm'],r['lambda_align'],r['identity']['seed'],r['identity']['order']):r for r in rows}
    if len(rows)!=28 or set(index)!=expected:raise ValueError('complete 28 rows required')
    counts={'historical_import':0,'new_execution':0};points=0
    for r in rows:
        counts[r['origin']]+=1
        if r['origin']=='historical_import':
            if r['lambda_align'] not in (0.,.05) or r['adam_diagnostics']!='NOT_MEASURED_HISTORICAL':raise ValueError('false historical diagnostics')
        else:
            i=r['identity'];required={str(n) for n in execution_plan()['diagnostics'][i['domain']]}
            if r['lambda_align']!=i['lambda_align'] or set(r['diagnostics'])!=required:raise ValueError('dose/diagnostic mismatch')
            for d in r['diagnostics'].values():
                if d['extra_vjps']!=4 or d['read_only_candidates']!=2 or not d['prediction']['passed']:raise ValueError('incomplete Adam diagnosis')
            points+=len(r['diagnostics'])
    if counts!={'historical_import':12,'new_execution':16} or points!=64:raise ValueError('origin/point coverage')
    pairs={};gates={}
    for w in (.05,.5,2.):
        contrasts={}
        for a,b in [('C3','C0'),('C2','C0'),('C3','C2')]:
            order={};seeds={}
            for s in (163,164):
                for o in (1,2):
                    ra=index[a,w,s,o];rb=index[b,0. if b=='C0' else w,s,o]
                    if ra['identity']['prefix_binding_sha256']!=rb['identity']['prefix_binding_sha256'] or ra['prefix_scores']!=rb['prefix_scores'] or ra['prefix_timeline']!=rb['prefix_timeline']:raise ValueError('common prefix mismatch')
                    ma,mb=metrics(ra),metrics(rb);delta={k:ma[k]-mb[k] for k in ma}
                    if abs(delta['Old']+delta['Forget'])>1e-10:raise ValueError('Forget/Old identity')
                    order[f'{s}/O{o}']=delta
                seeds[str(s)]={k:sum(order[f'{s}/O{o}'][k] for o in (1,2))/2 for k in ma}
            mean={k:sum(x[k] for x in seeds.values())/2 for k in ma};contrasts[a+'-'+b]=dict(per_seed_order=order,per_seed=seeds,mean=mean)
        pairs[str(w)]=contrasts;p=contrasts['C3-C0'];key=contrasts['C3-C2']
        perf=p['mean']['Final']>=.005 and all(x['Final']>0 for x in p['per_seed'].values()) and sum(p['per_seed_order'][f'{s}/O2']['Old'] for s in (163,164))/2>=0 and all(sum(p['per_seed_order'][f'{s}/O{o}']['Incoming'] for s in (163,164))/2>=-.005 for o in (1,2))
        gkey=key['mean']['Final']>=.001 and all(x['Final']>0 for x in key['per_seed'].values())
        gates[str(w)]=dict(G_perf=perf,G_key=gkey,eligible=w in (.5,2.),historical_only=w==.05)
    eligible=[w for w in (.5,2.) if gates[str(w)]['G_perf'] and gates[str(w)]['G_key']]
    return dict(pairs=pairs,gates=gates,selected_lambda=min(eligible) if eligible else None,
        decision='SEPARATE_COMPLETE_TRAJECTORY_REVIEW_ONLY' if eligible else ('STOP_DOSE_AND_CURRENT_NKA_SWD' if not any(gates[str(w)]['G_perf'] for w in (.5,2.)) else 'NO_KEY_SPECIFIC_ADVANCE'),automatic_followup=False)


def export(root,result):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);rows=result['rows'];final=[];domains=[];paired=[];grad={};adam={}
    def csv_file(name,fields,rs):
        with (root/name).open('w',newline='') as f:w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rs)
    for r in rows:
        i=r['identity'];meta=dict(node_id=r['node_id'],arm=i['arm'],lambda_align=r['lambda_align'],seed=i['seed'],order=i['order'],origin=r['origin'])
        final.append({**meta,**metrics(r)});first='RIM_ONE_r3' if i['order']==1 else 'Drishti_GS'
        for d,s in r['scores'].items():domains.append({**meta,'domain':d,**s,**{'first_target_early_'+k:r['prefix_scores'][d][k] if d==first else 'NA' for k in ('rim','cup','macro_Dice')}})
        if r['origin']=='historical_import':grad[r['node_id']]={'historical_gradients':r['diagnostics']};continue
        grad[r['node_id']]={step:dict(gradients=d['gradients'],alignment=d['alignment'],losses=d['losses'],extra_vjps=d['extra_vjps']) for step,d in r['diagnostics'].items()}
        for step,d in r['diagnostics'].items():adam[r['node_id']+'/'+step]={**meta,**{k:v for k,v in d.items() if k not in ('gradients','alignment')}}
    for dose,contrasts in result['pairs'].items():
        for contrast,p in contrasts.items():
            for label,key in [('seed_order','per_seed_order'),('seed','per_seed')]:
                for ident,vals in p[key].items():paired.append(dict(lambda_align=float(dose),contrast=contrast,aggregation=label,id=ident,**vals))
            paired.append(dict(lambda_align=float(dose),contrast=contrast,aggregation='overall',id='mean',**p['mean']))
    cols=['node_id','arm','lambda_align','seed','order','origin']
    csv_file('FINAL_METRICS.csv',cols+['Final','Old','Incoming','Forget'],final)
    csv_file('DOMAIN_METRICS.csv',cols+['domain','rim','cup','disc_union','macro_Dice','first_target_early_rim','first_target_early_cup','first_target_early_macro_Dice'],domains)
    csv_file('PAIRED_COMPARISONS.csv',['lambda_align','contrast','aggregation','id','Final','Old','Incoming','Forget'],paired)
    if (len(final),len(domains),len(paired),len(adam))!=(28,84,63,64):raise ValueError('export row coverage')
    write(root/'GRADIENT_DIAGNOSTICS.json',grad);write(root/'ADAM_COUNTERFACTUAL.json',adam)
    write(root/'COST_AND_COMPLETION.json',dict(status=result['status'],costs=result['costs']));write(root/'PUBLIC_RESULTS.json',result)
    (root/'FINAL_REPORT.md').write_text('# NKA_DOSE fixed development study\n\n16 new +12 historical trajectories; not independent patients or complete new-method trajectories.\n\n'+str(result['gates'])+'\n\nDecision: '+result['decision']+'; selected lambda for separate review only: '+str(result['selected_lambda'])+'\n\nAll positive/negative contrasts retained. DeltaForget=-DeltaOld is one piece of evidence. No high-dose C1 comparison exists. Adam counterfactual is local to each new dose state, not a no-auxiliary trajectory; historical Adam is NOT_MEASURED_HISTORICAL. No automatic follow-up.\n')


def report(root,config):
    from .execution import stage_state,proof_binding,check_costs,require_qualification
    from ..f5_confirmation_v1.assurance import integrity_current,session_totals
    root=Path(root);plan=canonical_plan();rows=historical(plan);costs={'formal':{},'integrity':{}};proofs={}
    for n in plan['nodes']:
        if stage_state(n,root,config)!='METADATA_SEALED':return dict(status='PENDING_FULL_MATRIX')
        nr=root/n['id'];r=read(nr/'receipt.json');proof=read(nr/'INTEGRITY.json')
        if r['resolved_options']!=plan['options'][n['domain']] or not integrity_current(nr/'student.pt',r,proof,proof_binding(n,config,plan)):return dict(status='PENDING_INTEGRITY',node=n['id'])
        rows.append({**r,'lambda_align':n['lambda_align']});proofs[n['id']]={k:proof[k] for k in ('status','file_sha256','student_hash','transform_hash','receipt_sha256')}
        costs['formal'][n['id']]=session_totals(nr/'costs');costs['integrity'][n['id']]=session_totals(nr/'integrity_costs')
    require_qualification(root,config,plan);costs['physical']=check_costs(root,plan)
    if costs['physical']['formal']!=42400:raise ValueError('formal count mismatch')
    new=[r for r in rows if r['origin']=='new_execution']
    for k,expected in [('diagnostic_vjps',256),('adam_candidates',128),('clean_U_forwards',33920)]:
        if sum(r['alignment_cost'][k] for r in new)!=expected:raise ValueError('meter/forward count mismatch')
    costs['qualifications']={n:session_totals(root/'costs'/n) for n in ('CUDA_QUALIFICATION','SMOKE')}
    costs['prefix']={p.name:session_totals(p) for p in (root/'costs').glob('PREFIX_*')}
    costs['CPU_original_other_study']=64;costs['CPU_new']=read(DOC/'CPU/TEST_REPORT.json')['cost']
    result=dict(study_id=STUDY,status='COMPLETE_DOSE_AWAITING_SCIENTIFIC_REVIEW',execution_commit=config['execution_commit'],plan_sha256=plan['plan_sha256'],rows=rows,costs=costs,integrity=proofs,**summarize(rows))
    export(root,result);write(root/'COMPLETION_PROOF.json',dict(status='PASS',new_nodes=16,historical=12,integrity=proofs,counts={'final':28,'domain':84,'paired':63,'adam':64}))
    write(root/'STATUS.json',dict(status=result['status'],publication_status='LOCAL_REPORT_READY'));return result
