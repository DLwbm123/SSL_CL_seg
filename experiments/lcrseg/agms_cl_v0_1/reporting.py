"""Complete 12/36/27/40 aggregate reports; no patient or model tensor IO."""
import copy
import csv
from pathlib import Path
from .protocol import STUDY,DOC,ARMS,read,write,digest,canonical_plan,anchored,execution_plan
from ..native_key_alignment_v0_1.reporting import metrics
CONTRASTS={'A1-A0':{'A1':1,'A0':-1},'A2-A0':{'A2':1,'A0':-1},'A3-A0':{'A3':1,'A0':-1},
           'A5-A0':{'A5':1,'A0':-1},'A5-A1':{'A5':1,'A1':-1},'A5-A2':{'A5':1,'A2':-1},
           'A5-A3':{'A5':1,'A3':-1},'A5-A4':{'A5':1,'A4':-1},'A3-A1-A2+A0':{'A3':1,'A1':-1,'A2':-1,'A0':1}}


def historical(plan):
    public=anchored('B2_PUBLIC_RESULTS.json');rows=[]
    for bound in plan['imports']:
        r=copy.deepcopy(next(r for r in public['rows'] if r['node_id']==bound['node_id']))
        if digest(r)!=bound['row_sha256']:raise ValueError('BASELINE_REUSE_BLOCKED: historical row')
        prefix=next(p for p in plan['prefixes'] if p['identity']['order']==bound['order'])
        pr=next(x for x in public['rows'] if x['node_id']==prefix['node_id'])
        r.update(arm='A0',origin='historical_import',source_commit=bound['source_commit'],
                 prefix_binding_sha256=prefix['binding_sha256'],prefix_scores=pr['scores'],prefix_timeline=pr['timeline'],
                 diagnostics='NOT_MEASURED_HISTORICAL')
        rows.append(r)
    return rows


def summarize(rows):
    index={(r['arm'],r['identity']['order']):r for r in rows}
    expected={(a,o) for a in ARMS for o in (1,2)}
    if len(rows)!=12 or set(index)!=expected:return dict(status='NOT_ASSESSED_REDUCED_SCOPE',gates=None,pairs=[],automatic_followup=False)
    for r in rows:
        if r['identity']['seed']!=163:raise ValueError('wrong seed')
        if set(r['scores'])!={'REFUGE','RIM_ONE_r3','Drishti_GS'}:raise ValueError('domain coverage')
        if r['arm']=='A0':
            if r['origin']!='historical_import' or r['diagnostics']!='NOT_MEASURED_HISTORICAL':raise ValueError('historical evidence fabricated')
        else:
            if r['origin']!='new_execution':raise ValueError('wrong origin')
            required={str(n) for n in execution_plan()['diagnostics'][r['identity']['domain']]}
            if set(r['diagnostics'])!=required:raise ValueError('diagnostic coverage')
            if any(d['extra_vjps']!=4 for d in r['diagnostics'].values()):raise ValueError('diagnostic VJP count')
    pairs=[]
    for name,terms in CONTRASTS.items():
        per_order=[]
        for order in (1,2):
            base=index['A0',order]
            for a in terms:
                r=index[a,order]
                if r['prefix_binding_sha256']!=base['prefix_binding_sha256'] or r['prefix_scores']!=base['prefix_scores'] or r['prefix_timeline']!=base['prefix_timeline']:
                    raise ValueError('common prefix mismatch')
            delta={k:sum(c*metrics(index[a,order])[k] for a,c in terms.items()) for k in ('Final','Old','Incoming','Forget')}
            if abs(delta['Forget']+delta['Old'])>1e-9:raise ValueError('DeltaForget identity')
            per_order.append(delta);pairs.append(dict(contrast=name,order=str(order),**delta))
        pairs.append(dict(contrast=name,order='mean',**{k:sum(r[k] for r in per_order)/2 for k in per_order[0]}))
    p=[r for r in pairs if r['contrast']=='A5-A0'];means={a:sum(metrics(index[a,o])['Final'] for o in (1,2))/2 for a in ARMS}
    best=max(('A1','A2','A3','A4'),key=lambda a:(means[a],-int(a[1])))
    perf=p[2]['Final']>=.003 and all(r['Final']>=-.002 and r['Incoming']>=-.005 for r in p[:2]) and p[1]['Old']>=0
    joint=means['A5']-means[best]>=.001
    return dict(status='ASSESSED_DEVELOPMENT_ONLY',pairs=pairs,gates={'P_perf':perf,'P_joint':joint},
                strongest_simple=best,joint_delta=means['A5']-means[best],automatic_followup=False)


def export(root,result):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);final=[];domains=[];diagnostics=[];coverage=[]
    for r in result['rows']:
        meta=dict(node_id=r['node_id'],arm=r['arm'],seed=163,order=r['identity']['order'],origin=r['origin'])
        final.append({**meta,**metrics(r)})
        for domain,values in r['scores'].items():domains.append({**meta,'domain':domain,**{k:values[k] for k in ('rim','cup','disc_union','macro_Dice')}})
        if r['origin']=='historical_import':coverage.append({**meta,'status':'NOT_MEASURED_HISTORICAL'});continue
        for step,d in r['diagnostics'].items():diagnostics.append({**meta,'step':int(step),**d})
        coverage.append({**meta,'status':'measured','summary':r['support'],'points':[{k:d[k] for k in ('coverage','risk','alpha','shadow_L','L_errors','uniform_risk_disagreement')} for d in r['diagnostics'].values()]})
    if (len(final),len(domains),len(result['pairs']),len(diagnostics))!=(12,36,27,40):raise ValueError('report coverage')
    def output(name,rows):
        with (root/name).open('w',newline='') as f:
            w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    output('FINAL_METRICS.csv',final);output('DOMAIN_METRICS.csv',domains);output('PAIRED_COMPARISONS.csv',result['pairs'])
    write(root/'GRADIENT_DIAGNOSTICS.json',diagnostics);write(root/'COVERAGE_AND_RISK.json',coverage)
    write(root/'COST_AND_COMPLETION.json',dict(status=result['status'],costs=result['costs']))
    write(root/'PUBLIC_RESULTS.json',result)
    (root/'FINAL_REPORT.md').write_text('# AGMS fixed P1 development results\n\n'+str(result['gates'])+
        '\n\nSingle observed seed, two orders; not independent-seed significance, independent patients, SOTA, or original KI reproduction. '
        'Training-L diagnostics are not independent calibration. All 10 new nodes retained; A0 historical. '
        'Common-prefix DeltaForget=-DeltaOld is one observation. Deployment discards heads/risk. No automatic follow-up.\n')


def report(root,config):
    from .execution import stage_state,proof_binding,check_costs,require_qualification
    from .p0 import validate_report
    from ..f5_confirmation_v1.assurance import integrity_current,session_totals
    root=Path(root);plan=canonical_plan();rows=historical(plan);costs={'formal':{},'integrity':{}};proofs={}
    for n in plan['nodes']:
        if stage_state(n,root,config)!='METADATA_SEALED':return dict(status='PENDING_FULL_MATRIX')
        nr=root/n['id'];r=read(nr/'receipt.json');proof=read(nr/'INTEGRITY.json')
        if r['resolved_options']!=plan['options'][n['domain']] or not integrity_current(nr/'student.pt',r,proof,proof_binding(n,config,plan)):
            return dict(status='PENDING_INTEGRITY',node=n['id'])
        rows.append({**r,'arm':n['arm'],'prefix_binding_sha256':n['prefix_binding_sha256']})
        proofs[n['id']]={k:proof[k] for k in ('status','file_sha256','student_hash','transform_hash','receipt_sha256')}
        costs['formal'][n['id']]=session_totals(nr/'costs');costs['integrity'][n['id']]=session_totals(nr/'integrity_costs')
    require_qualification(root,config,plan);p0=validate_report(read(root/'P0_REPORT.json'),config,plan)
    costs['physical']=check_costs(root,plan)
    if costs['physical']['formal']!=26500 or sum(r['extra_cost']['diagnostic_vjps'] for r in rows if r['origin']=='new_execution')!=160:
        raise ValueError('physical/VJP count mismatch')
    costs['qualifications']={n:session_totals(root/'costs'/n) for n in ('CUDA_QUALIFICATION','SMOKE','P0')}
    costs['prefix']={p.name:session_totals(p) for p in (root/'costs').glob('PREFIX_*')}
    costs['CPU_old']=134;costs['CPU_new']=read(DOC/'CPU/TEST_REPORT.json')['cost']
    analysis=summarize(rows)
    if analysis['status']!='ASSESSED_DEVELOPMENT_ONLY':raise ValueError('full coverage required')
    analysis.pop('status')
    result=dict(study_id=STUDY,status='COMPLETE_AGMS_P1_AWAITING_SCIENTIFIC_REVIEW',execution_commit=config['execution_commit'],
                plan_sha256=plan['plan_sha256'],rows=rows,costs=costs,integrity=proofs,P0=p0,**analysis)
    export(root,result);write(root/'COMPLETION_PROOF.json',dict(status='PASS',new_nodes=10,historical=2,integrity=proofs,
                                                             counts={'final':12,'domain':36,'paired':27,'diagnostics':40}))
    write(root/'STATUS.json',dict(status=result['status'],publication_status='LOCAL_REPORT_READY'))
    return result
