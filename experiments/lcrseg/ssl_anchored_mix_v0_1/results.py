"""Compile fixed phases and bootstrap existing patient scores; never access raw GT."""
import csv,json,statistics as st
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from .core import DOMAINS,ARMS,COUNTS,write_json
from .statistics import summary,q,costs,select,replication,intervals
from experiments.lcrseg.ssl_head_control_v0_1.results import table_write,read,rows

def verify_phase(base,tasks):
    b=Path(base);receipts=[];deploy=[];snap=[];exits=[];operations=[];groups=defaultdict(list)
    for s,d,a in tasks:
        p=b/f'seed{s}'/d/a;r=read(p/'receipt.json');de=read(p/'deployment.json')
        n=sum(1 for _ in (p/'steps.jsonl').open());assert n==r['optimizer_updates']==(3200 if d==DOMAINS[0] else 2100)
        assert r['status']=='TRAINING_COMPLETE' and r['models']==2 and de['models']==1 and de['status']=='PASS'
        assert r['source']==de['source'] and r['student_hash']==de['student_hash']
        assert r['counters']['backward']==r['counters']['ema_updates']==n
        assert not r['sigma_requires_grad'] and not r['sigma_gradient'] and not r['grad_update_in_Adam']
        if a.startswith('SUP'):assert r['unlabeled_opens']==0
        if a=='MIX_CTX':assert r['counters'].get('ema_u_forward',0)==0
        warm=read(p/'warmup.json');assert warm['u_opens']==0
        groups[s,d].append((a,warm,read(p/'initialization.json'),r['label_order_hash']))
        for ep in (20,40,60,80,100):
            ss=read(p/f'eval{ep}/receipt.json');assert ss['state_preserved'] and ss['additional_backward']==0 and ss['prediction_seals']==COUNTS[d][2]*5;snap.append(dict(seed=s,domain=d,arm=a,epoch=ep,**ss))
        for kind in ('train','deploy'):
            ex=read(p.parent/(a+'_'+kind+'_exit.json'));assert ex['exit_code']==0;exits.append(ex)
        op=read(p.parent/(a+'_operations/operation_counts.json'));dop=read(p.parent/(a+'_deploy_operations/operation_counts.json'))
        assert op['status']==dop['status']=='PASS' and op['counts']['optimizer_steps']==op['counts']['backward']==op['counts']['ema_updates']==n
        assert dop['counts'].get('optimizer_steps',0)==0 and dop['counts'].get('sample_val',0)==0
        receipts.append(r);deploy.append(de);operations.append(dict(seed=s,domain=d,arm=a,training=op,deployment=dop))
    for key,group in groups.items():
        for field in ('student_hash','ema_hash','optimizer_hash','label_order_hash'):
            assert len({w[field] for a,w,i,o in group if a!='SUP_CE'})==1
        assert len({i['student_hash'] for a,w,i,o in group})==1 and len({o for a,w,i,o in group})==1
    return dict(status='PASS',receipts=receipts,deployments=deploy,snapshots=snap,exits=exits,operations=operations)

def collect(phase,base,tasks):
    final=[];epochs=[];source=[];mechanism=[];ledger=[]
    for s,d,a in tasks:
        p=Path(base)/f'seed{s}'/d/a;n=0;bucket={}
        if (p/'steps.jsonl').exists():
            for line in (p/'steps.jsonl').open():
                r=json.loads(line);n+=1
                for cls in range(3):
                    k=(r['epoch'],cls);v=bucket.setdefault(k,Counter())
                    v.update(steps=1,L_class_pixels=r['labeled_class_exposure'][cls],L_records=r['labeled_image_records'],U_unique_batch_sources=r['unlabeled_unique_batch_sources'],U_context_records=r['unlabeled_context_records'],mix_steps=bool(r['mix_rectangles']),all_ignore_L=r['all_ignore_L'])
                    # Global loss/weight fields repeat across class rows; never sum these across classes.
                    for f in ['CE','GT_Dice','dice_weight','SCE','target_entropy','KL','lambda_cons']:v[f]+=r[f]
                    v['weighted_GT_Dice']+=r['dice_weight']*r['GT_Dice'];v['weighted_U_SCE']+=r['lambda_cons']*r['SCE']
                    for u in r['U_class']:
                        if u['class_id']==cls:v['U_predicted']+=u['predicted'];v['U_accepted']+=u['accepted']
            for (ep,cls),v in bucket.items():source.append(dict(phase=phase,seed=s,domain=d,arm=a,epoch=ep,class_id=cls,**dict(v),U_target_applicable=a in ('MT_CED','MIX_CED') and ep>20,unit='sums over steps; batch record counts repeat across classes, not distinct cohort patients; divide losses by steps'))
        ledger.append(dict(phase=phase,seed=s,domain=d,arm=a,updates=n,status='TRAINING_COMPLETE' if (p/'receipt.json').exists() else 'INCOMPLETE'))
        for ep in (20,40,60,80,100):
            for name,target in [('metrics.csv',epochs),('probability_aggregate.csv',mechanism),('quality_aggregate.csv',mechanism),('strata_aggregate.csv',mechanism)]:
                f=p/f'eval{ep}'/name
                if f.exists():target.extend(dict(phase=phase,kind=name,**r) for r in rows(f))
    final=[r for r in epochs if int(r['epoch'])==100 and r['model']=='student']
    return final,epochs,source,mechanism,ledger

def paired_tables(table,seeds,comparisons,phase):
    effects=[];class_cost=[]
    for a,b in comparisons:
        v=[q(table,s,a)-q(table,s,b) for s in seeds]
        effects.append(dict(phase=phase,arm=a,baseline=b,seed='all',**summary(v)))
        for s,x in zip(seeds,v):effects.append(dict(phase=phase,arm=a,baseline=b,seed=s,delta=x))
        class_cost.extend(dict(phase=phase,arm=a,baseline=b,**r) for r in costs(table,seeds,a,b))
    return effects,class_cost

def patient_arrays(base,old,seeds,arms):
    result={}
    for d in DOMAINS:
        out=[]
        for s in seeds:
            data=[]
            for a in arms:
                p=Path(old)/f'seed{s}'/d/'SUP' if a=='SUP_CE' and s in (31,32,33) else Path(base)/f'seed{s}'/d/a
                rr=[r for r in rows(p/'eval100/private_metrics.csv') if r['model']=='student'];rr.sort(key=lambda r:int(r['patient']))
                assert [int(r['patient']) for r in rr]==list(range(COUNTS[d][2]))
                data.append([float(r['macro_fg_dice']) for r in rr])
            out.append(data)
        result[d]=np.asarray(out)
    return result

def summarize(base,phase,tasks,old,selected=None):
    b=Path(base);out=b/'public_results';out.mkdir(exist_ok=True)
    final,epochs,source,mechanism,ledger=collect(phase,b,tasks);error=None;verification=None
    try:verification=verify_phase(b,tasks)
    except Exception as exc:error=repr(exc)
    complete=error is None
    if phase=='P1':
        oldtable=rows(Path(__file__).parents[1]/'docs/ssl_target_path_v0_1/FINAL_METRICS.csv')
        table=final+[dict(r,arm='SUP_CE',phase='P0_HISTORICAL') for r in oldtable if r['arm']=='SUP']
        seeds=(31,32,33);arms=('SUP_CE',*ARMS);comparisons=[('SUP_CED','SUP_CE'),('MT_CED','SUP_CED'),('MIX_CTX','SUP_CED'),('MIX_CED','MIX_CTX'),('MIX_CED','MT_CED'),('MIX_CED','SUP_CED'),('MIX_CED','SUP_CE')]
        decision=select(table,complete) if len(final)==24 else dict(status='INCOMPLETE',selected=None)
    else:
        table=final;seeds=(41,42);arms=tuple(dict.fromkeys(t[2] for t in tasks));comparisons=[('SUP_CED','SUP_CE'),(selected,'SUP_CE'),(selected,'SUP_CED')]+([(selected,'MIX_CTX')] if selected=='MIX_CED' else [])
        decision=replication(table,selected,complete) if len(final)==len(tasks) else dict(status='INCOMPLETE',selected=selected)
    effects=[];cc=[];ci=[]
    if complete:
        effects,cc=paired_tables(table,seeds,comparisons,phase)
        arrays=patient_arrays(b,old,seeds,arms)
        ci=intervals(arrays,seeds,arms,comparisons,phase)
        # Patient scores and epoch100 aggregates must agree before reporting intervals.
        for d,arr in arrays.items():
            for i,s in enumerate(seeds):
                for j,a in enumerate(arms):
                    expected=next(float(r['macro_fg_dice']) for r in table if int(r['seed'])==s and r['arm']==a and r['domain']==d)
                    assert abs(arr[i,j].mean()-expected)<1e-12
    for name,rr in [('FINAL_METRICS.csv',table),('EPOCH_METRICS.csv',epochs),('SUPERVISION_SOURCE_ACCOUNTING.csv',source),('MECHANISM_METRICS.csv',mechanism),('RUN_LEDGER.csv',ledger),('PAIRED_EFFECTS.csv',effects),('SEED_DOMAIN_CLASS_COST.csv',cc),('UNCERTAINTY_SUMMARY.csv',ci)]:
        if rr:table_write(out/name,rr)
    write_json(out/'MEMORY_AND_DEPLOYMENT.json',dict(verification=verification,error=error))
    write_json(out/('P1_SELECTION.json' if phase=='P1' else 'P2_REPLICATION.json'),decision)
    decision.update(engineering_complete=complete,engineering_error=error,actual_formal_updates=sum(r['updates'] for r in ledger))
    write_json(b/'phase_terminal.json',decision);return decision

def finish(base,p1,p2):
    b=Path(base);out=b/'public_results';out.mkdir();all_rows=defaultdict(list)
    for phase in ['P1']+(['P2'] if (b/'P2').exists() else []):
        for file in (b/phase/'public_results').glob('*.csv'):all_rows[file.name].extend(rows(file))
    if p1.get('engineering_complete') and p2.get('engineering_complete'):
        table=all_rows['FINAL_METRICS.csv'];selected=p1['selected'];comparisons=[('SUP_CED','SUP_CE'),(selected,'SUP_CE'),(selected,'SUP_CED')]+([(selected,'MIX_CTX')] if selected=='MIX_CED' else [])
        effects,cc=paired_tables(table,(31,32,33,41,42),comparisons,'ALL5_SELECTED_DESCRIPTIVE')
        all_rows['PAIRED_EFFECTS.csv'].extend(effects);all_rows['SEED_DOMAIN_CLASS_COST.csv'].extend(cc)
    for name,rr in all_rows.items():table_write(out/name,rr)
    write_json(out/'P1_SELECTION.json',p1);write_json(out/'P2_REPLICATION.json',p2)
    actual=sum(int(r['updates']) for r in all_rows['RUN_LEDGER.csv'])
    expected=63600+(42400 if p1.get('selected')=='MIX_CED' else 31800 if p1.get('selected')=='MT_CED' else 0)
    complete=p1.get('engineering_complete',False) and (p2['status']=='NOT_ADMITTED' or p2.get('engineering_complete',False)) and actual==expected
    terminal=dict(status=p2['status'] if p2['status']!='NOT_ADMITTED' else p1['status'],engineering='ENGINEERING_COMPLETE' if complete else 'INCOMPLETE',formal_updates=actual,historical_formal_total=248108+actual,selected=p1.get('selected'),stopped=True,further_experiments=False,CL='NOT_EVALUATED')
    write_json(b/'TERMINAL.json',terminal)
    write_json(out/'MEMORY_AND_DEPLOYMENT.json',{phase:read(b/phase/'public_results/MEMORY_AND_DEPLOYMENT.json') for phase in ['P1','P2'] if (b/phase).exists()})
    write_json(out/'TEST_AND_RUNTIME_REPORT.json',dict(qualification=read(b/'qualification_ledger.json'),smoke=read(b/'real_smoke/receipt.json'),terminal=terminal))
    (out/'FINAL_REPORT.md').write_text('# Anchored mix final execution state\n\n'+json.dumps(terminal,indent=2)+'\n\nScientific effects, uncertainty and mechanism narrative are finalized at publication from all fixed-phase tables. No additional training is admitted.\n')
    return terminal
