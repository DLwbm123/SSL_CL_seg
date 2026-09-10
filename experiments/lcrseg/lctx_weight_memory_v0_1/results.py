"""Pre-specified value and source-geometry decisions on complete paired trajectories."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from . import core as c
from .contract import read,verify,protocol
from experiments.lcrseg.ams_seq_transfer_v0_1.results import csv_read,trajectory,effect_summary
from experiments.lcrseg.ssl_foundation_v0_1.diagnostics import csv_write,METRICS
MEASURES=('Final','Incoming','Old','Forget')
ARMS=(*c.ARMS,'STATIC_SOURCE')
COMPARISONS=tuple(dict.fromkeys([('LR_SRC_A',a) for a in ARMS if a!='LR_SRC_A']+[('F_CONV','F_FULL'),('LR_FREE','F_CONV'),('LR_FREE','F_FULL'),('LR_RAND','LR_FREE'),('LR_SRC_AB','F_FULL')]+[(a,'STATIC_SOURCE') for a in c.ARMS]))

def gate(values,costs,baseline,kind):
    delta=values['LR_SRC_A']-values[baseline];mean,sd=effect_summary(delta)
    costs=[r for r in costs if r['arm']=='LR_SRC_A' and r['baseline']==baseline];groups=defaultdict(list)
    for r in costs:groups[(r['order'],r['role'],r['metric'])].append(r['delta'])
    checks=dict(Final=bool(mean[0]>=(.005 if kind=='VALUE' else .003)),Old=bool(mean[2]>=.020 if kind=='VALUE' else mean[2]>0),Incoming=bool(mean[1]>=-.005),order_Final=bool(np.all(delta[:,:,0].mean(1)>0) if kind=='VALUE' else np.all(delta[:,:,0].mean(1)>=0)),positive_seeds=int((delta.mean(0)[:,0]>0).sum())>=2,class_single=all(r['delta']>=-.050 for r in costs))
    if kind=='VALUE':checks.update(order_Old=bool(np.all(delta[:,:,2].mean(1)>0)),old_class_means=all(np.mean(v)>=-.005 for (o,r,m),v in groups.items() if r=='old'),current_class_means=all(np.mean(v)>=-.010 for (o,r,m),v in groups.items() if r=='current'))
    return dict(kind=kind,arm='LR_SRC_A',baseline=baseline,passed=all(checks.values()),checks=checks,mean=dict(zip(MEASURES,map(float,mean))),paired_seed_SD=dict(zip(MEASURES,map(float,sd))))

def finish(base):
    b=Path(base);source=verify();p=protocol();binding=read(b/'SOURCE_BINDING_AND_LAYER_MANIFEST.json');parent=Path(read(b/'private_inputs.json')['parent']);out=b/'public_results';out.mkdir()
    seal=read(b/'TARGET_WEIGHT_SEAL.json');assert seal['source']==source and len(seal['students'])==36
    aggregate=[];private={};source_scores={};ledger=[];memory=[];parameters=[];ops=Counter();supervision=[];init=[]
    for src in p['sources']:
        sid=src['task_id'];d=src['domain'];root=parent/'tasks'/sid/'evaluation';pr=csv_read(root/'private_patient_metrics.csv')
        assert all(r['domain']==d for r in pr) and [int(r['patient_index']) for r in pr]==list(range(c.COUNTS[d][2]))
        source_scores[(src['order'],src['seed'])]=np.array([[float(r[k]) for k in METRICS] for r in pr])
        for r in csv_read(root/'metrics.csv'):aggregate.append(dict(r,arm='SOURCE_STAGE_REFERENCE'))
    for task in p['tasks']:
        tid=task['task_id'];root=b/'tasks'/tid;r=read(root/'receipt.json');ev=read(root/'evaluation/receipt.json')
        assert r['source']==source and r['task']==task and r['updates']==task['updates'] and r['status']=='TRAINING_COMPLETE'
        assert seal['students'][tid]==r['student_hash']==ev['student_hash'] and ev['source']==source and ev['models']==1 and r['memory']['models']==2
        assert r['frozen_unchanged'] and r['U_opens']==0 and r['merge_max_abs']<=1e-5
        entry=binding['sources'][task['source_task_id']]
        assert r['boundary']['source_student_hash']==r['boundary']['initial_effective_hash']==r['boundary']['initial_EMA_hash']==entry['student_hash']
        if task['arm']=='F_FULL':assert {k:r[k] for k in entry['parent_F_FULL_reference']}==entry['parent_F_FULL_reference']
        for phase in ['train','eval']:
            assert read(b/'logs'/(tid+'_'+phase+'_exit.json'))['exit_code']==0
            oc=read(b/'tasks'/(tid+('_operations' if phase=='train' else '_eval_operations'))/'operation_counts.json');assert oc['status']=='PASS';ops.update(oc['counts'])
        rr=csv_read(root/'evaluation/private_patient_metrics.csv')
        for d in (task['source_domain'],task['domain']):
            rows=[x for x in rr if x['domain']==d];assert [int(x['patient_index']) for x in rows]==list(range(c.COUNTS[d][2]));private[(tid,d)]=np.array([[float(x[k]) for k in METRICS] for x in rows])
        aggregate+=csv_read(root/'evaluation/metrics.csv');ledger.append(dict(**task,status='COMPLETE',actual_updates=r['updates']));init.append(dict(task_id=tid,**r['boundary'],final_student_hash=r['student_hash']));memory.append(dict(task_id=tid,**r['memory'],merge_max_abs=r['merge_max_abs'],deployment=ev))
        for row in read(root/'PARAMETER_DIAGNOSTICS.json'):
            meta=next(z for z in entry['layers'] if z['layer']==row['layer']);parameters.append(dict(row,task_id=tid,seed=task['seed'],order=task['order'],k=meta['k'],rank=8,selected_source_energy=meta['selected_energy'],parameterized_right_leak=row.get('parameterized_right_leak'),parameterized_left_leak=row.get('parameterized_left_leak')))
        counts=Counter();n=0
        with (root/'steps.jsonl').open() as f:
            import json
            for line in f:
                z=json.loads(line);n+=1
                assert z['U_image_records']==0 and z['labeled_source_scoring_multiplicity']==1
                for k in ['anchor_label_records','donor_context_records','donor_extra_image_reads','U_image_records','CE','GT_Dice','SCE']:counts[k]+=z[k]
        assert n==task['updates'];supervision.append(dict(task_id=tid,updates=n,**counts))
    for src in p['sources']:
        tid=src['task_id'].replace('_SRC_CE','_STATIC_SOURCE');root=b/'tasks'/tid/'evaluation';ev=read(root/'receipt.json');assert ev['student_hash']==src['student_hash'] and ev['optimizer_updates']==ev['backward']==0
        assert read(b/'logs'/(tid+'_eval_exit.json'))['exit_code']==0
        oc=read(b/'tasks'/(tid+'_eval_operations')/'operation_counts.json');assert oc['status']=='PASS';ops.update(oc['counts'])
        d=next(d for d in c.DOMAINS if d!=src['domain']);rr=csv_read(root/'private_patient_metrics.csv');assert [int(x['patient_index']) for x in rr]==list(range(c.COUNTS[d][2]))
        private[(tid,d)]=np.array([[float(x[k]) for k in METRICS] for x in rr]);private[(tid,src['domain'])]=source_scores[(src['order'],src['seed'])]
        aggregate+=csv_read(root/'metrics.csv');source_row=next(r for r in aggregate if r['task_id']==src['task_id'] and r['domain']==src['domain']);aggregate.append(dict(source_row,task_id=tid,arm='STATIC_SOURCE',role='old'))
    assert len(ledger)==36 and sum(r['actual_updates'] for r in ledger)==95400 and ops['optimizer_steps']==ops['backward']==ops['ema_updates']==95400 and ops['sample_train_unlabeled']==0
    def measurements(weights=None):
        def mean(x,d):return float(x[:,3].mean() if weights is None else x[:,3]@weights[d])
        v={a:np.empty((2,3,4)) for a in ARMS}
        for oi,(order,src,dst) in enumerate([('O1',c.DOMAINS[0],c.DOMAINS[1]),('O2',c.DOMAINS[1],c.DOMAINS[0])]):
            for si,seed in enumerate(p['seeds']):
                ss=mean(source_scores[(order,seed)],src)
                for arm in ARMS:
                    tid=f'{order}_s{seed}_{arm}';v[arm][oi,si]=trajectory(ss,mean(private[(tid,src)],src),mean(private[(tid,dst)],dst))
        return v
    v=measurements();effects=[];costs=[];absolute=[];site={(r['task_id'],r['domain']):r for r in aggregate}
    for a,ref in COMPARISONS:
        delta=v[a]-v[ref];assert np.allclose(delta[:,:,3],-delta[:,:,2],atol=1e-14,rtol=0);assert np.allclose(delta[:,:,0],(delta[:,:,1]+delta[:,:,2])/2,atol=1e-14,rtol=0)
        mean,sd=effect_summary(delta)
        for mi,m in enumerate(MEASURES):
            for oi,o in enumerate(['O1','O2']):
                for si,s in enumerate(p['seeds']):effects.append(dict(arm=a,baseline=ref,metric=m,scope='seed_order',order=o,seed=s,delta=float(delta[oi,si,mi]),sample_SD=None))
                effects.append(dict(arm=a,baseline=ref,metric=m,scope='order_mean',order=o,seed=None,delta=float(delta[oi,:,mi].mean()),sample_SD=float(delta[oi,:,mi].std(ddof=1))))
            for si,s in enumerate(p['seeds']):effects.append(dict(arm=a,baseline=ref,metric=m,scope='seed_order_average',order='both',seed=s,delta=float(delta[:,si,mi].mean()),sample_SD=None))
            effects.append(dict(arm=a,baseline=ref,metric=m,scope='overall',order='both',seed=None,delta=float(mean[mi]),sample_SD=float(sd[mi])))
        for o in ['O1','O2']:
            for s in p['seeds']:
                for d in c.DOMAINS:
                    x,y=site[(f'{o}_s{s}_{a}',d)],site[(f'{o}_s{s}_{ref}',d)]
                    for m in ['rim_dice','cup_dice']:costs.append(dict(arm=a,baseline=ref,order=o,seed=s,domain=d,role=x['role'],metric=m,delta=float(x[m])-float(y[m])))
    for a in ARMS:
        for src in p['sources']:
            d=src['domain'];r=site[(f"{src['order']}_s{src['seed']}_{a}",d)];ss=site[(src['task_id'],d)]
            for m in ['rim_dice','cup_dice']:absolute.append(dict(arm=a,order=src['order'],seed=src['seed'],domain=d,metric=m,source_score=float(ss[m]),old_score=float(r[m]),absolute_forget=float(ss[m])-float(r[m])))
    draws=defaultdict(list);rng=np.random.default_rng(p['analysis_seed'])
    for _ in range(2000):
        weights={d:rng.multinomial(c.COUNTS[d][2],np.full(c.COUNTS[d][2],1/c.COUNTS[d][2]))/c.COUNTS[d][2] for d in c.DOMAINS};vv=measurements(weights)
        for a,ref in COMPARISONS:
            for oi,o in [(None,'both'),(0,'O1'),(1,'O2')]:draws[(a,ref,o)].append((vv[a]-vv[ref]).mean((0,1)) if oi is None else (vv[a]-vv[ref])[oi].mean(0))
    intervals=[]
    for (a,ref,o),z in draws.items():
        lo,hi=np.percentile(z,[2.5,97.5],axis=0)
        for i,m in enumerate(MEASURES):intervals.append(dict(arm=a,baseline=ref,order=o,metric=m,lower_95=float(lo[i]),upper_95=float(hi[i]),contains_zero=bool(lo[i]<=0<=hi[i]),resampling='paired_patient_within_physical_domain',replicates=2000,analysis_seed=p['analysis_seed'],condition='fixed trained models and repeatedly exposed development cohort'))
    value=gate(v,costs,'F_FULL','VALUE');geometry=[gate(v,costs,r,'SOURCE_GEOMETRY') for r in ['LR_FREE','LR_RAND']]
    status='RETENTION_VALUE_NOT_ESTABLISHED' if not value['passed'] else 'WEIGHT_SUBSPACE_VALUE_OBSERVED' if all(g['passed'] for g in geometry) else 'RETENTION_VALUE_WITH_GEOMETRY_UNCERTAIN'
    controls=[dict(arm=a,mean=dict(zip(MEASURES,map(float,(v[a]-v['LR_SRC_A']).mean((0,1)))))) for a in c.ARMS if a!='LR_SRC_A']
    terminal=dict(engineering='ENGINEERING_COMPLETE',scientific_state=status,tasks=36,formal_updates=95400,source_recovery_updates=0,static_evaluations=6,source=source,stopped=True,further_experiments=False)
    decisions=dict(terminal=terminal,VALUE=value,SOURCE_GEOMETRY=geometry,control_minus_main=controls)
    for name,rows in [('RUN_LEDGER',ledger),('SOURCE_INITIALIZATION_LEDGER',init),('FINAL_SITE_CLASS_METRICS',aggregate),('PAIRED_EFFECTS',effects),('DOMAIN_CLASS_COSTS',costs),('ABSOLUTE_CLASS_FORGET',absolute),('PATIENT_INTERVALS',intervals),('PARAMETER_DIAGNOSTICS',parameters),('SUPERVISION_SOURCE_ACCOUNTING',supervision)]:csv_write(out/(name+'.csv'),rows)
    c.write_json(out/'SOURCE_BINDING_AND_LAYER_MANIFEST.json',binding);c.write_json(out/'DECISIONS.json',decisions)
    c.write_json(out/'QUALIFICATION_AND_EXECUTION.json',dict(terminal=terminal,formal_counts=dict(ops),memory=memory,qualification_attempts=read(b/'qualification_ledger.json'),successful_training_exits=36,successful_evaluation_exits=42,STATIC_SOURCE_after_all_targets_sealed=True,old_updates_recounted=False,full_parent_trajectories_equal=6))
    lines=['# LCTX Weight Memory V0.1','',f'36 new targets / 95400 formal updates; source recovery 0; static reference 0 updates. {status}.','','| Arm | Final | Incoming | Old | Absolute Forget |','|---|---:|---:|---:|---:|']
    for a in ARMS:lines.append('|'+a+'|'+'|'.join(f'{x:.6f}' for x in v[a].mean((0,1)))+'|')
    lines+=['','| Main candidate minus control | Final | Incoming | Old | Forget |','|---|---:|---:|---:|---:|']
    for a in ARMS:
        if a!='LR_SRC_A':lines.append('|LR_SRC_A - '+a+'|'+'|'.join(f'{x:+.6f}' for x in (v['LR_SRC_A']-v[a]).mean((0,1)))+'|')
    lines+=['','VALUE failed checks: '+', '.join(k for k,val in value['checks'].items() if not val)+'.']
    for g in geometry:lines.append(g['kind']+' against '+g['baseline']+' failed checks: '+', '.join(k for k,val in g['checks'].items() if not val)+'.')
    lines+=['','This is a new weight-only protection implementation on supervised LCTX, not KI recovery or SSL confirmation. Main candidate remains LR_SRC_A regardless of control outcomes. A and B both train; input-only refers to the constraint. Effective rank-manifold dimensions and raw parameter bytes are in the source/layer manifest; AB has less effective capacity. The same Adam weight decay acts on different parameterizations and is not equal function regularization.','','Absolute old-class forgetting and every seed/order/class cost remain in the CSV tables. STATIC_SOURCE uses source-stage Old and six target-val evaluations only after the target-weight seal. Zero updates is a reference, not a trained candidate. F_FULL hashes match all six parent LCTX trajectories.','','Two fixed orders are averaged within each of three optimization seeds before SD. The 2000 paired-patient bootstrap draws share weights across methods, seeds, orders and stages within each physical domain. Intervals condition on already trained models and repeatedly exposed development patients; a zero-crossing interval is not equivalence. Local protected patch equations do not imply invariant old predictions or zero forgetting.','','Only current L images/labels entered training; U image reads 0. Parent source receipt/student hashes bind all six arms. No source EMA/Adam/RNG was inherited. Training retains student plus current dense EMA, whose updates use effective weights. Merge probes pass <=1e-5; deployment is one ordinary student. Parameter, basis, gradient, Adam, transient workspace and measured CUDA peaks are recorded separately; two model objects do not imply only twice the parameter memory.','','The fixed matrix has stopped. No new seeds, domains, rank/k/LR tuning or AMS follow-up was started. Public export contains aggregate evidence only; weights, patient rows, raw paths and logs remain on NAS.']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n');return terminal
