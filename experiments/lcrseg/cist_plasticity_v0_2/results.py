"""One pre-specified primary comparison; descriptive tradeoffs never veto its mean signal."""
from collections import Counter,defaultdict
from pathlib import Path
import time
import numpy as np
from experiments.lcrseg.ams_seq_transfer_v0_1.results import csv_read,trajectory,effect_summary,MEASURES
from experiments.lcrseg.ssl_foundation_v0_1.diagnostics import csv_write,METRICS
from . import core as c,contract as ct

ARMS=(*c.ARMS,'F_FULL','F_CONV','ISO_COND_L','ISO_COND_LU','STATIC_SOURCE')
COMPARISONS=((c.MAIN,'F_CONV'),(c.MAIN,'F_FULL'),('GN_CIST_L','GN_L'),('GN_L','F_CONV'),('GN_CIST_L','F_CONV'),('GN_L','ISO_COND_L'),('GN_CIST_L','ISO_COND_L'))

def primary_decision(delta,interval):
    mean,sd=effect_summary(delta);value=float(mean[0])
    state='PRIMARY_FINAL_SIGNAL_AT_LEAST_0_005' if value>=.005 else 'SMALL_POSITIVE_PRIMARY_SIGNAL' if value>0 else 'NO_PRIMARY_ACCURACY_GAIN'
    return dict(arm=c.MAIN,baseline='F_CONV',scientific_state=state,mean=dict(zip(MEASURES,map(float,mean))),
                paired_seed_SD=dict(zip(MEASURES,map(float,sd))),seed_Final_deltas=delta.mean(0)[:,0].tolist(),
                worst_order_averaged_seed_Final=float(delta.mean(0)[:,0].min()),patient_interval=interval,
                evidence_uncertain=interval['contains_zero'],secondary_outcomes_are_not_conjunctive_gates=True)

def finish(base):
    b=Path(base);source=ct.verify();p=ct.protocol();seal=ct.read(b/'TARGET_WEIGHT_SEAL.json');assert seal['source']==source and len(seal['students'])==18
    inputs=ct.read(b/'private_inputs.json');sb=ct.read(b/'SOURCE_BINDING.json');bb=ct.read(b/'BASELINE_BINDING.json');parent=Path(inputs['parent']);out=b/'public_results';out.mkdir()
    private={};source_scores={};aggregate=[];ledger=[];memory=[];mechanisms=[];training=[];ops=Counter()
    def load_scores(path,tid,domains,checksum=None):
        if checksum:ct.check_hash(path,checksum)
        rows=csv_read(path)
        for d in domains:
            rr=[r for r in rows if r['domain']==d];assert [int(r['patient_index']) for r in rr]==list(range(c.COUNTS[d][2]))
            private[(tid,d)]=np.array([[float(r[k]) for k in METRICS] for r in rr])
    for src in p['sources']:
        tid=src['task_id'];entry=sb['sources'][tid];root=parent/'tasks'/tid/'evaluation'
        load_scores(root/'private_patient_metrics.csv',tid,[src['domain']],entry['source_scores_sha256'])
        source_scores[(src['order'],src['seed'])]=private[(tid,src['domain'])]
        aggregate.extend(dict(r,arm='SOURCE_STAGE_REFERENCE') for r in csv_read(root/'metrics.csv'))
        prefix=tid.removesuffix('SRC_CE')
        for arm in ('F_FULL','F_CONV','ISO_COND_L','ISO_COND_LU','STATIC_SOURCE'):
            bid=prefix+arm;bound=bb['baselines'][bid];run=Path(inputs['baseline_'+bound['storage']]);root=run/'tasks'/bid/'evaluation'
            target=next(d for d in c.DOMAINS if d!=src['domain']);domains=[target] if arm=='STATIC_SOURCE' else list(c.DOMAINS)
            load_scores(root/'private_patient_metrics.csv',bid,domains,bound['score_sha256']);aggregate+=csv_read(root/'metrics.csv')
            if arm=='STATIC_SOURCE':
                private[(bid,src['domain'])]=source_scores[(src['order'],src['seed'])]
                sr=next(r for r in aggregate if r['task_id']==tid);aggregate.append(dict(sr,task_id=bid,arm=arm,role='old'))
    for task in p['tasks']:
        tid=task['task_id'];root=b/'tasks'/tid;r=ct.read(root/'receipt.json');ev=ct.read(root/'evaluation/receipt.json')
        assert r['status']=='TRAINING_COMPLETE' and r['source']==ev['source']==source and r['task']==task and r['updates']==task['updates']
        assert r['student_hash']==ev['student_hash']==seal['students'][tid]['student_hash'] and r['memory']['full_models']==ev['models']==1
        assert r['boundary']['source_student_hash']==sb['sources'][task['source_task_id']]['student_hash'] and r['fixed_subset_unchanged'] and r['EMA_updates']==r['pseudo_labels']==0
        assert r['U_opens']==0
        for phase in ('train','eval'):
            assert ct.read(b/'logs'/(tid+'_'+phase+'_exit.json'))['exit_code']==0
            oc=ct.read(b/'tasks'/(tid+('_operations' if phase=='train' else '_eval_operations'))/'operation_counts.json');assert oc['status']=='PASS';ops.update(oc['counts'])
        load_scores(root/'evaluation/private_patient_metrics.csv',tid,list(c.DOMAINS));aggregate+=csv_read(root/'evaluation/metrics.csv')
        ledger.append(dict(**task,status='COMPLETE',actual_updates=r['updates'],training_source=source,student_hash=r['student_hash'],source_student_hash=r['boundary']['source_student_hash']))
        memory.append(dict(task_id=tid,**r['memory'],L_opens=r['L_opens'],U_opens=r['U_opens'],training_seconds=r['seconds'],deployment=ev))
        mechanisms+=ct.read(root/'evaluation/MECHANISMS.json');byepoch=defaultdict(Counter);n=0
        with (root/'steps.jsonl').open() as f:
            import json
            for line in f:
                z=json.loads(line);n+=1;assert z['position']==n and z['labeled_source_scoring_multiplicity']==1 and z['donor_extra_image_reads']==z['pseudo_label_records']==z['EMA_updates']==0
                rr=byepoch[z['epoch']];rr['steps']+=1
                for k in ('loss','CE','GT_Dice','anchor_label_records','U_image_records','donor_context_records'):rr[k]+=z[k]
                assert z['U_image_records']==0
                assert all(x['feature_requires_grad'] for x in z['current_H_geometry'])
                rr['current_H_distance_relative_max_sum']+=max(x['distance_relative_max'] for x in z['current_H_geometry'])
        assert n==task['updates']
        training.extend(dict(task_id=tid,arm=task['arm'],epoch=ep,steps=rr['steps'],**{k+'_mean_per_step':v/rr['steps'] for k,v in rr.items() if k!='steps'}) for ep,rr in byepoch.items())
    assert len(ledger)==18 and sum(r['actual_updates'] for r in ledger)==47700 and ops['optimizer_steps']==ops['backward']==47700 and ops['ema_updates']==0
    assert ops['sample_train_labeled']==95400 and ops['sample_train_unlabeled']==0 and ops['sample_val']==1170
    def measurements(weights=None):
        def mean(x,d):return float(x[:,3].mean() if weights is None else x[:,3]@weights[d])
        values={a:np.empty((2,3,4)) for a in ARMS}
        for oi,(o,src,dst) in enumerate((('O1',c.DOMAINS[0],c.DOMAINS[1]),('O2',c.DOMAINS[1],c.DOMAINS[0]))):
            for si,seed in enumerate(p['seeds']):
                ss=mean(source_scores[(o,seed)],src)
                for a in ARMS:
                    tid=f'{o}_s{seed}_{a}';values[a][oi,si]=trajectory(ss,mean(private[(tid,src)],src),mean(private[(tid,dst)],dst))
        return values
    values=measurements();summaries=[];trajectories=[];effects=[];costs=[];distributions=[];absolute=[];site={(r['task_id'],r['domain']):r for r in aggregate}
    for a in ARMS:
        mean,sd=effect_summary(values[a]);summaries.append(dict(arm=a,**dict(zip(MEASURES,map(float,mean))),**{k+'_seed_SD':float(sd[i]) for i,k in enumerate(MEASURES)}))
        for oi,o in enumerate(('O1','O2')):
            for si,seed in enumerate(p['seeds']):trajectories.append(dict(arm=a,order=o,seed=seed,**dict(zip(MEASURES,map(float,values[a][oi,si])))))
    for a,ref in COMPARISONS:
        delta=values[a]-values[ref];assert np.allclose(delta[:,:,0],(delta[:,:,1]+delta[:,:,2])/2,atol=1e-14,rtol=0) and np.allclose(delta[:,:,3],-delta[:,:,2],atol=1e-14,rtol=0)
        mean,sd=effect_summary(delta)
        for mi,metric in enumerate(MEASURES):
            for oi,o in enumerate(('O1','O2')):
                for si,seed in enumerate(p['seeds']):effects.append(dict(arm=a,baseline=ref,metric=metric,scope='seed_order',order=o,seed=seed,delta=float(delta[oi,si,mi]),sample_SD=None))
            for si,seed in enumerate(p['seeds']):effects.append(dict(arm=a,baseline=ref,metric=metric,scope='seed_order_average',order='both',seed=seed,delta=float(delta[:,si,mi].mean()),sample_SD=None))
            effects.append(dict(arm=a,baseline=ref,metric=metric,scope='overall',order='both',seed=None,delta=float(mean[mi]),sample_SD=float(sd[mi])))
        for o in ('O1','O2'):
            for seed in p['seeds']:
                for d in c.DOMAINS:
                    x,y=site[(f'{o}_s{seed}_{a}',d)],site[(f'{o}_s{seed}_{ref}',d)]
                    for k in ('rim_dice','cup_dice'):costs.append(dict(arm=a,baseline=ref,order=o,seed=seed,domain=d,role=x['role'],metric=k,delta=float(x[k])-float(y[k])))
                    dd=private[(f'{o}_s{seed}_{a}',d)][:,3]-private[(f'{o}_s{seed}_{ref}',d)][:,3]
                    distributions.append(dict(arm=a,baseline=ref,order=o,seed=seed,domain=d,n=len(dd),mean=float(dd.mean()),**{f'q{q:02d}':float(np.percentile(dd,q)) for q in (0,5,25,50,75,95,100)},harm_below_minus005=int((dd<-.05).sum()),harm_below_minus001=int((dd<-.01).sum())))
    for a in ARMS:
        for src in p['sources']:
            d=src['domain'];r=site[(f"{src['order']}_s{src['seed']}_{a}",d)];sr=site[(src['task_id'],d)]
            for metric in ('rim_dice','cup_dice'):absolute.append(dict(arm=a,order=src['order'],seed=src['seed'],domain=d,metric=metric,source_score=float(sr[metric]),old_score=float(r[metric]),absolute_forget=float(sr[metric])-float(r[metric])))
    rng=np.random.default_rng(p['analysis_seed']);draws=defaultdict(list)
    for _ in range(2000):
        weights={d:rng.multinomial(c.COUNTS[d][2],np.full(c.COUNTS[d][2],1/c.COUNTS[d][2]))/c.COUNTS[d][2] for d in c.DOMAINS};vv=measurements(weights)
        for a,ref in COMPARISONS:
            for oi,o in ((None,'both'),(0,'O1'),(1,'O2')):draws[(a,ref,o)].append((vv[a]-vv[ref]).mean((0,1)) if oi is None else (vv[a]-vv[ref])[oi].mean(0))
    intervals=[]
    for (a,ref,o),arr in draws.items():
        low,high=np.percentile(arr,[2.5,97.5],axis=0)
        for mi,k in enumerate(MEASURES):intervals.append(dict(arm=a,baseline=ref,order=o,metric=k,lower_95=float(low[mi]),upper_95=float(high[mi]),contains_zero=bool(low[mi]<=0<=high[mi]),replicates=2000,analysis_seed=p['analysis_seed'],condition='paired patients within physical domain; fixed trained models and repeatedly exposed development cohort'))
    def ci(ref):return next(x for x in intervals if x['arm']==c.MAIN and x['baseline']==ref and x['order']=='both' and x['metric']=='Final')
    primary=primary_decision(values[c.MAIN]-values['F_CONV'],ci('F_CONV'))
    component_delta=values['GN_CIST_L']-values['GN_L'];component_mean,component_sd=effect_summary(component_delta)
    component_ci=next(x for x in intervals if x['arm']=='GN_CIST_L' and x['baseline']=='GN_L' and x['order']=='both' and x['metric']=='Final')
    ablations=[dict(arm='GN_CIST_L',control='GN_L',Final_delta=float(component_mean[0]),paired_seed_SD=float(component_sd[0]),patient_interval=component_ci,claim='whole module increment in GN update space; does not rename primary')]
    pareto=[]
    for a in ARMS:
        x=values[a].mean((0,1));dominated=any(np.all(values[z].mean((0,1))[[1,2]]>=x[[1,2]]) and np.any(values[z].mean((0,1))[[1,2]]>x[[1,2]]) for z in ARMS if z!=a)
        pareto.append(dict(arm=a,Incoming=float(x[1]),Old=float(x[2]),dominated_on_mean_Incoming_Old=bool(dominated)))
    terminal=dict(engineering='ENGINEERING_COMPLETE',scientific_state=primary['scientific_state'],primary=primary,formal_tasks=18,formal_updates=47700,
                  PRIMARY_ACCURACY_EFFECT=primary,COMPONENT_EFFECT=ablations,PATIENT_UNCERTAINTY=primary['evidence_uncertain'],source_training_updates=0,baseline_retraining_updates=0,source=source,stopped=True,further_experiments=False,old_terminal_unchanged=True)
    for name,rows in [('RUN_LEDGER',ledger),('ARM_SUMMARY',summaries),('TRAJECTORIES',trajectories),('FINAL_SITE_CLASS_METRICS',aggregate),('PAIRED_EFFECTS',effects),('DOMAIN_CLASS_COSTS',costs),('PATIENT_DELTA_DISTRIBUTIONS',distributions),('ABSOLUTE_CLASS_FORGET',absolute),('PATIENT_INTERVALS',intervals),('PARETO',pareto),('TRAINING_ACCOUNTING',training)]:csv_write(out/(name+'.csv'),rows,fields=list(dict.fromkeys(k for r in rows for k in r)))
    c.write_json(out/'DECISIONS.json',dict(terminal=terminal,component_comparisons=ablations,new_primary_rule=True,old_negative_findings_preserved=True))
    c.write_json(out/'MECHANISMS.json',mechanisms);c.write_json(out/'MEMORY_AND_COMPUTE.json',dict(tasks=memory,formal_counts=dict(ops),qualification=ct.read(b/'qualification_ledger.json'),smoke=ct.read(b/'smoke/receipt.json'),historical_training_not_recounted=True))
    for name in ('SOURCE_BINDING.json','BASELINE_BINDING.json','TARGET_WEIGHT_SEAL.json'):c.write_json(out/name,ct.read(b/name))
    c.write_json(out/'TERMINAL.json',terminal)
    changes=[]
    for task in p['tasks']:
        for ep in (20,100):
            report=ct.read(b/'tasks'/task['task_id']/('diagnostics_'+str(ep*task['steps_per_epoch'])+'.json'))
            changes.extend(dict(task_id=task['task_id'],epoch=ep,**r) for r in report['parameter_changes'])
    csv_write(out/'PARAMETER_CHANGES.csv',changes)
    lines=['# CIST Plasticity Add-on V0.2 final report','',
           f"18/18 target tasks;47700 formal updates. Engineering complete. {terminal['scientific_state']}.",
           f"Unique primary CONV_CIST_L minus reused F_CONV Final: {primary['mean']['Final']:+.6f}; development-patient95% interval [{ci('F_CONV')['lower_95']:+.6f},{ci('F_CONV')['upper_95']:+.6f}]. Seed order-averaged effects: {primary['seed_Final_deltas']}.",'',
           '| Arm | Final | Incoming | Old | Absolute Forget |','|---|---:|---:|---:|---:|']
    for r in summaries:lines.append('|'+r['arm']+'|'+'|'.join(f'{r[k]:.6f}' for k in MEASURES)+'|')
    lines+=['',f"GN_CIST_L minus GN_L Final: {component_mean[0]:+.6f}; interval [{component_ci['lower_95']:+.6f},{component_ci['upper_95']:+.6f}]. This control finding does not replace the pre-specified primary.",'',
            'GN_L learns1408 GN affine scalars. GN_CIST_L learns4328 scalars. CONV_CIST_L learns the exact F_CONV438192 convolution weights plus2920 controller scalars. Other core parameters and source-derived bases stayed fixed. The full gradient flows through current features, context and fixed readout; there is no global no_grad on the training feature path. These are not equal-parameter comparisons.',
            'All18 final students were sealed before any new patient evaluation. Only current L images and own train_labeled GT were used in training. U image access, source training-image access, EMA and teachers are zero. Original complementary LCTX donor/source collection and CE/Dice arithmetic were reused. Epoch20 is an unevaluated recovery/diagnostic checkpoint; epoch100 is the only deployment evaluated.',
            'Every dynamic GN/conv/controller parameter, Adam, RNG, counters and position was checkpointed; pending diagnosis precedes another update. Deployment includes updated core parameters and the controller where present. Frozen subsets, rather than whole source-core equality, are enforced. Parameter drift and isometry describe current H versus its same-forward transport, not source functional invariance.',
            'TRAINING_ACCOUNTING explicitly divides epoch sums by steps. PARAMETER_CHANGES, MECHANISMS and MEMORY_AND_COMPUTE disclose parameter changes, controller behavior, within-image distortion, real updates, memory and runtime. Temporary current-step recovery matrices are removed after commit; no historical patient Q/b library is maintained.',
            'The new primary mean threshold0.005 is not conjoined with seed/domain/class requirements. No individual class loss automatically vetoes it. Reported2000 patient-bootstrap intervals use shared physical-domain weights and analysis_seed2026091202, conditional on repeatedly exposed development patients and trained models. They do not establish independent-patient confirmation or clinical safety.',
            'Any main gain supports the entire module added to the F_CONV update space; old frozen-core ablations do not isolate geometry, conditioning or basis contributions here. GN updating is not a novelty claim. Old CIST is not a complete GOLD reproduction, and its negative terminal remains unchanged. The old global-isometry U invariance is an analytic limitation of that comparison; no GLOBAL rerun was performed.',
            'The fixed experiment has stopped. No new seed, rank, layer, U objective or source-weight search was started. Public outputs contain aggregate metrics and cost/provenance only. Images/GT, patient identifiers, per-case scores, weights and raw NAS paths remain private.']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    files=[dict(name=x.name,bytes=x.stat().st_size) for x in out.iterdir() if x.is_file()]
    c.write_json(out/'NAS_ARCHIVE_RECEIPT.json',dict(status='REPORTS_ARCHIVED_ON_NAS',run_id=b.name,public_files=files,retained_deployments=18,
                 deployed_bytes=sum((b/'tasks'/t['task_id']/'deploy_student.pt').stat().st_size for t in p['tasks']),publication='GitHub verification separate',completed=time.time()))
    return terminal
