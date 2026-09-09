"""Fixed paired analyses; private patient arrays never enter the public export."""
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .contract import protocol,read,verify
from .core import write_json,DOMAINS
from experiments.lcrseg.ssl_foundation_v0_1.diagnostics import csv_write,METRICS
ARMS=('T_CE','T_CED','T_LCTX','T_UCTX','T_AMS')
COMPARISONS=(('T_AMS','T_CE'),('T_AMS','T_CED'),('T_LCTX','T_CED'),('T_UCTX','T_LCTX'),('T_AMS','T_UCTX'),('T_LCTX','T_CE'),('T_UCTX','T_CE'),('T_UCTX','T_CED'))
MEASURES=('Final','Incoming','Old','Forget')
def csv_read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def trajectory(source,old,current):
    return np.array([.5*(old+current),current,old,source-old])

def effect_summary(deltas):
    # Input order x seed x outcome; order is fixed, not an independent seed.
    byseed=deltas.mean(0)
    return byseed.mean(0),byseed.std(0,ddof=1)

def verdict(values,costs,arm,baseline):
    delta=values[arm]-values[baseline];mean,sd=effect_summary(delta)
    cs=[r for r in costs if r['arm']==arm and r['baseline']==baseline]
    buckets=defaultdict(list)
    for r in cs:buckets[(r['order'],r['role'],r['metric'])].append(r['delta'])
    checks=dict(Final=bool(mean[0]>=.005),Incoming=bool(mean[1]>=0),orders=bool((delta[:,:,0].mean(1)>0).all()),positive_seeds=int((delta.mean(0)[:,0]>0).sum())>=2,Old=bool(mean[2]>=-.005),class_means=all(np.mean(v)>=-.020 for v in buckets.values()),class_single=all(r['delta']>=-.050 for r in cs))
    return dict(arm=arm,baseline=baseline,passed=all(checks.values()),checks=checks,mean=dict(zip(MEASURES,map(float,mean))),paired_seed_SD=dict(zip(MEASURES,map(float,sd))))

def finish(base):
    b=Path(base);source=verify();p=protocol();out=b/'public_results';out.mkdir()
    tasks=p['tasks'];aggregate=[];private={};ledger=[];init=[];memory=[];opcounts=Counter();supervision=[]
    for t in tasks:
        root=b/'tasks'/t['task_id'];r=read(root/'receipt.json');ev=read(root/'evaluation/receipt.json')
        assert r['source']==source and r['task']==t and r['updates']==t['updates'] and r['status']=='TRAINING_COMPLETE'
        assert ev['source']==source and ev['student_hash']==r['student_hash'] and ev['models']==1 and r['models']==2
        assert not r['sigma_requires_grad'] and not r['sigma_has_grad']
        for phase in ('train','eval'):
            ex=read(b/'logs'/(t['task_id']+'_'+phase+'_exit.json'));assert ex['exit_code']==0
            o=read(b/'tasks'/(t['task_id']+('_operations' if phase=='train' else '_eval_operations'))/'operation_counts.json')
            assert o['status']=='PASS';opcounts.update(o['counts'])
        aggregate+=csv_read(root/'evaluation/metrics.csv')
        for domain in ev['seen_domains']:
            rows=[x for x in csv_read(root/'evaluation/private_patient_metrics.csv') if x['domain']==domain]
            assert [int(x['patient_index']) for x in rows]==list(range(len(rows)))
            private[(t['task_id'],domain)]=np.array([[float(x[k]) for k in METRICS] for x in rows])
        ledger.append(dict(**t,status='COMPLETE',actual_updates=r['updates']))
        init.append(dict(task_id=t['task_id'],source_task_id=t['source_task_id'],**r['boundary'],final_student_hash=r['student_hash']))
        memory.append(dict(task_id=t['task_id'],**{k:r[k] for k in ('models','student_parameter_bytes','optimizer_bytes','sigma_bytes','grad_update_bytes','peak_allocated','peak_reserved')},deployment=ev))
        byepoch={}
        with (root/'steps.jsonl').open() as f:
            n=0
            for line in f:
                x=json.loads(line);n+=1;z=byepoch.setdefault(x['epoch'],Counter())
                for k in ('CE','GT_Dice','dice_weight','SCE','target_entropy','KL','loss','labeled_image_records','anchor_label_records','U_image_records','donor_context_records','donor_extra_image_reads','labeled_source_pixels'):
                    z[k]+=x[k]
                z['steps']+=1;z['mix_steps']+=bool(x['mix_rectangles'])
                for cls,nc in enumerate(x['labeled_class_exposure']):z[f'L_pixels_{cls}']+=nc
                for rr in x['U_class']:z[f'U_predicted_{rr["class_id"]}']+=rr['predicted'];z[f'U_accepted_{rr["class_id"]}']+=rr['accepted']
            assert n==r['updates']
        for epoch,z in byepoch.items():
            supervision.append(dict(task_id=t['task_id'],arm=t['arm'],domain=t['domain'],seed=t['seed'],epoch=epoch,context_source='L_BATCH_REVERSED' if t['arm']=='T_LCTX' and epoch>20 else 'CURRENT_U' if t['arm'] in ('T_UCTX','T_AMS') and epoch>20 else 'NONE',**{k:z[k] for k in ('steps','CE','GT_Dice','dice_weight','SCE','target_entropy','KL','loss','labeled_image_records','anchor_label_records','U_image_records','donor_context_records','donor_extra_image_reads','labeled_source_pixels','mix_steps','L_pixels_0','L_pixels_1','L_pixels_2','U_predicted_0','U_predicted_1','U_predicted_2','U_accepted_0','U_accepted_1','U_accepted_2')}))
    assert len(ledger)==36 and sum(x['actual_updates'] for x in ledger)==95400 and opcounts['optimizer_steps']==95400
    for seed in p['seeds']:
        for order in ('O1','O2'):
            rr=[read(b/'tasks'/f'{order}_s{seed}_{a}'/'warmup.json') for a in ARMS[1:]]
            for k in ('student_hash','EMA_hash','optimizer_hash','label_order_hash'):assert len({x[k] for x in rr})==1
            assert all(x['U_opens']==0 for x in rr)
            source_hash=read(b/'tasks'/f'{order}_s{seed}_SRC_CE'/'receipt.json')['student_hash']
            for a in ARMS:assert read(b/'tasks'/f'{order}_s{seed}_{a}'/'boundary.json')['student_hash']==source_hash
    def measurements(weights=None):
        def score(tid,d):
            x=private[(tid,d)][:,3]
            return float(x.mean() if weights is None else np.dot(x,weights[d]))
        values={a:np.empty((2,3,4)) for a in ARMS}
        for oi,(order,src,dst) in enumerate((('O1',DOMAINS[0],DOMAINS[1]),('O2',DOMAINS[1],DOMAINS[0]))):
            for si,seed in enumerate(p['seeds']):
                prefix=f'{order}_s{seed}_';ss=score(prefix+'SRC_CE',src)
                for a in ARMS:values[a][oi,si]=trajectory(ss,score(prefix+a,src),score(prefix+a,dst))
        return values
    values=measurements();effects=[];costs=[];site={(r['task_id'],r['domain']):r for r in aggregate}
    for arm,ref in COMPARISONS:
        delta=values[arm]-values[ref];assert np.allclose(delta[:,:,3],-delta[:,:,2],atol=1e-14,rtol=0);assert np.allclose(delta[:,:,0],.5*(delta[:,:,1]+delta[:,:,2]),atol=1e-14,rtol=0)
        mean,sd=effect_summary(delta)
        for mi,m in enumerate(MEASURES):
            for oi,order in enumerate(('O1','O2')):
                for si,seed in enumerate(p['seeds']):effects.append(dict(arm=arm,baseline=ref,metric=m,scope='seed_order',order=order,seed=seed,delta=float(delta[oi,si,mi]),sample_SD=None))
                effects.append(dict(arm=arm,baseline=ref,metric=m,scope='order_mean',order=order,seed=None,delta=float(delta[oi,:,mi].mean()),sample_SD=float(delta[oi,:,mi].std(ddof=1))))
            for si,seed in enumerate(p['seeds']):effects.append(dict(arm=arm,baseline=ref,metric=m,scope='seed_order_average',order='both',seed=seed,delta=float(delta[:,si,mi].mean()),sample_SD=None))
            effects.append(dict(arm=arm,baseline=ref,metric=m,scope='overall',order='both',seed=None,delta=float(mean[mi]),sample_SD=float(sd[mi])))
        for order in ('O1','O2'):
            for seed in p['seeds']:
                for d in DOMAINS:
                    ra,rb=site[(f'{order}_s{seed}_{arm}',d)],site[(f'{order}_s{seed}_{ref}',d)]
                    for m in ('rim_dice','cup_dice'):
                        costs.append(dict(arm=arm,baseline=ref,order=order,seed=seed,domain=d,role=ra['role'],metric=m,delta=float(ra[m])-float(rb[m])))
    intervals=[];draws=defaultdict(list);rng=np.random.default_rng(p['analysis_seed'])
    sizes={d:len(next(v for (tid,dom),v in private.items() if dom==d)) for d in DOMAINS}
    for _ in range(2000):
        weights={d:rng.multinomial(n,np.full(n,1/n))/n for d,n in sizes.items()};v=measurements(weights)
        for a,ref in COMPARISONS:
            for oi,order in [(None,'both'),(0,'O1'),(1,'O2')]:draws[(a,ref,order)].append((v[a]-v[ref]).mean((0,1)) if oi is None else (v[a]-v[ref])[oi].mean(0))
    for (a,ref,order),arr in draws.items():
        lo,hi=np.percentile(arr,[2.5,97.5],axis=0)
        for mi,m in enumerate(MEASURES):intervals.append(dict(arm=a,baseline=ref,order=order,metric=m,resampling='paired_patient_within_physical_domain',replicates=2000,analysis_seed=p['analysis_seed'],lower_95=float(lo[mi]),upper_95=float(hi[mi]),contains_zero=bool(lo[mi]<=0<=hi[mi]),condition='fixed trained models and development cohort; shared weights across methods/seeds/orders/stages'))
    gates=[verdict(values,costs,a,r) for a in ('T_AMS','T_UCTX','T_LCTX') for r in ('T_CE','T_CED')]
    main=[g for g in gates if g['arm']=='T_AMS'];passed=all(g['passed'] for g in main)
    states=['SEQ_TRANSFER_VALUE_OBSERVED' if passed else 'SEQ_TRANSFER_VALUE_NOT_ESTABLISHED']
    if any(g['mean']['Incoming']>0 and not g['checks']['Old'] for g in main):states.append('PLASTICITY_RETENTION_TRADEOFF')
    inc=float((values['T_AMS']-values['T_UCTX']).mean((0,1))[0]);ci=next(x for x in intervals if x['arm']=='T_AMS' and x['baseline']=='T_UCTX' and x['metric']=='Final' and x['order']=='both')
    context=any(all(g['passed'] for g in gates if g['arm']==a) for a in ('T_UCTX','T_LCTX'))
    if context and (inc<.003 or ci['contains_zero']):states.append('CONTEXT_VALUE_WITH_PSEUDO_UNCERTAINTY')
    grouped=defaultdict(list)
    for r in costs:grouped[tuple(r[k] for k in ('arm','baseline','order','domain','role','metric'))].append(r['delta'])
    amber=[dict(zip(('arm','baseline','order','domain','role','metric'),key),mean_delta=float(np.mean(v))) for key,v in grouped.items() if key[3]==DOMAINS[1] and key[5]=='cup_dice' and np.mean(v)<-.010]
    terminal=dict(engineering='ENGINEERING_COMPLETE',scientific_states=states,formal_tasks=36,formal_updates=95400,source=source,pseudo_component='PRACTICAL_INCREMENT_MET' if inc>=.003 else 'PRACTICAL_INCREMENT_NOT_MET',pseudo_Final_increment=inc,pseudo_patient_CI=ci,stopped=True,further_experiments=False,old_KI='NOT_APPLICABLE')
    for name,rr in [('RUN_LEDGER',ledger),('SOURCE_INITIALIZATION_LEDGER',init),('FINAL_SITE_CLASS_METRICS',aggregate),('PAIRED_EFFECTS',effects),('DOMAIN_CLASS_COSTS',costs),('UNCERTAINTY_SUMMARY',intervals),('SUPERVISION_SOURCE_ACCOUNTING',supervision)]:csv_write(out/(name+'.csv'),rr)
    write_json(out/'MEMORY_AND_DEPLOYMENT.json',memory);write_json(out/'RESOURCE_ACCOUNTING.json',dict(formal_counts=dict(opcounts),qualification_local=read(b/'qualification_local/qualification.json'),qualification_server=read(b/'qualification_server/qualification.json'),smoke=read(b/'smoke/receipt.json'),qualification_attempt_ledger=read(b/'qualification_ledger.json'),historical_counts_reused=False))
    write_json(out/'DECISIONS.json',dict(terminal=terminal,gates=gates,Drishti_cup_amber=amber))
    overall=[r for r in effects if r['scope']=='overall'];lines=['# AMS Sequential Transfer V0.1','',f'Completed36 tasks /95400 formal optimizer updates. Engineering:ENGINEERING_COMPLETE. Scientific states:{", ".join(states)}.','','Full-parameter sequential control; not KI, parameter isolation, independent-patient confirmation or clinical safety. Fixed epoch100 students; no source-score selection.','','| Comparison | Final | Incoming | Old | Forget |','|---|---:|---:|---:|---:|']
    for a,ref in COMPARISONS:lines.append('|'+a+' - '+ref+'|'+'|'.join(f'{next(r["delta"] for r in overall if r["arm"]==a and r["baseline"]==ref and r["metric"]==m):+.6f}' for m in MEASURES)+'|')
    lines+=['',f'Pseudo Final increment:{inc:+.6f}; paired-patient95% interval [{ci["lower_95"]:+.6f},{ci["upper_95"]:+.6f}]. '+terminal['pseudo_component']+'.',f'Drishti cup amber groups:{len(amber)}; all retained in DECISIONS.json.','', 'PAIRED_EFFECTS.csv contains every seed/order and order-averaged seed SD. Shared physical-domain patient weights are used for all2000 bootstrap draws; two orders do not create six independent seeds. DeltaForget=-DeltaOld and DeltaFinal=(DeltaIncoming+DeltaOld)/2 verified.','', 'DOMAIN_CLASS_COSTS.csv and DECISIONS.json retain all class costs and each frozen guard. Source-only source score does not filter target work. LCTX donor rows are reuse of current L images, not extra U reads or extra donor GT supervision. Source image/GT access is absent from target training; current EMA and Adam reset.','', 'Operation attempts and successes, synthetic/smoke costs, peak allocation/reservation and single-student deployment are in RESOURCE_ACCOUNTING.json and MEMORY_AND_DEPLOYMENT.json. Patient rows, weights and raw runtime paths stay on NAS. No additional experiment starts.']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n');return terminal
