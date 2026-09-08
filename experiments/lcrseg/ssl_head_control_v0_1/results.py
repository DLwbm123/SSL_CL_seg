"""All gates use unrounded final-student metrics and mutually exclusive P2 paths."""
import csv,json
from pathlib import Path
from .core import DOMAINS,ARMS,head_mode,LINEAR,NORMAL,write_json
from experiments.lcrseg.ssl_foundation_v0_1.results import score,q,read,rows
OLD_DOC=Path('experiments/lcrseg/docs/ssl_foundation_v0_1')

def table_write(path,rr):
    rr=list(rr)
    keys=list(dict.fromkeys(k for r in rr for k in r)) if rr else ['status']
    with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)

def screen(table,old,complete=True):
    candidates=[];supQ=q(table,11,'LIN_SUP');strong=max(supQ,q(old,11,'SUP_G0'),q(old,11,'SUP_G1'))
    for arm in ('LIN_MT_CONF','LIN_MT_PAS'):
        delta=[score(table,11,arm,d)-score(table,11,'LIN_SUP',d) for d in DOMAINS]
        classes=[score(table,11,arm,d,k)-score(table,11,'LIN_SUP',d,k) for d in DOMAINS for k in ('rim_dice','cup_dice')]
        checks=dict(mean_SSL_gain=q(table,11,arm)-supQ>=.01,domain_guard=min(delta)>=-.005,class_guard=min(classes)>=-.02,stronger_SUP=q(table,11,arm)>=strong+.005,complete=complete)
        candidates.append(dict(arm=arm,Q=q(table,11,arm),mean_delta=q(table,11,arm)-supQ,min_delta=min(delta),delta=delta,class_delta=classes,checks=checks,passed=all(checks.values())))
    win=sorted([x for x in candidates if x['passed']],key=lambda r:(-r['min_delta'],-r['Q'],r['arm']!='LIN_MT_CONF',r['arm']))
    delta=[score(table,11,'LIN_SUP',d)-score(old,11,'SUP_G1',d) for d in DOMAINS];classes=[score(table,11,'LIN_SUP',d,k)-score(old,11,'SUP_G1',d,k) for d in DOMAINS for k in ('rim_dice','cup_dice')]
    supchecks=dict(mean_gain=supQ-max(q(old,11,'SUP_G0'),q(old,11,'SUP_G1'))>=.01,domain_guard=min(delta)>=-.005,class_guard=min(classes)>=-.02,complete=complete)
    path='P2_SSL' if win else 'P2_SUP' if all(supchecks.values()) else 'NOT_ADMITTED'
    return dict(path=path,selected=win[0]['arm'] if win else None,candidates=candidates,supervised_checks=supchecks,supervised_delta=delta,supervised_class_delta=classes,supervised_gain_over_stronger=supQ-max(q(old,11,'SUP_G0'),q(old,11,'SUP_G1')),old_reference_Q={a:q(old,11,a) for a in ('SUP_G0','SUP_G1','MT_PAS_G1')},P0_used_for_selection=False)

def confirmation(table,gate,complete=True):
    path=gate['path'];a=gate['selected'] if path=='P2_SSL' else 'LIN_SUP';ref='LIN_SUP' if path=='P2_SSL' else 'SUP_G1';per=[];ds=[];cs=[];domain_means=[]
    for seed in (21,22):
        delta=[score(table,seed,a,d)-score(table,seed,ref,d) for d in DOMAINS];classes=[score(table,seed,a,d,k)-score(table,seed,ref,d,k) for d in DOMAINS for k in ('rim_dice','cup_dice')]
        per.append(dict(seed=seed,arm=a,reference=ref,Q=q(table,seed,a),reference_Q=q(table,seed,ref),Q_delta=q(table,seed,a)-q(table,seed,ref),delta=delta,class_delta=classes));ds+=delta;cs+=classes
    checks=dict(each_seed_positive=all(r['Q_delta']>0 for r in per),mean_gain=sum(r['Q_delta'] for r in per)/2>=.01,domain_guard=min(ds)>=-.005,class_guard=min(cs)>=-.02,complete=complete)
    if path=='P2_SSL':
        domain_means=[sum(score(table,s,a,d)-score(table,s,ref,d) for s in (21,22))/2 for d in DOMAINS]
        checks.update(each_domain_mean_nonnegative=min(domain_means)>=0,stronger_SUP=sum(q(table,s,a) for s in (21,22))/2>=max(sum(q(table,s,b) for s in (21,22))/2 for b in ('LIN_SUP','SUP_G1'))+.005)
    success=all(checks.values())
    return dict(path=path,passed=success,checks=checks,per_seed=per,status=('LINEAR_HEAD_SSL_INCREMENT_REPLICATED' if path=='P2_SSL' else 'SUPERVISED_HEAD_REFERENCE_SIGNAL_REPLICATED') if success else ('LINEAR_HEAD_SSL_REPLICATION_NOT_ESTABLISHED' if path=='P2_SSL' else 'SUPERVISED_HEAD_REFERENCE_NOT_REPLICATED'),domain_mean_delta=domain_means)

def validate_phase(base,seeds,arms):
    for seed in seeds:
        for domain in DOMAINS:
            initial=[];linear_warm=[];orders=[]
            for arm in arms:
                r=Path(base)/f'seed{seed}'/domain/arm;receipt=read(r/'receipt.json');dep=read(r/'deployment.json');init=read(r/'initialization.json');warm=read(r/'warmup.json')
                assert receipt['optimizer_updates']==(3200 if domain==DOMAINS[0] else 2100)
                assert receipt['head_mode']==dep['head_mode']==init['head_mode']==head_mode(arm)
                assert receipt['source']==dep['source']==init['source'];assert receipt['student_hash']==dep['student_hash'] and dep['status']=='PASS' and dep['models']==1 and receipt['models']==2
                assert sum(1 for _ in (r/'steps.jsonl').open())==receipt['optimizer_updates']
                assert warm['u_opens']==0
                if not arm.startswith('LIN_MT'):assert receipt['unlabeled_opens']==0
                assert receipt['counters']['backward']==receipt['counters']['ema_updates']==receipt['optimizer_updates'] and not receipt['sigma_gradient']
                if arm.startswith('LIN_'):assert receipt['counters'].get('gas_autograd',0)==0;linear_warm.append(warm)
                for e in (20,40,60,80,100):
                    dr=read(r/f'eval{e}/receipt.json');assert dr['status']=='COMPLETE' and dr['state_preserved']
                    for name in ('quality','probability','reliability','gradient_logit','correctness_strata'):assert (r/f'eval{e}/{name}_aggregate.csv').stat().st_size>0
                    if arm=='LIN_MT_PAS' and e>20:assert dr['matched_mass_completed']
                initial.append(init);orders.append(receipt['label_order_hash'])
            assert len({x['student_hash'] for x in initial})==1 and len(set(orders))==1
            for k in ('student_hash','ema_hash','optimizer_hash','label_order_hash'):assert len({w[k] for w in linear_warm})<=1
            if seed==11:
                old=read(OLD_DOC/'CLOSEOUT_VERIFICATION.json')['initializations']
                ref=next(x for x in old if x['domain']==domain and x['arm']=='SUP_G0')
                assert all(x['backbone_hash']==ref['backbone_hash'] and x['head_hash']==ref['head_hash'] for x in initial)
    return True

def collect(base):
    tables={n:[] for n in ('metrics','quality','probability','reliability','gradient_logit','correctness_strata','mass','training_class')};ledger=[];receipts=[];deploy=[]
    for r in sorted(Path(base).glob('seed*/*/*')):
        if not r.is_dir():continue
        n=0;class_totals={}
        if (r/'steps.jsonl').exists():
            for line in (r/'steps.jsonl').open():
                n+=1;step=json.loads(line)
                for stat in step['by_teacher_class']:
                    key=(step['epoch'],stat['class_id']);acc=class_totals.setdefault(key,dict(steps=0,accepted=0,squared_probability_sum=0.,MSE_loss_contribution=0.,weighted_MSE_loss_contribution=0.,weighted_local_logit_gradient_norm=0.));acc['steps']+=1
                    for field in acc:
                        if field!='steps':acc[field]+=stat[field]
        for (epoch,cls),acc in class_totals.items():
            tables['training_class'].append(dict(seed=int(r.parent.parent.name[4:]),domain=r.parent.name,arm=r.name,epoch=epoch,class_id=cls,steps=acc['steps'],accepted=acc['accepted'],probability_MSE_accepted_mean=acc['squared_probability_sum']/acc['accepted'] if acc['accepted'] else None,mean_MSE_loss_contribution=acc['MSE_loss_contribution']/acc['steps'],mean_weighted_MSE_loss_contribution=acc['weighted_MSE_loss_contribution']/acc['steps'],mean_weighted_local_logit_gradient_norm=acc['weighted_local_logit_gradient_norm']/acc['steps'],context='actual training; local logit derivative, not network gradient'))
        ok=(r/'receipt.json').exists() and (r/'deployment.json').exists();ledger.append(dict(seed=int(r.parent.parent.name[4:]),domain=r.parent.name,arm=r.name,updates=n,status='COMPLETE' if ok else 'INCOMPLETE'))
        if (r/'receipt.json').exists():receipts.append(read(r/'receipt.json'))
        if (r/'deployment.json').exists():deploy.append(dict(domain=r.parent.name,arm=r.name,seed=int(r.parent.parent.name[4:]),**read(r/'deployment.json')))
        for e in (20,40,60,80,100):
            for k in tables:
                file=r/f'eval{e}'/('metrics.csv' if k=='metrics' else k+'_aggregate.csv')
                if file.exists():tables[k]+=rows(file)
    for file in Path(base).glob('P0/*/*/mass_aggregate.csv'):tables['mass']+=rows(file)
    return tables,ledger,receipts,deploy

def finish(base,gate,confirm,error=None):
    base=Path(base);out=base/'public_results';out.mkdir();tables,ledger,receipts,deploy=collect(base)
    updates=sum(r['updates'] for r in ledger);expected={'NOT_ADMITTED':6,'P2_SUP':14,'P2_SSL':22}.get(gate['path'],6)
    complete=not error and len(ledger)==expected and all(r['status']=='COMPLETE' for r in ledger)
    p0=[read(p) for p in base.glob('P0/*/*/P0_RECEIPT.json')]
    status=dict(ENGINEERING='COMPLETE' if complete else 'INCOMPLETE',PAS_MATCHED_COUNT_EVIDENCE='DESCRIPTIVE_COMPLETE' if len(p0)==4 else 'INCOMPLETE_OLD_MODEL_DIAGNOSTIC',HEAD_SUPERVISED_EFFECT=confirm['status'] if confirm and gate['path']=='P2_SUP' else 'SCREEN_SIGNAL' if all(gate.get('supervised_checks',{}).values()) and gate.get('supervised_checks') else 'NOT_ESTABLISHED',SSL_INCREMENT=confirm['status'] if confirm and gate['path']=='P2_SSL' else 'NOT_ESTABLISHED',PAS_INCREMENT='DESCRIPTIVE_COMPONENT_CONTRASTS_ONLY',CL_AND_SOTA='NOT_EVALUATED',formal_updates=updates,cumulative_formal_updates=131508+updates,path=gate['path'],stopped=True,error=error)
    for name,key in [('EPOCH_CURVES.csv','metrics'),('PSEUDO_QUALITY_BY_CLASS.csv','quality'),('PROBABILITY_DIAGNOSTICS.csv','probability'),('RELIABILITY_BINS.csv','reliability'),('GRADIENT_LOGIT_DIAGNOSTICS.csv','gradient_logit'),('CORRECTNESS_STRATA.csv','correctness_strata'),('PAS_FIXED_MASS_AUDIT.csv','mass'),('TRAINING_CLASS_LOSS.csv','training_class')]:table_write(out/name,tables[key])
    final=[x for x in tables['metrics'] if int(x['epoch'])==100];table_write(out/'SINGLE_DOMAIN_METRICS.csv',[x for x in final if x['model']=='student']);table_write(out/'PER_CLASS_METRICS.csv',final);table_write(out/'RUN_LEDGER.csv',ledger)
    old=rows(OLD_DOC/'SINGLE_DOMAIN_METRICS.csv');contrasts=[]
    for seed in sorted({int(x['seed']) for x in final}):
        present={x['arm'] for x in final if int(x['seed'])==seed}
        pairs=[(a,'LIN_SUP') for a in ('LIN_MT_CONF','LIN_MT_PAS') if a in present]+([('LIN_MT_PAS','LIN_MT_CONF')] if {'LIN_MT_PAS','LIN_MT_CONF'}<=present else [])
        if 'SUP_G1' in present:pairs+=[('LIN_SUP','SUP_G1')]
        for a,b in pairs:
            for d in DOMAINS:
                if all(any(int(x['seed'])==seed and x['domain']==d and x['arm']==arm and x['model']=='student' for x in final) for arm in (a,b)):contrasts.append(dict(seed=seed,domain=d,arm=a,reference=b,**{k+'_delta':score(final,seed,a,d,k)-score(final,seed,b,d,k) for k in ('macro_fg_dice','rim_dice','cup_dice')}))
    if len([x for x in final if int(x['seed'])==11 and x['model']=='student'])==6:
        for d in DOMAINS:
            contrasts.append(dict(seed=11,domain=d,arm='LIN_SUP',reference='old_SUP_G0',**{k+'_delta':score(final,11,'LIN_SUP',d,k)-score(old,11,'SUP_G0',d,k) for k in ('macro_fg_dice','rim_dice','cup_dice')}))
            for a in ('CONF','PAS'):contrasts.append(dict(seed=11,domain=d,arm='LIN_MT_'+a,reference='head_by_SSL_interaction_vs_G0',**{k+'_delta':(score(final,11,'LIN_MT_'+a,d,k)-score(final,11,'LIN_SUP',d,k))-(score(old,11,'MT_'+a+'_G0',d,k)-score(old,11,'SUP_G0',d,k)) for k in ('macro_fg_dice','rim_dice','cup_dice')}))
    table_write(out/'HEAD_SSL_CONTRASTS.csv',contrasts)
    table_write(out/'REPLICATION_BY_SEED.csv',confirm['per_seed'] if confirm else [dict(status='NOT_ADMITTED' if gate['path']=='NOT_ADMITTED' else 'INCOMPLETE')])
    for name,obj in [('SCREEN_DECISION.json',gate),('CONFIRMATION.json',confirm or dict(status='NOT_ADMITTED')),('TRAINING_AND_MEMORY_ACCOUNTING.json',dict(tasks=receipts,P0=p0,formal_updates=updates,qualification=read(base/'qualification_server/qualification.json'),real_smoke=read(base/'real_smoke/receipt.json'))),('DEPLOYMENT_REPORT.json',deploy),('TEST_REPORT.json',dict(local=read(base/'qualification_local/qualification.json'),server=read(base/'qualification_server/qualification.json'),smoke=read(base/'real_smoke/receipt.json'))),('SOURCE_AND_INPUT_LINEAGE.json',read(base/'INPUT_LINEAGE.json')),('STATUS.json',status)]:write_json(out/name,obj)
    (out/'PAS_FIXED_MASS_AUDIT.md').write_text('Fixed teacher-predicted class composition and geometric K; scoring support equality explicitly reported after ignoring GT255. Permutations then prediction draws average within patient. P0 is descriptive and never selects a recipe. LR is teacher correctness conditional on C; undefined and infinite rows are disclosed. Full numeric interpretation follows at public closeout.\n')
    (out/'FINAL_REPORT.md').write_text('# Head control final state\n\n'+json.dumps(status,indent=2)+'\n\nNo post-hoc parameter, seed, threshold or primary-output switching. Full-precision tables and all candidate gates are attached. P0 is separate from segmentation evidence. CL/SOTA not evaluated.\n')
    write_json(base/'TERMINAL.json',status);return status
