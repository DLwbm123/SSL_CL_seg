"""Fixed final-student decision; complete factorial effects stay separate from the candidate."""
import json,statistics
from pathlib import Path
from .core import *
from experiments.lcrseg.ssl_head_control_v0_1.results import table_write,read,rows
from experiments.lcrseg.ssl_foundation_v0_1.results import score,q

def summary(values):
    return dict(mean=statistics.mean(values),sample_SD=statistics.stdev(values),minimum=min(values),maximum=max(values),values=values)
def decision(table,complete):
    delta=[q(table,s,'T_SCE')-q(table,s,'SUP') for s in SEEDS];old=[q(table,s,'T_SCE')-q(table,s,'J_MSE') for s in SEEDS]
    domains=[dict(seed=s,domain=d,macro_delta=score(table,s,'T_SCE',d)-score(table,s,'SUP',d),rim_delta=score(table,s,'T_SCE',d,'rim_dice')-score(table,s,'SUP',d,'rim_dice'),cup_delta=score(table,s,'T_SCE',d,'cup_dice')-score(table,s,'SUP',d,'cup_dice')) for s in SEEDS for d in DOMAINS]
    checks=dict(mean_gain=statistics.mean(delta)>=.01,each_seed_positive=min(delta)>0,mean_gain_over_J_MSE=statistics.mean(old)>=.005,domain_guard=min(r['macro_delta'] for r in domains)>=-.005,class_guard=min(r[k] for r in domains for k in ('rim_delta','cup_delta'))>=-.02,engineering_complete=complete)
    return dict(main_candidate='T_SCE',checks=checks,delta_SUP=summary(delta),delta_J_MSE=summary(old),seed_domain=domains,status='TARGET_PATH_SSL_DEVELOPMENT_SIGNAL' if all(checks.values()) else 'TARGET_PATH_SSL_NOT_ESTABLISHED',ENGINEERING='ENGINEERING_COMPLETE' if complete else 'INCOMPLETE',CL='NOT_EVALUATED',further_experiments=False)
def factorial(table):
    result=[]
    for s in SEEDS:
        for d in DOMAINS:
            for metric in ('macro_fg_dice','rim_dice','cup_dice'):
                v={a:score(table,s,a,d,metric) for a in ARMS}
                terms=dict(mask_effect_MSE=v['T_MSE']-v['J_MSE'],mask_effect_SCE=v['T_SCE']-v['J_SCE'],loss_effect_joint=v['J_SCE']-v['J_MSE'],loss_effect_teacher=v['T_SCE']-v['T_MSE'],interaction=(v['T_SCE']-v['J_SCE'])-(v['T_MSE']-v['J_MSE']))
                result.extend(dict(seed=s,domain=d,metric=metric,effect=k,value=x) for k,x in terms.items())
    original=list(result)
    for d in DOMAINS:
        for metric in ('macro_fg_dice','rim_dice','cup_dice'):
            for effect in ('mask_effect_MSE','mask_effect_SCE','loss_effect_joint','loss_effect_teacher','interaction'):
                rr=[r['value'] for r in original if (r['domain'],r['metric'],r['effect'])==(d,metric,effect)]
                result.append(dict(seed='all',domain=d,metric=metric,effect=effect,**summary(rr)))
    return result

def verify_complete(base):
    receipts=[];exits=[];snapshots=[]
    for s in SEEDS:
        for d in DOMAINS:
            warm=[];initial=[];order=[]
            for a in ARMS:
                p=Path(base)/f'seed{s}'/d/a;r=read(p/'receipt.json');deploy=read(p/'deployment.json');n=sum(1 for _ in (p/'steps.jsonl').open())
                assert r['source']==deploy['source'] and r['head_mode']==deploy['head_mode']==LINEAR
                assert n==r['optimizer_updates']==(3200 if d==DOMAINS[0] else 2100)
                assert r['models']==2 and deploy['models']==1 and deploy['status']=='PASS' and deploy['student_hash']==r['student_hash']
                assert r['counters']['backward']==r['counters']['ema_updates']==n and r['training_prototype_bytes']==0 and not r['sigma_requires_grad'] and not r['sigma_gradient'] and not r['grad_update_in_Adam']
                assert r['counters'].get('gas_autograd',0)==0
                if a=='SUP':assert r['unlabeled_opens']==0 and r['counters'].get('student_u_images',0)==0
                w=read(p/'warmup.json');assert w['u_opens']==0;warm.append(w);initial.append(read(p/'initialization.json'));order.append(r['label_order_hash']);receipts.append(r)
                for e in (20,40,60,80,100):
                    snap=read(p/f'eval{e}/receipt.json');assert snap['status']=='COMPLETE' and snap['state_preserved'] and snap['additional_backward']==0 and snap['head_mode']==LINEAR
                    assert snap['prediction_seals']==COUNTS[d][2]*5
                    for file in ('correction_aggregate.csv','gradient_aggregate.csv','quality_aggregate.csv','probability_aggregate.csv'):assert (p/f'eval{e}'/file).stat().st_size>0
                    snapshots.append(dict(seed=s,domain=d,arm=a,epoch=e,**snap))
                for k in ('train','deploy'):
                    ex=read(p.parent/(a+'_'+k+'_exit.json'));assert ex['exit_code']==0;exits.append(ex)
            for k in ('student_hash','ema_hash','optimizer_hash','label_order_hash'):assert len({x[k] for x in warm})==1
            assert len({x['student_hash'] for x in initial})==1 and len(set(order))==1
    return receipts,exits,snapshots

def finish(base,tasks):
    b=Path(base);out=b/'public_results';out.mkdir();metrics=[];correction=[];grad=[];quality=[];training=[];counts=[]
    for p in sorted(b.glob('seed*/*/*')):
        if not p.is_dir():continue
        n=0;acc={}
        if (p/'steps.jsonl').exists():
            for line in (p/'steps.jsonl').open():
                row=json.loads(line);n+=1
                for x in row['by_teacher_class']:
                    k=(row['epoch'],x['class_id']);r=acc.setdefault(k,dict(steps=0,accepted=0,raw_loss_contribution=0.,weighted_loss_contribution=0.,SCE=0.,target_entropy=0.,KL=0.,local_logit_gradient_norm=0.));r['steps']+=1
                    for field in r:
                        if field!='steps':r[field]+=x[field]
        for (epoch,cls),r in acc.items():training.append(dict(kind='actual_training',seed=int(p.parent.parent.name[4:]),domain=p.parent.name,arm=p.name,epoch=epoch,class_id=cls,**r,unit='accepted summed; per-batch contributions summed over epoch; divide by steps for mean'))
        counts.append(dict(seed=p.parent.parent.name,domain=p.parent.name,arm=p.name,updates=n))
        for e in (20,40,60,80,100):
            root=p/f'eval{e}'
            for file,target,kind in [('metrics.csv',metrics,None),('correction_aggregate.csv',correction,None),('gradient_aggregate.csv',grad,'evaluator_local_logit'),('quality_aggregate.csv',quality,'pseudo_quality'),('probability_aggregate.csv',quality,'probability')]:
                if (root/file).exists():target.extend(dict(**r,kind=kind) if kind else r for r in rows(root/file))
    engineering_error=None;receipts=[];exits=[];snapshots=[]
    try:receipts,exits,snapshots=verify_complete(b)
    except Exception as exc:engineering_error=repr(exc)
    final=[r for r in metrics if int(r['epoch'])==100 and r['model']=='student'];complete=engineering_error is None
    if len(final)==30:dec=decision(final,complete);contrasts=factorial(final)
    else:dec=dict(ENGINEERING='INCOMPLETE',status='INCOMPLETE',main_candidate='T_SCE',error=engineering_error);contrasts=[]
    dec.update(CORRECTION_OPPORTUNITY_RETENTION='DESCRIPTIVE_RECORDED_NOT_A_MODEL_BENEFIT_GATE',TARGET_ERROR_PROPAGATION='DESCRIPTIVE_RECORDED',formal_updates=sum(r['updates'] for r in counts),historical_total=168608+sum(r['updates'] for r in counts),stopped=True,error=engineering_error)
    for file,rr in [('FINAL_METRICS.csv',final),('EPOCH_METRICS.csv',metrics),('FACTORIAL_CONTRASTS.csv',contrasts),('CORRECTION_PATH_DIAGNOSTICS.csv',correction),('CLASS_LOSS_AND_GRADIENT.csv',training+grad),('PROBABILITY_AND_PSEUDO_QUALITY.csv',quality),('RUN_LEDGER.csv',counts)]:table_write(out/file,rr)
    for file,obj in [('DECISION.json',dec),('TRAINING_AND_MEMORY_ACCOUNTING.json',dict(receipts=receipts,child_exits=exits,snapshots=snapshots,tasks=tasks,actual_counts=counts,qualification=read(b/'qualification_ledger.json'),smoke=read(b/'real_smoke/receipt.json'))),('SOURCE_AND_INPUT_LINEAGE.json',read(b/'INPUT_LINEAGE.json')),('TEST_REPORT.json',read(b/'qualification_ledger.json'))]:write_json(out/file,obj)
    (out/'FINAL_REPORT.md').write_text('# Target-path fixed matrix final state\n\n'+json.dumps(dec,indent=2)+'\n\nFinal student only; all three seeds fixed in advance. Raw public tables preserve every factor, class and seed. Narrative interpretation is completed at publication; no additional experiment is admitted.\n')
    write_json(b/'TERMINAL.json',dec);return dec
