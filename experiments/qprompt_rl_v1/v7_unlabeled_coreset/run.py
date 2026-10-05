"""Finite pilot coordinator with a campaign-wide freeze before validation."""
import csv
import fcntl
import os
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from experiments.qprompt_rl_v1.v6_low_label_grpo import run as base
from . import protocol as p, controller as q

read,write=base.read,base.write


def report(root,jobs,evaluations):
    rows=[];calls=Counter();cost=Counter();audits=[];groups=[]
    for j in jobs+evaluations:
        dest=base.jobroot(root,j);final=read(dest/'FINAL.json');calls.update(final['physical_calls'])
        audits.append(dict(job=j['id'],**read(dest/'COMPLETION_AUDIT.json')))
        cost.update(read(dest/'MEASURED_COSTS.json'))
        if j['job']=='evaluate':rows.extend(read(dest/'RESULTS.json'))
        if j['job']=='learn':groups.append(dict(domain=j['domain'],controller=j['controller'],groups=read(dest/'GROUP_DIAGNOSTICS.json')))
    assert len(rows)==18 and dict(calls)==p.scope()['calls']
    write(root/'RESULTS.json',rows)
    with (root/'RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    paired=[]
    for r in rows:
        if r['method']!='GRPO':continue
        domain=[v for v in rows if v['domain']==r['domain']]
        fixed=[v for v in domain if v['method'] in p.FIXED]
        random=next(v for v in domain if v['method']=='RANDOM' and v['controller']==r['controller'])
        allu=next(v for v in fixed if v['method']=='ALL_U')
        paired.append(dict(domain=r['domain'],controller=r['controller'],vs_best_nonRL=r['macro_Dice']-max(v['macro_Dice'] for v in fixed+[random]),vs_random=r['macro_Dice']-random['macro_Dice'],old_vs_allU=r['old_REFUGE']-allu['old_REFUGE'],paired={v['method']:dict(new=r['macro_Dice']-v['macro_Dice'],old=r['old_REFUGE']-v['old_REFUGE']) for v in fixed+[random]}))
    new=mean(v['vs_best_nonRL'] for v in paired);rnd=mean(v['vs_random'] for v in paired);old=mean(v['old_vs_allU'] for v in paired)
    gate=new>=.005 and rnd>=.005 and old>=-.005 and min(v['old_vs_allU'] for v in paired)>=-.01 and min(v['vs_best_nonRL'] for v in paired)>=-.0025 and all(mean(v['vs_best_nonRL'] for v in paired if v['controller']==s)>0 for s in p.SEEDS)
    write(root/'DECISION.json',dict(status='SCREEN_CANDIDATE' if gate else 'NO_POSITIVE_EVIDENCE',screen_gate=gate,mean_new_vs_best_nonRL=new,mean_new_vs_random=rnd,mean_old_vs_allU=old,paired=paired,independent_patient_confirmation=False,automatic_followup=False,publication='pending'))
    evals=[]
    for j in evaluations:
        for path in (base.jobroot(root,j)/'evaluation_costs').glob('*/RESOURCE.json'):
            evals.append(dict(job=j['id'],endpoint=path.parent.name,**read(path)))
    write(root/'COSTS.json',dict(physical_calls=dict(calls),synthetic_actor_calls=16,measured=dict(cost),evaluators=evals,wall_clock_limit=None,note='CUDA event intervals include host gaps; selection VJPs and all development calls are additional costs, not speedup claims.'))
    write(root/'GROUP_DIAGNOSTICS.json',groups)
    write(root/'COMPLETION_AUDIT.json',dict(status='PASS',endpoints=len(rows),expected=18,all_training_locked_before_val=True,jobs=audits))
    text='# V7 unlabeled coreset pilot results\n\n20% original label pool; reward holdout removed from all student fit sets. Two controller repeats, one fixed source and historical development validation patients. No independent-patient confirmation.\n\n| Domain | Method | Controller | New Dice % | Old REFUGE Dice % | U images visited |\n|---|---|---|---:|---:|---:|\n'
    for r in rows:text+=f"| {r['domain']} | {r['method']} | {r['controller']} | {100*r['macro_Dice']:.4f} | {100*r['old_REFUGE']:.4f} | {r['selected_union']} |\n"
    text+='\nDecision: '+('SCREEN_CANDIDATE' if gate else 'NO_POSITIVE_EVIDENCE')+'. All adverse cells are retained; no automatic new experiment. RETRIEVE_FO is a first-order adaptation, not an official reproduction. See DECISION.json, COSTS.json and EXPERIMENT_PLAN.md for comparisons, scope and limitations.\n'
    (root/'RESULT_REPORT.md').write_text(text)
    write(root/'FINAL.json',dict(status='COMPLETE_PRIVATE',time=time.time(),publication='pending GitHub verification',endpoints=18))


def coordinator(root,c):
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'FINAL.json').exists()
    jobs=p.matrix()
    if (root/'RUN_LOCK.json').exists():assert read(root/'RUN_LOCK.json')['jobs']==jobs
    else:write(root/'RUN_LOCK.json',dict(jobs=jobs,scope=p.scope(),created=time.time()))
    (root/'scopes/pilot/jobs').mkdir(parents=True,exist_ok=True)
    write(root/'PROCESS.json',dict(pid=os.getpid(),time=time.time(),process_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    if not (root/'CONTROLLER_CHECK.json').exists():write(root/'CONTROLLER_CHECK.json',q.self_check())
    write(root/'status.json',dict(status='RUNNING',phase='TRAIN',time=time.time()))
    _,failed=base.schedule(root,c,jobs)
    if failed:
        write(root/'status.json',dict(status='TECHNICAL_FAILURE',failed=failed,time=time.time()));return
    write(root/'scopes/pilot/V7_ENDPOINT_LOCK.json',dict(time=time.time(),all_training_complete=True))
    evaluations=[dict(j,id='pilot__eval_'+j['name'],name='eval_'+j['name'],job='evaluate',stage='v7',deps=[],caps={},target=str(base.jobroot(root,j))) for j in jobs if j['job']=='endpoint']
    write(root/'status.json',dict(status='RUNNING',phase='EVALUATE',time=time.time()))
    _,failed=base.schedule(root,c,evaluations)
    if failed:
        write(root/'status.json',dict(status='TECHNICAL_FAILURE',failed=failed,time=time.time()));return
    report(root,jobs,evaluations);write(root/'status.json',dict(status='COMPLETE_PRIVATE',time=time.time()))


if __name__=='__main__':
    r=Path(os.environ['EXEC_RUN']);c=read(os.environ['EXEC_CONFIG'])
    try:coordinator(r,c)
    except BaseException as exc:
        import traceback
        write(r/'status.json',dict(status='FAILED',time=time.time(),error=repr(exc),traceback=traceback.format_exc()));raise
