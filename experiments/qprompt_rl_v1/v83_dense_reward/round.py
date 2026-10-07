"""Complete action rewards diagnose policy optimization, not GRPO success."""
import fcntl
import json
import math
import os
import shutil
import statistics
import time
import traceback
from pathlib import Path
from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import run_b as run
from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.continue_action import read,write

JOBS=[dict(id='qualification',job='qualification',caps=dict(qualification=8,actor_qualification=9)),
      dict(id='decision_entries',job='entries',caps=dict(entries=800)),
      *[dict(id=f'audit_{s}',job='audit',stream=s,caps=dict(audit=14400)) for s in (1,2)],
      *[dict(id=f'prior_{s}',job='fit',controller=s,caps=dict(actor_prior=64)) for s in (601,602)],
      dict(id='development',job='development',caps=dict(entries=400,development=4000))]


def preferences(rewards,floor):
    assert len(rewards)==9 and all(math.isfinite(x) for x in rewards)
    mean=statistics.mean(rewards);scale=max(statistics.pstdev(rewards),floor)
    exp=[math.exp(max(-3.,min(3.,(r-mean)/scale))) for r in rewards]
    return [x/sum(exp) for x in exp]


def dataset(rows,floor):
    table={(r['context'],r['entry_step'],r['stream'],r['action']):r for r in rows}
    contexts=sorted({r['context'] for r in rows})
    assert len(rows)==len(table)==288 and len(contexts)==8
    out=[]
    for key in contexts:
        for step in (100,200):
            group=[table[key,step,s,a] for s in (1,2) for a in range(9)]
            state=group[0]['state'];assert all(r['state']==state for r in group)
            assert len(state)==24 and abs(state[0]-step/300)<1e-6
            rewards=[statistics.mean(table[key,step,s,a]['reward'] for s in (1,2)) for a in range(9)]
            out.append(dict(context=key,step=step,state=state,rewards=rewards,preferences=preferences(rewards,floor)))
    return out


def screen(current,baseline):
    gates={};means=current['means'];uniform=means['UNIFORM_ACTION']
    for seed in (601,602):
        name=f'PRE_FROZEN_{seed}';row=means[name];previous=baseline['means'][name]
        comparisons={m:row['utility']-means[m]['utility'] for m in ('NATIVE','FIXED_BEST','UNIFORM_ACTION')}
        du=row['utility']-previous['utility'];dn=row['new']-uniform['new'];do=row['old']-uniform['old']
        gates[str(seed)]=dict(passed=min(comparisons.values())>=.0005 and du>=.0005 and dn>=-.0025 and do>=-.005,
                             utility_vs_controls=comparisons,utility_vs_V82_prior=du,new_vs_uniform=dn,old_vs_uniform=do)
    return dict(status='PASS_DENSE_REWARD_DIAGNOSTIC' if any(v['passed'] for v in gates.values()) else 'STOP_DENSE_REWARD_NO_PRACTICAL_GAIN',gates=gates)


def selfcheck():
    a=preferences(list(range(9)),1e-4);b=preferences([x+100 for x in range(9)],1e-4)
    assert max(abs(x-y) for x,y in zip(a,b))<1e-12 and abs(sum(a)-1)<1e-12 and a[8]>a[0]
    assert preferences([0.]*9,1e-4)==[1/9]*9
    rows=[dict(context=str(k),entry_step=step,stream=s,action=a,reward=float(a),state=[step/300]+[0.]*23)
          for k in range(8) for step in (100,200) for s in (1,2) for a in range(9)]
    assert len(dataset(rows,1e-4))==16
    try:dataset(rows[:-1],1e-4)
    except AssertionError:pass
    else:raise AssertionError('incomplete action table accepted')
    base={'means':{m:dict(new=.7,old=.8,utility=0.) for m in ('NATIVE','FIXED_BEST','UNIFORM_ACTION','PRE_FROZEN_601','PRE_FROZEN_602')}}
    assert screen(base,base)['status'].startswith('STOP')


def main():
    root=Path(os.environ['EXEC_RUN']);config=read(Path(os.environ['EXEC_CONFIG']));prior=Path(config['prior_campaign'])
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),commit=config['commit'],protocol=config['protocol']),f)
    assert config['protocol']=='V83_DENSE_REWARD' and config['training_horizon']==300
    assert read(prior/'DECISION.json')['status']=='STOP_EPISODE_ALIGNMENT_NO_PRACTICAL_GAIN'
    shutil.copyfile(prior/'PHYSICAL_LEDGER.jsonl',root/'PHYSICAL_LEDGER.jsonl');(root/'jobs').mkdir(exist_ok=False)
    previous=read(prior/'COSTS.json')['attempts'];caps=dict(previous)
    for k,n in dict(qualification=8,actor_qualification=9,entries=1200,audit=28800,actor_prior=128,development=4000).items():caps[k]=caps.get(k,0)+n
    write(root/'FROZEN_EXECUTION.json',dict(protocol=config['protocol'],cumulative_caps=caps,previous_attempts=previous,supervision='complete action reward vectors, not GRPO',decision_steps=[100,200]))
    account=run.Accounting(root,caps,JOBS)
    for jobs in (JOBS[:1],JOBS[1:2],JOBS[2:4]):
        if not run.schedule(root,config,jobs,account):return
    rows=[]
    for stream in (1,2):rows.extend(json.loads(l) for l in (root/f'jobs/audit_{stream}/ACTION_ROWS.jsonl').read_text().splitlines())
    table=dataset(rows,config['sigma_floor_override'])
    means={a:statistics.mean(row['rewards'][a] for row in table) for a in range(9)}
    fixed=sorted(means,key=lambda a:(-means[a],a!=0,a//3,a))[0]
    write(root/'FIXED_BEST.json',dict(action=fixed,reward_means=means,selection='global mean training reward; same complete information as dense actor; frozen before development'))
    for jobs in (JOBS[4:6],JOBS[6:]):
        if not run.schedule(root,config,jobs,account):return
    current=read(root/'jobs/development/DEVELOPMENT_RESULTS.json');baseline=read(prior/'jobs/development/DEVELOPMENT_RESULTS.json')
    outcome=screen(current,baseline);outcome.update(physical_cumulative=dict(account.count),C='NOT_RUN',D='NOT_RUN',development='REUSED_DIAGNOSTIC_NOT_INDEPENDENT',grpo_success_claim=False,time=time.time())
    write(root/'DECISION.json',outcome);write(root/'STATUS.json',dict(outcome,publication='PENDING'))


if __name__=='__main__':
    selfcheck()
    if os.environ.get('EXEC_SELFCHECK')!='1':
        try:main()
        except BaseException as exc:
            write(Path(os.environ['EXEC_RUN'])/'STATUS.json',dict(status='ENGINEERING_STOP',error=repr(exc),traceback=traceback.format_exc()));raise
