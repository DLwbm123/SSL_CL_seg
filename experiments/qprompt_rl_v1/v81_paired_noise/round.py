"""Independent paired-noise diagnostic; reuses qualified V8 jobs unchanged."""
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


def paired_floor(rows):
    table={(r['context'],r['stream'],r['action']):r['reward'] for r in rows}
    contexts=sorted({r['context'] for r in rows})
    assert len(contexts)==8 and len(table)==len(rows)==144
    contrasts=[abs((table[k,1,a]-table[k,1,0])-(table[k,2,a]-table[k,2,0]))/math.sqrt(2)
               for k in contexts for a in range(1,9)]
    return max(1e-4,statistics.median(contrasts))


def screen(current,baseline):
    uniform=current['means']['UNIFORM_ACTION'];gates={}
    for seed in (601,602):
        name=f'PRE_FROZEN_{seed}';now=current['means'][name];old=baseline['means'][name]
        du=now['utility']-uniform['utility'];db=now['utility']-old['utility']
        dn=now['new']-uniform['new'];do=now['old']-uniform['old']
        gates[str(seed)]=dict(passed=du>=.0005 and db>=.0005 and dn>=-.0025 and do>=-.005,
                             utility_vs_uniform=du,utility_vs_original_prior=db,new_vs_uniform=dn,old_vs_uniform=do)
    return dict(status='PASS_PAIRED_NOISE_DIAGNOSTIC' if any(v['passed'] for v in gates.values()) else 'STOP_PAIRED_NOISE_NO_PRACTICAL_GAIN',gates=gates)


def selfcheck():
    rows=[dict(context=str(k),stream=s,action=a,reward=(100 if s==2 else 0)+a*.001)
          for k in range(8) for s in (1,2) for a in range(9)]
    assert abs(paired_floor(rows)-1e-4)<1e-12 # common stream shift cancels
    rows[-1]['reward']+=1.;assert abs(paired_floor(rows)-1e-4)<1e-12
    try:paired_floor(rows[:-1])
    except AssertionError:pass
    else:raise AssertionError('missing paired row accepted')
    base={'means':{k:dict(new=.7,old=.8,utility=0.) for k in ('UNIFORM_ACTION','PRE_FROZEN_601','PRE_FROZEN_602')}}
    assert screen(base,base)['status'].startswith('STOP')
    better=json.loads(json.dumps(base));better['means']['PRE_FROZEN_601']['utility']=.001
    assert screen(better,base)['status'].startswith('PASS')


def main():
    root=Path(os.environ['EXEC_RUN']);config=read(Path(os.environ['EXEC_CONFIG']));prior=Path(config['prior_campaign'])
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),commit=config['commit'],protocol=config['protocol']),f)
    assert config['protocol']=='V81_PAIRED_NOISE'
    assert read(prior/'DECISION.json')['status']=='STOP_PRIOR_NOT_TRANSFERABLE_IN_D1_SCREEN'
    rows=[json.loads(s) for s in (Path(config['action_root'])/'ACTION_ROWS.jsonl').read_text().splitlines()]
    assert abs(config['sigma_floor_override']-paired_floor(rows))<1e-12
    shutil.copyfile(prior/'PHYSICAL_LEDGER.jsonl',root/'PHYSICAL_LEDGER.jsonl');(root/'jobs').mkdir(exist_ok=False)
    previous=read(prior/'COSTS.json')['attempts'];caps=dict(previous)
    for k,n in dict(qualification=8,actor_qualification=8,prior=51200,actor_prior=512,entries=400,development=4000).items():caps[k]=caps.get(k,0)+n
    write(root/'FROZEN_EXECUTION.json',dict(protocol=config['protocol'],sigma_floor=config['sigma_floor_override'],cumulative_caps=caps,previous_attempts=previous))
    account=run.Accounting(root,caps)
    for jobs in (run.JOBS[:1],run.JOBS[1:3],run.JOBS[3:]):
        if not run.schedule(root,config,jobs,account):return
    current=read(root/'jobs/development/DEVELOPMENT_RESULTS.json');baseline=read(prior/'jobs/development/DEVELOPMENT_RESULTS.json')
    outcome=screen(current,baseline);diagnostics={}
    for seed in (601,602):
        groups=[json.loads(s) for s in (root/f'jobs/prior_{seed}/GROUPS.jsonl').read_text().splitlines()]
        diagnostics[str(seed)]=dict(skipped=sum(g['actor']['skipped'] for g in groups),groups=len(groups))
    outcome.update(diagnostics=diagnostics,physical_cumulative=dict(account.count),C='NOT_RUN',D='NOT_RUN',development='REUSED_DIAGNOSTIC_NOT_INDEPENDENT',time=time.time())
    write(root/'DECISION.json',outcome);write(root/'STATUS.json',dict(outcome,publication='PENDING'))


if __name__=='__main__':
    selfcheck()
    if os.environ.get('EXEC_SELFCHECK')!='1':
        try:main()
        except BaseException as exc:
            write(Path(os.environ['EXEC_RUN'])/'STATUS.json',dict(status='ENGINEERING_STOP',error=repr(exc),traceback=traceback.format_exc()));raise
