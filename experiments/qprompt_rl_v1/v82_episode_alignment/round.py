"""Independent first-domain diagnostic with matched 300-step episodes."""
import fcntl
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import run_b as run
from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.continue_action import read,write
from experiments.qprompt_rl_v1.v81_paired_noise.round import paired_floor,screen

JOBS=[run.JOBS[0],dict(id='training_entries',job='training_entries',caps=dict(entries=800)),*run.JOBS[1:]]


def selfcheck():
    # The same context is visited eight times: four complete two-decision episodes.
    steps=[100+100*(visit%2) for visit in range(8)]
    assert steps==[100,200]*4 and [s/300 for s in steps]==[1/3,2/3]*4
    assert [100+100*(visit%8) for visit in range(8)]==list(range(100,900,100))
    assert sum(j['caps'].get('entries',0) for j in JOBS)==1200
    assert sum(j['caps'].get('prior',0) for j in JOBS)==51200


def main():
    root=Path(os.environ['EXEC_RUN']);config=read(Path(os.environ['EXEC_CONFIG']));prior=Path(config['prior_campaign'])
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),commit=config['commit'],protocol=config['protocol']),f)
    assert config['protocol']=='V82_EPISODE_ALIGNMENT'
    assert read(prior/'DECISION.json')['status']=='STOP_PAIRED_NOISE_NO_PRACTICAL_GAIN'
    assert config['training_horizon']==300 and config['episode_groups']==2
    rows=[json.loads(s) for s in (Path(config['action_root'])/'ACTION_ROWS.jsonl').read_text().splitlines()]
    assert abs(config['sigma_floor_override']-paired_floor(rows))<1e-12
    shutil.copyfile(prior/'PHYSICAL_LEDGER.jsonl',root/'PHYSICAL_LEDGER.jsonl');(root/'jobs').mkdir(exist_ok=False)
    previous=read(prior/'COSTS.json')['attempts'];caps=dict(previous)
    for k,n in dict(qualification=8,actor_qualification=8,prior=51200,actor_prior=512,entries=1200,development=4000).items():caps[k]=caps.get(k,0)+n
    write(root/'FROZEN_EXECUTION.json',dict(protocol=config['protocol'],sigma_floor=config['sigma_floor_override'],horizon=300,decisions=[100,200],cumulative_caps=caps,previous_attempts=previous))
    account=run.Accounting(root,caps,JOBS)
    for jobs in (JOBS[:1],JOBS[1:2],JOBS[2:4],JOBS[4:]):
        if not run.schedule(root,config,jobs,account):return
    current=read(root/'jobs/development/DEVELOPMENT_RESULTS.json');baseline=read(prior/'jobs/development/DEVELOPMENT_RESULTS.json')
    outcome=screen(current,baseline)
    outcome['status']='PASS_EPISODE_ALIGNMENT_DIAGNOSTIC' if outcome['status'].startswith('PASS') else 'STOP_EPISODE_ALIGNMENT_NO_PRACTICAL_GAIN'
    outcome.update(physical_cumulative=dict(account.count),C='NOT_RUN',D='NOT_RUN',development='REUSED_DIAGNOSTIC_NOT_INDEPENDENT',time=time.time())
    write(root/'DECISION.json',outcome);write(root/'STATUS.json',dict(outcome,publication='PENDING'))


if __name__=='__main__':
    selfcheck()
    if os.environ.get('EXEC_SELFCHECK')!='1':
        try:main()
        except BaseException as exc:
            write(Path(os.environ['EXEC_RUN'])/'STATUS.json',dict(status='ENGINEERING_STOP',error=repr(exc),traceback=traceback.format_exc()));raise
