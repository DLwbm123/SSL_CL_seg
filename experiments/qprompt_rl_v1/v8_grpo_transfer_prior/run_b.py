"""Qualification -> two finite priors -> all frozen development endpoints."""
import fcntl
import json
import os
import shutil
import subprocess
import time
import traceback
from collections import Counter
from pathlib import Path
from .continue_action import read,write

JOBS=[dict(id='qualification',job='qualification',caps=dict(qualification=8,actor_qualification=8)),
      *[dict(id=f'prior_{seed}',job='prior',controller=seed,caps=dict(prior=25600,actor_prior=256)) for seed in (601,602)],
      dict(id='development',job='development',caps=dict(entries=400,development=4000))]


class Accounting:
    def __init__(self,root,caps=None):
        self.root=root;self.offsets={};self.count=Counter();self.success=Counter();self.failure=Counter()
        self.caps=caps or dict(qualification=128,actor_qualification=128,auxiliary=8000,entries=1200,audit=14400,prior=51200,actor_prior=512,development=4000)
        for line in (root/'PHYSICAL_LEDGER.jsonl').read_text().splitlines():
            row=json.loads(line);self.consume(row)
    def consume(self,row):
        {'attempt':self.count,'success':self.success,'failure':self.failure}[row['event']][row['category']]+=1
    def refresh(self):
        for job in JOBS:
            source=self.root/'jobs'/job['id']/'PHYSICAL_LEDGER.jsonl'
            if not source.exists():continue
            with source.open() as f:
                f.seek(self.offsets.get(job['id'],0))
                while True:
                    pos=f.tell();line=f.readline()
                    if not line or not line.endswith('\n'):self.offsets[job['id']]=pos;break
                    row=dict(json.loads(line),job=job['id']);self.consume(row)
                    with (self.root/'PHYSICAL_LEDGER.jsonl').open('a') as out:out.write(json.dumps(row)+'\n');out.flush()
        assert all(n<=self.caps[k] for k,n in self.count.items())
        write(self.root/'COSTS.json',dict(attempts=dict(self.count),success=dict(self.success),failures=dict(self.failure),includes_stage_A=True,common_source_historical_student_updates=8000,updated=time.time()))


def schedule(root,config,jobs,account):
    pending=list(jobs);active={};failed=[];done=[]
    while pending or active:
        for gpu,(p,j) in list(active.items()):
            if p.poll() is None:continue
            path=root/'jobs'/j['id'];write(path/'PROCESS_EXIT.json',dict(exit_code=p.returncode,time=time.time()))
            if p.returncode or not (path/'FINAL.json').exists():failed.append(j['id'])
            else:done.append(j['id'])
            del active[gpu]
        if not failed:
            for gpu in (5,6,7):
                if not pending:break
                if gpu in active:continue
                free=int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<12000:continue
                j=pending.pop(0);path=root/'jobs'/j['id'];path.mkdir(exist_ok=False);cfg=dict(config,**j,campaign=str(root),gpu=gpu)
                write(path/'CONFIG.private.json',cfg)
                env=dict(os.environ,EXEC_RUN=str(path),EXEC_CONFIG=str(path/'CONFIG.private.json'),EXEC_MODULE='experiments.qprompt_rl_v1.v8_grpo_transfer_prior.stage_b',PYTHONPATH=config['code'],CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8')
                with (path/'worker.log').open('a') as log:
                    p=subprocess.Popen(['bash','./with_nas_storage.sh',config['python'],'-c','import os,runpy;runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")'],cwd=config['code']+'/experiments/lcrseg/scripts',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(path/'LAUNCH.json',dict(pid=p.pid,gpu=gpu,commit=config['commit'],time=time.time()));active[gpu]=(p,j)
        account.refresh()
        states={j['id']:read(root/'jobs'/j['id']/'STATUS.json') if (root/'jobs'/j['id']/'STATUS.json').exists() else {} for j in jobs}
        write(root/'STATUS.json',dict(status='ENGINEERING_STOP' if failed else 'RUNNING',phase='STAGE_B',active=[j['id'] for _,j in active.values()],pending=[j['id'] for j in pending],completed=done,failed=failed,job_progress=states,physical=dict(account.count),updated=time.time()))
        if failed and not active:return False
        if active or pending:time.sleep(10)
    return True


def main():
    root=Path(os.environ['EXEC_RUN']);config=read(Path(os.environ['EXEC_CONFIG']));lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),commit=config['commit']),f)
    assert read(Path(config['action_root'])/'ACTION_DECISION.json')['status']=='PASS_ACTION_SIGNAL'
    shutil.copyfile(Path(config['action_root'])/'PHYSICAL_LEDGER.jsonl',root/'PHYSICAL_LEDGER.jsonl');(root/'jobs').mkdir(exist_ok=False)
    account=Accounting(root)
    for jobs in (JOBS[:1],JOBS[1:3],JOBS[3:]):
        if not schedule(root,config,jobs,account):return
    outcome=read(root/'jobs/development/FINAL.json');write(root/'DECISION.json',outcome);write(root/'STATUS.json',dict(outcome,physical_cumulative=dict(account.count),publication='PENDING'))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(Path(os.environ['EXEC_RUN'])/'STATUS.json',dict(status='ENGINEERING_STOP',error=repr(exc),traceback=traceback.format_exc()));raise
