"""Create-only jobs, no retry, phase gates and independent physical ledger."""
import fcntl,json,os,time,traceback,subprocess
from collections import Counter
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from experiments.qprompt_rl_v1.v6_low_label_grpo import run as base
from . import protocol as p
read,write=base.read,base.write

def stamp(epoch=None):
    now=datetime.fromtimestamp(time.time() if epoch is None else epoch,timezone.utc)
    return dict(epoch=now.timestamp(),utc=now.isoformat(),asia_shanghai=now.astimezone(ZoneInfo('Asia/Shanghai')).isoformat())

class Accounting:
    def __init__(self,root):self.root=root;self.offsets={};self.count=Counter();self.success=Counter();self.failures=Counter()
    def refresh(self,jobs):
        manifest=[]
        for j in jobs:
            path=base.jobroot(self.root,j);log=path/'PHYSICAL_LEDGER.jsonl'
            if log.exists():
                with log.open() as f:
                    f.seek(self.offsets.get(j['id'],0))
                    while True:
                        pos=f.tell();line=f.readline()
                        if not line or not line.endswith('\n'):self.offsets[j['id']]=pos;break
                        row=json.loads(line)
                        with (self.root/'PHYSICAL_LEDGER.jsonl').open('a') as dest:dest.write(json.dumps(dict(job=j['id'],**row))+'\n');dest.flush()
                        target={'attempt':self.count,'success':self.success,'failure':self.failures}[row['event']];target[row['category']]+=1
            state=read(path/'status.json') if (path/'status.json').exists() else {'status':'PENDING'}
            record=dict(job=j['id'],kind=j['job'],domain=j['domain'],method=j.get('method'),controller=j.get('controller'),status=state['status'])
            if (path/'SESSION.json').exists():record['started']=stamp(read(path/'SESSION.json')['start'])
            if (path/'PROCESS_EXIT.json').exists():record['ended']=stamp(read(path/'PROCESS_EXIT.json')['time'])
            manifest.append(record)
        assert sum(self.count.values())<=21232 and not set(self.count)-{'smoke','endpoint'}
        write(self.root/'JOB_MANIFEST.json',dict(updated=stamp(),jobs=manifest))
        write(self.root/'COSTS.json',dict(prior_failed_attempt=read(self.root/'PRIOR_ATTEMPT_COSTS.json') if (self.root/'PRIOR_ATTEMPT_COSTS.json').exists() else None,updated=stamp(),physical_attempts=dict(self.count),physical_successes=dict(self.success),physical_failures=dict(self.failures),student_budget=21232,controller_updates=0))

def schedule(root,c,jobs,account,alljobs,phase):
    pending=list(jobs);active={};failed=[];done=[]
    for j in jobs:assert not base.jobroot(root,j).exists(),'create-only; no automatic retry'
    while active or pending:
        for gpu,item in list(active.items()):
            proc=item['process']
            if proc.poll() is None:continue
            path=base.jobroot(root,item['node']);write(path/'PROCESS_EXIT.json',dict(time=time.time(),exit_code=proc.returncode))
            if proc.returncode==0 and (path/'FINAL.json').exists():done.append(item['node']['id'])
            else:failed.append(item['node']['id'])
            del active[gpu]
        if not failed:
            for gpu in (5,6,7):
                if gpu in active or not pending:continue
                raw=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
                free,util,used=map(int,raw.split(','))
                if free<6144 or util!=0 or used>100:continue
                j=pending.pop(0);active[gpu]=base.start(root,c,j,gpu)
        account.refresh(alljobs)
        write(root/'STATUS.json',dict(status='FAILED' if failed else 'RUNNING',phase=phase,updated=stamp(),active=[x['node']['id'] for x in active.values()],pending=[j['id'] for j in pending],completed=done,failed=failed))
        if failed and not active:break
        if active or pending:time.sleep(5)
    return failed

def coordinator(root,c):
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'RUN_MANIFEST.json').exists(),'never overwrite frozen audit'
    write(root/'PROCESS.json',dict(pid=os.getpid(),started=stamp(),process_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    (root/'scopes/v71/jobs').mkdir(parents=True,exist_ok=False)
    write(root/'STATUS.json',dict(status='RUNNING',phase='PROVENANCE',updated=stamp()))
    if c.get('prior_attempt'):
        prior=read(Path(c['prior_attempt'])/'stopped_report/COSTS.json')
        assert prior['physical_student_updates']==prior['controller_updates']==0, 'no budget reset across attempts'
        write(root/'PRIOR_ATTEMPT_COSTS.json',prior)
        write(root/'RESUMPTION.json',dict(authorized_by='User: 为什么失败，你解决一下',prior_attempt=Path(c['prior_attempt']).name,prior_student_updates=0,prior_controller_updates=0,prior_diagnostic_costs_preserved=True,scope_and_budget_unchanged=True,automatic_retry=False,started=stamp()))
    from .provenance import run
    try:run(root,c)
    except BaseException as exc:
        write(root/'PROVENANCE_AUDIT.json',dict(status='BLOCKED_BASELINE_PROVENANCE',error=repr(exc),traceback=traceback.format_exc(),student_updates=0,controller_updates=0));raise
    account=Accounting(root);jobs=p.matrix();alljobs=list(jobs)
    for phase,kind in [('A_DIAGNOSIS_C1','diagnosis'),('C2_QUALIFICATION','qualification'),('C3_ENDPOINTS','endpoint')]:
        failed=schedule(root,c,[j for j in jobs if j['job']==kind],account,alljobs,phase)
        if failed:return
        if kind=='qualification':
            checks=[read(base.jobroot(root,j)/'QUALIFICATION.json') for j in jobs if j['job']==kind]
            assert all(v['status']=='PASS' for v in checks) and account.count['smoke']==32
            write(root/'QUALIFICATION.json',dict(status='PASS',domains=checks,zero_checks=[read(base.jobroot(root,j)/'ZERO_CHECKS.json') for j in jobs if j['job']=='diagnosis'],student_updates=32,controller_updates=0))
    assert dict(account.count)==dict(smoke=32,endpoint=21200)
    write(root/'scopes/v71/V7_1_ENDPOINT_LOCK.json',dict(all_eight_frozen=True,time=time.time(),updated=stamp()))
    evals=[dict(j,id='v71__eval_'+j['name'],name='eval_'+j['name'],job='evaluate',stage='v7_1',caps={},target=str(base.jobroot(root,j))) for j in jobs if j['job']=='endpoint'];alljobs+=evals
    if schedule(root,c,evals,account,alljobs,'FINAL_EVALUATION'):return
    from .report import report
    report(root,c,jobs,evals);write(root/'STATUS.json',dict(status='COMPLETE_PRIVATE',publication='pending',updated=stamp()))

if __name__=='__main__':
    root=Path(os.environ['EXEC_RUN']);c=read(os.environ['EXEC_CONFIG'])
    try:coordinator(root,c)
    except BaseException as exc:
        write(root/'STATUS.json',dict(status='FAILED',updated=stamp(),error=repr(exc),traceback=traceback.format_exc()));raise
