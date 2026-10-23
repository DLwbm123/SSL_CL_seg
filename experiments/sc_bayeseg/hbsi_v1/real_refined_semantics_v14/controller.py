"""Neutral-entry background queue; one original deadline and full-run storage cap."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


root = Path(os.environ['RUN_ROOT'])
ops = Path(os.environ['OPS_ROOT'])
results = root/'results'
from experiments.sc_bayeseg.hbsi_v1.budgets import ResourceLedger
ResourceLedger(results,max_seconds=5400,max_updates=3400,max_bytes=1024**3)
ledger = json.loads((results/'RESOURCE_LEDGER.json').read_text())
campaign_path=Path(os.environ['CAMPAIGN_FILE'])
campaign=json.loads(campaign_path.read_text())
started = time.time()
deadline = min(ledger['created'] + 5400, started + 5400-ledger['seconds_charged'], campaign['deadline']-30)
attempt = os.environ.get('CONTROLLER_ATTEMPT', '1')
original_start = root/'CONTROLLER_START.json'
if original_start.exists():
    deadline = min(deadline, json.loads(original_start.read_text())['deadline'])
child = None


def tree_size(directory):
    total = 0
    for path in directory.rglob('*'):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # Atomic checkpoint publication/retention can remove a temporary
            # between enumeration and stat; count the published file instead.
            continue
    return total


def status(value):
    value.update(updated=time.time(), original_started=ledger['created'], controller_started=started,
                 controller_deadline=deadline, pid=os.getpid())
    temp=root/'STATUS.tmp'
    temp.write_text(json.dumps(value,indent=2))
    os.replace(temp,root/'STATUS.json')


def stop(*_):
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise TimeoutError('controller original time/storage budget reached')


signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGALRM,stop)
signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.time()))
jobs=[('preflight',None,3),('train','mixed',3),('train','aligned',3),
      ('evaluate','mixed',0),('evaluate','mixed',3),('evaluate','aligned',0),('evaluate','aligned',3)]
completed=json.loads((root/'STATUS.json').read_text()).get('completed',[]) if (root/'STATUS.json').exists() else []
try:
    with (original_start if not original_start.exists() else root/f'CONTROLLER_START_{attempt}.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),started=started,deadline=deadline,
                       argv=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True).strip()),f)
    for command,arm,K in jobs:
        key=f'{command}_{arm}_K{K}'
        if key in completed:continue
        if time.time()>=deadline:stop()
        env=dict(os.environ,TASK=command,ARM=arm or '',INNER=str(K))
        before_charge=json.loads((results/'RESOURCE_LEDGER.json').read_text())['seconds_charged']
        job_started=time.monotonic()
        with (root/(key+f'.attempt{attempt}.log')).open('x') as log:
            child=subprocess.Popen([str(ops/'python'),str(ops/'run.py')],env=env,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status(dict(state='RUNNING',job=key,child_pid=child.pid,completed=completed))
            while child.poll() is None:
                # Reserve space for a partial and enforce the whole campaign cap.
                size=tree_size(root)
                campaign_size=sum(tree_size(Path(r)) for r in campaign['run_roots'])
                if size>=int(.9*1024**3) or campaign_size>=campaign['storage_stop_bytes']:stop()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
        accounting=json.loads((results/'RESOURCE_LEDGER.json').read_text())
        overhead=max(0.,time.monotonic()-job_started-(accounting['seconds_charged']-before_charge))
        accounting['seconds_charged']+=overhead
        accounting.setdefault('controller_overhead',[]).append(dict(job=key,attempt=attempt,seconds=overhead))
        temporary=results/'RESOURCE_LEDGER.controller.tmp';temporary.write_text(json.dumps(accounting,indent=2))
        os.replace(temporary,results/'RESOURCE_LEDGER.json')
        if child.returncode:
            raise RuntimeError(f'{key} exited {child.returncode}')
        completed.append(key)
    status(dict(state='COMPLETE',completed=completed,scientific_state='DIAGNOSTIC_ONLY'))
except BaseException as exc:
    status(dict(state='FAILED',completed=completed,error=repr(exc),scientific_state='NOT_QUALIFIED'))
    raise
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
