"""Neutral-entry background queue; one original deadline and full-run storage cap."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


root = Path(os.environ['RUN_ROOT'])
ops = Path(__file__).parent
results = root/'results'
ledger = json.loads((results/'RESOURCE_LEDGER.json').read_text())
started = time.time()
deadline = min(ledger['created'] + 24*3600, started + 3600-ledger['seconds_charged'])
attempt = os.environ.get('CONTROLLER_ATTEMPT', '1')
original_start = root/'CONTROLLER_START.json'
if original_start.exists():
    deadline = min(deadline, json.loads(original_start.read_text())['deadline'])
child = None


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
jobs=[('evaluate','initial',0),('train','L',3),('train','LU',3),
      ('evaluate','L',0),('evaluate','L',3),('evaluate','LU',0),('evaluate','LU',3)]
completed=json.loads((root/'STATUS.json').read_text()).get('completed',[]) if (root/'STATUS.json').exists() else []
try:
    with (original_start if not original_start.exists() else root/f'CONTROLLER_START_{attempt}.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),started=started,deadline=deadline,
                       argv=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True).strip()),f)
    for command,arm,K in jobs:
        key=f'{command}_{arm}_K{K}'
        if key in completed:continue
        if time.time()>=deadline:stop()
        env=dict(os.environ,TASK=command,ARM=arm,INNER=str(K))
        before_charge=json.loads((results/'RESOURCE_LEDGER.json').read_text())['seconds_charged']
        job_started=time.monotonic()
        with (root/(key+f'.attempt{attempt}.log')).open('x') as log:
            child=subprocess.Popen([str(ops/'python'),str(ops/'run.py')],env=env,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status(dict(state='RUNNING',job=key,child_pid=child.pid,completed=completed))
            while child.poll() is None:
                # 1.9 GiB stop leaves room for an emergency partial below 2 GiB.
                size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                if size>=int(1.9*1024**3):stop()
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
