"""Read-only checkpoint diagnostics, gated on completion of frozen replication."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path
import torch
from .model import initialize
from .train import setup, evaluate
from sc_bayeseg.window_v1.r_start import write


def worker(s):
    setup();out=Path(s['output']);out.mkdir(exist_ok=False)
    model=initialize(s,'B')
    ck=torch.load(s['checkpoint'],map_location='cpu',weights_only=False)
    assert ck['step']==s['step'];model.load_state_dict(ck['model'])
    def probability(x):
        logits,_=model(x,draws=s['draws'],image_loss=False)
        return torch.stack([v.softmax(1) for v in logits]).mean(0)
    model.probability=probability
    write(out/'CONTEXT.json',s);metrics=evaluate(model,s,out)
    write(out/'RESULTS.json',dict(status='COMPLETE',optimizer_updates=0,step=s['step'],draws=s['draws'],evaluation_seed=s['evaluation_seed'],metrics=metrics))
    write(out/'EXIT.json',dict(exit_code=0))


def controller(s):
    root=Path(s['root']);out=root/'diagnostics';out.mkdir(exist_ok=False)
    write(out/'QUEUE.json',s['diagnostic_tasks']);active={};done={};attempted=set();summary_done=False
    def stop(signum=None,frame=None):
        for p,g in active.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        raise SystemExit('diagnostic controller stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while True:
        elapsed=time.time()-s['original_started_unix']
        total=sum(int(subprocess.check_output(['du','-s','-B1',p],text=True).split()[0]) for p in (s['root'],s['ops']))
        if elapsed>=85500 or total>=49_000_000_000:stop()
        gate=root/'CONTROLLER_EXIT.json'
        if not gate.exists():
            write(out/'STATUS.json',dict(status='WAITING_FOR_REPLICATION',elapsed=elapsed));time.sleep(30);continue
        assert json.loads(gate.read_text())['exit_code']==0, 'Replication failed; do not run downstream diagnostics'
        if not summary_done:
            env=dict(os.environ,SOURCE_RUN=str(Path(s['source_paths']['B']).parent.parent),REPLICA_RUN=s['root'])
            with (out/'replication_summary.log').open('w') as log:
                subprocess.run(['python','summary.py'],cwd=s['ops'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            summary_done=True
        for name,(p,gpu) in list(active.items()):
            if p.poll() is not None:done[name]=p.returncode;del active[name]
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        free={u.strip():int(m) for u,m in (line.split(',') for line in rows.splitlines())}
        available=[g for g in s['gpus'] if g not in [v[1] for v in active.values()] and free[g]>=4096]
        for task in s['diagnostic_tasks']:
            name=task['name']
            if name in attempted or not available:continue
            config=dict(s,**task,output=str(out/name),diagnostic_worker=True)
            path=Path(s['ops'])/('diag_'+name+'.json');write(path,config)
            env=dict(os.environ,TASK_SETTINGS=str(path),CUDA_VISIBLE_DEVICES=available.pop(0))
            with (out/(name+'.log')).open('ab') as log:
                p=subprocess.Popen(['python','diag.py'],cwd=s['ops'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=(p,env['CUDA_VISIBLE_DEVICES']);attempted.add(name)
        write(out/'STATUS.json',dict(status='RUNNING' if active else 'WAITING_FOR_GPU',active={k:dict(pid=p.pid,gpu=g) for k,(p,g) in active.items()},finished=done,elapsed=elapsed,allocated_bytes=total))
        if len(done)==len(s['diagnostic_tasks']):
            write(out/'EXIT.json',dict(exit_code=0 if all(v==0 for v in done.values()) else 1,finished=done));return
        time.sleep(30)


if __name__=='__main__':
    s=json.loads(Path(os.environ['TASK_SETTINGS']).read_text())
    (worker if s.get('diagnostic_worker') else controller)(s)
