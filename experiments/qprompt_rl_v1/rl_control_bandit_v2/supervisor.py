"""Frozen dependency queue; one serialized budget coordinator, at most four workers."""
import json,os,subprocess,sys,time,hashlib,signal
from pathlib import Path
from .runner import RUN,CODE,old,authorize
from .runtime import PLAN,TASKS,broker,process_identity
from r1_12h.core import atomic,append,events


def spawn(mode,gpu,**extra):
    e=dict(os.environ,EXEC_MODE=mode,CUDA_VISIBLE_DEVICES=str(gpu),**{k:str(v) for k,v in extra.items()});source=json.loads((RUN/'ACTIVE_CODE.json').read_text())['source']
    log=RUN/'logs'/f'{mode}_{extra.get("EXEC_TASK",extra.get("EXEC_BACKBONE","main"))}_{time.time_ns()}.log'
    with log.open('w') as stream:
        p=subprocess.Popen(['python3.12','-'],executable=sys.executable,cwd=source,env=e,stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT,text=True);p.stdin.write('from rl_control_bandit_v2.runner import main;main()\n');p.stdin.close()
    row=dict(pid=p.pid,identity=process_identity(p.pid),mode=mode,gpu=gpu,log=str(log),**extra);append(RUN/'PROCESS_LEDGER.jsonl',row);return p,row


def available():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(a):int(b) for a,b in (s.split(',') for s in text.splitlines())}


def supervise():
    authorize()
    for row in events(RUN/'PROCESS_LEDGER.jsonl'):
        if process_identity(row['pid'])==row['identity']:raise RuntimeError('live worker requires audited recovery')
    server,budget=broker(RUN);deadline=json.loads((RUN/'SESSION.json').read_text())['deadline'];atomic(RUN/'SUPERVISOR.json',dict(pid=os.getpid(),identity=process_identity(os.getpid()),code_commit=CODE))
    queue={t:dict(status='PENDING',failures=[]) for t in TASKS};active={};fingerprints={}
    if (RUN/'QUEUE.json').exists():queue=json.loads((RUN/'QUEUE.json').read_text())
    for task,row in queue.items():
        folder=RUN/'tasks'/task
        if (folder/'EVALUATION.json').exists() or (folder/'R3A_DONE.json').exists():row['status']='DONE'
        elif (folder/'TRAIN_DONE.json').exists():row['status']='TRAINED'
        elif row['status'] in ('RUNNING','EVALUATING'):row['status']='PENDING'
    try:
        for backbone in ('UNET_QUERY_128','DINOV2_VITS14_QUERY'):
            folder=RUN/'qualification'/backbone
            if (folder/'PASSED.json').exists():continue
            if (folder/'FAILURE.json').exists():raise RuntimeError('qualification requires repair; no blind retries')
            while time.time()<deadline:
                free=available();g=max(free,key=free.get)
                if free[g]>=18000:break
                time.sleep(20)
            if time.time()>=deadline:raise RuntimeError('qualification deadline')
            p,row=spawn('qualify',g,EXEC_BACKBONE=backbone)
            while p.poll() is None and time.time()<deadline:time.sleep(5)
            if p.poll() is None:
                if process_identity(p.pid)==row['identity']:p.terminate()
                p.wait(timeout=30)
            if p.returncode or not (folder/'PASSED.json').exists():atomic(folder/'FAILURE.json',dict(returncode=p.returncode,code_commit=CODE));raise RuntimeError('qualification failure')
        while True:
            for task,(p,entry) in list(active.items()):
                if p.poll() is None:continue
                del active[task];folder=RUN/'tasks'/task;mode=entry['mode'];receipt=folder/({'audit':'R3A_DONE.json','train':'TRAIN_DONE.json','evaluate':'EVALUATION.json'}[mode])
                if p.returncode==0 and receipt.exists():queue[task]['status']='TRAINED' if mode=='train' else 'DONE'
                elif (folder/'PAUSED.json').exists():queue[task]['status']='PENDING'
                elif (folder/'TIME_LIMIT.json').exists() or time.time()>=deadline:queue[task]['status']='TIME_LIMIT'
                else:
                    reason=Path(entry['log']).read_text(errors='replace').strip().splitlines()[-1];fp=hashlib.sha256((CODE+reason).encode()).hexdigest();fingerprints[fp]=fingerprints.get(fp,0)+1;queue[task]['failures'].append(dict(fingerprint=fp,reason=reason,mode=mode,code_commit=CODE));queue[task]['status']='BLOCKED' if fingerprints[fp]>=2 else ('TRAINED' if mode=='evaluate' else 'PENDING')
                    if fingerprints[fp]>=2:
                        atomic(RUN/'ENGINEERING_PAUSE.json',dict(fingerprint=fp,reason=reason,mode=mode))
            expired=time.time()>=deadline;paused=(RUN/'ENGINEERING_PAUSE.json').exists()
            free=available()
            if not expired and not paused:
                for task,spec in TASKS.items():
                    row=queue[task]
                    if len(active)>=4:break
                    if row['status'] not in ('PENDING','TRAINED'):continue
                    binding=json.loads((RUN/'PREFIX_BINDINGS.private.json').read_text())[spec['dependencies'][0]]
                    if not binding['valid']:row['status']='INVALID_PREFIX';continue
                    if task.startswith('R3B') and queue[spec['dependencies'][1]]['status']!='DONE':continue
                    need=18000;eligible=[g for g,n in free.items() if n>=need]
                    if not eligible:continue
                    g=max(eligible,key=free.get);mode='evaluate' if row['status']=='TRAINED' else 'audit' if task.startswith('R3A') else 'train';active[task]=spawn(mode,g,EXEC_TASK=task);free[g]-=need;row['status']='EVALUATING' if mode=='evaluate' else 'RUNNING'
            atomic(RUN/'QUEUE.json',queue);atomic(RUN/'BUDGET.json',budget.state)
            from .report import aggregate
            aggregate(RUN,queue,budget.state,False)
            if time.time()>deadline+120:
                for p,row in active.values():
                    if process_identity(p.pid)==row['identity']:p.terminate()
            if not active:
                if paused:raise RuntimeError('repeated failure requires repair')
                if expired or all(v['status'] in ('DONE','INVALID_PREFIX','BLOCKED','TIME_LIMIT') for v in queue.values()):break
            time.sleep(15)
        end=time.time()+1800
        for task,spec in TASKS.items():
            folder=RUN/'tasks'/task
            if (folder/'final.pt').exists() and not (folder/'EVALUATION.json').exists() and time.time()<end:
                free=available();g=max(free,key=free.get)
                if free[g]<16000:continue
                p,row=spawn('evaluate',g,EXEC_TASK=task)
                try:p.wait(timeout=min(300,end-time.time()))
                except subprocess.TimeoutExpired:
                    if process_identity(p.pid)==row['identity']:p.terminate()
                    p.wait(timeout=30)
                if p.returncode==0:queue[task]['status']='DONE'
        aggregate(RUN,queue,budget.state,True);atomic(RUN/'QUEUE.json',queue);atomic(RUN/'BUDGET.json',budget.state);atomic(RUN/'FINAL.json',dict(status='COMPLETE' if all(v['status']=='DONE' for v in queue.values()) else 'INCOMPLETE',code_commit=CODE,finished=time.time()))
    except BaseException as e:atomic(RUN/'SUPERVISOR_FAILURE.json',dict(error=repr(e),time=time.time(),code_commit=CODE));raise
    finally:server.shutdown();server.server_close()
