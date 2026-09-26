"""One coordinator owns scheduling and budget. Workers always use neutral argv."""
import os,sys,json,time,subprocess,hashlib,signal
from pathlib import Path
from .runner import RUN,CODE,provenance
from .runtime import PLAN,TASKS,broker,process_identity,disk_bytes
from r1_12h.core import atomic,append,events


def spawn(mode,gpu,**extra):
    env=dict(os.environ,EXEC_MODE=mode,CUDA_VISIBLE_DEVICES=str(gpu),**{k:str(v) for k,v in extra.items()})
    source=json.loads((RUN/'ACTIVE_CODE.json').read_text())['source']
    log=RUN/'logs'/f'{mode}_{extra.get("EXEC_TASK",extra.get("EXEC_BACKBONE","main"))}_{time.time_ns()}.log'
    with log.open('w') as stream:
        p=subprocess.Popen(['python3.12','-'],executable=sys.executable,cwd=source,env=env,stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT,text=True)
        p.stdin.write('from r1_6_readout_v1.runner import main;main()\n');p.stdin.close()
    entry=dict(pid=p.pid,identity=process_identity(p.pid),mode=mode,gpu=gpu,log=str(log),**extra)
    append(RUN/'PROCESS_LEDGER.jsonl',entry)
    return p,entry


def free_gpus():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    return {int(a):int(b) for a,b in (x.split(',') for x in lines)}


def failure(entry):
    text=Path(entry['log']).read_text(errors='replace')
    tail=text.strip().splitlines()[-1] if text.strip() else 'EMPTY_LOG'
    return hashlib.sha256((CODE+tail).encode()).hexdigest(),tail


def supervise():
    provenance();server,budget=broker(RUN)
    # Refuse restart around live workers: reclaim only after explicit ownership audit.
    for row in events(RUN/'PROCESS_LEDGER.jsonl'):
        if process_identity(row['pid'])==row['identity']:raise RuntimeError('existing worker requires audited reattachment')
    session=json.loads((RUN/'SESSION.json').read_text());deadline=session['deadline']
    atomic(RUN/'SUPERVISOR.json',dict(pid=os.getpid(),identity=process_identity(os.getpid()),code_commit=CODE))
    active={};fingerprints={};blocked_modes=set();stage='qualify';queue={t:dict(status='PENDING',failures=[]) for t in TASKS}
    if (RUN/'QUEUE.json').exists():queue=json.loads((RUN/'QUEUE.json').read_text())
    order=sorted(TASKS,key=lambda t:(TASKS[t]['seed'],{'QUERY_PREFIX':-1,'B0':0,'B2':1,'B1':2,'B3':3}[TASKS[t]['arm']],PLAN['backbones'].index(TASKS[t]['backbone']),PLAN['domains'].index(TASKS[t]['domain'])))
    try:
        for stage in ('qualify','smoke','forensic'):
            for backbone in PLAN['backbones']:
                folder=RUN/({'qualify':'qualification'}.get(stage,stage))/backbone
                if (folder/'PASSED.json').exists():continue
                # A failed qualification is repaired explicitly, never silently repeated.
                if (folder/'ENGINEERING_FAILURE.json').exists():raise RuntimeError('qualification repair required')
                while time.time()<deadline:
                    free=free_gpus();eligible=[g for g,n in free.items() if n>=16000]
                    if eligible:break
                    time.sleep(20)
                if time.time()>=deadline:raise RuntimeError('WALL_CLOCK_LIMIT')
                gpu=max(eligible,key=free.get);p,entry=spawn(stage,gpu,EXEC_BACKBONE=backbone)
                while p.poll() is None and time.time()<deadline:time.sleep(5)
                if p.poll() is None:
                    if process_identity(p.pid)==entry['identity']:p.send_signal(signal.SIGTERM)
                    p.wait(timeout=30)
                if p.returncode or not (folder/'PASSED.json').exists():
                    fp,reason=failure(entry);atomic(folder/'ENGINEERING_FAILURE.json',dict(fingerprint=fp,reason=reason,code_commit=CODE));raise RuntimeError(stage+' engineering failure')
        atomic(RUN/'NATIVE_QUALIFICATION.json',dict(status='PASSED',synthetic=budget.state['synthetic'],smoke=budget.state['smoke'],forensic_batches=sum(json.loads((RUN/'forensic'/b/'PASSED.json').read_text())['batches'] for b in PLAN['backbones'])))
        while True:
            for task,(p,entry) in list(active.items()):
                if p.poll() is None:continue
                del active[task];folder=RUN/'tasks'/task;mode=entry['mode']
                success=(folder/('EVALUATION.json' if mode=='evaluate' else 'TRAIN_DONE.json')).exists()
                if p.returncode==0 and success:queue[task]['status']='DONE' if mode=='evaluate' or not TASKS[task]['is_final_endpoint'] else 'TRAINED'
                elif (folder/'ENGINEERING_STOP.json').exists():queue[task]['status']='ENGINEERING_BLOCKED'
                elif (folder/'TIME_LIMIT.json').exists():queue[task]['status']='TIME_LIMIT'
                else:
                    fp,reason=failure(entry);queue[task]['failures'].append(dict(fingerprint=fp,reason=reason,mode=mode,code_commit=CODE))
                    fingerprints[fp]=fingerprints.get(fp,0)+1;count=fingerprints[fp]
                    if count>=2:
                        blocked_modes.add(mode);atomic(RUN/'BLOCKED_MODES.json',sorted(blocked_modes))
                        for other in queue:
                            if queue[other]['status']==('PENDING' if mode=='train' else 'TRAINED'):queue[other]['status']='ENGINEERING_BLOCKED'
                    queue[task]['status']='ENGINEERING_BLOCKED' if count>=2 else ('TRAINED' if mode=='evaluate' else 'PENDING')
            expired=time.time()>=deadline
            if time.time()>deadline+120:
                for task,(p,entry) in active.items():
                    if process_identity(p.pid)==entry['identity']:p.terminate()
            if expired and not active:break
            free=free_gpus()
            for task in order:
                if len(active)>=4:break
                status=queue[task]['status'];spec=TASKS[task]
                if status not in ('PENDING','TRAINED'):continue
                if ('train' if status=='PENDING' else 'evaluate') in blocked_modes:queue[task]['status']='ENGINEERING_BLOCKED';continue
                if expired:continue
                if status=='PENDING' and spec['depends_on']:
                    dep=spec['depends_on'][0]
                    if spec['seed']==261:
                        if not json.loads((RUN/'PREFIX_AUDIT.private.json').read_text())[dep]['valid']:queue[task]['status']='INVALID_PREFIX';continue
                    elif queue[dep]['status']!='DONE':continue
                measured=json.loads((RUN/'qualification'/spec['backbone']/'PASSED.json').read_text())['results']
                need=max(12000,max(x['peak_reserved'] for x in measured)/1024**2*1.3+2000)
                eligible=[g for g,n in free.items() if n>=need]
                if not eligible:continue
                gpu=max(eligible,key=free.get);mode='evaluate' if status=='TRAINED' else 'train';p,entry=spawn(mode,gpu,EXEC_TASK=task);active[task]=(p,entry);free[gpu]-=need;queue[task]['status']='EVALUATING' if mode=='evaluate' else 'RUNNING'
            atomic(RUN/'QUEUE.json',queue);atomic(RUN/'BUDGET.json',budget.state)
            from .analysis import aggregate
            aggregate(RUN,queue,budget.state,False)
            if not active and all(v['status'] in ('DONE','ENGINEERING_BLOCKED','INVALID_PREFIX','TIME_LIMIT') or (TASKS[t]['depends_on'] and TASKS[t]['depends_on'][0] in queue and queue[TASKS[t]['depends_on'][0]]['status']=='ENGINEERING_BLOCKED') for t,v in queue.items()):break
            time.sleep(15)
        # End-of-window evaluation of already complete endpoints is bounded to 30 minutes.
        maintenance_end=time.time()+1800
        for task in order:
            folder=RUN/'tasks'/task
            if TASKS[task]['is_final_endpoint'] and (folder/'final.pt').exists() and not (folder/'EVALUATION.json').exists() and time.time()<maintenance_end:
                free=free_gpus();gpu=max(free,key=free.get)
                if free[gpu]<10000:continue
                p,entry=spawn('evaluate',gpu,EXEC_TASK=task)
                try:p.wait(timeout=min(300,maintenance_end-time.time()))
                except subprocess.TimeoutExpired:
                    if process_identity(p.pid)==entry['identity']:p.terminate()
                    p.wait(timeout=30)
                if p.returncode==0:queue[task]['status']='DONE'
        from .analysis import aggregate
        aggregate(RUN,queue,budget.state,True)
        atomic(RUN/'QUEUE.json',queue);atomic(RUN/'BUDGET.json',budget.state)
        atomic(RUN/'FINAL.json',dict(status='COMPLETE' if all(v['status']=='DONE' for v in queue.values()) else 'INCOMPLETE_ENGINEERING_OR_BUDGET',finished=time.time(),code_commit=CODE))
    except BaseException as e:
        atomic(RUN/'SUPERVISOR_FAILURE.json',dict(error=repr(e),time=time.time(),code_commit=CODE));raise
    finally:server.shutdown();server.server_close()
