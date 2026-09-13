"""Finite shared-GPU queue: 30 targets, seal, 30 evaluations, reports, stop."""
import os,sys,json,time,queue,threading,subprocess,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from . import core as c,contract as ct

def child(b,t,phase,gpu,data,reference):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    args=['worker','--base',str(b),'--task-id',t['task_id'],'--data',data,'--reference',reference,'--phase',phase]
    env['R_ENTRY']='import sys,runpy;sys.argv='+repr(args)+';runpy.run_module("experiments.lcrseg.lctx_paper_closeout_v1.engine",run_name="__main__")'
    command=['bash','-s','--',sys.executable,'-c','import os;exec(os.environ["R_ENTRY"])'];started=time.time();stem=t['task_id']+'_'+phase
    with (b/'logs'/(stem+'.log')).open('x') as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env)
        proc.stdin.write(Path('experiments/lcrseg/scripts/with_nas_storage.sh').read_bytes());proc.stdin.close()
        c.write_json(b/'logs'/(stem+'_started.json'),dict(pid=proc.pid,gpu=gpu,started=started,command=command));code=proc.wait()
    c.write_json(b/'logs'/(stem+'_exit.json'),dict(pid=proc.pid,gpu=gpu,exit_code=code,started=started,finished=time.time()))
    if code:raise RuntimeError(stem+' failed; attempt retained')

def execute(base):
    b=ct.nas(base);source=ct.verify();p=ct.protocol();ct.neutral_subprocess_paths();inputs=ct.read(b/'private_inputs.json')
    qual=ct.read(b/'qualification/qualification.json');smoke=ct.read(b/'smoke/receipt.json')
    if any(r['status']!='PASS' or r['source']!=source for r in (qual,smoke)):raise PermissionError('qualification/source mismatch')
    ledger=ct.read(b/'qualification_ledger.json')
    if ledger['synthetic_updates']>128 or ledger['real_smoke_updates']>12:raise PermissionError('qualification budget exceeded')
    with (b/'reservation.json').open('x') as f:json.dump(dict(source=source,tasks=p['tasks'],formal_updates=79500,created=time.time()),f,indent=2)
    (b/'logs').mkdir();minimum=smoke['admission_mib']
    def status(state,**kw):c.write_json(b/'CONTROLLER_STATUS.json',dict(status=state,source=source,updated=time.time(),**kw))
    def phase(kind):
        jobs=queue.Queue();stop=threading.Event()
        for t in p['tasks']:jobs.put(t)
        def lane(gpu):
            try:
                while not stop.is_set() and not jobs.empty():
                    free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                    c.write_json(b/'logs'/f'gpu{gpu}_resource.json',dict(gpu=gpu,free_mib=free,minimum_mib=minimum,checked=time.time()))
                    if free<minimum:stop.wait(55);continue
                    try:t=jobs.get_nowait()
                    except queue.Empty:return
                    child(b,t,kind,gpu,inputs['data'],os.environ['R_REFERENCE'])
            except BaseException:stop.set();raise
        with ThreadPoolExecutor(max_workers=len(p['GPUs'])) as pool:
            futures=[pool.submit(lane,g) for g in p['GPUs']]
            for f in futures:f.result()
    try:
        status('RUNNING_TRAINING',GPUs=p['GPUs'],formal_tasks=30,formal_updates=79500)
        phase('train');students={}
        for t in p['tasks']:
            root=b/'tasks'/t['task_id'];r=ct.read(root/'receipt.json')
            if r['source']!=source or r['updates']!=t['updates'] or ct.read(root/f"diagnostics_pass_{t['updates']}.json")['status']!='PASS':raise RuntimeError('final training receipt mismatch')
            students[t['task_id']]=r['student_hash']
        c.write_json(b/'TARGET_WEIGHT_SEAL.json',dict(source=source,students=students,created=time.time()))
        status('RUNNING_EVALUATION');phase('eval')
        from .results import finish
        terminal=finish(b);c.write_json(b/'TERMINAL.json',terminal);status('COMPLETE',terminal=terminal)
    except BaseException as exc:
        status('INCOMPLETE_ENGINEERING',error=repr(exc),automatic_retry=False);raise
