"""Detached memory-admitted queue, one task per GPU, no score-based dispatch."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from . import core as c,contract as ct

def child(base,tid,phase,gpu,data,reference):
    b=Path(base);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    module='experiments.lcrseg.dpr_v0_1.'+('engine' if phase=='train' else 'evaluate')
    args=['worker','--base',str(b),'--task-id',tid,'--data',data,'--reference',reference]
    env['R_ENTRY']='import runpy,sys;sys.argv='+repr(args)+';runpy.run_module('+repr(module)+',run_name="__main__")'
    command=['bash','-s','--',sys.executable,'-c','import os;exec(os.environ["R_ENTRY"])'];started=time.time()
    with (b/'logs'/(tid+'_'+phase+'.log')).open('x') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env)
        process.stdin.write(Path('experiments/lcrseg/scripts/with_nas_storage.sh').read_bytes());process.stdin.close()
        c.write_json(b/'logs'/(tid+'_'+phase+'_started.json'),dict(pid=process.pid,gpu=gpu,started=started,neutral_command=command))
        code=process.wait()
    c.write_json(b/'logs'/(tid+'_'+phase+'_exit.json'),dict(pid=process.pid,gpu=gpu,exit_code=code,started_unix=started,finished_unix=time.time()))
    if code:raise RuntimeError(f'{tid} {phase} exited {code}')

def execute(base,data,reference):
    source=ct.verify();b=ct.nas(base);p=ct.protocol();ct.neutral_subprocess_paths()
    for name in ('SOURCE_BINDING.json','BASELINE_BINDING.json','qualification_local/qualification.json','qualification_server/qualification.json','smoke/receipt.json'):
        r=ct.read(b/name)
        if r['status']!='PASS' or r['source']!=source:raise PermissionError('exact-source admission incomplete: '+name)
    ledger=ct.read(b/'qualification_ledger.json');assert ledger['synthetic_updates']<=ct.protocol()['qualification_synthetic_cap'] and ledger['real_smoke_updates']==12
    reservation=dict(source=source,tasks=p['tasks'],formal_updates=47700,source_training_updates=0,baseline_retraining_updates=0,
                     input_hashes={n:ct.sha256(b/n) for n in ('SOURCE_BINDING.json','BASELINE_BINDING.json','private_inputs.json','qualification_ledger.json','smoke/receipt.json')},created=time.time())
    with (b/'reservation.json').open('x') as f:json.dump(reservation,f,indent=2)
    (b/'logs').mkdir();jobs=queue.Queue();stop=threading.Event();phase='train';minimum=ct.read(b/'smoke/receipt.json')['admission_mib']
    for t in p['tasks']:jobs.put(t['task_id'])
    def worker(gpu):
        try:
            while not stop.is_set() and not jobs.empty():
                free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                c.write_json(b/'logs'/('gpu'+str(gpu)+'_resource.json'),dict(gpu=gpu,free_mib=free,minimum_mib=minimum,status='ADMITTED' if free>=minimum else 'WAITING_GPU_MEMORY'))
                if free<minimum:stop.wait(55);continue
                try:tid=jobs.get_nowait()
                except queue.Empty:return
                child(b,tid,phase,gpu,data,reference)
        except BaseException:
            stop.set();raise
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(worker,g) for g in p['GPUs']]
            for f in futures:f.result()
        if stop.is_set():raise RuntimeError('new dispatch stopped after engineering failure')
        students={}
        for t in p['tasks']:
            root=b/'tasks'/t['task_id'];r=ct.read(root/'receipt.json');diag=ct.read(root/('diagnostics_pass_'+str(t['updates'])+'.json'))
            assert r['source']==source and r['task']==t and r['updates']==t['updates'] and diag['status']=='PASS'
            students[t['task_id']]=dict(student_hash=r['student_hash'],source_student_hash=r['boundary']['source_student_hash'],checkpoint_sha256=ct.sha256(root/('checkpoint_'+str(t['updates'])+'.pt')))
        with (b/'TARGET_WEIGHT_SEAL.json').open('x') as f:json.dump(dict(source=source,students=students,created=time.time()),f,indent=2)
        phase='eval'
        for t in p['tasks']:jobs.put(t['task_id'])
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(worker,g) for g in p['GPUs']]
            for f in futures:f.result()
        if stop.is_set():raise RuntimeError('final evaluation engineering failure')
        from .results import finish
        terminal=finish(b);c.write_json(b/'TERMINAL.json',terminal);c.write_json(b/'parent_receipt.json',dict(status='COMPLETE',source=source,finished=time.time(),terminal=terminal));print(json.dumps(terminal),flush=True)
    except BaseException as exc:
        c.write_json(b/'parent_failure.json',dict(status='INCOMPLETE_ENGINEERING',source=source,error=repr(exc),new_dispatch_stopped=True,automatic_retry=False));raise
