"""Durable recovery queue. Only the exact 420 engineering seal releases the 17 tasks."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from experiments.lcrseg.lctx_weight_memory_v0_1 import core as c
from experiments.lcrseg.di_dmpa_gate1.binding import sha256
from . import contract as ct

NEUTRAL_CODE='import os;exec(os.environ["R_ENTRY"])'
WRAPPER=Path('experiments/lcrseg/scripts/with_nas_storage.sh')

def child(base,tid,phase,gpu,data,reference):
    b=Path(base);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    args=['worker','--base',str(b),'--task-id',tid,'--data',data,'--reference',reference,'--recovery']
    module='experiments.lcrseg.lctx_weight_memory_v0_1.'+('engine' if phase=='train' else 'evaluate')
    env['R_ENTRY']='import runpy,sys;sys.argv='+repr(args)+';runpy.run_module('+repr(module)+',run_name="__main__")'
    command=['bash','-s','--',sys.executable,'-c',NEUTRAL_CODE]
    started=time.time();log=b/'logs'/(tid+'_'+phase+'.log')
    with log.open('x') as f:
        p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT,env=env)
        p.stdin.write(WRAPPER.read_bytes());p.stdin.close()
        c.write_json(b/'logs'/(tid+'_'+phase+'_started.json'),dict(pid=p.pid,gpu=gpu,started=started,command=command))
        code=p.wait()
    c.write_json(b/'logs'/(tid+'_'+phase+'_exit.json'),dict(pid=p.pid,exit_code=code,gpu=gpu,started_unix=started,finished_unix=time.time(),command=command))
    if code:raise RuntimeError('child exited '+str(code)+': '+tid+' '+phase)

def execute(base,data,reference):
    source=ct.verify();b=ct.nas(base);m=ct.read(b/'RECOVERY_MANIFEST.json');assert m['source']==source and m['status']=='PASS'
    for n in ('qualification_local','qualification_server'):
        q=ct.read(b/n/'qualification.json');assert q['status']=='PASS' and q['source']==source and q['real_updates']==0
    qual=ct.read(b/'qualification_ledger.json')
    assert qual['new_synthetic_updates']<=120 and qual['new_real_updates']==0
    reservation=dict(source=source,new_task_ids=m['new_task_ids'],new_updates=49900,scientific_updates=95400,physical_updates_if_complete=95820,
                     input_hashes={n:sha256(b/n) for n in ['RECOVERY_MANIFEST.json','SOURCE_BINDING_AND_LAYER_MANIFEST.json','private_inputs.json','qualification_ledger.json']})
    with (b/'reservation.json').open('x') as f:json.dump(reservation,f,indent=2)
    (b/'logs').mkdir();stop=threading.Event();ready=threading.Event();jobs=queue.Queue();failed='O1_s62_LR_SRC_AB'
    for t in ct.protocol()['tasks']:
        if t['task_id'] in m['new_task_ids'] and t['task_id']!=failed:jobs.put(t['task_id'])
    def run_task(tid,gpu,static=False):
        while not stop.is_set():
            free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            c.write_json(b/'logs'/(tid+'_resource.json'),dict(gpu=gpu,free_mib=free,minimum_mib=2048,status='ADMITTED' if free>=2048 else 'WAITING_GPU_MEMORY'))
            if free>=2048:break
            stop.wait(55)
        if stop.is_set():return False
        if not static:child(b,tid,'train',gpu,data,reference)
        child(b,tid,'eval',gpu,data,reference);return True
    def worker(gpu):
        try:
            if gpu==4:run_task(failed,gpu)
            else:
                while not stop.is_set() and not ready.wait(2):
                    if (b/'RECOVERY_420_GATE.json').exists():
                        assert ct.read(b/'RECOVERY_420_GATE.json')['status']=='PASS';ready.set()
            while not stop.is_set():
                try:tid=jobs.get_nowait()
                except queue.Empty:return
                run_task(tid,gpu)
        except BaseException:
            stop.set();raise
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(worker,g) for g in (4,5,6,7)]
            for f in futures:f.result()
        if stop.is_set():raise RuntimeError('recovery queue failed')
        students={}
        for t in ct.protocol()['tasks']:
            tid=t['task_id'];root=ct.task_root(b,tid);r=ct.read(root/'receipt.json');ev=ct.read(root/'evaluation/receipt.json')
            expected=ct.OLD_SOURCE if tid in m['reused_task_ids'] else source
            assert r['source']==ev['source']==expected and r['student_hash']==ev['student_hash'] and r['updates']==t['updates']
            if tid in m['reused_task_ids']:
                audit=ct.read(b/'reaudit'/(tid+'.json'));assert audit['status']=='PASS' and audit['student_hash']==r['student_hash']
            else:assert ct.read(root/('diagnostics_pass_'+str(t['updates'])+'.json'))['status']=='PASS'
            students[tid]=r['student_hash']
        with (b/'TARGET_WEIGHT_SEAL.json').open('x') as f:
            json.dump(dict(source=source,created=time.time(),students=students,training_sources={tid:ct.OLD_SOURCE if tid in m['reused_task_ids'] else source for tid in students},reaudit_source=source),f,indent=2)
        static=queue.Queue()
        for src in ct.protocol()['sources']:static.put(src['task_id'].replace('_SRC_CE','_STATIC_SOURCE'))
        def static_worker(gpu):
            while not stop.is_set():
                try:tid=static.get_nowait()
                except queue.Empty:return
                try:run_task(tid,gpu,True)
                except BaseException:stop.set();raise
        with ThreadPoolExecutor(max_workers=4) as pool:
            for f in [pool.submit(static_worker,g) for g in (4,5,6,7)]:f.result()
        from experiments.lcrseg.lctx_weight_memory_v0_1.results import finish
        terminal=finish(b,recovery=True)
        c.write_json(b/'TERMINAL.json',terminal);c.write_json(b/'parent_receipt.json',dict(status='COMPLETE',source=source,finished=time.time(),terminal=terminal))
        print(json.dumps(terminal),flush=True)
    except BaseException as exc:
        c.write_json(b/'parent_failure.json',dict(status='INCOMPLETE_ENGINEERING',source=source,error=repr(exc),new_dispatch_stopped=True,automatic_retry=False))
        raise
