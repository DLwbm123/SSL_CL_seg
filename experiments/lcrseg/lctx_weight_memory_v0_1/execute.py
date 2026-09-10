"""Detached, fixed matrix. No scientific-score dispatch gates."""
import argparse,json,os,subprocess,sys,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .contract import verify,read,protocol,nas
from .core import write_json
from experiments.lcrseg.di_dmpa_gate1.binding import sha256
from experiments.lcrseg.single_teacher_scd_v0_1.execute import child
MODULE='experiments.lcrseg.lctx_weight_memory_v0_1.'

def execute(base,data,reference):
    source=verify();b=nas(base);p=protocol()
    for name in ['SOURCE_BINDING_AND_LAYER_MANIFEST.json','qualification_local/qualification.json','qualification_server/qualification.json','smoke/receipt.json']:
        r=read(b/name)
        if r['status']!='PASS' or r['source']!=source:raise RuntimeError('exact-source qualification required')
    with (b/'reservation.json').open('x') as f:json.dump(dict(source=source,tasks=p['tasks'],formal_updates=95400,source_recovery_updates=0,input_hashes={n:sha256(b/n) for n in ['SOURCE_BINDING_AND_LAYER_MANIFEST.json','private_inputs.json']},created=time.time()),f,indent=2)
    minimum=read(b/'smoke/receipt.json')['admission_mib'];logs=b/'logs';logs.mkdir();stop=threading.Event()
    def lane(gpu,tasks,static=False):
        done=[]
        for task in tasks:
            if stop.is_set():break
            tid=task['task_id']
            try:
                while True:
                    if stop.is_set():return done
                    free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    write_json(logs/(tid+'_resource.json'),dict(gpu=gpu,free_mib=free,minimum_mib=minimum,status='ADMITTED' if free>=minimum else 'WAITING_GPU_MEMORY'))
                    if free>=minimum:break
                    time.sleep(55)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                args=['--base',str(b),'--task-id',tid,'--data',data,'--reference',reference]
                if not static:child([sys.executable,'-m',MODULE+'engine',*args],logs/(tid+'_train.log'),logs/(tid+'_train_exit.json'),env)
                child([sys.executable,'-m',MODULE+'evaluate',*args],logs/(tid+'_eval.log'),logs/(tid+'_eval_exit.json'),env)
                done.append(dict(task_id=tid,status='COMPLETE'))
            except BaseException as e:
                stop.set();write_json(logs/(tid+'_failure.json'),dict(status='INCOMPLETE_ENGINEERING',error=repr(e)));raise
        return done
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(lane,g,p['tasks'][i::4]) for i,g in enumerate((4,5,6,7))]
        done=[r for f in futures for r in f.result()]
    if len(done)!=36:raise RuntimeError('incomplete matrix')
    seal=dict(source=source,created=time.time(),students={t['task_id']:read(b/'tasks'/t['task_id']/'receipt.json')['student_hash'] for t in p['tasks']})
    with (b/'TARGET_WEIGHT_SEAL.json').open('x') as f:json.dump(seal,f,indent=2)
    static=[dict(task_id=s['task_id'].replace('_SRC_CE','_STATIC_SOURCE')) for s in p['sources']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(lane,g,static[i::4],True) for i,g in enumerate((4,5,6,7))]
        done += [r for f in futures for r in f.result()]
    write_json(b/'parent_receipt.json',dict(source=source,tasks=done,finished=time.time()))
    from .results import finish
    terminal=finish(b);write_json(b/'TERMINAL.json',terminal);print(json.dumps(terminal),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['base','data','reference']:p.add_argument('--'+k,required=True)
    a=vars(p.parse_args())
    try:execute(**a)
    except BaseException as e:write_json(Path(a['base'])/'parent_failure.json',dict(status='INCOMPLETE_ENGINEERING',error=repr(e)));raise
