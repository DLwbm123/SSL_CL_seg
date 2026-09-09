"""Detached fixed DAG: all six new sources precede all thirty targets."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .contract import verify,protocol,read
from .core import write_json
from experiments.lcrseg.single_teacher_scd_v0_1.execute import child
GPUS=(4,5,6,7);MODULE='experiments.lcrseg.ams_seq_transfer_v0_1.'

def phase(base,tasks,data,reference,minimum):
    def lane(gpu,work):
        results=[]
        for task in work:
            tid=task['task_id'];logs=base/'logs';logs.mkdir(exist_ok=True)
            try:
                while True:
                    free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    write_json(logs/(tid+'_resource.json'),dict(gpu=gpu,free_mib=free,minimum_mib=minimum,status='ADMITTED' if free>=minimum else 'WAITING_GPU_MEMORY'))
                    if free>=minimum:break
                    time.sleep(55)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                args=['--base',str(base),'--task-id',tid,'--data',data,'--reference',reference]
                child([sys.executable,'-m',MODULE+'engine',*args],logs/(tid+'_train.log'),logs/(tid+'_train_exit.json'),env)
                child([sys.executable,'-m',MODULE+'evaluate',*args],logs/(tid+'_eval.log'),logs/(tid+'_eval_exit.json'),env)
                results.append(dict(task_id=tid,status='COMPLETE'))
            except BaseException as e:
                results.append(dict(task_id=tid,status='INCOMPLETE_ENGINEERING',error=repr(e)))
                write_json(logs/(tid+'_failure.json'),results[-1])
        return results
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(lane,g,tasks[i::4]) for i,g in enumerate(GPUS)]
        return [r for f in futures for r in f.result()]

def execute(base,data,reference):
    source=verify();b=Path(base);p=protocol()
    if not str(b.resolve()).startswith('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/'):raise PermissionError('NAS required')
    for name in ('SOURCE_AND_INPUT_LINEAGE.json','qualification_local/qualification.json','qualification_server/qualification.json','smoke/receipt.json'):
        r=read(b/name)
        if r['status']!='PASS' or r['source']!=source:raise RuntimeError('exact-source qualification required')
    if (b/'reservation.json').exists():raise FileExistsError('do not duplicate matrix')
    with (b/'reservation.json').open('x') as f:json.dump(dict(source=source,tasks=p['tasks'],formal_updates=95400,GPUs=GPUS,created=time.time()),f,indent=2)
    minimum=read(b/'smoke/receipt.json')['admission_mib']
    sources=[t for t in p['tasks'] if t['arm']=='SRC_CE'];targets=[t for t in p['tasks'] if t['arm']!='SRC_CE']
    a=phase(b,sources,data,reference,minimum);write_json(b/'P1_receipt.json',dict(source=source,tasks=a))
    if any(r['status']!='COMPLETE' for r in a):raise RuntimeError('source engineering failure; no source-score gate used')
    a+=phase(b,targets,data,reference,minimum);write_json(b/'parent_receipt.json',dict(source=source,tasks=a,finished=time.time()))
    if any(r['status']!='COMPLETE' for r in a):raise RuntimeError('target engineering incomplete')
    from .results import finish
    terminal=finish(b);write_json(b/'TERMINAL.json',terminal);print(json.dumps(terminal),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','data','reference'):p.add_argument('--'+k,required=True)
    a=vars(p.parse_args())
    try:execute(**a)
    except BaseException as e:
        write_json(Path(a['base'])/'parent_failure.json',dict(status='INCOMPLETE_ENGINEERING',error=repr(e)));raise
