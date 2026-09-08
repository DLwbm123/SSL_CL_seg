"""Detached finite matrix: every registered seed/arm runs, independent of val results."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .core import *
from .freeze import verify
from .results import read,finish
from experiments.lcrseg.single_teacher_scd_v0_1.execute import child
GPUS=(4,5,6);MODULE='experiments.lcrseg.ssl_target_path_v0_1.'
def execute(base,data,reference):
    b=Path(base);source=verify()
    assert str(b.resolve()).startswith('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/')
    for path in ('INPUT_LINEAGE.json','qualification_local/qualification.json','qualification_server/qualification.json','real_smoke/receipt.json'):
        r=read(b/path);assert r['status']=='PASS' and r['source']==source
    ledger=read(b/'qualification_ledger.json');assert ledger['synthetic_successful_updates']<=1000 and ledger['real_smoke_updates']<=10 and ledger['source']==source
    tasks=[(s,d,a) for s in SEEDS for a in ARMS for d in DOMAINS]
    assert len(tasks)==30
    with (b/'reservation.json').open('x') as f:json.dump(dict(source=source,tasks=tasks,GPUs=GPUS,total_updates=79500,previous_formal=168608,main_candidate='T_SCE',all_seeds_unconditionally_admitted=True),f,indent=2)
    def task(s,d,a,gpu):
        parent=b/f'seed{s}'/d;parent.mkdir(parents=True,exist_ok=True);root=parent/a
        try:
            while True:
                free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=4096:break
                write_json(parent/(a+'_resource.json'),dict(status='WAITING_GPU_MEMORY',gpu=gpu,free_mib=free));time.sleep(55)
            write_json(parent/(a+'_resource.json'),dict(status='ADMITTED',gpu=gpu,free_mib=free,minimum_mib=4096))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
            child([sys.executable,'-m',MODULE+'engine','--data',data,'--reference',reference,'--output',str(root),'--domain',d,'--arm',a,'--seed',str(s)],parent/(a+'_train.log'),parent/(a+'_train_exit.json'),env)
            child([sys.executable,'-m',MODULE+'deploy','--root',str(root),'--data',data,'--reference',reference],parent/(a+'_deploy.log'),parent/(a+'_deploy_exit.json'),env)
            return dict(seed=s,domain=d,arm=a,status='COMPLETE')
        except BaseException as error:
            r=dict(seed=s,domain=d,arm=a,status='INCOMPLETE_ENGINEERING',error=repr(error));write_json(parent/(a+'_failure.json'),r);return r
    def lane(gpu,tt):return [task(*t,gpu) for t in tt]
    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        ff=[pool.submit(lane,gpu,tasks[i::len(GPUS)]) for i,gpu in enumerate(GPUS)];results=[r for f in ff for r in f.result()]
    write_json(b/'parent_receipt.json',dict(source=source,tasks=results,finished_unix=time.time()))
    print(json.dumps(finish(b,results)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','data','reference'):p.add_argument('--'+k,required=True)
    a=vars(p.parse_args())
    try:execute(**a)
    except BaseException as error:
        write_json(Path(a['base'])/'parent_failure.json',dict(status='INCOMPLETE_ENGINEERING',error=repr(error)));raise
