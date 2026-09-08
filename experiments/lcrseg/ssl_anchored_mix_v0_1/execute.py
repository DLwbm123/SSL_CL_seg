"""Detached P1 matrix and conditional P2; fixed gates, one task per GPU 3/4/5."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .core import *
from .freeze import verify
from .results import read,summarize,finish
from experiments.lcrseg.single_teacher_scd_v0_1.execute import child
GPUS=(3,4,5);MODULE='experiments.lcrseg.ssl_anchored_mix_v0_1.'
def phase(base,name,tasks,data,reference,source):
    b=Path(base)/name;b.mkdir()
    with (b/'reservation.json').open('x') as f:json.dump(dict(source=source,tasks=tasks,GPUs=GPUS,expected_updates=sum(3200 if d==DOMAINS[0] else 2100 for s,d,a in tasks)),f,indent=2)
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
    with ThreadPoolExecutor(max_workers=3) as pool:
        ff=[pool.submit(lane,g,tasks[i::3]) for i,g in enumerate(GPUS)];rr=[r for f in ff for r in f.result()]
    write_json(b/'parent_receipt.json',dict(source=source,tasks=rr,finished_unix=time.time()))

def execute(base,data,reference,old):
    b=Path(base);source=verify()
    assert str(b.resolve()).startswith('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/')
    protocol=read(Path('experiments/lcrseg/docs/ssl_anchored_mix_v0_1/PROTOCOL.json'))
    assert old==protocol['old_run_root']
    for name in ['INPUT_LINEAGE.json','qualification_local/qualification.json','qualification_server/qualification.json','real_smoke/receipt.json']:
        r=read(b/name);assert r['status']=='PASS' and r['source']==source
    ledger=read(b/'qualification_ledger.json');assert ledger['synthetic_successful_updates']<=1000 and ledger['real_smoke_updates']<=8
    if (b/'P1').exists():raise FileExistsError('existing matrix is protected')
    tasks=[(s,d,a) for s in (31,32,33) for a in ARMS for d in DOMAINS]
    phase(b,'P1',tasks,data,reference,source);p1=summarize(b/'P1','P1',tasks,old)
    selected=p1.get('selected');p2=dict(status='NOT_ADMITTED',selected=None)
    if selected:
        arms=('SUP_CE','SUP_CED','MT_CED') if selected=='MT_CED' else ('SUP_CE','SUP_CED','MIX_CTX','MIX_CED')
        tasks=[(s,d,a) for s in (41,42) for a in arms for d in DOMAINS]
        phase(b,'P2',tasks,data,reference,source);p2=summarize(b/'P2','P2',tasks,old,selected)
    terminal=finish(b,p1,p2);write_json(b/'parent_receipt.json',dict(source=source,finished_unix=time.time(),terminal=terminal));print(json.dumps(terminal),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','data','reference','old'):p.add_argument('--'+k,required=True)
    a=vars(p.parse_args())
    try:execute(**a)
    except BaseException as error:
        write_json(Path(a['base'])/'parent_failure.json',dict(status='INCOMPLETE_ENGINEERING',error=repr(error)));raise
