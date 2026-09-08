"""Durable finite background DAG. P2 is unconditional; only the sealed P1 gate admits P3."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from experiments.lcrseg.single_teacher_scd_v0_1.execute import child
from experiments.lcrseg.single_teacher_scd_v0_1.engine import write_json
from .freeze import verify
from .results import CONFIG,read,seal_p_gate,finish

def run_replication(seed,common,train,decision):
    parent=common(seed)
    # Training results/val values never control whether another mandatory arm or seed runs.
    results=[train(seed,arm,parent) for arm in ('S','L05')]
    admitted=decision()
    if admitted:results.append(train(seed,'L05_SSL',parent))
    else:results.append(dict(arm='L05_SSL',optimization_seed=seed,status='NOT_ADMITTED_BY_FROZEN_GATE'))
    return results

def available(gpu,receipt):
    if gpu not in (4,5,6,7):raise PermissionError('unauthorized GPU')
    start=time.time()
    while True:
        free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
        state=dict(status='READY' if free[gpu]>=CONFIG['minimum_free_mib'] else 'WAITING_GPU_MEMORY',gpu=gpu,free_mib=free[gpu],required_mib=CONFIG['minimum_free_mib'],wait_seconds=time.time()-start)
        write_json(receipt,state)
        if state['status']=='READY':return state
        time.sleep(55)  # Queue the authorized finite task; never terminate another process.

def execute(base,data,reference,qualification):
    base=Path(base);source=verify();lineage=read(base/'INPUT_LINEAGE.json');q=read(qualification)
    assert q['status']=='PASS' and q['source']==source==lineage['source']
    if not str(base.resolve()).startswith('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/'):raise ValueError('NAS only')
    with (base/'reservation.json').open('x') as f:json.dump(dict(source=source,created_unix=time.time(),protocol=CONFIG,qualification=str(qualification)),f,indent=2)
    for seed in (0,1,2):(base/f'optimization_seed{seed}').mkdir()
    module='experiments.lcrseg.l05_ssl_replication.'
    def task(seed,arm,parent):
        root=base/f'optimization_seed{seed}'/arm;root.mkdir()
        gpu=CONFIG['lanes']['P1' if seed==0 else f'seed{seed}']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
        try:
            if arm!='common' and parent is None:raise RuntimeError('same-seed common failed; no substitute initialization')
            for t in ((0,) if arm=='common' else (1,2)):
                available(gpu,root/f'stage{t}_resource.json')
                sr=root/f'stage{t}'
                cmd=[sys.executable,'-m',module+'engine','--data',data,'--reference',reference,'--output',str(sr),'--arm','S' if arm=='common' else arm,'--optimization-seed',str(seed),'--stage',str(t)]
                if parent is not None:cmd+=['--parent',str(parent)]
                child(cmd,root/f'stage{t}_train.log',root/f'stage{t}_train_exit.json',env)
                child([sys.executable,'-m','experiments.lcrseg.single_teacher_scd_v0_1.evaluate','--data',data,'--reference',reference,'--checkpoint',str(sr/'student_latest.pt'),'--output',str(sr/'val.json')],root/f'stage{t}_eval.log',root/f'stage{t}_eval_exit.json',env)
                parent=sr/'student_latest.pt'
            if arm!='common':child([sys.executable,'-m','experiments.lcrseg.single_teacher_scd_v0_1.deploy','--checkpoint',str(parent),'--reference',reference],root/'deploy.log',root/'deploy_exit.json',env)
            result=dict(status='COMPLETE',source=source,arm=arm,optimization_seed=seed,successful_updates=8000 if arm=='common' else 5300)
        except BaseException as error:result=dict(status='ENGINEERING_FAILURE',source=source,arm=arm,optimization_seed=seed,error=repr(error))
        write_json(root/'parent_receipt.json',result);return result
    def p1():
        task(0,'L05_SSL',Path(lineage['common_seed0_checkpoint']))
        try:return seal_p_gate(base)
        except BaseException as error:
            result=dict(status='NOT_EVALUATED_ENGINEERING_ERROR',passed=False,source=source,error=repr(error))
            if not (base/'P_GATE.json').exists():write_json(base/'P_GATE.json',result)
            return result
    def common(seed):
        result=task(seed,'common',None)
        return base/f'optimization_seed{seed}/common/stage0/student_latest.pt' if result['status']=='COMPLETE' else None
    with ThreadPoolExecutor(max_workers=3) as pool:
        gate_future=pool.submit(p1)
        futures=[pool.submit(run_replication,seed,common,task,lambda:gate_future.result()['passed']) for seed in (1,2)]
        results=[f.result() for f in futures];gate_future.result()
    write_json(base/'parent_receipt.json',dict(status='FIXED_TASKS_TERMINAL',source=source,replication_lanes=results,finished_unix=time.time()))
    terminal=finish(base);write_json(base/'TERMINAL.json',terminal)
    print(json.dumps(terminal),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('base','data','reference','qualification'):p.add_argument('--'+name,required=True)
    a=vars(p.parse_args())
    try:execute(**a)
    except BaseException as error:
        write_json(Path(a['base'])/'parent_failure.json',dict(status='INCOMPLETE_ENGINEERING',error=repr(error)))
        raise
