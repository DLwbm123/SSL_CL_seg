"""Durable finite two-GPU queue; all P1 tasks finish before fixed P2 admission."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from experiments.lcrseg.single_teacher_scd_v0_1.execute import child
from .core import *
from .freeze import verify
from .results import read,collect,screen,confirmation,phase_complete,finish

def execute(base,data,reference,qualification):
    base=Path(base);source=verify();q=read(qualification);smoke=read(base/'real_smoke/receipt.json')
    assert q['status']=='PASS' and smoke['status']=='PASS' and q['source']==smoke['source']==read(base/'INPUT_LINEAGE.json')['source']==source
    local_q=read(base/'qualification_local/qualification.json')
    assert local_q['status']=='PASS' and local_q['source']==source
    assert str(base.resolve()).startswith('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/')
    with (base/'reservation.json').open('x') as f:json.dump(dict(source=source,started_unix=time.time(),P1_budget=31800,P2_budget=42400,seeds=[11,12,13],GPUs=[4,5]),f)
    module='experiments.lcrseg.ssl_foundation_v0_1.'
    def task(seed,domain,arm,gpu):
        parent=base/f'seed{seed}'/domain;parent.mkdir(parents=True,exist_ok=True);root=parent/arm
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
        try:
            start=time.time()
            while True:
                free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=4096:break
                write_json(parent/(arm+'_queue.json'),dict(status='WAITING_GPU_MEMORY',gpu=gpu,free_mib=free));time.sleep(55)
            write_json(parent/(arm+'_resource.json'),dict(gpu=gpu,free_mib=free,minimum_mib=4096,wait_seconds=time.time()-start))
            cmd=[sys.executable,'-m',module+'engine','--data',data,'--reference',reference,'--output',str(root),'--domain',domain,'--arm',arm,'--seed',str(seed)]
            child(cmd,parent/(arm+'_train.log'),parent/(arm+'_train_exit.json'),env)
            child([sys.executable,'-m',module+'deploy','--root',str(root),'--data',data,'--reference',reference],parent/(arm+'_deploy.log'),parent/(arm+'_deploy_exit.json'),env)
            return dict(seed=seed,domain=domain,arm=arm,status='COMPLETE')
        except BaseException as e:
            result=dict(seed=seed,domain=domain,arm=arm,status='INCOMPLETE_ENGINEERING',error=repr(e));write_json(parent/(arm+'_failure.json'),result);return result
    def phase(seeds,arms):
        tasks=[(seed,domain,arm) for seed in seeds for arm in arms for domain in DOMAINS]
        # One independent lane per GPU; never more than one study process on that GPU.
        def lane(gpu,tt):return [task(*t,gpu) for t in tt]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(lane,gpu,tasks[i::2]) for i,gpu in enumerate((4,5))]
            return [r for f in futures for r in f.result()]
    results=phase((11,),ARMS)
    gate=dict(status='INCOMPLETE_ENGINEERING',passed=False,selected=None);confirmed=None
    try:
        phase_complete(base,(11,),ARMS);table=collect(base)[0];gate=screen(table)
        write_json(base/'SCREEN_DECISION.json',gate)
        if gate['passed']:
            arms=('SUP_G0','SUP_G1','MT_CONF_'+gate['selected'][-2:],'MT_PAS_'+gate['selected'][-2:])
            results+=phase((12,13),arms)
            phase_complete(base,(12,13),arms);table=collect(base)[0];confirmed=confirmation(table,gate)
    except BaseException as e:
        gate['engineering_failure']=repr(e)
        write_json(base/'phase_failure.json',dict(status='INCOMPLETE_ENGINEERING',error=repr(e)))
        if not (base/'SCREEN_DECISION.json').exists():write_json(base/'SCREEN_DECISION.json',gate)
    write_json(base/'parent_receipt.json',dict(source=source,tasks=results,finished_unix=time.time()))
    print(json.dumps(finish(base,gate,confirmed)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','data','reference','qualification'):p.add_argument('--'+k,required=True)
    a=vars(p.parse_args())
    try:execute(**a)
    except BaseException as e:
        write_json(Path(a['base'])/'parent_failure.json',dict(status='INCOMPLETE_ENGINEERING',error=repr(e)));raise
