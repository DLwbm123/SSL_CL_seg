"""Finite four-lane queue; immutable P0 budget, then P1 and one conditional P2."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from experiments.lcrseg.single_teacher_scd_v0_1.execute import child
from .core import *
from .freeze import verify
from .results import read,rows,collect,screen,confirmation,validate_phase,finish,OLD_DOC
GPUS=(4,5,6,7);MODULE='experiments.lcrseg.ssl_head_control_v0_1.'

def execute(base,data,reference,qualification):
    base=Path(base);source=verify()
    assert str(base.resolve()).startswith('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/')
    for file in (Path(qualification),base/'qualification_local/qualification.json',base/'real_smoke/receipt.json',base/'INPUT_LINEAGE.json'):
        record=read(file);assert record['status']=='PASS' and record['source']==source
    with (base/'reservation.json').open('x') as f:json.dump(dict(source=source,started_unix=time.time(),P0_sample_model_forward_cap=1300,P1_budget=15900,P2_SSL_budget=42400,P2_SUP_budget=21200,P2_mutually_exclusive=True,seeds=[11,21,22],GPUs=GPUS),f)
    def resource(gpu,receipt):
        started=time.time()
        while True:
            free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=4096:break
            write_json(receipt,dict(status='WAITING_GPU_MEMORY',gpu=gpu,free_mib=free));time.sleep(55)
        write_json(receipt,dict(status='ADMITTED',gpu=gpu,free_mib=free,minimum_mib=4096,wait_seconds=time.time()-started))
        return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
    def audit(domain,arm,gpu):
        parent=base/'P0'/domain;parent.mkdir(parents=True,exist_ok=True);root=parent/arm
        try:
            env=resource(gpu,parent/(arm+'_resource.json'))
            child([sys.executable,'-m',MODULE+'p0','--output',str(root),'--data',data,'--reference',reference,'--domain',domain,'--arm',arm],parent/(arm+'_audit.log'),parent/(arm+'_exit.json'),env)
            return dict(domain=domain,arm=arm,status='COMPLETE')
        except BaseException as error:
            result=dict(domain=domain,arm=arm,status='INCOMPLETE_OLD_MODEL_DIAGNOSTIC',error=repr(error),automatic_retry=False);write_json(parent/(arm+'_failure.json'),result);return result
    def task(seed,domain,arm,gpu):
        parent=base/f'seed{seed}'/domain;parent.mkdir(parents=True,exist_ok=True);root=parent/arm
        try:
            env=resource(gpu,parent/(arm+'_resource.json'))
            child([sys.executable,'-m',MODULE+'engine','--data',data,'--reference',reference,'--output',str(root),'--domain',domain,'--arm',arm,'--seed',str(seed)],parent/(arm+'_train.log'),parent/(arm+'_train_exit.json'),env)
            child([sys.executable,'-m',MODULE+'deploy','--root',str(root),'--data',data,'--reference',reference],parent/(arm+'_deploy.log'),parent/(arm+'_deploy_exit.json'),env)
            return dict(seed=seed,domain=domain,arm=arm,status='COMPLETE')
        except BaseException as error:
            result=dict(seed=seed,domain=domain,arm=arm,status='INCOMPLETE_ENGINEERING',error=repr(error));write_json(parent/(arm+'_failure.json'),result);return result
    def parallel(fn,tasks):
        def lane(gpu,tt):return [fn(*t,gpu) for t in tt]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(lane,gpu,tasks[i::4]) for i,gpu in enumerate(GPUS)]
            return [r for f in futures for r in f.result()]
    p0=parallel(audit,[(d,a) for a in ('MT_PAS_G0','MT_PAS_G1') for d in DOMAINS]);write_json(base/'P0_SUMMARY.json',p0)
    p1=('LIN_SUP','LIN_MT_CONF','LIN_MT_PAS')
    results=parallel(task,[(11,d,a) for a in p1 for d in DOMAINS]);gate=dict(path='INCOMPLETE',selected=None);confirmed=None;error=None
    try:
        validate_phase(base,(11,),p1);gate=screen(collect(base)[0]['metrics'],rows(OLD_DOC/'SINGLE_DOMAIN_METRICS.csv'));write_json(base/'SCREEN_DECISION.json',gate)
        if gate['path']!='NOT_ADMITTED':
            arms=ARMS if gate['path']=='P2_SSL' else ('LIN_SUP','SUP_G1')
            results+=parallel(task,[(s,d,a) for s in (21,22) for a in arms for d in DOMAINS])
            validate_phase(base,(21,22),arms);confirmed=confirmation(collect(base)[0]['metrics'],gate)
    except BaseException as exc:
        error=repr(exc);write_json(base/'phase_failure.json',dict(status='INCOMPLETE_ENGINEERING',error=error))
    write_json(base/'parent_receipt.json',dict(source=source,P0=p0,tasks=results,error=error,finished_unix=time.time()))
    print(json.dumps(finish(base,gate,confirmed,error)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in ('base','data','reference','qualification'):parser.add_argument('--'+k,required=True)
    args=vars(parser.parse_args())
    try:execute(**args)
    except BaseException as error:
        write_json(Path(args['base'])/'parent_failure.json',dict(status='INCOMPLETE_ENGINEERING',error=repr(error)));raise
