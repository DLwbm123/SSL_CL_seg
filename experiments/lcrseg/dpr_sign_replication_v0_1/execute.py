"""Finite dependency queue: new sources, paired targets, seal, evaluation, stop."""
import os,sys,json,time,queue,threading,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from . import core as c,contract as ct

def child(base,tid,phase,gpu,data,reference):
    b=Path(base);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    module='experiments.lcrseg.dpr_sign_replication_v0_1.'+('engine' if phase=='train' else 'evaluate')
    args=['worker','--base',str(b),'--task-id',tid,'--data',data,'--reference',reference]
    env['R_ENTRY']='import runpy,sys;sys.argv='+repr(args)+';runpy.run_module('+repr(module)+',run_name="__main__")'
    command=['bash','-s','--',sys.executable,'-c','import os;exec(os.environ["R_ENTRY"])'];started=time.time()
    with (b/'logs'/(tid+'_'+phase+'.log')).open('x') as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env)
        proc.stdin.write(Path('experiments/lcrseg/scripts/with_nas_storage.sh').read_bytes());proc.stdin.close()
        c.write_json(b/'logs'/(tid+'_'+phase+'_started.json'),dict(pid=proc.pid,gpu=gpu,started=started,neutral_command=command));code=proc.wait()
    c.write_json(b/'logs'/(tid+'_'+phase+'_exit.json'),dict(pid=proc.pid,gpu=gpu,exit_code=code,started_unix=started,finished_unix=time.time()))
    if code:raise RuntimeError(tid+' '+phase+' exit '+str(code))

def execute(base,data,reference):
    b=ct.nas(base);source=ct.verify();p=ct.protocol();ct.neutral_subprocess_paths()
    for n in ('qualification_local/qualification.json','qualification_server/qualification.json','smoke/receipt.json'):
        r=ct.read(b/n);assert r['status']=='PASS' and r['source']==source
    assert ct.read(b/'qualification_ledger.json')['real_smoke_updates']==8
    with (b/'reservation.json').open('x') as f:json.dump(dict(source=source,tasks=p['tasks'],formal_updates=47700,input_hashes={'private_inputs.json':ct.sha256(b/'private_inputs.json')},created=time.time()),f,indent=2)
    (b/'logs').mkdir();minimum=ct.read(b/'smoke/receipt.json')['admission_mib']
    def phase(tasks,kind):
        jobs=queue.Queue();stop=threading.Event()
        for t in tasks:jobs.put(t['task_id'])
        def worker(gpu):
            try:
                while not stop.is_set() and not jobs.empty():
                    free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                    c.write_json(b/'logs'/f'gpu{gpu}_resource.json',dict(gpu=gpu,free_mib=free,minimum_mib=minimum))
                    if free<minimum:stop.wait(55);continue
                    try:tid=jobs.get_nowait()
                    except queue.Empty:return
                    child(b,tid,kind,gpu,data,reference)
            except BaseException:stop.set();raise
        with ThreadPoolExecutor(max_workers=4) as pool:
            fs=[pool.submit(worker,g) for g in p['GPUs']]
            for f in fs:f.result()
    try:
        src=[t for t in p['tasks'] if t['arm']=='SRC_CE'];phase(src,'train');phase(src,'eval')
        binding={}
        for t in src:
            root=b/'tasks'/t['task_id'];r=ct.read(root/'receipt.json');ev=ct.read(root/'evaluation/receipt.json');assert r['student_hash']==ev['student_hash'] and r['updates']==t['updates'] and r['source']==source
            binding[t['task_id']]=dict(student_hash=r['student_hash'],source=source,seed=t['seed'],domain=t['domain'],receipt_sha256=ct.sha256(root/'receipt.json'),deploy_sha256=ct.sha256(root/'deploy_student.pt'),source_scores_sha256=ct.sha256(root/'evaluation/private_patient_metrics.csv'))
        c.write_json(b/'SOURCE_BINDING.json',dict(status='PASS',source=source,sources=binding,new_source_updates=15900,old_sources_loaded=0))
        phase([t for t in p['tasks'] if t['arm']=='F_CONV'],'train')
        phase([t for t in p['tasks'] if t['arm']==c.MAIN],'train')
        targets=[t for t in p['tasks'] if t['arm']!='SRC_CE'];students={}
        for t in targets:
            root=b/'tasks'/t['task_id'];r=ct.read(root/'receipt.json');assert r['updates']==t['updates'] and r['source']==source and ct.read(root/f"diagnostics_pass_{t['updates']}.json")['status']=='PASS'
            students[t['task_id']]=dict(student_hash=r['student_hash'],source_student_hash=r['boundary']['source_student_hash'],checkpoint_sha256=ct.sha256(root/f"checkpoint_{t['updates']}.pt"))
        c.write_json(b/'TARGET_WEIGHT_SEAL.json',dict(source=source,students=students,created=time.time()))
        phase(targets,'eval')
        from .results import finish
        terminal=finish(b);c.write_json(b/'TERMINAL.json',terminal);c.write_json(b/'parent_receipt.json',dict(status='COMPLETE',source=source,finished=time.time()));print(json.dumps(terminal),flush=True)
    except BaseException as exc:
        c.write_json(b/'parent_failure.json',dict(status='INCOMPLETE_ENGINEERING',source=source,error=repr(exc),automatic_retry=False));raise
