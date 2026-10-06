"""Finish the running source, then hand off at a zero-update action boundary."""
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path


def read(p):return json.loads(p.read_text())
def write(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');os.replace(temp,p)


def main():
    root=Path(os.environ['EXEC_RUN']);config=read(Path(os.environ['EXEC_CONFIG']));prior=Path(config['prior_attempt'])
    assert not (root/'CONTINUATION_STARTED.json').exists()
    write(root/'CONTINUATION_STARTED.json',dict(pid=os.getpid(),time=time.time()))
    write(root/'STATUS.json',dict(status='WAITING_AUXILIARY',phase='CLEAN_AUXILIARY',dependency=prior.name))
    while True:
        state=read(prior/'STATUS.json')
        if state['status']=='ENGINEERING_STOP':break
        time.sleep(10)
    hold=prior/'m2000_n2_brightness/STAGE_HOLD.json'
    if not hold.exists() or not state.get('error','').startswith('FileExistsError') or not (prior/'auxiliary/receipt.json').exists():
        raise RuntimeError('upstream did not stop at the reserved clean phase boundary')
    receipt=read(prior/'auxiliary/receipt.json');assert receipt['status']=='SEALED' and receipt['step']==8000
    assert read(prior/'QUALIFICATION.json')['status']=='PASS'
    counts=Counter()
    for line in (prior/'PHYSICAL_LEDGER.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['event']=='attempt':counts[row['category']]+=1
    assert counts['auxiliary']==8000 and counts['audit']==counts['entries']==0
    (root/'auxiliary').symlink_to(prior/'auxiliary',target_is_directory=True)
    for name in ('QUALIFICATION.json','PHYSICAL_LEDGER.jsonl','ROLES.private.json'):
        shutil.copyfile(prior/name,root/name)
    write(root/'A_CONTINUATION.json',dict(reason='Complete both old/new step-25 readouts before action training; source is reused without retraining.',roles_match="PENDING_RUNTIME_COMPARE",source_commit=receipt['identity']['execution_commit'],action_commit=config['commit'],physical_carry=dict(counts),previous=prior.name))
    write(prior/'PHASE_HOLD_RECEIPT.json',state)
    write(prior/'STATUS.json',dict(status='AUXILIARY_COMPLETE_HANDED_OFF',next_run=root.name,updates=8000))
    while True:
        free,util,used=map(int,subprocess.check_output(['nvidia-smi','--id=5','--query-gpu=memory.free,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(','))
        if free>=12000 and util==0 and used<100:break
        time.sleep(10)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',CUBLAS_WORKSPACE_CONFIG=':4096:8',EXEC_MODULE='experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker',PYTHONPATH=config['code'])
    with (root/'worker.log').open('a') as log:
        p=subprocess.Popen(['bash','./with_nas_storage.sh',config['python'],'-c','import os,runpy;runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")'],cwd=config['code']+'/experiments/lcrseg/scripts',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(root/'LAUNCH.json',dict(pid=p.pid,gpu=5,time=time.time(),commit=config['commit']))
    time.sleep(1)
    command=Path(f'/proc/{p.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if any(x in command.lower() for x in ('wangbomin','ssl_cl','grpo')):raise RuntimeError('non-neutral child command')
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    write(root/'PROCESS_CHECK.json',dict(command=command,gpu_processes=output,time=time.time()))
    p.wait();write(root/'PROCESS_EXIT.json',dict(exit_code=p.returncode,time=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(Path(os.environ['EXEC_RUN'])/'STATUS.json',dict(status='HANDOFF_STOP',error=repr(exc)));raise
