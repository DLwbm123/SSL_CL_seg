"""Operational GPU amendment only. Trainers and report math execute from frozen r0."""
import json,os,sys,time,shutil,subprocess
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def prepare_attempt(b,row):
    tid=row['task_id'];old=b/'tasks'/tid;ops=b/'tasks'/(tid+'_operations');archive=b/'gpu_relocation_attempts'/tid
    archive.mkdir(parents=True)
    old.rename(archive/'task');ops.rename(archive/'operations');old.mkdir()
    for name in ('latest.pt','boundary.json','warmup.json'):
        p=archive/'task'/name
        if p.exists():shutil.copy2(p,old/name)
    for p in (archive/'task').glob('checkpoint_*.pt'):os.link(p,old/p.name)
    lines=(archive/'task/steps.jsonl').read_text().splitlines();pos=row['checkpoint_position']
    assert len(lines)>=pos and json.loads(lines[pos-1])['update']==pos
    (old/'steps.jsonl').write_text('\n'.join(lines[:pos])+'\n')
    for p in list((b/'logs').glob(tid+'_train*')):p.rename(archive/p.name)
    (archive/'RECOVERY_BINDING.json').write_text(json.dumps(row,indent=2)+'\n')


def main():
    controller_repo=Path(__file__).resolve().parents[3]
    controller_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=controller_repo,text=True).strip()
    b=Path(os.environ['R_BASE']);runtime=b/'r0';os.chdir(runtime);sys.path.insert(0,str(runtime))
    from experiments.lcrseg.dpr_sign_replication_v0_1 import core as c,contract as ct
    from experiments.lcrseg.dpr_sign_replication_v0_1.execute import child
    from experiments.lcrseg.dpr_sign_replication_v0_1.results import finish
    source=ct.verify();p=ct.protocol();ct.neutral_subprocess_paths()
    assert source=='87bb70ee05609e77935b1bab5b9a44c9532dc491'
    c.write_json(b/'GPU67_CONTROLLER_STATUS.json',dict(status='WAITING_EXISTING_GPU67_SOURCE_WORKERS',controller_source=controller_source,training_source=source,GPUs=[6,7],started=time.time()))
    # Original parent stops new dispatch after the two user-requested terminations;
    # its GPU6/7 children finish normally before this controller takes ownership.
    while not (b/'parent_exit.json').exists():time.sleep(20)
    assert ct.read(b/'parent_exit.json')['exit_code']==1
    rows=ct.read(b/'GPU_RELOCATION_CAPTURE.json');byid={r['task_id']:r for r in rows}
    for row in rows:prepare_attempt(b,row)
    minimum=ct.read(b/'smoke/receipt.json')['admission_mib'];data=ct.read(b/'private_inputs.json')['data'];reference='/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg/third_party/JASCL_REFERENCE'
    def job(t,phase,gpu):
        root=b/'tasks'/t['task_id'];receipt=root/('receipt.json' if phase=='train' else 'evaluation/receipt.json')
        if receipt.exists():
            rr=ct.read(receipt);assert rr['source']==source
            assert ct.read(b/'logs'/(t['task_id']+'_'+phase+'_exit.json'))['exit_code']==0
            return
        while int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<minimum:time.sleep(20)
        if phase!='train' or t['task_id'] not in byid:return child(b,t['task_id'],phase,gpu,data,reference)
        row=byid[t['task_id']]
        entry="""from collections import Counter
import torch
from experiments.lcrseg.dpr_sign_replication_v0_1 import engine as e,contract as ct
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations
ct.neutral_subprocess_paths()
with Operations(%r) as op:
 op.counts=Counter(%r)
 e.train(%r,%r,%r,%r,torch.device('cuda:0'),resume=True)
"""%(str(b/'tasks'/(t['task_id']+'_operations')),row['checkpoint_operation_counts'],str(b),t['task_id'],data,reference)
        env=dict(os.environ,R_ENTRY=entry,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        start=time.time()
        with (b/'logs'/(t['task_id']+'_train.log')).open('x') as log:
            proc=subprocess.Popen(['bash','-s','--',sys.executable,'-c','import os;exec(os.environ["R_ENTRY"])'],cwd=runtime,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
            proc.stdin.write((runtime/'experiments/lcrseg/scripts/with_nas_storage.sh').read_bytes());proc.stdin.close()
            c.write_json(b/'logs'/(t['task_id']+'_train_started.json'),dict(pid=proc.pid,gpu=gpu,started=start,resume_position=row['checkpoint_position'],controller_source=controller_source))
            code=proc.wait()
        c.write_json(b/'logs'/(t['task_id']+'_train_exit.json'),dict(exit_code=code,pid=proc.pid,gpu=gpu,started_unix=start,finished_unix=time.time(),resumed=True))
        if code:raise RuntimeError('GPU relocation resume failed: '+t['task_id'])
    def phase(tasks,kind):
        # Two sequential lanes, no thread can dispatch onto an unapproved GPU.
        def lane(gpu,items):
            for t in items:job(t,kind,gpu)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(lane,g,tasks[i::2]) for i,g in enumerate((6,7))]
            for f in futures:f.result()
    try:
        c.write_json(b/'GPU67_CONTROLLER_STATUS.json',dict(status='RUNNING_GPU67_ONLY',controller_source=controller_source,training_source=source,GPUs=[6,7]))
        sources=[t for t in p['tasks'] if t['arm']=='SRC_CE'];phase(sources,'train');phase(sources,'eval');binding={}
        for t in sources:
            root=b/'tasks'/t['task_id'];r=ct.read(root/'receipt.json');ev=ct.read(root/'evaluation/receipt.json');assert r['student_hash']==ev['student_hash'] and r['source']==source and r['updates']==t['updates']
            binding[t['task_id']]=dict(student_hash=r['student_hash'],source=source,seed=t['seed'],domain=t['domain'],receipt_sha256=ct.sha256(root/'receipt.json'),deploy_sha256=ct.sha256(root/'deploy_student.pt'),source_scores_sha256=ct.sha256(root/'evaluation/private_patient_metrics.csv'))
        c.write_json(b/'SOURCE_BINDING.json',dict(status='PASS',source=source,sources=binding,new_source_updates=15900,old_sources_loaded=0))
        phase([t for t in p['tasks'] if t['arm']=='F_CONV'],'train');phase([t for t in p['tasks'] if t['arm']==c.MAIN],'train')
        targets=[t for t in p['tasks'] if t['arm']!='SRC_CE'];students={}
        for t in targets:
            root=b/'tasks'/t['task_id'];r=ct.read(root/'receipt.json');assert ct.read(root/f"diagnostics_pass_{t['updates']}.json")['status']=='PASS'
            students[t['task_id']]=dict(student_hash=r['student_hash'],source_student_hash=r['boundary']['source_student_hash'],checkpoint_sha256=ct.sha256(root/f"checkpoint_{t['updates']}.pt"))
        c.write_json(b/'TARGET_WEIGHT_SEAL.json',dict(source=source,students=students,created=time.time()));phase(targets,'eval')
        terminal=finish(b);extra=sum(x['discarded_completed_optimizer_updates'] for x in rows)
        cost=dict(controller_source=controller_source,training_source=source,GPUs=[6,7],formal_scientific_updates=47700,extra_completed_optimizer_updates=extra,physical_completed_optimizer_updates=47700+extra,qualification_updates_separate=True,prefix_counters='actual persisted operation events at saved checkpoints; original tails and logs archived',attempts=rows)
        c.write_json(b/'public_results/GPU_RELOCATION_COST.json',cost)
        r=ct.read(b/'public_results/RESOURCE_ACCOUNTING.json');r['GPU_relocation']=cost;c.write_json(b/'public_results/RESOURCE_ACCOUNTING.json',r)
        terminal['controller_source']=controller_source;terminal['GPU_relocation_extra_updates']=extra;c.write_json(b/'public_results/TERMINAL.json',terminal);c.write_json(b/'TERMINAL.json',terminal)
        c.write_json(b/'GPU67_CONTROLLER_STATUS.json',dict(status='COMPLETE',controller_source=controller_source,terminal=terminal,finished=time.time()))
    except BaseException as exc:
        c.write_json(b/'GPU67_CONTROLLER_FAILURE.json',dict(status='INCOMPLETE_ENGINEERING',error=repr(exc),controller_source=controller_source,automatic_retry=False));raise

if __name__=='__main__':main()
