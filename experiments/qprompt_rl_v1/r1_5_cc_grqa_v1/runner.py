"""Dedicated R1.5 queue using the frozen R1 data/model/state primitives."""
import csv
import fcntl
import gc
import io
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from r1_12h import core
from r1_12h import runner as r
from r1_12h.core import atomic,append,events,file_sha,tensor_sha,restore,snapshot,make_reference,PrototypeBank,ema_update
from .method import components,gradient_interaction,weight

ROOT=Path(os.environ['EXEC_ROOT']);OLD=Path(os.environ['EXEC_OLD']);HERE=Path(__file__).parent
PLAN=json.loads((HERE/'DEVELOPMENT_PLAN.json').read_text());AUTH=json.loads((HERE/'EXECUTION_AUTHORIZATION.json').read_text())
CODE=r.CODE;SEED=int(os.environ.get('EXEC_SEED','261'));PHASE=os.environ.get('EXEC_PHASE','development')
CONFIG_SHA=file_sha(HERE/'DEVELOPMENT_PLAN.json')
R1_CODE=AUTH['r1_training_commit']


def configure():
    def seeded(*parts):
        return int.from_bytes(core.hashlib.sha256(json.dumps([SEED,*parts]).encode()).digest()[:8],'big')%(2**63-1)
    core.seed=seeded;r.seed=seeded
    r.CONFIG_SHA=CONFIG_SHA;r.RUN=ROOT/PHASE/str(SEED)
    r.RUN.mkdir(parents=True,exist_ok=True)
    return r.RUN

RUN=configure()


def provenance():return dict(code_commit=CODE,config_digest=CONFIG_SHA,phase=PHASE,seed=SEED)


def charge(task,step):
    with (ROOT/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p=ROOT/'BUDGET.json';b=json.loads(p.read_text()) if p.exists() else {}
        v=b.setdefault(PHASE,dict(attempted=0,replay=0,seen={}))
        key=str(SEED)+'__'+task;replay=step<v['seen'].get(key,0)
        cap=AUTH[PHASE+('_replay_cap' if replay else '_committed_cap')]
        if PHASE=='confirmation':assert json.loads((ROOT/'reports/DEVELOPMENT_COMPLETION.json').read_text())['gate']['passes']
        if (v['replay'] if replay else v['attempted']-v['replay'])>=cap:raise RuntimeError('optimizer budget exhausted')
        v['attempted']+=1;v['replay']+=int(replay);v['seen'][key]=max(v['seen'].get(key,0),step+1)
        atomic(p,b)
        return 'replay' if replay else 'formal'


def setup_task():
    r.setup();random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);r.provenance();task=os.environ['EXEC_TASK'];backbone=os.environ['EXEC_BACKBONE'];domain=os.environ['EXEC_DOMAIN'];arm=os.environ['EXEC_ARM']
    folder=r.task_paths(task);schedule,sha=r.schedule_for(domain)
    ds=r.dataset(domain,'train_labeled');assert ds.rows==schedule['rows'];cache=[ds[i] for i in range(len(ds))]
    model,init_sha=r.build(backbone,False,r.ASSETS,r.SOURCE);model.cuda().train();opt,sch=r.optimizer_for(model,backbone)
    bank=ref=None;step=0;elapsed=0.;attempts=0;prefix_sha=None
    if arm!='QUERY_PREFIX':
        bank=PrototypeBank(128 if backbone=='UNET_QUERY_128' else 384).cuda();ref=make_reference(model) if arm!='A0' else None
    latest=folder/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        approved=json.loads((ROOT/'APPROVED_ENGINEERING_RESUMES.json').read_text()) if (ROOT/'APPROVED_ENGINEERING_RESUMES.json').exists() else []
        assert state['code_commit']==CODE or [state['code_commit'],CODE] in approved
        assert state['config_digest']==CONFIG_SHA and state['data_schedule_digest']==sha
        step=restore(state,model,opt,sch,ref,bank);elapsed=state['training_seconds'];attempts=state['physical_attempts'];prefix_sha=state.get('prefix_sha')
    elif arm!='QUERY_PREFIX':
        prefix=(OLD if PHASE=='development' else RUN)/'tasks'/f'{backbone}__{domain}__QUERY_PREFIX'/'prefix.pt'
        state=torch.load(prefix,map_location='cpu',weights_only=False)
        prefix_sha=file_sha(prefix)
        assert state['code_commit']==(R1_CODE if PHASE=='development' else CODE)
        assert state['global_step']==state['data_cursor']==state['scheduler']['last_epoch']==2000 and state['data_schedule_digest']==sha
        if PHASE=='development':
            verified=json.loads((ROOT/'PREFIX_AUDIT.json').read_text())[backbone+'__'+domain]
            assert prefix_sha==verified['checkpoint_sha256']
        step=restore(state,model,opt,sch)
        bank=r.initialize_bank(model,cache,backbone,torch.device('cuda:0'))
        if ref is not None:ref=make_reference(model)
        atomic(folder/'PREFIX_BINDING.json',dict(prefix_sha=prefix_sha,student_digest=tensor_sha(model.state_dict()),restored_step=step,scheduler_step=sch.last_epoch,**provenance()))
    metadata=dict(**provenance(),data_schedule_digest=sha,model_initialization_digest=init_sha,prefix_sha=prefix_sha,task=task,backbone=backbone,domain=domain,arm=arm)
    return folder,cache,schedule,model,opt,sch,bank,ref,step,elapsed,attempts,metadata


def diagnostics(model,x,y,bank,ref,arm,step,both=False):
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        total,terms,_,_,info=components(model,x,y,bank,ref,arm,step,True,both)
    named={'seg':terms['Lseg'],'img':10*terms['Limg']}
    if both:named.update(grqa_orig=5*terms['GRQA'],grqa_cc=5*terms['GRQA_CC'])
    else:named['grqa']=terms['regularizer']
    grads=gradient_interaction(terms,model,named)
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
    return {k:float(v.detach()) for k,v in terms.items()},info,grads


def train():
    folder,cache,schedule,model,opt,sch,bank,ref,step,elapsed,attempts,meta=setup_task()
    arm=meta['arm'];end=2000 if arm=='QUERY_PREFIX' else 3000;latest=folder/'latest.pt'
    if not latest.exists():atomic(latest,snapshot(model,opt,sch,ref,bank,step,training_seconds=elapsed,physical_attempts=attempts,**meta),binary=True)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        while step<end:
            x,y=r.load_batch(cache,schedule['steps'][step],torch.device('cuda:0'))
            diagnostic=(step+1)%100==0;diag={}
            if diagnostic and arm!='QUERY_PREFIX':
                vals,routing,grads=diagnostics(model,x,y,bank,ref,arm,step)
                diag={**vals,**routing,**grads}
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):total,terms,current,support,_=components(model,x,y,bank,ref,arm,step)
            if not torch.isfinite(total):raise FloatingPointError('nonfinite loss')
            total.backward();norm=sum(p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None).sqrt()
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
            category=charge(meta['task'],step);attempts+=1
            append(folder/'TASK_LEDGER.jsonl',dict(event='optimizer_attempt',step=step,category=category,**provenance()))
            try:opt.step()
            except BaseException:
                append(folder/'TASK_LEDGER.jsonl',dict(event='optimizer_failed',step=step,category=category));raise
            sch.step()
            if bank is not None:bank.update(current.detach(),support)
            if ref is not None:ema_update(model,ref,.99)
            step+=1
            append(folder/'TASK_LEDGER.jsonl',dict(event='committed',step=step,category=category,**provenance()))
            if diagnostic:
                append(folder/'DIAGNOSTICS.jsonl',dict(diag,global_step=step,local_step=step-(0 if arm=='QUERY_PREFIX' else 2000),total_loss=float(total.detach()),Lseg=float(terms['Lseg'].detach()),lambda_grqa=0 if arm=='QUERY_PREFIX' else weight(arm,step-2000),lr=[g['lr'] for g in opt.param_groups],grad_norm=float(norm),training_seconds=elapsed+time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),**provenance()))
            if step%250==0 or step==end:
                atomic(latest,snapshot(model,opt,sch,ref,bank,step,training_seconds=elapsed+time.monotonic()-start,physical_attempts=attempts,**meta),binary=True)
                atomic(folder/'PROGRESS.json',dict(global_step=step,**meta))
        target=folder/('prefix.pt' if arm=='QUERY_PREFIX' else 'final.pt')
        atomic(target,snapshot(model,opt,sch,ref,bank,step,training_seconds=elapsed+time.monotonic()-start,physical_attempts=attempts,**meta),binary=True)
        atomic(folder/'TRAIN_DONE.json',dict(**meta,status='COMPLETED',global_step=step,committed_updates=2000 if arm=='QUERY_PREFIX' else 1000,training_seconds=elapsed+time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),checkpoint_bytes=target.stat().st_size,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    except BaseException:
        atomic(folder/f'FAILURE_{time.time_ns()}.private.json',dict(step=step,traceback=traceback.format_exc(),**meta));raise


def evaluate():
    r.evaluate()
    p=RUN/'tasks'/os.environ['EXEC_TASK']/'EVALUATION.json';value=json.loads(p.read_text())
    done=json.loads(p.with_name('TRAIN_DONE.json').read_text())
    value.update({k:done[k] for k in ('prefix_sha','data_schedule_digest','phase','seed')});atomic(p,value)


def forensic():
    assert PHASE=='development' and os.environ['EXEC_ARM']=='A1'
    folder,cache,schedule,model,opt,sch,bank,ref,step,elapsed,attempts,meta=setup_task()
    before=tensor_sha(model.state_dict());bank_before=tensor_sha(bank.state_dict())
    output=ROOT/'forensic'/meta['task'];output.mkdir(parents=True,exist_ok=True)
    for step in range(2000,3000,10):
        x,y=r.load_batch(cache,schedule['steps'][step],torch.device('cuda:0'))
        terms,routing,grads=diagnostics(model,x,y,bank,ref,'A1',step,True)
        append(output/'BATCHES.jsonl',dict(step=step,**meta,terms=terms,routing=routing,gradients=grads,optimizer_calls=0))
    assert before==tensor_sha(model.state_dict()) and bank_before==tensor_sha(bank.state_dict()) and sch.last_epoch==2000
    atomic(output/'DONE.json',dict(**meta,batches=100,student_forward_calls=100,reference_forward_calls=100,autograd_grad_calls=400,optimizer_calls=0,state_unchanged=True))


def prepare():
    r.setup();r.provenance()
    if PHASE=='confirmation':r.prepare();return
    (RUN/'schedules').mkdir(parents=True,exist_ok=True)
    for domain in PLAN['domains']:shutil.copyfile(OLD/'schedules'/f'{domain}.json',RUN/'schedules'/f'{domain}.json')
    for name in ('INITIALIZATION.json','ENVIRONMENT.json'):shutil.copyfile(OLD/name,RUN/name)
    audit={}
    for b in PLAN['backbones']:
        for d in PLAN['domains']:
            task=f'{b}__{d}__QUERY_PREFIX';p=OLD/'tasks'/task/'prefix.pt';sha=file_sha(p)
            bindings=[json.loads(x.read_text()) for x in (OLD/'tasks').glob(f'{b}__{d}__*/PREFIX_BINDING.json')]
            assert len(bindings)==5 and all(x['prefix_sha']==sha for x in bindings)
            s=torch.load(p,map_location='cpu',weights_only=False);assert s['code_commit']==R1_CODE
            assert s['global_step']==s['data_cursor']==s['scheduler']['last_epoch']==2000
            assert s['data_schedule_digest']==file_sha(RUN/'schedules'/f'{d}.json')
            assert all(torch.isfinite(v).all() for v in s['student'].values() if torch.is_floating_point(v))
            audit[b+'__'+d]=dict(checkpoint_sha256=sha,training_code_commit=s['code_commit'],global_step=2000,scheduler_step=2000,data_schedule_digest=s['data_schedule_digest'],student_digest=tensor_sha(s['student']),readable_finite=True)
    atomic(ROOT/'PREFIX_AUDIT.json',audit)
    atomic(RUN/'DATA_BINDING.private.json',dict(**provenance(),old_receipt_sha=file_sha(OLD/'DATA_BINDING.private.json'),prefix_audit_sha=file_sha(ROOT/'PREFIX_AUDIT.json')))
    print('FOUR_PREFIXES_VERIFIED',flush=True)


def spawn(mode,gpu,phase='development',seed=261,**extra):
    e=dict(os.environ,EXEC_MODE=mode,CUDA_VISIBLE_DEVICES=str(gpu),EXEC_PHASE=phase,EXEC_SEED=str(seed),**extra)
    active=json.loads((ROOT/'ACTIVE_CODE.json').read_text());cwd=active['source'];e['EXEC_RUN']=str(ROOT/phase/str(seed))
    log=ROOT/'logs'/f'{mode}_{extra.get("EXEC_TASK","main")}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
    with log.open('w') as f:
        p=subprocess.Popen(['python3.12','-'],executable=sys.executable,cwd=cwd,env=e,stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT,text=True)
        p.stdin.write('import runpy;runpy.run_module("r1_5_cc_grqa_v1.runner",run_name="__main__")\n');p.stdin.close()
    append(ROOT/'PROCESS_LEDGER.jsonl',dict(pid=p.pid,identity=r.process_identity(p.pid),gpu=gpu,mode=mode,phase=phase,seed=seed,log=str(log),**extra));return p


def queue_run(phase,seeds,forensics=False):
    path=ROOT/('FORENSIC_QUEUE.json' if forensics else phase.upper()+'_QUEUE.json')
    arms=['A1'] if forensics else (PLAN['arms'] if phase=='development' else ['QUERY_PREFIX','A0','A4'])
    queue=json.loads(path.read_text()) if path.exists() else [dict(task=f'{b}__{d}__{a}',backbone=b,domain=d,arm=a,seed=s,status='PENDING',attempts=0,eval_attempts=0) for a in arms for s in seeds for b in PLAN['backbones'] for d in PLAN['domains']]
    active={};records={(x['phase'],x['seed'],x.get('EXEC_TASK')):x for x in events(ROOT/'PROCESS_LEDGER.jsonl') if x['mode'] in ('train','evaluate','forensic')}
    for t in queue:
        if t['status'] in ('RUNNING','RUNNING_EVAL'):
            prev=records.get((phase,t['seed'],t['task']))
            if prev:active[(t['seed'],t['task'])]=(r.ExistingChild(prev['pid'],prev['identity']),prev['gpu'],prev['mode'])
            else:t['status']='PENDING'
    while True:
        for key,(p,gpu,mode) in list(active.items()):
            if p.poll() is None:continue
            t=next(t for t in queue if (t['seed'],t['task'])==key);folder=ROOT/phase/str(t['seed'])/'tasks'/t['task']
            success=(ROOT/'forensic'/t['task']/'DONE.json').exists() if forensics else (folder/('EVALUATION.json' if mode=='evaluate' else 'TRAIN_DONE.json')).exists()
            t['status']=('COMPLETED' if forensics or mode=='evaluate' or t['arm']=='QUERY_PREFIX' else 'EVAL_PENDING') if success else ('EVAL_PENDING' if mode=='evaluate' else 'PENDING')
            if not success and t['eval_attempts' if mode=='evaluate' else 'attempts']>=3:t['status']='NEEDS_REPAIR'
            append(ROOT/'SESSION_LEDGER.jsonl',dict(phase=phase,seed=t['seed'],task=t['task'],mode=mode,returncode=p.returncode,status=t['status']));del active[key]
        busy={g for _,g,_ in active.values()};free=r.gpu_free()
        for t in queue:
            if t['status'] not in ('PENDING','EVAL_PENDING'):continue
            if phase=='confirmation' and t['arm']!='QUERY_PREFIX':
                parent=next(v for v in queue if v['seed']==t['seed'] and v['backbone']==t['backbone'] and v['domain']==t['domain'] and v['arm']=='QUERY_PREFIX')
                if parent['status']!='COMPLETED':continue
            # R1 measured <=1.5GB; allow 3.5GB for extra diagnostic gradients and runtime.
            candidates=[g for g,m in free.items() if g not in busy and m>=3500]
            if not candidates:continue
            gpu=max(candidates,key=free.get);mode='forensic' if forensics else ('evaluate' if t['status']=='EVAL_PENDING' else 'train')
            t['eval_attempts' if mode=='evaluate' else 'attempts']+=1
            p=spawn(mode,gpu,phase,t['seed'],EXEC_TASK=t['task'],EXEC_BACKBONE=t['backbone'],EXEC_DOMAIN=t['domain'],EXEC_ARM=t['arm'])
            active[(t['seed'],t['task'])]=(p,gpu,mode);busy.add(gpu);t['status']='RUNNING_EVAL' if mode=='evaluate' else 'RUNNING'
        atomic(path,queue)
        if all(t['status']=='COMPLETED' for t in queue):return
        if not active and any(t['status']=='NEEDS_REPAIR' for t in queue):raise RuntimeError('queue needs autonomous agent repair; preserved all state')
        time.sleep(5)


def supervise():
    with (ROOT/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (ROOT/'PREFIX_AUDIT.json').exists():
            p=spawn('prepare',-1);assert p.wait()==0
        queue_run('development',[261],True)
        from .report import report_forensic,report_phase
        report_forensic(ROOT)
        queue_run('development',[261])
        passed=report_phase(ROOT,'development')
        if passed:
            for seed in (262,263):
                if not (ROOT/'confirmation'/str(seed)/'INITIALIZATION.json').exists():
                    p=spawn('prepare',-1,'confirmation',seed);assert p.wait()==0
            queue_run('confirmation',[262,263]);report_phase(ROOT,'confirmation')
        atomic(ROOT/'FINAL.json',dict(time=time.time(),development_passed=passed,status='CONFIRMATION_COMPLETED' if passed else 'R1_5_DEVELOPMENT_NOT_MET',**provenance()))


def service():
    while not (ROOT/'FINAL.json').exists():
        p=spawn('supervise',-1);p.wait();append(ROOT/'WATCHDOG.jsonl',dict(pid=p.pid,returncode=p.returncode))
        if not (ROOT/'FINAL.json').exists():time.sleep(30)

if __name__=='__main__':
    {'prepare':prepare,'forensic':forensic,'train':train,'evaluate':evaluate,'supervise':supervise,'service':service}[os.environ['EXEC_MODE']]()
