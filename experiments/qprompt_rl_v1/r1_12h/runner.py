"""One neutral process per role; durable queue and bounded optimizer accounting."""
import csv
import fcntl
import gc
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from qprompt.data import CanonicalM1, MANIFEST_SHA, SPLIT_SHA
from qprompt.metrics import image_dice
from .augmentation import geometry, strong
from .core import *

SOURCE = Path(__file__).resolve().parents[1]
RUN = Path(os.environ['EXEC_RUN']).resolve()
ASSETS = Path(os.environ['EXEC_ASSETS']).resolve()
DATA = Path(os.environ['EXEC_DATA']).resolve()
MIGRATION = Path(os.environ['EXEC_MANIFEST']).resolve()
CODE = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
CONFIG_SHA = file_sha(Path(__file__).with_name('EXECUTION_AUTHORIZATION.json'))


def provenance():
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=SOURCE, check=True)
    subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=SOURCE, check=True)
    if subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip():
        raise RuntimeError('dirty scientific checkout')
    return dict(code_commit=CODE, config_digest=CONFIG_SHA)


def setup():
    if (RUN/'ENVIRONMENT.json').exists():
        pinned=json.loads((RUN/'ENVIRONMENT.json').read_text())
        assert pinned['torch']==torch.__version__ and pinned['numpy']==np.__version__ and pinned['python']==sys.version
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    random.seed(261); np.random.seed(261); torch.manual_seed(261)


def dataset(domain, role):
    return CanonicalM1(DATA, MIGRATION, domain=domain, role=role,
                       purpose='train' if role == 'train_labeled' else 'evaluate', current_domain=domain)


def prepare():
    """No optimizer: independent qualification process can inspect M1 L and val."""
    provenance(); RUN.mkdir(parents=True, exist_ok=True)
    m = json.loads(MIGRATION.read_text())
    selected = [x for x in m['files'] if x['root_id'] == 'data']
    assert len(selected) == 182
    assert all(x['domain'] in CONFIG['domains'] and x['role'] in ('train_labeled','val') for x in selected)
    binding = dict(**provenance(), source_migration_digest=file_sha(MIGRATION), records={}, payload_files=182)
    for domain, expected in zip(CONFIG['domains'], [(16,40),(10,25)]):
        for role, count in zip(('train_labeled','val'), expected):
            ds = dataset(domain,role); assert len(ds) == count
            valid_pixels = 0
            for i in range(len(ds)):
                item = ds[i]  # Rechecks payload SHA, keys, geometry, and classes.
                assert torch.isfinite(item['image']).all() and 0 <= item['image'].min() <= item['image'].max() <= 1
                assert (item['label'] != 255).any()
                valid_pixels += int((item['label'] != 255).sum())
            binding['records'][domain+'/'+role] = dict(count=count, valid_pixels=valid_pixels)
            if role == 'train_labeled':
                schedule=[]
                for step in range(3000):
                    g=torch.Generator().manual_seed(seed(domain,step,'indices'))
                    schedule.append(dict(global_step=step,indices=torch.randperm(len(ds),generator=g)[:2].tolist(),
                        geometry_seed=seed(domain,step,'geometry'),photometric_seed=seed(domain,step,'photometric')))
                path=RUN/'schedules'/f'{domain}.json'
                if path.exists(): raise RuntimeError('schedule already frozen')
                atomic(path,dict(domain=domain,rows=ds.rows,steps=schedule))
    atomic(RUN/'DATA_BINDING.private.json',binding)
    init={}
    for backbone in CONFIG['backbones']:
        a,ha=build(backbone,True,ASSETS,SOURCE);del a
        b,hb=build(backbone,False,ASSETS,SOURCE);del b
        assert ha==hb
        init[backbone]=dict(body_digest=ha,dense_head_seed=seed(backbone,'dense-head'),
            query_head_seed=seed(backbone,'query-head'))
    atomic(RUN/'INITIALIZATION.json',dict(**provenance(),models=init,
        schedules={d:file_sha(RUN/'schedules'/f'{d}.json') for d in CONFIG['domains']}))
    atomic(RUN/'ENVIRONMENT.json',dict(**provenance(),python=sys.version,torch=torch.__version__,
        numpy=np.__version__,cuda_build=torch.version.cuda,precision=CONFIG['precision'],
        deterministic=True,tf32=False,autocast='bfloat16',shared_packages=True))
    print('DATA_BINDING_AND_INITIALIZATION_PASSED',flush=True)


def schedule_for(domain):
    path=RUN/'schedules'/f'{domain}.json'
    expected=json.loads((RUN/'INITIALIZATION.json').read_text())['schedules'][domain]
    assert file_sha(path)==expected
    return json.loads(path.read_text()),expected


def load_batch(cache, item, device, augment=True):
    g=torch.Generator().manual_seed(item['geometry_seed']);pairs=[]
    for i in item['indices']:
        x,y=cache[i]['image'],cache[i]['label']
        if augment:x,y,_=geometry(x,y,torch.ones_like(y,dtype=torch.bool),g)
        pairs.append((x,y))
    x=torch.stack([p[0] for p in pairs]).to(device)
    y=torch.stack([p[1] for p in pairs]).to(device)
    if augment:
        x=strong(x,torch.Generator(device=device).manual_seed(item['photometric_seed']))
    return x,y


@torch.no_grad()
def initialize_bank(model, cache, backbone, device):
    bank=PrototypeBank(128 if backbone=='UNET_QUERY_128' else 384).to(device)
    vectors=[];supports=[];was_training=model.training;model.eval()
    for item in cache:
        x=item['image'][None].to(device);y=item['label'][None].to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16): out=model(x)
        v,s=image_prototypes(out['pixels'].float(),y,vit_pad=isinstance(model,DINOv2Adapter))
        vectors.append(v);supports.append(s)
    bank.update(torch.cat(vectors),torch.cat(supports));model.train(was_training)
    return bank


def charge(kind, task, step):
    """Global lock: <=28 jobs, fsynced before every optimizer call."""
    with (RUN/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=RUN/'BUDGET.json'
        b=json.loads(path.read_text()) if path.exists() else dict(synthetic=0,smoke=0,formal=0,recovery=0,seen={})
        category=kind
        if kind=='formal':
            category='recovery' if step < b['seen'].get(task,0) else 'formal'
        cap={'synthetic':32,'smoke':16,'formal':40000,'recovery':4000}[category]
        if b[category]>=cap:raise RuntimeError(category+' optimizer budget exhausted')
        if kind=='smoke':
            per=b.setdefault('smoke_by_backbone',{})
            if per.get(task,0)>=8:raise RuntimeError('backbone smoke budget exhausted')
            per[task]=per.get(task,0)+1
        b[category]+=1
        if kind=='formal':b['seen'][task]=max(b['seen'].get(task,0),step+1)
        atomic(path,b)
    return category


def transaction(model,opt,scheduler,bank,reference,x,y,arm,kind,task,step,log,diagnose=False):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        total,terms,current,support,queries=losses(model,x,y,arm,bank,reference)
    if not torch.isfinite(total):raise FloatingPointError('nonfinite loss before optimizer')
    details={k:float(v.detach()) for k,v in terms.items()}
    if diagnose:
        details['Lseg_gradient_norm']=grad_norm(terms['Lseg'],model)
        if 'regularizer' in terms:details['regularizer_gradient_norm']=grad_norm(terms['regularizer'],model)
        if queries is not None and bank is not None:details.update(diagnostics(*queries,bank))
    total.backward()
    grads=[p.grad.detach().float() for p in model.parameters() if p.grad is not None]
    norm=torch.stack([g.square().sum() for g in grads]).sum().sqrt()
    if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient before optimizer')
    category=charge(kind,task,step)
    append(log,dict(event='optimizer_attempt',step=step,category=category,code_commit=CODE))
    try:opt.step()
    except BaseException:
        append(log,dict(event='optimizer_failed',step=step,category=category));raise
    # All mutable training state commits only after optimizer success.
    scheduler.step()
    if bank is not None:bank.update(current.detach(),support)
    if reference is not None:ema_update(model,reference,.99)
    append(log,dict(event='committed',step=step+1,category=category,code_commit=CODE))
    details.update(total_loss=float(total.detach()),grad_norm=float(norm),lr=[p['lr'] for p in opt.param_groups])
    if bank is not None:details['prototype_support']=bank.supported.tolist()
    return details


def task_paths(task):
    p=RUN/'tasks'/task;p.mkdir(parents=True,exist_ok=True)
    return p


def train():
    setup();provenance()
    task=os.environ['EXEC_TASK'];backbone=os.environ['EXEC_BACKBONE'];domain=os.environ['EXEC_DOMAIN'];arm=os.environ['EXEC_ARM']
    folder=task_paths(task);device=torch.device('cuda:0')
    schedule,sha=schedule_for(domain)
    ds=dataset(domain,'train_labeled');assert ds.rows==schedule['rows']
    cache=[ds[i] for i in range(len(ds))]  # Only current-domain L payload enters this process.
    model,init_sha=build(backbone,arm=='D0',ASSETS,SOURCE);model.to(device).train()
    opt,scheduler=optimizer_for(model,backbone)
    bank=reference=None;step=0;elapsed=0.;attempts=0
    if arm in ('Q1','Q2','Q3','Q4'):
        bank=PrototypeBank(128 if backbone=='UNET_QUERY_128' else 384).to(device)
        reference=make_reference(model) if arm in ('Q2','Q3','Q4') else None
    latest=folder/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        assert state['config_digest']==CONFIG_SHA and state['data_schedule_digest']==sha
        assert state['code_commit']==CODE, 'repair requires explicit validated checkpoint rebinding'
        step=restore(state,model,opt,scheduler,reference,bank);elapsed=state['training_seconds'];attempts=state['physical_attempts']
    elif arm.startswith('Q') and arm!='QUERY_PREFIX':
        prefix=RUN/'tasks'/f'{backbone}__{domain}__QUERY_PREFIX'/'prefix.pt'
        state=torch.load(prefix,map_location='cpu',weights_only=False)
        assert state['global_step']==2000 and state['data_schedule_digest']==sha and state['code_commit']==CODE
        step=restore(state,model,opt,scheduler)
        if bank is not None:bank=initialize_bank(model,cache,backbone,device)
        if reference is not None:reference=make_reference(model)
        atomic(folder/'PREFIX_BINDING.json',dict(prefix_sha=file_sha(prefix),student_digest=tensor_sha(model.state_dict()),
            restored_step=step,scheduler_step=scheduler.last_epoch,**provenance()))
    end=2000 if arm=='QUERY_PREFIX' else 3000
    metadata=dict(**provenance(),data_schedule_digest=sha,model_initialization_digest=init_sha,
                  task=task,backbone=backbone,domain=domain,arm=arm)
    if not latest.exists():
        atomic(latest,snapshot(model,opt,scheduler,reference,bank,step,training_seconds=elapsed,
                               physical_attempts=attempts,**metadata),binary=True)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    deadline=json.loads((RUN/'SESSION.json').read_text())['deadline']
    try:
        while step<end and time.time()<deadline:
            x,y=load_batch(cache,schedule['steps'][step],device)
            info=transaction(model,opt,scheduler,bank,reference,x,y,arm,'formal',task,step,folder/'TASK_LEDGER.jsonl',
                             diagnose=(step+1)%100==0)
            step+=1;attempts+=1
            if step%100==0:
                append(folder/'DIAGNOSTICS.jsonl',dict(step=step,**info,wall_clock=time.time(),
                    training_seconds=elapsed+time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),
                    peak_reserved=torch.cuda.max_memory_reserved(),code_commit=CODE))
            if step%250==0 or step==end:
                atomic(latest,snapshot(model,opt,scheduler,reference,bank,step,
                    training_seconds=elapsed+time.monotonic()-start,physical_attempts=attempts,**metadata),binary=True)
                atomic(folder/'PROGRESS.json',dict(global_step=step,**metadata))
        duration=elapsed+time.monotonic()-start
        state=snapshot(model,opt,scheduler,reference,bank,step,training_seconds=duration,physical_attempts=attempts,**metadata)
        atomic(latest,state,binary=True)
        if step==end:
            target=folder/('prefix.pt' if arm=='QUERY_PREFIX' else 'final.pt')
            atomic(target,state,binary=True)
            atomic(folder/'TRAIN_DONE.json',dict(**metadata,status='COMPLETED',global_step=step,
                committed_updates=2000 if arm=='QUERY_PREFIX' else (3000 if arm=='D0' else 1000),
                training_seconds=duration,peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),checkpoint_bytes=target.stat().st_size,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        else:atomic(folder/'TIME_LIMIT.json',dict(**metadata,global_step=step,status='WALL_CLOCK_LIMIT'))
    except BaseException as error:
        atomic(folder/f'FAILURE_{time.time_ns()}.private.json',dict(**metadata,step=step,
            schedule=schedule['steps'][min(step,2999)],error=repr(error),traceback=traceback.format_exc()))
        if isinstance(error,FloatingPointError):
            atomic(folder/'forensic.pt',snapshot(model,opt,scheduler,reference,bank,step,
                training_seconds=elapsed+time.monotonic()-start,physical_attempts=attempts,**metadata),binary=True)
        raise


def qualify():
    setup();provenance();backbone=os.environ['EXEC_BACKBONE'];device=torch.device('cuda:0')
    folder=RUN/'qualification'/backbone;folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for dense in (True,False):
        model,sha=build(backbone,dense,ASSETS,SOURCE);model.to(device).train()
        opt,sch=optimizer_for(model,backbone)
        x=torch.rand(2,3,384,384,device=device);y=torch.zeros(2,384,384,device=device,dtype=torch.long)
        y[:,64:256,64:256]=1;y[:,128:224,128:224]=2
        bank=ref=None
        if not dense:
            bank=initialize_bank(model,[dict(image=x[i].cpu(),label=y[i].cpu()) for i in range(2)],backbone,device)
            ref=make_reference(model)
        arm='D0' if dense else 'Q3';torch.cuda.reset_peak_memory_stats()
        for step in range(2):
            transaction(model,opt,sch,bank,ref,x,y,arm,'synthetic',backbone+arm,step,folder/'TASK_LEDGER.jsonl',True)
            if step==0:
                atomic(folder/'recovery.pt',snapshot(model,opt,sch,ref,bank,1,**provenance()),binary=True)
                expected=tensor_sha(model.state_dict());rng=torch.cuda.get_rng_state().clone()
        restore(torch.load(folder/'recovery.pt',map_location='cpu',weights_only=False),model,opt,sch,ref,bank)
        assert tensor_sha(model.state_dict())==expected and torch.equal(torch.cuda.get_rng_state(),rng)
        assert sch.last_epoch==1
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            before=model(x);before=before if dense else before['semantic']
            del bank,ref
            after=model(x);after=after if dense else after['semantic']
            torch.testing.assert_close(before,after,rtol=0,atol=0)
        results.append(dict(arm=arm,peak_allocated=torch.cuda.max_memory_allocated(),
                            peak_reserved=torch.cuda.max_memory_reserved(),restore_equal=True,deployment_equal=True))
        del model,opt,sch,x,y,before,after;gc.collect();torch.cuda.empty_cache()
    atomic(folder/'PASSED.json',dict(**provenance(),results=results,synthetic_optimizer_calls=4))


def smoke():
    setup();provenance();backbone=os.environ['EXEC_BACKBONE'];device=torch.device('cuda:0')
    folder=RUN/'smoke'/backbone;folder.mkdir(parents=True,exist_ok=True)
    ds=dataset('RIM_ONE_r3','train_labeled');cache=[ds[i] for i in range(len(ds))]
    schedule,_=schedule_for('RIM_ONE_r3')
    for dense in (True,False):
        model,_=build(backbone,dense,ASSETS,SOURCE);model.to(device).train();opt,sch=optimizer_for(model,backbone)
        bank=None if dense else initialize_bank(model,cache,backbone,device)
        ref=None if dense else make_reference(model)
        for step in range(2):
            x,y=load_batch(cache,schedule['steps'][step],device)
            transaction(model,opt,sch,bank,ref,x,y,'D0' if dense else 'Q3','smoke',backbone,step,folder/'TASK_LEDGER.jsonl')
        del model,opt,sch,bank,ref;gc.collect();torch.cuda.empty_cache()
    atomic(folder/'PASSED.json',dict(**provenance(),optimizer_calls=4,discarded=True))


def evaluate():
    setup();provenance();task=os.environ['EXEC_TASK'];folder=task_paths(task)
    state=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    backbone,domain,arm=(state[k] for k in ('backbone','domain','arm'))
    assert state['global_step']==3000
    model,_=build(backbone,arm=='D0',ASSETS,SOURCE);model.load_state_dict(state['student']);model.cuda().eval()
    ds=dataset(domain,'val');values=[];equal=None
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(len(ds)):
            item=ds[i];x=item['image'][None].cuda();out=model(x)
            p=out if arm=='D0' else out['semantic']
            if i==0:
                state.pop('reference',None);state.pop('bank',None)
                other=model(x);other=other if arm=='D0' else other['semantic']
                equal=bool(torch.equal(p,other));assert equal
            values.append(image_dice(p[0].argmax(0).cpu(),item['label']))
        x=torch.zeros(1,3,384,384,device='cuda');times=[]
        for i in range(25):
            torch.cuda.synchronize();start=time.perf_counter();model(x);torch.cuda.synchronize()
            if i>=5:times.append((time.perf_counter()-start)*1000)
    scores={k:float(np.mean([v[k] for v in values if v[k] is not None]))
            if any(v[k] is not None for v in values) else None for k in ('rim','cup','macro','disc_union')}
    atomic(folder/'EVALUATION.json',dict(**provenance(),training_code_commit=state['code_commit'],task=task,
        backbone=backbone,domain=domain,arm=arm,**scores,supported_images=sum(v['macro'] is not None for v in values),
        latency_ms_mean=float(np.mean(times)),latency_ms_p95=float(np.percentile(times,95)),
        deployment_equal=equal,checkpoint_sha256=file_sha(folder/'final.pt')))


def gpu_free():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    return {int(a):int(b) for a,b in (line.split(',') for line in lines)}


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:return None


class ExistingChild:
    def __init__(self,pid,identity):self.pid,self.identity=pid,identity;self.returncode=None
    def poll(self):
        if self.identity is not None and process_identity(self.pid)==self.identity:return None
        self.returncode=255;return self.returncode


def spawn(mode, gpu, **extra):
    e=dict(os.environ);e.update(EXEC_MODE=mode,CUDA_VISIBLE_DEVICES=str(gpu),**{k:str(v) for k,v in extra.items()})
    active=RUN/'ACTIVE_CODE.json'
    cwd=Path(json.loads(active.read_text())['source']) if active.exists() else SOURCE
    pth=RUN/'logs'/f'{mode}_{extra.get("EXEC_TASK",extra.get("EXEC_BACKBONE","main"))}_{time.time_ns()}.log'
    pth.parent.mkdir(exist_ok=True)
    with pth.open('w') as log:
        p=subprocess.Popen(['python3.12','-'],executable=sys.executable,cwd=cwd,env=e,
                           stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True)
        p.stdin.write('import runpy;runpy.run_module("r1_12h.runner",run_name="__main__")\n');p.stdin.close()
    append(RUN/'PROCESS_LEDGER.jsonl',dict(pid=p.pid,identity=process_identity(p.pid),gpu=gpu,mode=mode,log=str(pth),**extra))
    return p


def aggregate(queue, final=False):
    evaluations=[];resources=[];statuses=[]
    for task in queue:
        folder=RUN/'tasks'/task['task'];done=folder/'TRAIN_DONE.json';ev=folder/'EVALUATION.json'
        row=dict(task=task['task'],status=task['status'],backbone=task['backbone'],domain=task['domain'],arm=task['arm'])
        if done.exists():row.update(json.loads(done.read_text()))
        elif (folder/'TIME_LIMIT.json').exists():row.update(json.loads((folder/'TIME_LIMIT.json').read_text()))
        elif (folder/'PROGRESS.json').exists():row.update(json.loads((folder/'PROGRESS.json').read_text()))
        statuses.append(row)
        if ev.exists():evaluations.append(json.loads(ev.read_text()))
        if done.exists():
            r=json.loads(done.read_text());r['updates_per_second']=r['committed_updates']/max(r['training_seconds'],1e-9)
            if ev.exists():r.update({k:v for k,v in json.loads(ev.read_text()).items() if k.startswith('latency') or k=='deployment_equal'})
            resources.append(r)
    def csv_write(name,rows):
        out=io.StringIO();fields=sorted({k for r in rows for k in r})
        if fields:
            writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        path=RUN/'reports'/name;path.parent.mkdir(exist_ok=True);path.write_text(out.getvalue())
    csv_write('RESULTS.csv',evaluations);csv_write('RESOURCE_REPORT.csv',resources)
    means={};deltas=[]
    for backbone in CONFIG['backbones']:
        for arm in ('D0','Q0','Q1','Q2','Q3','Q4'):
            rows=[r for r in evaluations if r['backbone']==backbone and r['arm']==arm and r['macro'] is not None]
            means[(backbone,arm)]=sum(r['macro'] for r in rows)/2 if len(rows)==2 else None
        for a,b in [('Q0','D0'),('Q3','Q0'),('Q3','Q1'),('Q3','Q2'),('Q3','Q4')]:
            x,y=means[(backbone,a)],means[(backbone,b)]
            deltas.append(dict(backbone=backbone,contrast=a+'-'+b,delta=None if x is None or y is None else x-y))
    csv_write('DELTAS.csv',deltas)
    progression={}
    for backbone in CONFIG['backbones']:
        per_domain=[]
        for domain in CONFIG['domains']:
            rows={r['arm']:r['macro'] for r in evaluations if r['backbone']==backbone and r['domain']==domain}
            per_domain.append(rows['Q3']-rows['Q0'] if rows.get('Q3') is not None and rows.get('Q0') is not None else None)
        x,y=means[(backbone,'Q3')],means[(backbone,'Q0')]
        progression[backbone]=dict(observe_only=True,per_domain_delta=per_domain,
            passes=None if x is None or y is None or None in per_domain else x-y>=.005 and min(per_domain)>=-.01)
    session=json.loads((RUN/'SESSION.json').read_text());budget=json.loads((RUN/'BUDGET.json').read_text()) if (RUN/'BUDGET.json').exists() else {}
    attempts=[]
    for p in (RUN/'tasks').glob('*/TASK_LEDGER.jsonl'):attempts.extend(events(p))
    committed=sum(r.get('committed_updates',max(0,r.get('global_step',0)-(2000 if r['arm'] not in ('D0','QUERY_PREFIX') else 0))) for r in statuses)
    report=dict(**provenance(),status='RUNNING' if not final else ('COMPLETED' if len(evaluations)==24 else 'SESSION_ENDED_INCOMPLETE'),
        session=session,planned_committed_updates=40000,actual_committed_updates=committed,
        failed_optimizer_attempts=sum(e['event']=='optimizer_failed' for e in attempts),
        recovery_replay_attempts=budget.get('recovery',0),synthetic_qualification_calls=budget.get('synthetic',0),
        l_only_smoke_calls=budget.get('smoke',0),endpoints_completed=len(evaluations),tasks=statuses,deltas=deltas,
        full_40000=committed==40000,progression=progression,
        scope='development seed and exposed validation; no independent confirmation')
    atomic(RUN/'reports/R1_12H_COMPLETION.json',report)
    (RUN/'reports/R1_12H_COMPLETION.md').write_text('# R1 sustained session\n\n'+
        f'Status: {report["status"]}. Endpoints: {len(evaluations)}/24. Committed updates: {committed}/40000.\n\n'+
        f'Synthetic calls: {budget.get("synthetic",0)}; smoke: {budget.get("smoke",0)}; recovery: {budget.get("recovery",0)}.\n\n'+
        'See RESULTS.csv, DELTAS.csv, RESOURCE_REPORT.csv and completion JSON for every endpoint and missing task. '
        'Performance thresholds are observe-only. R2/R3 remain unauthorized.\n')
    with (RUN/'reports/TASK_LEDGER.jsonl').open('w') as f:
        for item in statuses:f.write(json.dumps(item)+'\n')
    (RUN/'reports/PATCH_LOG.jsonl').touch(exist_ok=True)


def supervise():
    provenance();lock=(RUN/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (RUN/'SESSION.json').exists():
        assert (RUN/'DATA_BINDING.private.json').exists()
        atomic(RUN/'SESSION.json',dict(start=time.time(),deadline=time.time()+43200,**provenance()))
    deadline=json.loads((RUN/'SESSION.json').read_text())['deadline']
    for previous in events(RUN/'PROCESS_LEDGER.jsonl'):
        if previous['mode'] in ('qualify','smoke'):
            child=ExistingChild(previous['pid'],previous.get('identity'))
            while child.poll() is None:time.sleep(5)
    # Qualification failures consume their physical budget and are retried independently.
    readiness={}
    for backbone in CONFIG['backbones']:
        readiness[backbone]=False
        for mode in ('qualify','smoke'):
            passed=RUN/('qualification' if mode=='qualify' else 'smoke')/backbone/'PASSED.json'
            if passed.exists():continue
            for attempt in range(3):
                if time.time()>=deadline:break
                free=gpu_free();gpu=max(free,key=free.get)
                if free[gpu]<10000:time.sleep(30);continue
                p=spawn(mode,gpu,EXEC_BACKBONE=backbone);p.wait()
                append(RUN/'SESSION_LEDGER.jsonl',dict(mode=mode,backbone=backbone,attempt=attempt,returncode=p.returncode))
                if p.returncode==0:break
            if not passed.exists():break
        else:readiness[backbone]=True
    qpath=RUN/'QUEUE.json'
    if qpath.exists():queue=json.loads(qpath.read_text())
    else:
        queue=[dict(task=f'{b}__{d}__{a}',backbone=b,domain=d,arm=a,status='PENDING',attempts=0,eval_attempts=0)
               for a in CONFIG['task_order'] for b in CONFIG['backbones'] for d in CONFIG['domains']]
    active={};last_report=0
    records={r.get('EXEC_TASK'):r for r in events(RUN/'PROCESS_LEDGER.jsonl') if r['mode'] in ('train','evaluate')}
    for task in queue:
        if task['status'] in ('RUNNING','RUNNING_EVAL'):
            previous=records.get(task['task'])
            if previous:
                active[task['task']]=(ExistingChild(previous['pid'],previous.get('identity')),previous['gpu'],previous['mode'])
            else:task['status']='EVAL_PENDING' if task['status']=='RUNNING_EVAL' else 'PENDING'
    while time.time()<deadline or active:
        for task_id,(p,gpu,mode) in list(active.items()):
            if p.poll() is None:continue
            task=next(t for t in queue if t['task']==task_id);folder=RUN/'tasks'/task_id
            if mode=='train':
                if (folder/'TRAIN_DONE.json').exists():task['status']='COMPLETED' if task['arm']=='QUERY_PREFIX' else 'EVAL_PENDING'
                elif time.time()>=deadline:task['status']='WALL_CLOCK_LIMIT'
                else:task['status']='PENDING' if task['attempts']<3 else 'FAILED_ENGINEERING_CONTINUE_QUEUE'
            else:task['status']='COMPLETED' if (folder/'EVALUATION.json').exists() else ('EVAL_PENDING' if task['eval_attempts']<3 else 'FAILED_EVALUATION_CONTINUE_QUEUE')
            append(RUN/'SESSION_LEDGER.jsonl',dict(task=task_id,mode=mode,returncode=p.returncode,status=task['status']))
            del active[task_id]
        busy={g for _,g,_ in active.values()};free=gpu_free()
        for task in queue:
            if time.time()>=deadline:break
            if task['status'] not in ('PENDING','EVAL_PENDING'):continue
            if not readiness[task['backbone']]:
                task['status']='FAILED_QUALIFICATION_CONTINUE_QUEUE';continue
            if task['arm'] not in ('D0','QUERY_PREFIX'):
                prefix=next(t for t in queue if t['backbone']==task['backbone'] and t['domain']==task['domain'] and t['arm']=='QUERY_PREFIX')
                if prefix['status'].startswith('FAILED'):
                    task['status']='FAILED_PREFIX_DEPENDENCY';continue
                if prefix['status']!='COMPLETED':continue
            qual=json.loads((RUN/'qualification'/task['backbone']/'PASSED.json').read_text())
            needed=max(r['peak_reserved'] for r in qual['results'])/(1024**2)*1.35+2048
            candidates=[g for g,m in free.items() if g not in busy and m>=needed]
            if not candidates:continue
            gpu=max(candidates,key=free.get);mode='evaluate' if task['status']=='EVAL_PENDING' else 'train'
            task['eval_attempts' if mode=='evaluate' else 'attempts']+=1
            p=spawn(mode,gpu,EXEC_TASK=task['task'],EXEC_BACKBONE=task['backbone'],EXEC_DOMAIN=task['domain'],EXEC_ARM=task['arm'])
            active[task['task']]=(p,gpu,mode);busy.add(gpu);task['status']='RUNNING_EVAL' if mode=='evaluate' else 'RUNNING'
        atomic(qpath,queue)
        if time.time()-last_report>60:aggregate(queue);last_report=time.time()
        terminal=all(t['status'] not in ('PENDING','EVAL_PENDING','RUNNING','RUNNING_EVAL') for t in queue)
        if terminal and not active and all(t['status']=='COMPLETED' for t in queue):break
        if time.time()>=deadline and not active:break
        time.sleep(5)
    for t in queue:
        if t['status'] in ('PENDING','EVAL_PENDING'):t['status']='WALL_CLOCK_LIMIT'
    # Final evaluation may finish after the training window; it adds no optimizer calls.
    for t in queue:
        folder=RUN/'tasks'/t['task']
        if not (folder/'final.pt').exists() or (folder/'EVALUATION.json').exists():continue
        free=gpu_free();gpu=max(free,key=free.get)
        if free[gpu]<10000:continue
        p=spawn('evaluate',gpu,EXEC_TASK=t['task']);p.wait()
        if (folder/'EVALUATION.json').exists():t['status']='COMPLETED'
    atomic(qpath,queue);aggregate(queue,final=True)
    atomic(RUN/'FINAL.json',dict(status='SESSION_FINISHED',time=time.time(),**provenance()))


def service():
    """Keep the supervisor alive; it reattaches to surviving workers after a crash."""
    while not (RUN/'FINAL.json').exists():
        child=spawn('supervise',-1);code=child.wait()
        append(RUN/'WATCHDOG.jsonl',dict(returncode=code,supervisor_pid=child.pid))
        if (RUN/'FINAL.json').exists():break
        time.sleep(15)


if __name__=='__main__':
    mode=os.environ['EXEC_MODE']
    {'prepare':prepare,'qualify':qualify,'smoke':smoke,'train':train,'evaluate':evaluate,'supervise':supervise,'service':service}[mode]()
