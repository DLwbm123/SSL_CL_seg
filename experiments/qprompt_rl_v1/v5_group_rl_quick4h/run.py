"""User-authorized four-hour screen; preserve prior work and its original clock."""
import csv
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT=Path(os.environ['EXEC_RUN'])
C=json.loads(Path(os.environ['EXEC_CONFIG']).read_text())
COMMAND=[sys.executable,'-c','import os,runpy;runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")']
DOMAINS=('RIM_ONE_r3','Drishti_GS')
HORIZONS=dict(zip(DOMAINS,(3200,2100)))
METHODS=('ORIGINAL','OFFLINE_25','GRPO_FS','PPO_MATCHED')
GPUS=(6,7,5)

def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)

def configure():
    from experiments.qprompt_rl_v1.v5_grpo_group_control import protocol as p
    p.DEVELOPMENT_SEEDS=(168,);p.CONFIRMATION_SEEDS=();p.CONTROLLERS=(401,)
    p.LEARNERS=('PPO_MATCHED','GRPO_FS');p.GROUPS=1;p.GROUP_SIZE=2
    # Worker admission uses this scope receipt instead of asserting full-V5 totals.
    p.budget=lambda:dict(protocol='V5_QUICK4H',performance_student_cap=74200,full_horizons=HORIZONS)

class UserTimeBudgetReached(RuntimeError):pass

def worker():
    configure()
    from experiments.qprompt_rl_v1.v5_grpo_group_control import worker as w
    original_update=w.v4.Ledger.update;original_call=w.v4.Ledger.call
    def update(ledger,t,*args,**kwargs):
        if time.time()>=C['training_deadline']:
            w.save_student(t,ROOT/'budget_stop.private.pt',reason='user four-hour budget')
            raise UserTimeBudgetReached('user training deadline; state preserved')
        return original_update(ledger,t,*args,**kwargs)
    def call(ledger,category,key,fn):
        if time.time()>=C['training_deadline']:raise UserTimeBudgetReached('user optimizer deadline')
        return original_call(ledger,category,key,fn)
    w.v4.Ledger.update=update;w.v4.Ledger.call=call
    def evaluate_file(path):
        remaining=C['hard_deadline']-60-time.time()
        if remaining<=0:raise UserTimeBudgetReached('evaluation time budget')
        env=dict(os.environ,EVAL_INPUT=str(path),EXEC_MODULE='experiments.qprompt_rl_v1.v5_grpo_group_control.evaluator')
        with (ROOT/'evaluator.log').open('a') as log:
            subprocess.run(COMMAND,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
    w.evaluate_file=evaluate_file
    try:
        with w.NativeOperations(ROOT/'operations'):w.main()
    except UserTimeBudgetReached as exc:
        previous=read(ROOT/'status.json');write(ROOT/'status.json',dict(previous,status='USER_TIME_BUDGET_STOP',reason=str(exc)))


def group_check():
    import torch
    from experiments.qprompt_rl_v1.v5_grpo_group_control import controller as q
    torch.set_num_threads(2);torch.manual_seed(518)
    model=q.Policy(q.ActorCritic().state_dict());opt=q.optimizers(model);x=torch.zeros(40)
    with torch.no_grad():
        dist=q.distribution(model.actor(x));row=dict(x=x,action=0,logp=float(dist.log_prob(torch.tensor(0))),probs=dist.probs.tolist(),value=0.,reward=1.)
    calls=[]
    def step(o,kind,i):calls.append((kind,i));o.step()
    diag=q.update(model,opt,[[dict(row) for _ in range(4)] for _ in range(2)],[0.,1.],'GRPO_FS',1.,519,step,group_size=2)
    assert len(calls)==16 and diag['loss_denominator']==8
    assert abs(diag['advantage_mean'])<1e-8 and abs(diag['advantage_std']-.5)<1e-8
    write(ROOT/'GROUP2_QUALIFICATION.json',dict(status='PASS',synthetic_actor_calls=16,student_calls=0,group_size=2,unchanged_G4_default=True,baseline_native_qualification_reused=690))


def nodes():
    result=[]
    # Full source168 calibration continues in its original processes/source snapshot.
    for d in DOMAINS:result.append(dict(id=f'cal_168_{d}',job='calibration',adopt=True,deps=[],domain=d,caps=dict(calibration=5*HORIZONS[d])))
    # Baselines can use the otherwise idle GPU before controller learning finishes.
    for m in ('ORIGINAL','OFFLINE_25'):
        for d in DOMAINS:result.append(endpoint_node(d,m))
    for d in DOMAINS:
        result.extend([dict(id=f'ref_0_{d}',job='reference',deps=[],domain=d,seed=168,group=0,caps=dict(reference=HORIZONS[d])),
                       dict(id=f'scale_{d}',job='scale',deps=[f'cal_168_{d}'],domain=d,caps=dict(critic_warmup=64))])
    for d in DOMAINS:
        for m in ('GRPO_FS','PPO_MATCHED'):
            result.append(dict(id=f'learn_401_{d}_{m}',job='learn',deps=[f'scale_{d}',f'ref_0_{d}'],domain=d,method=m,controller=401,caps=dict(development=2*HORIZONS[d],actor=16,**({'critic':16} if m=='PPO_MATCHED' else {}))))
            result.append(endpoint_node(d,m,[f'learn_401_{d}_{m}']))
    return result

def endpoint_node(d,m,deps=None):
    return dict(id=f'quick4h_168_{d}_{m}',job='endpoint',stage='quick4h',deps=deps or [],seed=168,domain=d,endpoint_methods=[[m,None if m in ('ORIGINAL','OFFLINE_25') else 401]],caps=dict(endpoint=HORIZONS[d]))

def alive(info):
    path=Path('/proc')/str(info['pid'])/'stat'
    if not path.exists():return False
    values=path.read_text().split()
    return values[2]!='Z' and values[21]==info['process_start_ticks']

def start(j,gpu):
    root=ROOT/'jobs'/j['id'];root.mkdir(exist_ok=False)
    conf=dict(C,**j,campaign=str(ROOT),run_root=str(root),gpu=gpu)
    write(root/'CONFIG.private.json',conf);write(root/'SESSION.json',dict(start=time.time(),original_start=C['original_start'],optimizer_deadline=C['training_deadline'],hard_deadline=C['hard_deadline'],fixed_caps=j['caps']))
    env=dict(os.environ,EXEC_RUN=str(root),EXEC_CONFIG=str(root/'CONFIG.private.json'),EXEC_MODULE='experiments.qprompt_rl_v1.v5_group_rl_quick4h.run',CUDA_VISIBLE_DEVICES=str(gpu))
    log=(root/'worker.log').open('a');cmd=['bash','./with_nas_storage.sh']+COMMAND
    proc=subprocess.Popen(cmd,cwd=Path(C['code'])/'experiments/lcrseg/scripts',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(job=j['id'],gpu=gpu,pid=proc.pid,time=time.time(),commit=C['commit'],process_start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21])
    write(root/'LAUNCH.json',info)
    return dict(info=info,process=proc,log=log,node=j)

def schedule(jobs,deadline,adopt=False):
    pending=[j for j in jobs if not j.get('adopt')];active={};done=set();failed=[];stopped=[]
    if adopt:
        for j in jobs:
            if not j.get('adopt'):continue
            root=ROOT/'jobs'/j['id'];info=read(root/'LAUNCH.json')
            if (root/'FINAL.json').exists():done.add(j['id'])
            elif alive(info):active[info['gpu']]=dict(info=info,process=None,log=None,node=j)
            else:failed.append(j['id'])
    while pending or active:
        now=time.time()
        for gpu,item in list(active.items()):
            root=ROOT/'jobs'/item['node']['id'];info=item['info'];proc=item['process']
            running=proc.poll() is None if proc is not None else alive(info)
            if running and now>=deadline:
                os.killpg(info['pid'],signal.SIGTERM)
                write(root/'USER_BUDGET_STOP.json',dict(time=now,original_deadline=C['hard_deadline'],checkpoint_preserved=True))
                stopped.append(info['job']);running=False
            if running:continue
            if proc is not None:
                try:code=proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(info['pid'],signal.SIGKILL);code=proc.wait(timeout=5)
                item['log'].close();write(root/'PROCESS_EXIT.json',dict(exit_code=code,time=time.time(),pid=info['pid']))
            if (root/'FINAL.json').exists():done.add(info['job'])
            elif info['job'] not in stopped:
                status=read(root/'status.json') if (root/'status.json').exists() else {}
                if status.get('status')=='USER_TIME_BUDGET_STOP':stopped.append(info['job'])
                else:failed.append(info['job'])
            del active[gpu]
        if time.time()<deadline and not failed and not stopped:
            for gpu in GPUS:
                if gpu in active:continue
                ready=next((j for j in pending if set(j['deps'])<=done),None)
                if ready is None:continue
                free=int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<6144:continue
                pending.remove(ready);active[gpu]=start(ready,gpu)
        write(ROOT/'QUEUE_STATUS.json',dict(time=time.time(),active=[v['info'] for v in active.values()],pending=[j['id'] for j in pending],completed=sorted(done),failed=failed,time_stopped=stopped,hard_deadline=C['hard_deadline']))
        if not active and (failed or stopped or time.time()>=deadline):break
        if not active and pending and not any(set(j['deps'])<=done for j in pending):raise RuntimeError('unresolvable dependency')
        if pending or active:time.sleep(2)
    return done,failed,stopped,[j['id'] for j in pending]

def ledger_counts(root):
    path=root/'PHYSICAL_LEDGER.jsonl';attempts=Counter();success=Counter()
    if path.exists():
        for line in path.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r['event']=='attempt':attempts[r['category']]+=1
            if r['event']=='success':success[r['category']]+=1
    return dict(attempts=dict(attempts),success=dict(success))

def report(jobs,done,failures,stops,pending,eval_done):
    rows=[]
    for name in sorted(eval_done):rows.extend(read(ROOT/'jobs'/name/'RESULTS.json'))
    if rows:
        with (ROOT/'SCREEN_RESULTS.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(ROOT/'SCREEN_RESULTS.json',rows)
    lookup={(r['domain'],r['method']):r for r in rows};pairs={}
    for reference in ('ORIGINAL','OFFLINE_25','PPO_MATCHED'):
        cells=[]
        for d in DOMAINS:
            if (d,'GRPO_FS') in lookup and (d,reference) in lookup:
                a,b=lookup[d,'GRPO_FS'],lookup[d,reference]
                cells.append(dict(domain=d,**{k:a[k]-b[k] for k in ('macro_Dice','old_REFUGE')}))
        pairs[reference]=dict(cells=cells,mean={k:mean(r[k] for r in cells) for k in ('macro_Dice','old_REFUGE')} if cells else None,complete_two_domains=len(cells)==2)
    audits=[];cost=Counter();operations=Counter();intervals=Counter();features=Counter();resources=[];evaluators=[]
    for root in sorted((ROOT/'jobs').iterdir()):
        counts=ledger_counts(root);cost.update(counts['attempts']);audits.append(dict(job=root.name,**counts,complete=(root/'FINAL.json').exists()))
        if (root/'operations'/'operation_counts.json').exists():operations.update(read(root/'operations'/'operation_counts.json')['counts'])
        if (root/'MEASURED_COSTS.json').exists():intervals.update(read(root/'MEASURED_COSTS.json'))
        if (root/'FEATURE_COST.jsonl').exists():
            for line in (root/'FEATURE_COST.jsonl').read_text().splitlines():
                v=json.loads(line);features.update(extractions=1,VJP=v['VJP'],virtual_previews=v['virtual_previews'])
        if (root/'FINAL.json').exists():
            final=read(root/'FINAL.json');launch=read(root/'LAUNCH.json');resources.append(dict(job=root.name,wall_seconds=final['time']-launch['time'],peak_cuda_allocated=final['peak_cuda_allocated']))
        for path in (root/'evaluation_costs').glob('*/RESOURCE.json'):evaluators.append(dict(job=root.name,endpoint=path.parent.name,resource=read(path),operations=read(path.parent/'operations'/'operation_counts.json')['counts']))
    retired=Path(C['parent_run'])/'jobs'/'cal_169_RIM_ONE_r3';retired_cost=ledger_counts(retired);cost.update(retired_cost['attempts'])
    prefix=Path(C['qualification_prefix']);operations.update(read(prefix/'operations'/'operation_counts.json')['counts'])
    student=sum(v for k,v in cost.items() if k in ('smoke','calibration','reference','development','endpoint'))
    write(ROOT/'ALL_COSTS.json',dict(charged_physical_calls=dict(cost),charged_student_updates=student,retired_user_scope_cost=retired_cost,synthetic_optimizer_calls=65916,additional_group2_check_calls=16,operation_counts_completed_scopes=dict(operations),measured_intervals=dict(intervals),feature_counts=dict(features),resources=resources,evaluators=evaluators,original_start=C['original_start'],wall_seconds_from_original_start=time.time()-C['original_start'],history_source_reused=24000,compute_note='CUDA intervals include host gaps; not exact kernel busy time. Retired/interrupted scopes remain charged.'))
    complete=len(rows)==8 and not failures and not stops and not pending
    write(ROOT/'COMPLETION_AUDIT.json',dict(status='COMPLETE_SCREEN' if complete else 'PARTIAL_SCREEN',endpoints=len(rows),required_endpoints=8,failures=failures,time_stopped=stops,unstarted=pending,jobs=audits,student_calls=student,retired_cost_preserved=True,val_barrier=True))
    write(ROOT/'DECISION.json',dict(status='EXPLORATORY_ONLY',paired_comparisons=pairs,automatic_promotion=False,stable_advantage_claim_allowed=False,multistep_credit_claim_allowed=False,segmentation_seeds=1,controller_seeds=1,group_size=2,groups=1))
    policy=[];reward=[]
    for root in sorted((ROOT/'jobs').glob('learn_*')):
        if (root/'GROUP_DIAGNOSTICS.json').exists():reward.append(dict(job=root.name,groups=read(root/'GROUP_DIAGNOSTICS.json')))
        if (root/'POLICY_PANEL.private.json').exists():policy.append(dict(job=root.name,groups=[{k:v for k,v in z.items() if k not in ('probabilities','top2_margin')} for z in read(root/'POLICY_PANEL.private.json')]))
    write(ROOT/'POLICY_DIAGNOSTICS.json',policy);write(ROOT/'REWARD_DIAGNOSTICS.json',reward)
    text='# V5 four-hour exploratory screen\n\n'+('All 8 full-horizon endpoints completed.\n\n' if complete else f'{len(rows)}/8 endpoints completed; unfinished cells are not endpoints.\n\n')
    text+='| Domain | Method | New Dice (%) | Old REFUGE (%) |\n|---|---|---:|---:|\n'
    for r in rows:text+=f"| {r['domain']} | {r['method']} | {r['macro_Dice']*100:.4f} | {r['old_REFUGE']*100:.4f} |\n"
    text+='\nOne source seed168, controller401, two domains, G=2 and one policy update group. This is a user-authorized budget revision, not completion of the original V5 pilot. No Bandit/reward-shuffle/repeated-controller evidence, no promotion and no independent-patient confirmation. Compare GRPO_FS with ORIGINAL, OFFLINE_25 and matched PPO separately in DECISION.json; probability or Q improvement alone is not Dice improvement. All consumed pre-amendment work remains charged.\n'
    (ROOT/'FINAL_INTERPRETATION.md').write_text(text)
    write(ROOT/'FINAL.json',dict(status='COMPLETE_PRIVATE' if complete else 'PARTIAL_PRIVATE',time=time.time(),endpoints=len(rows),hard_deadline=C['hard_deadline'],publication='pending GitHub verification'))

def coordinator():
    import fcntl
    lock=(ROOT/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'RUN_LOCK.json').exists(),'no automatic resume'
    assert C['hard_deadline']==C['original_start']+4*3600
    assert C['training_deadline']==C['hard_deadline']-900
    (ROOT/'jobs').mkdir(exist_ok=False);parent=Path(C['parent_run'])
    for name in ('qualification','init_RIM_ONE_r3','init_Drishti_GS','cal_168_RIM_ONE_r3','cal_168_Drishti_GS'):
        (ROOT/'jobs'/name).symlink_to(parent/'jobs'/name,target_is_directory=True)
    assert read(ROOT/'jobs'/'qualification'/'QUALIFICATION.json')['status']=='PASS'
    jobs=nodes();total=sum(sum(j['caps'].get(k,0) for k in ('calibration','reference','development','endpoint')) for j in jobs)
    assert total==74200
    write(ROOT/'RUN_LOCK.json',dict(protocol='V5_QUICK4H',commit=C['commit'],parent_execution='77cf65db6802ebbeaf2cd6b7c097fc11f0ebc2cc',original_start=C['original_start'],hard_deadline=C['hard_deadline'],training_deadline=C['training_deadline'],performance_student_cap=total,native_qualification_reused=690,retired_scope_additional_cost='charged separately, never reset',jobs=jobs))
    write(ROOT/'PROCESS.json',dict(pid=os.getpid(),time=time.time(),process_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    group_check()
    write(ROOT/'status.json',dict(status='RUNNING',phase='QUICK_SCREEN',time=time.time()))
    done,failed,stopped,pending=schedule(jobs,C['training_deadline'],adopt=True)
    endpoint_jobs=[j for j in jobs if j['job']=='endpoint' and j['id'] in done]
    write(ROOT/'QUICK4H_ENDPOINT_LOCK.json',dict(time=time.time(),jobs=[j['id'] for j in endpoint_jobs],complete_matrix=len(endpoint_jobs)==8,all_training_stopped_or_complete=True))
    evaluations=[dict(id='eval_'+j['id'],job='evaluate',stage='quick4h',target=str(ROOT/'jobs'/j['id']),deps=[],caps={}) for j in endpoint_jobs]
    ev_done,ev_failed,ev_stopped,ev_pending=schedule(evaluations,C['hard_deadline']-60)
    report(jobs,done,failed+ev_failed,stopped+ev_stopped,pending+ev_pending,ev_done)
    write(ROOT/'status.json',dict(status=read(ROOT/'FINAL.json')['status'],time=time.time()))

if __name__=='__main__':
    if C.get('job'):worker()
    else:
        try:coordinator()
        except BaseException as exc:
            write(ROOT/'status.json',dict(status='FAILED',time=time.time(),error=repr(exc),traceback=traceback.format_exc()));raise
