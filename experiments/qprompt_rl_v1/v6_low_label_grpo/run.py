"""Resumable finite scientific matrix, without a wall-clock cutoff."""
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

DOMAINS = {'RIM_ONE_r3':3200, 'Drishti_GS':2100}
BASELINES = ('ORIGINAL','FINE_05','OFFLINE_25')
VARIANTS = [dict(variant=f'{m}_G{g}', method=m, group_size=g)
            for g in (4,2,8) for m in ('GRPO_FS','PPO_MATCHED')]
COMMAND = [sys.executable,'-c','import os,runpy;runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")']

def read(path): return json.loads(Path(path).read_text())
def write(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def matrix(c):
    jobs=[]
    def add(scope,name,kind,deps,caps,**kw):
        j=dict(id=scope+'__'+name,scope=scope,name=name,job=kind,deps=[scope+'__'+d for d in deps],caps=caps,**kw)
        jobs.append(j);return j
    for seed in c.get('subsets',[5101]):
        for pct in c.get('label_percents',[20,10,5]):
            scope=f'b{pct:02d}_s{seed}'
            for d,h in DOMAINS.items():
                common=dict(domain=d,seed=168,subset_seed=seed,label_percent=pct)
                add(scope,'qualification_'+d,'qualification',[],dict(smoke=8),**common)
                add(scope,f'panel_168_{d}','panel',['qualification_'+d],dict(panel=h+h//50*75),**common)
                add(scope,'init_'+d,'initialize',[f'panel_168_{d}'],dict(controller=256),**common)
                add(scope,f'cal_168_{d}','calibration',['init_'+d],dict(calibration=5*h),**common)
                add(scope,'scale_'+d,'scale',[f'cal_168_{d}'],dict(critic_warmup=64),**common)
                add(scope,'ref_0_'+d,'reference',['qualification_'+d],dict(reference=h),group=0,**common)
                def endpoint(m,controller=None,variant=None,deps=()):
                    key=variant or m
                    name=f'endpoint_{d}_{key}'+(f'_{controller}' if controller else '')
                    base=f'endpoint_{d}_ORIGINAL'
                    ds=list(deps)+(['init_'+d] if m=='OFFLINE_25' else [])
                    if m!='ORIGINAL':ds.append(base)
                    j=add(scope,name,'endpoint',ds,dict(endpoint=h),stage='v6',method=m,variant=key,
                          endpoint_methods=[[m,controller]],controller=controller,**common)
                    if m!='ORIGINAL':j['entry_job']=base
                    return name
                for m in BASELINES: endpoint(m,deps=['qualification_'+d])
                for controller in c.get('controllers',[401]):
                    for v in c.get('variants',VARIANTS):
                        m,g=v['method'],v['group_size'];assert g in (2,4,8)
                        n=c.get('trajectories',8);assert n%g==0
                        name=f"learn_{controller}_{d}_{v['variant']}"
                        # minibatches=G, four epochs: identical sample exposure and 4*N optimizer calls.
                        cap=dict(development=n*h,actor=4*n)
                        if m=='PPO_MATCHED':cap['critic']=4*n
                        add(scope,name,'learn',['scale_'+d,'ref_0_'+d],cap,controller=controller,**v,**common)
                        endpoint(m,controller,v['variant'],[name])
    quals=[j['id'] for j in jobs if j['job']=='qualification']
    for j in jobs:
        if j['job']!='qualification':j['deps']=sorted(set(j['deps']+quals))
    return jobs

def alive(info):
    path=Path('/proc')/str(info['pid'])/'stat'
    if not path.exists():return False
    a=path.read_text().split();return a[2]!='Z' and a[21]==info['process_start_ticks']

def jobroot(root,j):return root/'scopes'/j['scope']/'jobs'/j['name']

def start(root,c,j,gpu):
    dest=jobroot(root,j);dest.mkdir(exist_ok=False)
    scope=dest.parent.parent
    cfg=dict(c,**j,campaign=str(scope),v4_root=str(scope),run_root=str(dest),gpu=gpu)
    if j.get('entry_job'):cfg['shared_entry_path']=str(dest.parent/j['entry_job']/'ENTRY.private.pt')
    write(dest/'CONFIG.private.json',cfg);write(dest/'SESSION.json',dict(start=time.time(),wall_clock_limit=None,fixed_caps=j['caps']))
    env=dict(os.environ,EXEC_RUN=str(dest),EXEC_CONFIG=str(dest/'CONFIG.private.json'),
             EXEC_MODULE=c.get('worker_module','experiments.qprompt_rl_v1.v6_low_label_grpo.worker'),CUDA_VISIBLE_DEVICES=str(gpu))
    with (dest/'worker.log').open('a') as log:
        proc=subprocess.Popen(['bash','./with_nas_storage.sh']+COMMAND,cwd=Path(c['code'])/'experiments/lcrseg/scripts',
                              env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(pid=proc.pid,job=j['id'],gpu=gpu,time=time.time(),commit=c['commit'],
              process_start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21])
    write(dest/'LAUNCH.json',info);return dict(node=j,info=info,process=proc)

def schedule(root,c,jobs):
    done=set();pending=[];active={};failed=[]
    request=root/'RETRY_REQUEST.json';retry=read(request) if request.exists() else {}
    for j in jobs:
        dest=jobroot(root,j)
        if (dest/'FINAL.json').exists():done.add(j['id']);continue
        if (dest/'LAUNCH.json').exists():
            info=read(dest/'LAUNCH.json')
            if alive(info):
                assert info['gpu'] not in active
                active[info['gpu']]=dict(node=j,info=info,process=None);continue
        if dest.exists():
            if j['id'] in retry.get('jobs',[]):
                assert retry.get('reason'), 'technical retry must state diagnosed reason'
                archive=root/'failed_attempts';archive.mkdir(exist_ok=True)
                dest.rename(archive/(j['id']+'_'+str(time.time_ns())))
            else:failed.append(j['id']);continue
        pending.append(j)
    if retry:
        write(root/('RETRY_APPLIED_'+str(time.time_ns())+'.json'),retry);request.unlink()
    while active or pending:
        for gpu,item in list(active.items()):
            proc=item['process'];running=proc.poll() is None if proc is not None else alive(item['info'])
            if running:continue
            dest=jobroot(root,item['node'])
            write(dest/'PROCESS_EXIT.json',dict(time=time.time(),exit_code=proc.returncode if proc else None))
            if (dest/'FINAL.json').exists():done.add(item['node']['id'])
            else:failed.append(item['node']['id'])
            del active[gpu]
        if not failed:
            for gpu in (6,7,5):
                if gpu in active:continue
                ready=next((j for j in pending if set(j['deps'])<=done),None)
                if ready is None:continue
                free=int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<6144:continue
                pending.remove(ready);active[gpu]=start(root,c,ready,gpu)
        write(root/'QUEUE_STATUS.json',dict(time=time.time(),active=[x['info'] for x in active.values()],pending=[j['id'] for j in pending],completed=sorted(done),failed=failed,wall_clock_limit=None))
        if failed and not active:break
        if not active and pending and not any(set(j['deps'])<=done for j in pending):raise RuntimeError('unresolvable dependencies')
        if pending or active:time.sleep(5)
    return done,failed

def report(root,c,jobs,evaluations):
    rows=[];cost=Counter();audits=[];policies=[];rewards=[];intervals=Counter();eval_resources=[]
    for j in jobs+evaluations:
        dest=jobroot(root,j)
        assert (dest/'FINAL.json').exists()
        audits.append(dict(job=j['id'],**read(dest/'COMPLETION_AUDIT.json')))
        cost.update(read(dest/'FINAL.json')['physical_calls'])
        if (dest/'MEASURED_COSTS.json').exists():intervals.update(read(dest/'MEASURED_COSTS.json'))
        for rp in (dest/'evaluation_costs').glob('*/RESOURCE.json'):
            eval_resources.append(dict(job=j['id'],endpoint=rp.parent.name,**read(rp)))
        if j['job']=='evaluate':
            for r in read(dest/'RESULTS.json'):
                r.update(label_percent=j['label_percent'],subset_seed=j['subset_seed'],variant=j['variant']);rows.append(r)
        if (dest/'POLICY_PANEL.private.json').exists():
            for r in read(dest/'POLICY_PANEL.private.json'):
                policies.append(dict(job=j['id'],**{k:v for k,v in r.items() if k not in ('probabilities','top2_margin')}))
        if (dest/'GROUP_DIAGNOSTICS.json').exists():
            for r in read(dest/'GROUP_DIAGNOSTICS.json'):
                fields=('group','domain','controller','method','relative_rewards','reward_mean','reward_population_std','unique_sequences','advantage_mean','advantage_std','positive_active_fraction')
                rewards.append(dict(job=j['id'],**{k:r[k] for k in fields}))
    # Failed work is never erased from total consumed computation.
    retired=Counter();partial_lines=0
    for f in (root/'failed_attempts').glob('*/PHYSICAL_LEDGER.jsonl'):
        for line in f.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:partial_lines+=1;continue
            if r['event']=='attempt':retired[r['category']]+=1
    total=cost+retired
    write(root/'RESULTS.json',rows)
    with (root/'RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(root/'COSTS.json',dict(completed_calls=dict(cost),failed_attempt_calls=dict(retired),total_calls=dict(total),unparseable_failed_ledger_lines=partial_lines,synthetic_check=read(root/'CONTROLLER_CHECK.json'),measured_completed_intervals=dict(intervals),evaluators=eval_resources,note='CUDA intervals include host gaps, not exact kernel busy time; failed detail intervals may be incomplete.'))
    write(root/'POLICY_DIAGNOSTICS.json',policies);write(root/'REWARD_SUMMARY.json',rewards)
    early = read(root/'EARLY_EVALUATION_AMENDMENT.json') if (root/'EARLY_EVALUATION_AMENDMENT.json').exists() else None
    write(root/'COMPLETION_AUDIT.json',dict(status='PASS' if partial_lines==0 else 'COST_RECONCILIATION_REQUIRED',endpoints=len(rows),expected=len(evaluations),all_training_locked_before_val=early is None,early_evaluation_amendment=early,jobs=audits))
    comparisons=[]
    for r in rows:
        if r['method']!='GRPO_FS':continue
        same=[v for v in rows if (v['label_percent'],v['subset_seed'],v['domain'])==(r['label_percent'],r['subset_seed'],r['domain'])]
        controls=[v for v in same if v['method'] in BASELINES]
        g=r['variant'].split('_G')[-1]
        ppo=next(v for v in same if v['variant']=='PPO_MATCHED_G'+g and v['controller']==r['controller'])
        original=next(v for v in controls if v['method']=='ORIGINAL')
        comparisons.append(dict(label_percent=r['label_percent'],subset_seed=r['subset_seed'],domain=r['domain'],controller=r['controller'],variant=r['variant'],versus_best_nonRL=r['macro_Dice']-max(v['macro_Dice'] for v in controls),versus_PPO=r['macro_Dice']-ppo['macro_Dice'],old_versus_ORIGINAL=r['old_REFUGE']-original['old_REFUGE'],paired={v['method']:dict(new=r['macro_Dice']-v['macro_Dice'],old=r['old_REFUGE']-v['old_REFUGE']) for v in controls}))
    candidates=[]
    for variant in sorted({r['variant'] for r in comparisons}):
        low=[r for r in comparisons if r['variant']==variant and r['label_percent'] in (5,10)]
        full=[r for r in comparisons if r['variant']==variant and r['label_percent']==20]
        avg=mean(r['versus_best_nonRL'] for r in low);pp=mean(r['versus_PPO'] for r in low);old=mean(r['old_versus_ORIGINAL'] for r in low)
        repeat=[]
        for s,ct in sorted({(r['subset_seed'],r['controller']) for r in low}):
            z=[r for r in low if r['subset_seed']==s and r['controller']==ct]
            repeat.append(dict(subset_seed=s,controller=ct,new=mean(r['versus_best_nonRL'] for r in z),PPO=mean(r['versus_PPO'] for r in z)))
        gate=avg>=.005 and pp>=.002 and old>=-.005 and min(r['versus_best_nonRL'] for r in low)>=-.0025 and min(r['old_versus_ORIGINAL'] for r in low)>=-.01
        replicated=gate and len(repeat)>=4 and all(r['new']>0 and r['PPO']>0 for r in repeat)
        candidates.append(dict(variant=variant,low_label_cells=len(low),mean_new_vs_best_nonRL=avg,mean_new_vs_PPO=pp,mean_old_vs_ORIGINAL=old,min_new_vs_best_nonRL=min(r['versus_best_nonRL'] for r in low),interaction_low_minus_20=avg-mean(r['versus_best_nonRL'] for r in full),screen_gate=gate,replication_gate=replicated,repeats=repeat))
    candidates.sort(key=lambda r:r['mean_new_vs_best_nonRL'],reverse=True)
    write(root/'DECISION.json',dict(status='REPLICATED_POSITIVE' if any(v['replication_gate'] for v in candidates) else 'SCREEN_CANDIDATE' if any(v['screen_gate'] for v in candidates) else 'NO_POSITIVE_EVIDENCE',candidates=candidates,paired=comparisons,independent_patient_confirmation=False,publication='pending'))
    text='# V6 low-label GRPO results\n\nAll registered full-horizon endpoints completed. Single fixed REFUGE source; validation patients are historical development patients. Label-subset/controller repeats are not independent-patient confirmation.\n\n| Budget | Subset | Domain | Variant | Controller | New Dice % | Old Dice % |\n|---|---|---|---|---|---:|---:|\n'
    for r in rows:text+=f"| {r['label_percent']} | {r['subset_seed']} | {r['domain']} | {r['variant']} | {r['controller']} | {100*r['macro_Dice']:.4f} | {100*r['old_REFUGE']:.4f} |\n"
    text+='\nAll candidates, adverse cells and consumed costs are retained. See DECISION.json for paired differences, prespecified thresholds and the low-label interaction; no outcome changes the frozen endpoint scoring.\n'
    (root/'RESULT_REPORT.md').write_text(text)
    write(root/'FINAL.json',dict(status='COMPLETE_PRIVATE',time=time.time(),endpoints=len(rows),publication='pending GitHub verification'))

def coordinator(root,c):
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'FINAL.json').exists(), 'already complete'
    jobs=matrix(c)
    if (root/'RUN_LOCK.json').exists():assert read(root/'RUN_LOCK.json')['jobs']==jobs
    else:write(root/'RUN_LOCK.json',dict(jobs=jobs,wall_clock_limit=None,created=time.time()))
    for scope in sorted({j['scope'] for j in jobs}):
        base=root/'scopes'/scope/'jobs';base.mkdir(parents=True,exist_ok=True)
        for d in DOMAINS:
            aliases={f'fit_{d}':f'init_{d}',**{f'ref_{i}_{d}':f'ref_0_{d}' for i in range(1,c.get('trajectories',8))}}
            for name,target in aliases.items():
                path=base/name
                if not path.is_symlink():path.symlink_to(target,target_is_directory=True)
    write(root/'PROCESS.json',dict(pid=os.getpid(),time=time.time(),process_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    if not (root/'CONTROLLER_CHECK.json').exists():
        from .self_check import controller_check
        write(root/'CONTROLLER_CHECK.json',controller_check())
    write(root/'status.json',dict(status='RUNNING',phase='TRAIN',time=time.time()))
    done,failed=schedule(root,c,jobs)
    if failed:
        write(root/'status.json',dict(status='TECHNICAL_FAILURE',failed=failed,time=time.time()));return
    for scope in sorted({j['scope'] for j in jobs}):write(root/'scopes'/scope/'V6_ENDPOINT_LOCK.json',dict(time=time.time(),all_training_complete=True))
    evaluations=[]
    for j in jobs:
        if j['job']!='endpoint':continue
        evaluations.append(dict(j,id=j['scope']+'__eval_'+j['name'],name='eval_'+j['name'],job='evaluate',deps=[],caps={},target=str(jobroot(root,j))))
    write(root/'status.json',dict(status='RUNNING',phase='EVALUATE',time=time.time()))
    _,failed=schedule(root,c,evaluations)
    if failed:
        write(root/'status.json',dict(status='TECHNICAL_FAILURE',failed=failed,time=time.time()));return
    report(root,c,jobs,evaluations)
    write(root/'status.json',dict(status='COMPLETE_PRIVATE',time=time.time()))

if __name__=='__main__':
    r=Path(os.environ['EXEC_RUN']);config=read(os.environ['EXEC_CONFIG'])
    try:coordinator(r,config)
    except BaseException as exc:
        import traceback
        write(r/'status.json',dict(status='FAILED',time=time.time(),error=repr(exc),traceback=traceback.format_exc()));raise
