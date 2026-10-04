"""Finite authorized stages with global freeze barriers and no retries/deadlines."""
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from statistics import mean
from . import protocol as p

ROOT=Path(os.environ['EXEC_RUN'])
C=json.loads(Path(os.environ['EXEC_CONFIG']).read_text())
COMMAND=[sys.executable,'-c','import os,runpy;runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")']

def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)
def read(path):return json.loads(path.read_text())
def csvwrite(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run_phase(name,queue):
    pending=list(queue);active={};failed=[];last_sample=0
    while pending or active:
        for gpu in p.GPUS:
            if gpu in active or not pending or failed:continue
            free=int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<6144:continue
            j=pending.pop(0);root=ROOT/'jobs'/j['id'];root.mkdir(exist_ok=False)
            conf=dict(C,**j,campaign=str(ROOT),run_root=str(root),gpu=gpu)
            write(root/'CONFIG.private.json',conf);write(root/'SESSION.json',dict(start=time.time(),optimizer_deadline=None,hard_deadline=None,fixed_caps=j['caps'],retries=0))
            env=dict(os.environ,EXEC_RUN=str(root),EXEC_CONFIG=str(root/'CONFIG.private.json'),EXEC_MODULE='experiments.qprompt_rl_v1.v5_grpo_group_control.worker',CUDA_VISIBLE_DEVICES=str(gpu))
            log=(root/'worker.log').open('a');cmd=['bash','./with_nas_storage.sh']+COMMAND
            process=subprocess.Popen(cmd,cwd=Path(C['code'])/'experiments/lcrseg/scripts',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            info=dict(job=j['id'],gpu=gpu,pid=process.pid,time=time.time(),initial_free_mib=free,commit=C['commit'],command=cmd,process_start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21])
            write(root/'LAUNCH.json',info);active[gpu]=(process,log,root,info)
        now=time.time()
        if active and now-last_sample>=30:
            # In-run cost instrumentation; no independent monitoring automation.
            result=subprocess.run(['nvidia-smi','pmon','-i','5,6,7','-s','um','-c','1'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            with (ROOT/'GPU_UTILIZATION.private.jsonl').open('a') as f:f.write(json.dumps(dict(time=now,exit_code=result.returncode,active=[v[3] for v in active.values()],sample=result.stdout))+'\n')
            last_sample=now
        for gpu,(proc,log,root,info) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();write(root/'PROCESS_EXIT.json',dict(exit_code=code,time=time.time(),pid=proc.pid))
            if code!=0 or not (root/'FINAL.json').exists():failed.append(info['job'])
            del active[gpu]
        write(ROOT/'QUEUE_STATUS.json',dict(phase=name,time=time.time(),active=[v[3] for v in active.values()],pending=[j['id'] for j in pending],failed=failed))
        if failed and not active:raise RuntimeError('Engineering failure; no retries/dependent launches: '+', '.join(failed))
        if pending or active:time.sleep(3)
    write(ROOT/(name.upper()+'_PHASE_COMPLETE.json'),dict(time=time.time(),jobs=[j['id'] for j in queue],commit=C['commit']))

def evaluation(stage,queue):
    assert all(read(ROOT/'jobs'/j['id']/'FREEZE.json')['status']=='FROZEN' for j in queue)
    write(ROOT/(stage.upper()+'_ENDPOINT_LOCK.json'),dict(time=time.time(),jobs=[j['id'] for j in queue],commit=C['commit'],all_models_frozen=True,controller_updates=0))
    jobs=[dict(id='eval_'+j['id'],job='evaluate',stage=stage,target=str(ROOT/'jobs'/j['id']),caps={}) for j in queue]
    run_phase(stage+'_evaluation',jobs)
    rows=[r for j in jobs for r in read(ROOT/'jobs'/j['id']/'RESULTS.json')]
    assert len(rows)==sum(len(p.endpoint_methods(stage)) for j in queue)
    csvwrite(ROOT/(stage.upper()+'_RESULTS.csv'),rows)
    write(ROOT/(stage.upper()+'_RESULTS.json'),rows)
    return rows

def clipping_gate():
    values={}
    for d in p.DOMAINS:
        rows=read(ROOT/'jobs'/f'learn_401_{d}_GRPO_FS'/'GROUP_DIAGNOSTICS.json')
        values[d]=[r['group'] for r in rows if r['positive_active_fraction']>=.05]
    trigger=any(len(v)>=2 for v in values.values())
    write(ROOT/'CLIPHI_TRIGGER.json',dict(status='TRIGGERED' if trigger else 'NOT_TRIGGERED',groups=values,criterion='positive-advantage actual upper-clipped fraction >= .05 in at least 2 groups in one domain; pooled across update opportunities',main_candidate_unchanged=True))
    return trigger

def report(pilot,promotion,confirmation=None,cliphi=None):
    allrows=confirmation if confirmation is not None else pilot
    decision=p.confirmation_decision(confirmation) if confirmation is not None else dict(practical_candidate=None,grpo_advantage_supported=None,multistep_control_supported=None,controller_reproduction_supported=None,status='PILOT_NOT_PROMOTED',confirmation_run=False)
    decision.update(primary='GRPO_FS',promotion=promotion,conditional_cliphi_run=cliphi is not None,no_candidate_replacement=True)
    write(ROOT/'DECISION.json',decision)
    summary=[]
    for method in sorted({r['method'] for r in allrows}):
        selected=[r for r in allrows if r['method']==method]
        summary.append(dict(method=method,**{k:mean(r[k] for r in selected) for k in ('macro_Dice','old_REFUGE','old_change')},cells=len(selected)))
    csvwrite(ROOT/'SUMMARY.csv',summary)
    costs=Counter();ops=Counter();measured=Counter();resources=[];audits=[];policy=[];rewards=[];synthetic={};evaluator=[];features=Counter()
    for root in sorted((ROOT/'jobs').iterdir()):
        a=read(root/'COMPLETION_AUDIT.json');assert a['status']=='PASS';audits.append(dict(job=root.name,**a));costs.update(a['physical_calls'])
        ops.update(read(root/'operations'/'operation_counts.json')['counts']);measured.update(read(root/'MEASURED_COSTS.json'))
        final,launch=read(root/'FINAL.json'),read(root/'LAUNCH.json')
        resources.append(dict(job=root.name,gpu=launch['gpu'],wall_seconds=final['time']-launch['time'],peak_cuda_allocated=final['peak_cuda_allocated']))
        if (root/'SYNTHETIC_CHECK.json').exists():synthetic=read(root/'SYNTHETIC_CHECK.json')
        if (root/'FEATURE_COST.jsonl').exists():
            for line in (root/'FEATURE_COST.jsonl').read_text().splitlines():
                row=json.loads(line);features.update(extractions=1,VJP=row['VJP'],virtual_previews=row['virtual_previews'])
        if (root/'GROUP_DIAGNOSTICS.json').exists():
            groups=read(root/'GROUP_DIAGNOSTICS.json');rewards.append(dict(job=root.name,groups=groups))
            panels=read(root/'POLICY_PANEL.private.json')
            # Publish only aggregate panel diagnostics; per-state probabilities stay private.
            policy.append(dict(job=root.name,groups=[{k:v for k,v in r.items() if k not in ('probabilities','top2_margin')} for r in panels]))
        for path in (root/'evaluation_costs').glob('*/RESOURCE.json'):
            resource=read(path);operation=read(path.parent/'operations'/'operation_counts.json')['counts']
            evaluator.append(dict(job=root.name,endpoint=path.parent.name,resources=resource,operations=operation))
        if (root/'V4_REWARD_DIAGNOSTICS.json').exists():rewards.append(dict(job=root.name,V4_panel=read(root/'V4_REWARD_DIAGNOSTICS.json')))
    student=sum(v for k,v in costs.items() if k in ('smoke','calibration','reference','development','endpoint','source'))
    expected=p.budget()['full_without_P0'] if confirmation is not None else p.budget()['pilot_without_P0']
    expected+=p.P0_CALLS+(143100 if cliphi is not None else 0)
    assert student==expected,(student,expected)
    synthetic_calls=sum(synthetic['optimizer_calls'].values())+268
    assert ops['optimizer_steps']==sum(costs.values())+synthetic_calls
    write(ROOT/'POLICY_DIAGNOSTICS.json',policy);write(ROOT/'REWARD_DIAGNOSTICS.json',rewards)
    write(ROOT/'ALL_COSTS.json',dict(physical_calls=dict(costs),student_calls=student,synthetic_calls=synthetic_calls,synthetic_detail=synthetic['optimizer_calls'],historical_source_updates_reused=24000,operation_counts=dict(ops),measured_intervals=dict(measured),feature_counts=dict(features),job_resources=resources,evaluator=evaluator,wall_seconds=time.time()-read(ROOT/'SESSION.json')['start'],GPU_measurement='per-process CUDA command intervals including host gaps; pmon samples private, not exact kernel occupancy/energy; other GPU jobs not charged',fairness='matched student rollouts/reward opportunities; PPO/Bandit additional critic calls explicitly charged'))
    write(ROOT/'COMPLETION_AUDIT.json',dict(status='PASS',jobs=audits,student_calls=student,expected=expected,pilot_endpoints=len(pilot),confirmation_endpoints=len(confirmation) if confirmation is not None else 0,cliphi_endpoints=len(cliphi) if cliphi is not None else 0,global_val_barriers=True,endpoint_controller_updates=0,no_retries=True,test_sealed=True,hidden_U_labels_read=False))
    text='# V5_GROUP_RL results\n\n'
    text+='Pilot completed with 54 endpoints. Promotion: '+str(promotion['promoted'])+'. '
    text+='Confirmation completed with 210 endpoints.\n\n' if confirmation is not None else 'No independent-controller or new-source confirmation was authorized by the pilot gate.\n\n'
    text+='| Method | New Dice (%) | Old REFUGE Dice (%) | Old change (pp) |\n|---|---:|---:|---:|\n'
    for r in summary:text+=f"| {r['method']} | {100*r['macro_Dice']:.4f} | {100*r['old_REFUGE']:.4f} | {100*r['old_change']:.4f} |\n"
    text+='\n## Required interpretation\n\n1. Probability/KL/entropy changes and argmax agreement are in POLICY_DIAGNOSTICS; action changes alone do not prove value. Actual full-horizon endpoint and phase-shuffle schedules are retained.\n2. REWARD_DIAGNOSTICS contains online and descriptive audit changes/rankings; pilot val comparisons are in PROMOTION_DECISION. Q units are not Dice.\n3. Supervised ORIGINAL comparison is mandatory and reported even when strong-U comparisons improve.\n4. Same-student-budget PPO comparison is separate from practical benefit.\n5. Bandit, group reward shuffle and within-phase action shuffle comparisons address different alternatives; do not conflate them.\n6. All development seeds/domains retained; controller and new-source reproducibility is unestablished unless conditional confirmation ran and its gates passed. Five source seeds are optimization seeds on reused val patients, not independent patients.\n7. ALL_COSTS separates real student updates, historical source reuse, actor/critic calls, features/VJP/virtual Adam, actual label/sample accesses, CUDA intervals, wall and peaks. Matching deployment horizons does not mean equal total cost.\n\n'
    text+='Frozen decision: `'+json.dumps(decision)+'`.\n\nOriginal KI identity remains unverified; this is an incremental V4-base experiment. No failed result implies all RL is ineffective. Raw private data/checkpoints/roles/case scores stay on NAS. Publication is pending separate GitHub verification.\n'
    (ROOT/'FINAL_INTERPRETATION.md').write_text(text)

def main():
    lock=(ROOT/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'RUN_LOCK.json').exists(),'no automatic resume'
    (ROOT/'jobs').mkdir(exist_ok=False);(ROOT/'sources').mkdir(exist_ok=False)
    write(ROOT/'RUN_LOCK.json',dict(protocol='V5_GROUP_RL',commit=C['commit'],baseline=p.BASELINE,budget=p.budget(),optimizer_deadline=None,hard_deadline=None,retries=0))
    write(ROOT/'PROCESS.json',dict(pid=os.getpid(),time=time.time(),process_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    matrix=p.jobs()
    try:
        for stage in ('qualification','initialize','calibration','references','scale','learn401'):
            write(ROOT/'status.json',dict(status='RUNNING',phase=stage,time=time.time()));run_phase(stage,matrix[stage])
            if stage=='qualification':assert read(ROOT/'jobs'/'qualification'/'QUALIFICATION.json')['status']=='PASS'
        trigger=clipping_gate()
        write(ROOT/'PILOT_POLICY_LOCK.json',dict(time=time.time(),controllers=[401],selection='group6 final; no feedback-based checkpoint choice',commit=C['commit']))
        run_phase('pilot',matrix['pilot']);pilot=evaluation('pilot',matrix['pilot'])
        promotion=p.pilot_decision(pilot,True);write(ROOT/'PROMOTION_DECISION.json',promotion)
        cliphi=None
        if trigger:
            run_phase('cliphi_learn',matrix['cliphi_learn']);run_phase('cliphi',matrix['cliphi']);cliphi=evaluation('cliphi',matrix['cliphi'])
        confirmation=None
        if promotion['promoted']:
            run_phase('extension',matrix['extension']);run_phase('sources',matrix['sources'])
            write(ROOT/'CONFIRMATION_POLICY_LOCK.json',dict(time=time.time(),controllers=p.CONTROLLERS,selection='all three group6 final policies; no ensemble',commit=C['commit']))
            run_phase('confirmation',matrix['confirmation']);confirmation=evaluation('confirmation',matrix['confirmation'])
        report(pilot,promotion,confirmation,cliphi)
        write(ROOT/'FINAL.json',dict(status='COMPLETE_PRIVATE',time=time.time(),commit=C['commit'],publication='pending verification and GitHub delivery'))
        write(ROOT/'status.json',dict(status='COMPLETE_PRIVATE',time=time.time()))
    except BaseException as exc:
        write(ROOT/'status.json',dict(status='FAILED',time=time.time(),error=repr(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
