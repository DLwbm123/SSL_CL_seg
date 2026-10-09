"""Audit completed sequential outputs and paired coverage differences; no model calls."""
import csv
import fcntl
import json
import math
import os
import statistics as st
from collections import Counter
from pathlib import Path


def read(path):return json.loads(path.read_text())
def events(path):return [json.loads(v) for v in path.read_text().splitlines()]
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def table(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main(root):
    lock=(root/'COORDINATOR.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(root/'FINAL.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    costs=read(root/'COSTS.json');caps=costs['physical'];assert sum(caps.values())==56905
    ledger=events(root/'PHYSICAL_LEDGER.jsonl');keys=lambda r:(r['event'],r['category'],r['key'],r['ordinal'])
    joint=Counter();counts=Counter();jobs=[];queries=probes=0;barrier=read(root/'ENDPOINT_LOCK.json')['time']
    for job in sorted((root/'jobs').iterdir()):
        assert read(job/'FINAL.json')['status']=='COMPLETE' and read(job/'PROCESS_EXIT.json')['exit_code']==0
        if (job/'COUNTS.json').exists():counts.update(read(job/'COUNTS.json'))
        if job.name.startswith('train'):assert read(job/'FINAL.json')['time']<barrier
        if job.name.startswith('evaluate'):assert read(job/'STARTED.json')['time']>barrier
        physical={}
        for label in ('PHYSICAL','QUERY','EXTRACTION','LINEAR_SOLVE'):
            path=job/(label+'_LEDGER.jsonl')
            if not path.exists():continue
            rows=events(path);attempts=Counter(json.dumps({k:v for k,v in r.items() if k!='event'},sort_keys=True) for r in rows if r['event']=='attempt');success=Counter(json.dumps({k:v for k,v in r.items() if k!='event'},sort_keys=True) for r in rows if r['event']=='success')
            assert attempts==success and all(v==1 for v in attempts.values()) and not any(r['event']=='failure' for r in rows)
            if label=='PHYSICAL':joint.update((job.name,*keys(r)) for r in rows);physical=dict(Counter(r['category'] for r in rows if r['event']=='success'))
            if label=='QUERY':queries+=len(attempts)
            if label=='EXTRACTION':probes+=len(attempts)
            if label=='LINEAR_SOLVE':assert len(attempts)==1
        jobs.append(dict(job=job.name,physical=physical,exit_code=0))
    assert len(jobs)==26 and joint==Counter((r['job'],*keys(r)) for r in ledger)
    assert Counter(r['category'] for r in ledger if r['event']=='success')==Counter(caps)
    assert counts==Counter(costs['counts']) and queries==792 and probes==2128 and counts['query_images']==3168
    data=read(root/'DEVELOPMENT_RESULTS.json');rows=data['rows'];lookup={(r['context'],r['stream'],r['method']):r for r in rows};assert len(rows)==len(lookup)==264
    traces=[]
    for r in rows:
        gain=.25*(r['mid_new']-r['new_entry'])+.75*(r['new']-r['new_entry']);penalty=max(0.,r['old_memory_reference']-r['old']-.005)
        assert abs(gain-r['gain'])<1e-14 and abs(penalty-r['forget_penalty'])<1e-14 and abs(gain-penalty-r['utility'])<1e-14
        src=root/f"jobs/train{r['context'][3:]}_{r['stream']}";trace=read(src/(r['method']+'_DECISIONS.private.json'))
        assert [t['step'] for t in trace]==[100,200] and [t['action'] for t in trace]==r['actions']
        for t in trace:
            p=t['probabilities'];assert len(p)==13 and all(math.isfinite(v) and v>=0 for v in p) and abs(sum(p)-1)<1e-6 and 0<=t['action']<13
            traces.append(dict(context=r['context'],stream=r['stream'],method=r['method'],step=t['step'],action=t['action'],probabilities=p))
    metrics=('new','old','utility','gain','forget_penalty','absolute_forgetting')
    methods=sorted({r['method'] for r in rows})
    for method in methods:
        for k in metrics:assert abs(st.mean(r[k] for r in rows if r['method']==method)-data['means'][method][k])<1e-14
    means=dict(data['means'])
    for mode in ('SAMPLE','ARGMAX'):
        for arm in ('WARM','CE','RL','DISTILL'):
            for k in metrics:assert abs(means[f'{mode}_{arm}'][k]-st.mean(means[f'{mode}_{arm}_{seed}'][k] for seed in (601,602)))<1e-14
    controls=('SAMPLE_WARM','SAMPLE_CE','SAMPLE_DISTILL','UNIFORM','GLOBAL','RIDGE','NN','TIME','OFF')
    delta={m:{k:means['SAMPLE_RL'][k]-means[m][k] for k in ('new','old','utility')} for m in controls}
    paired_gate={f'{a}_{seed}':means[f'SAMPLE_RL_{seed}']['utility']-means[f'SAMPLE_{a}_{seed}']['utility'] for a in ('WARM','CE','DISTILL') for seed in (601,602)}
    trade={m:(r['new']>=.002 and r['old']>=-.0025) or (r['old']>=.005 and r['new']>=-.0025) for m,r in delta.items()}
    passed=all(v>0 for v in paired_gate.values()) and all(v['utility']>=.0005 for v in delta.values()) and all(trade.values())
    assert delta==data['primary_deltas'] and paired_gate==data['paired_seed_utility'] and trade==data['practical_tradeoff']
    assert data['status']==('V111_POSITIVE_CANDIDATE_REQUIRES_CONFIRMATION' if passed else 'V111_NO_PRACTICAL_SEQUENTIAL_RL_GAIN')
    old=read(Path(read(root/'CONFIG.private.json')['v108'])/'DEVELOPMENT_RESULTS.json');prior={(r['context'],r['stream'],r['method']):r for r in old['rows']};assert len(prior)==240 and prior.keys()<lookup.keys()
    differences=[]
    for key in prior:
        r=lookup[key]
        prev=prior[key];assert r['new_entry']==prev['new_entry'] and r['old_memory_reference']==prev['old_memory_reference']
        differences.append(dict(context=key[0],stream=key[1],method=key[2],**{k:r[k]-prev[k] for k in metrics},old_actions=prev['actions'],new_actions=r['actions']))
    groups={m:[r for r in differences if r['method']==m] for m in methods if m in {r['method'] for r in differences}}
    for mode in ('SAMPLE','ARGMAX'):
        for arm in ('WARM','CE','RL','DISTILL'):groups[f'{mode}_{arm}']=[r for r in differences if r['method'].startswith(f'{mode}_{arm}_')]
    coverage=[dict(method=m,n=len(group),**{k:st.mean(r[k] for r in group) for k in metrics},positive_utility=sum(r['utility']>0 for r in group),negative_utility=sum(r['utility']<0 for r in group),tie_utility=sum(r['utility']==0 for r in group)) for m,group in groups.items()]
    paired=[]
    for stream in (3,4,5):
        for seed in (601,602):
            for arm in ('WARM','CE','DISTILL'):
                paired.append(dict(stream=stream,seed=seed,control=arm,**{k:st.mean(lookup[(f'dev{i}',stream,f'SAMPLE_RL_{seed}')][k]-lookup[(f'dev{i}',stream,f'SAMPLE_{arm}_{seed}')][k] for i in range(4)) for k in ('new','old','utility')}))
    for name,values in [('COVERAGE_PAIRED_RESULTS',differences),('COVERAGE_SUMMARY',coverage),('PAIRED_STREAM_RESULTS',paired)]:write(root/(name+'.json'),values);table(root/(name+'.csv'),values)
    write(root/'DECISION_TRACES.json',traces)
    write(root/'COMPLETION_AUDIT.json',dict(status='PASS',jobs=jobs,all_root_child_optimizer_keys_exact=True,optimizer_attempt_success_pairs=56905,failed_optimizer_calls=0,query_attempt_success_pairs=queries,probe_attempt_success_pairs=probes,all264_rewards_and_means_recomputed=True,all528_decision_probabilities_actions_valid=True,all240_paired_coverage_keys_reference_exact=True,all528_snapshots_before_queries=True,root_exit_code=0))
    print(json.dumps(dict(coverage_summary=coverage,paired_streams=paired)))


if __name__=='__main__':main(Path(os.environ['EXEC_RUN']))
