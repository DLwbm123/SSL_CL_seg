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
    recovered=(root/'METADATA_RECOVERY.json').exists()
    if recovered:
        assert read(root/'METADATA_RECOVERY.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==1 and read(root/'FAILURE.json')['error']=="FileExistsError(17, 'File exists')"
    else:assert read(root/'FINAL.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    costs=read(root/('COMPLETED_COSTS.json' if recovered else 'COSTS.json'));caps=costs['physical'];assert sum(caps.values())==7208
    ledger=events(root/'PHYSICAL_LEDGER.jsonl');keys=lambda r:(r['event'],r['category'],r['key'],r['ordinal'])
    joint=Counter();counts=Counter();jobs=[];queries=probes=0;barrier=read(root/'ENDPOINT_LOCK.json')['time']
    for job in sorted((root/'jobs').iterdir()):
        if job.is_symlink():continue
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
    assert len(jobs)==25 and joint==Counter((r['job'],*keys(r)) for r in ledger)
    assert Counter(r['category'] for r in ledger if r['event']=='success')==Counter(caps)
    assert counts==Counter(costs['counts']) and queries==108 and probes==304 and counts['query_images']==432
    cfg=read(root/'CONFIG.private.json');rows=read(root/'DEVELOPMENT_RESULTS.json')['rows'];assert len(rows)==36
    old={(r['context'],r['stream'],r['method']):r for r in read(Path(cfg['v111'])/'DEVELOPMENT_RESULTS.json')['rows']}
    entries={r['context']:r for r in read(Path(cfg['v112'])/'ENTRY_BASELINES.json')};paired=[]
    for r in rows:
        prev=old[(r['context'],r['stream'],r['method'])];assert r['actions']==prev['actions'] and r['new_entry']==prev['new_entry'] and r['old_memory_reference']==prev['old_memory_reference']
        gain=.25*(r['mid_new']-r['new_entry'])+.75*(r['new']-r['new_entry']);penalty=max(0.,r['old_memory_reference']-r['old']-.005)
        assert abs(gain-r['gain'])<1e-14 and abs(gain-penalty-r['utility'])<1e-14
        e=entries[r['context']]
        paired.append(dict(context=r['context'],stream=r['stream'],method=r['method'],new=r['new'],old=r['old'],utility=r['utility'],new_difference=r['new']-prev['new'],old_difference=r['old']-prev['old'],utility_difference=r['utility']-prev['utility'],mid_new_minus_entry=r['mid_new']-r['new_entry'],final_new_minus_entry=r['new']-r['new_entry'],weighted_new_gain=gain,old_final_minus_entry=r['old']-e['old_entry'],old_final_minus_memory=r['old']-r['old_memory_reference'],utility_minus_frozen=r['utility']-e['frozen_utility']))
    metrics=[k for k in paired[0] if k not in ('context','stream','method')];groups={m:[r for r in paired if r['method']==m] for m in ('OFF','GLOBAL','TIME')};groups['POOLED']=paired
    summary=[dict(method=m,n=len(rs),**{k:st.mean(r[k] for r in rs) for k in metrics}) for m,rs in groups.items()]
    ratecalls=0
    for job in (root/'jobs').iterdir():
        if job.is_symlink() or not (job/'RATE_LEDGER.jsonl').exists():continue
        rates=events(job/'RATE_LEDGER.jsonl');physical=[r for r in events(job/'PHYSICAL_LEDGER.jsonl') if r['event']=='success'];assert len(rates)==len(physical)
        assert Counter((r['category'],r['key']) for r in rates)==Counter((r['category'],r['key']) for r in physical)
        for i,r in enumerate(rates):assert r['ordinal']==i+1 and 100<=r['step']<300 and all(math.isfinite(a) and math.isfinite(b) and a>=0 and b==a*.25 for a,b in zip(r['base'],r['effective']))
        ratecalls+=len(rates)
    assert ratecalls==7208
    pooled=summary[-1];signal=pooled['utility_difference']>0 and pooled['new_difference']>=.002 and pooled['old_difference']>=-.0025
    for name,values in [('PAIRED_RESULTS',paired),('METHOD_SUMMARY',summary)]:write(root/(name+'.json'),values);table(root/(name+'.csv'),values)
    decision=dict(status='RATE_SENSITIVITY_SIGNAL' if signal else 'NO_BROAD_RATE_SENSITIVITY_SIGNAL',positive_absolute_policies=[r['method'] for r in summary[:-1] if r['weighted_new_gain']>0 and r['final_new_minus_entry']>0],campaign_success=False)
    write(root/'DECISION.json',decision)
    write(root/'COMPLETION_AUDIT.json',dict(status='PASS',jobs=jobs,all_root_child_optimizer_keys_exact=True,optimizer_attempt_success_pairs=7208,rate_invocations=7208,query_attempt_success_pairs=queries,probe_attempt_success_pairs=probes,all36_paired_actions_entry_references_exact=True,all72_snapshots_before_queries=True,original_root_exit_code=read(root/'EXIT.json')['exit_code'],metadata_recovery=recovered))
    print(json.dumps(dict(decision=decision,summary=summary)))


if __name__=='__main__':main(Path(os.environ['EXEC_RUN']))
