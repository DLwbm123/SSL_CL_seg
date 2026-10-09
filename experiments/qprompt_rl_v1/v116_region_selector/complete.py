"""Audit every operation and report the complete regional selector pilot."""
import csv
import fcntl
import json
import math
import os
import statistics as st
from collections import Counter
from pathlib import Path


def read(p):return json.loads(p.read_text())
def events(p):return [json.loads(v) for v in p.read_text().splitlines()]
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def table(p,rows):
    with p.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def decision(means):
    rl=means['RL'];controls=('CE','RANDOM','CONFIDENCE','COVERAGE','GLOBAL_REUSED','OFF_REUSED');comparisons=[]
    for name in controls:
        d={k:rl[k]-means[name][k] for k in ('new','old','utility')}
        comparisons.append(dict(control=name,**d,utility_pass=d['utility']>=.0005,practical_pass=(d['new']>=.002 and d['old']>=-.0025) or (d['old']>=.005 and d['new']>=-.0025)))
    seed_pass=all(means[f'RL_{s}']['utility']>means[f'CE_{s}']['utility'] for s in (601,602))
    relative=seed_pass and all(r['utility_pass'] and r['practical_pass'] for r in comparisons)
    absolute=all(means[f'RL_{s}']['weighted_new_gain']>0 and means[f'RL_{s}']['final_new_minus_entry']>0 and means[f'RL_{s}']['old_final_minus_entry']>=-.0025 for s in (601,602))
    return dict(status='PILOT_CANDIDATE_REQUIRES_MATCHED_TARGET_AND_FRESH_CONFIRMATION' if relative and absolute else 'NO_RELIABLE_REGION_SELECTION_SIGNAL',relative_gate=relative,absolute_gate=absolute,each_seed_beats_imitation=seed_pass,comparisons=comparisons,campaign_success=False)


def selfcheck():
    base=dict(new=.7,old=.8,utility=0.,weighted_new_gain=.01,final_new_minus_entry=.01,old_final_minus_entry=0.)
    names=['RL','CE','RL_601','RL_602','CE_601','CE_602','RANDOM','CONFIDENCE','COVERAGE','GLOBAL_REUSED','OFF_REUSED'];means={n:dict(base) for n in names}
    assert not decision(means)['relative_gate']
    for n in ('RL','RL_601','RL_602'):means[n].update(new=.703,utility=.003)
    assert decision(means)['relative_gate'] and decision(means)['absolute_gate'] and not decision(means)['campaign_success']
    means['RL_602']['final_new_minus_entry']=-.01;assert not decision(means)['absolute_gate']


def main(root):
    selfcheck();lock=(root/'COORDINATOR.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(root/'FINAL.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    caps=dict(qualification=8,prior=12800,development=16800,actor_qualification=2,selector_RL=64,selector_CE=64);costs=read(root/'COSTS.json');assert costs['physical']==costs['success']==caps and not costs['failures']
    assert costs['native_updates']==29608 and costs['actor_optimizer_updates']==130 and costs['linear_solves']==0
    actorbarrier=read(root/'ACTOR_LOCK.json')['time'];endpointbarrier=read(root/'ENDPOINT_LOCK.json')['time'];assert read(root/'ENDPOINT_LOCK.json')['snapshots']==168
    joint=Counter();counts=Counter();jobs=[];queries=rates=selections=0;selection_groups={};groups=[]
    for job in sorted((root/'jobs').iterdir()):
        assert not job.is_symlink() and read(job/'FINAL.json')['status']=='COMPLETE' and read(job/'PROCESS_EXIT.json')['exit_code']==0 and not (job/'FAILURE.json').exists()
        assert not (job/'RESET_LEDGER.jsonl').exists();counts.update(read(job/'COUNTS.json'))
        physical={}
        for label in ('PHYSICAL','QUERY'):
            path=job/(label+'_LEDGER.jsonl')
            if not path.exists():continue
            rows=events(path);identity=lambda r:json.dumps({k:v for k,v in r.items() if k!='event'},sort_keys=True)
            attempts=Counter(identity(r) for r in rows if r['event']=='attempt');success=Counter(identity(r) for r in rows if r['event']=='success')
            assert attempts==success and all(v==1 for v in attempts.values()) and not any(r['event']=='failure' for r in rows)
            if label=='PHYSICAL':
                joint.update(json.dumps(dict(r,job=job.name),sort_keys=True) for r in rows);physical=dict(Counter(r['category'] for r in rows if r['event']=='success'))
            else:queries+=len(attempts)
        native=physical.get('qualification',0)+physical.get('prior',0)+physical.get('development',0)
        if native:
            rr=events(job/'RATE_LEDGER.jsonl');ss=events(job/'SELECTION_LEDGER.jsonl');assert len(rr)==len(ss)==native
            assert Counter((r['category'],r['key']) for r in rr)==Counter((r['category'],r['key']) for r in ss)
            for i,(r,s) in enumerate(zip(rr,ss)):
                assert r['ordinal']==s['ordinal']==i+1 and r['step']==s['step'] and 100<=r['step']<300
                assert all(math.isfinite(a) and a>=0 and b==a*.25 for a,b in zip(r['base'],r['effective']))
                assert s['selected']==sum(s['classes'])==sum(b for a,b in s['image_budgets']) and s['eligible']==sum(a for a,b in s['image_budgets'])
                assert all(b==a//2 for a,b in s['image_budgets']) and 0<=s['tiles']<=16*s['images']
                key=(job.name,s['mode']);v=selection_groups.setdefault(key,Counter());v.update({k:s[k] for k in ('images','eligible','selected','tiles')});v['invocations']+=1
                for c in range(3):v[f'class{c}']+=s['classes'][c]
            rates+=len(rr);selections+=len(ss)
            if job.name=='qualification':
                strip=lambda r:{k:v for k,v in r.items() if k!='ordinal'}
                assert [strip(v) for v in ss[:2]]==[strip(v) for v in ss[2:4]] and [strip(v) for v in ss[4:6]]==[strip(v) for v in ss[6:8]]
        if job.name.startswith('learn'):
            assert read(job/'ACTOR_LOCK.json')['time']<actorbarrier
            gs=events(job/'GROUPS.jsonl');assert len(gs)==32
            for i,g in enumerate(gs):
                assert g['group']==i and g['context']==i%8 and g['cycle']==i//8 and g['entry_step']==100+50*(i//8) and g['retained_branch']==0 and len(g['branches'])==4
                for j,r in enumerate(g['branches']):
                    assert r['branch']==j and abs(r['reward']-(.25*(r['mid_new']-r['new_entry'])+.75*(r['new']-r['new_entry'])+r['old']-r['old_entry']))<1e-14
                assert g['actor']['winner']==max(range(4),key=lambda j:(g['branches'][j]['reward'],-j));groups.append(g)
        if job.name.startswith('train'):assert read(job/'STARTED.json')['time']>actorbarrier and read(job/'FINAL.json')['time']<endpointbarrier and len(list(job.glob('*_TRAINING.json')))==7
        if job.name.startswith('evaluate'):assert read(job/'STARTED.json')['time']>endpointbarrier
        jobs.append(dict(job=job.name,physical=physical,exit_code=0))
    assert len(jobs)==27 and joint==Counter(json.dumps(r,sort_keys=True) for r in events(root/'PHYSICAL_LEDGER.jsonl'))
    assert counts==Counter(costs['counts']) and rates==selections==29608 and queries==1148
    assert counts['query_Q_train_new']==2304 and counts['query_Q_train_old']==1280 and counts['query_Q_dev_new']==672 and counts['query_Q_dev_old']==336
    cfg=read(root/'CONFIG.private.json');rows=read(root/'DEVELOPMENT_RESULTS.json')['rows'];assert len(rows)==len({(r['context'],r['stream'],r['method']) for r in rows})==84
    entries={r['context']:r for r in read(Path(cfg['v112'])/'ENTRY_BASELINES.json')};baseline=[dict(r,method=r['method']+'_REUSED') for r in read(Path(cfg['v113'])/'DEVELOPMENT_RESULTS.json')['rows'] if r['method'] in ('GLOBAL','OFF')]
    extended=[]
    for r in rows+baseline:
        gain=.25*(r['mid_new']-r['new_entry'])+.75*(r['new']-r['new_entry']);assert abs(gain-r['gain'])<1e-14 and abs(gain-max(0.,r['old_memory_reference']-r['old']-.005)-r['utility'])<1e-14
        e=entries[r['context']];extended.append(dict(r,mid_new_minus_entry=r['mid_new']-r['new_entry'],final_new_minus_entry=r['new']-r['new_entry'],weighted_new_gain=gain,old_final_minus_entry=r['old']-e['old_entry'],old_final_minus_memory=r['old']-r['old_memory_reference'],utility_minus_frozen=r['utility']-e['frozen_utility']))
    metrics=('new','old','utility','mid_new_minus_entry','final_new_minus_entry','weighted_new_gain','old_final_minus_entry','old_final_minus_memory','utility_minus_frozen');means={}
    for method in sorted({r['method'] for r in extended})+['RL','CE']:
        rs=[r for r in extended if r['method']==method or (method in ('RL','CE') and r['method'].startswith(method+'_'))];means[method]={k:st.mean(r[k] for r in rs) for k in metrics}
    summary=[dict(method=m,n=24 if m in ('RL','CE') else 12,**v) for m,v in means.items()];out=decision(means)
    common={(r['context'],r['stream'],r['method']):r for r in extended};paired=[]
    for r in extended:
        if not r['method'].startswith('RL_'):continue
        for method in ('CE_'+r['method'].split('_')[1],'RANDOM','CONFIDENCE','COVERAGE','GLOBAL_REUSED','OFF_REUSED'):
            other=common[(r['context'],r['stream'],method)];paired.append(dict(context=r['context'],stream=r['stream'],method=r['method'],control=method,**{k:r[k]-other[k] for k in metrics}))
    for name,values in [('METHOD_SUMMARY',summary),('PAIRED_RESULTS',paired)]:write(root/(name+'.json'),values);table(root/(name+'.csv'),values)
    for name,key in [('CONTEXT_SUMMARY','context'),('STREAM_SUMMARY','stream')]:
        result=[]
        for val in sorted({r[key] for r in extended}):
            for method in means:
                rs=[r for r in extended if r[key]==val and (r['method']==method or (method in ('RL','CE') and r['method'].startswith(method+'_')))];result.append({key:val,'method':method,'n':len(rs),**{k:st.mean(r[k] for r in rs) for k in metrics}})
        write(root/(name+'.json'),result);table(root/(name+'.csv'),result)
    write(root/'ABSOLUTE_RESULTS.json',extended);write(root/'TRAINING_GROUPS.json',groups);write(root/'SELECTION_SUMMARY.json',[dict(job=k[0],method=k[1],**v) for k,v in selection_groups.items()]);write(root/'DECISION.json',out)
    write(root/'COMPLETION_AUDIT.json',dict(status='PASS',jobs=jobs,native_optimizer_pairs=29608,actor_optimizer_pairs=130,all_root_child_keys_exact=True,rate_records=rates,selection_records=selections,query_pairs=queries,query_images=4592,all_pixel_budgets_exact=True,all64_groups256_rewards_exact=True,all168_snapshots_before_Qdev=True,all_actors_sealed_before_deployment=True,campaign_success=False))
    print(json.dumps(dict(decision=out,summary=summary)))


if __name__=='__main__':main(Path(os.environ['EXEC_RUN']))
