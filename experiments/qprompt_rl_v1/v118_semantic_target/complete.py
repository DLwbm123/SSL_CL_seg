"""Zero-model completion audit of all physical pairs and all training-only outputs."""
import fcntl
import importlib.util
import json
import math
import os
from collections import Counter
from pathlib import Path


def read(p):return json.loads(p.read_text())
def events(p):return [json.loads(v) for v in p.read_text().splitlines()]


def main(root):
    lock=(root/'COORDINATOR.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(root/'FINAL.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    costs=read(root/'COSTS.json');assert costs['physical']==costs['success']==dict(qualification=12,prior=9600) and not costs['failures']
    joint=Counter();total=Counter();queries=rates=selections=prototypes=0;jobnames=[]
    for job in sorted((root/'jobs').iterdir()):
        jobnames.append(job.name);assert not job.is_symlink() and read(job/'FINAL.json')['status']=='COMPLETE' and read(job/'PROCESS_EXIT.json')['exit_code']==0 and not (job/'FAILURE.json').exists()
        assert not (job/'RESET_LEDGER.jsonl').exists();total.update(read(job/'COUNTS.json'))
        physical={}
        for label in ('PHYSICAL','QUERY'):
            path=job/(label+'_LEDGER.jsonl')
            if not path.exists():continue
            ev=events(path);ident=lambda r:json.dumps({k:v for k,v in r.items() if k!='event'},sort_keys=True)
            aa=Counter(ident(r) for r in ev if r['event']=='attempt');ss=Counter(ident(r) for r in ev if r['event']=='success')
            assert aa==ss and all(v==1 for v in aa.values()) and not any(r['event']=='failure' for r in ev)
            if label=='PHYSICAL':
                joint.update(json.dumps(dict(r,job=job.name),sort_keys=True) for r in ev);physical=dict(Counter(r['category'] for r in ev if r['event']=='success'))
            else:
                assert all(r['role'] in ('Q_train_new','Q_train_old') for r in ev);queries+=len(aa)
        rr=events(job/'RATE_LEDGER.jsonl') if (job/'RATE_LEDGER.jsonl').exists() else [];ss=events(job/'SELECTION_LEDGER.jsonl') if (job/'SELECTION_LEDGER.jsonl').exists() else [];assert len(rr)==len(ss)==sum(physical.values())
        assert len(ss)==(12 if job.name=='qualification' else 600 if job.name.startswith('train') else 0)
        expected=Counter((r['category'],r['key'],r['step']) for r in rr)
        assert expected==Counter((r['category'],r['key'],r['step']) for r in ss)
        for i,r in enumerate(rr):
            assert r['ordinal']==i+1 and 100<=r['step']<300 and all(math.isfinite(a) and a>=0 and b==a*.25 for a,b in zip(r['base'],r['effective']))
        for i,s in enumerate(ss):
            assert s['ordinal']==i+1 and s['selected']==sum(s['classes'])==sum(b for a,b in s['image_budgets']) and s['eligible']==sum(a for a,b in s['image_budgets'])
            assert all(b==a//2 for a,b in s['image_budgets']) and 0<=s['tiles']<=16*s['images']
        rates+=len(rr);selections+=len(ss)
        pe=events(job/'PROTOTYPE_LEDGER.jsonl') if (job/'PROTOTYPE_LEDGER.jsonl').exists() else []
        key=lambda r:(r['ordinal'],r['category'],r['key'],r['step'],r['mode'])
        pa=Counter(key(r) for r in pe if r['event']=='attempt');ps=Counter(key(r) for r in pe if r['event']=='success')
        assert pa==ps and all(v==1 for v in pa.values()) and len(pe)==2*len(pa)
        assert len(pa)==(10 if job.name=='qualification' else 600 if job.name.startswith('train') else 0)
        for r in pe:
            if r['event']=='success':assert sum(r['target_class_counts'])==r['selected'] and 0<=r['changed']<=r['selected'] and (r['mode']!='BASE' or r['changed']==0)
        prototypes+=len(pa)
    assert set(jobnames)=={'qualification',*[f'{kind}{i}_{s}' for kind in ('train','evaluate') for i in range(8) for s in (3,4)]}
    assert joint==Counter(json.dumps(r,sort_keys=True) for r in events(root/'PHYSICAL_LEDGER.jsonl'))
    assert total==Counter(costs['counts']) and rates==selections==9612 and prototypes==9610 and queries==384
    assert total['query_images']==1536 and total['query_Q_train_new']==total['query_Q_train_old']==768
    assert costs['native_updates']==9612 and costs['actor_optimizer_updates']==costs['linear_solves']==costs['Q_dev_images']==0
    assert costs['additional_EMA_forward_calls']==9610 and costs['additional_EMA_image_forwards']==19220
    seal=read(root/'ENDPOINT_LOCK.json');assert seal['snapshots']==192 and seal['trajectories']==48
    for i in range(8):
        for s in (3,4):
            for method in ('BASE','SEMANTIC','ROTATED'):
                receipt=read(root/f'jobs/train{i}_{s}/{method}_TRAINING.json');assert receipt['updates']==200 and receipt['time']<=seal['time']
                assert all((root/f'jobs/train{i}_{s}/{method}_{step}.private.pt').is_file() for step in (150,200,250,300))
            assert read(root/f'jobs/evaluate{i}_{s}/STARTED.json')['time']>=seal['time']
    rows=read(root/'RESULTS.json')
    spec=importlib.util.spec_from_file_location('readout',root/'analysis.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    for name,value in a.analyze(rows).items():assert value==read(root/(name+'.json'))
    out=dict(status='PASS',jobs=33,native_optimizer_pairs=9612,rate_records=9612,selector_records=9612,prototype_pairs=9610,query_pairs=384,query_images=1536,Q_dev_images=0,all_root_child_keys_exact=True,all192_timepoints_and48_trajectories_exact=True,all_pixel_budgets_exact=True,all_snapshots_sealed_before_query=True,readout_recomputed=True,campaign_success=False)
    with (root/'COMPLETION_AUDIT.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out))


if __name__=='__main__':main(Path(os.environ['EXEC_RUN']))
