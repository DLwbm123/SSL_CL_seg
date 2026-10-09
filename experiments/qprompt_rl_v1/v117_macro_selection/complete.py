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
    costs=read(root/'COSTS.json');assert costs['physical']==costs['success']==dict(qualification=8,prior=12800) and not costs['failures']
    joint=Counter();total=Counter();queries=rates=selections=0;jobnames=[]
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
        rr=events(job/'RATE_LEDGER.jsonl');ss=events(job/'SELECTION_LEDGER.jsonl');assert len(rr)==sum(physical.values())
        assert len(ss)==(4 if job.name=='qualification' else 1200)
        expected=Counter((r['category'],r['key'],r['step']) for r in rr if (r['key']=='0' if job.name=='qualification' else not r['key'].endswith('/OFF')))
        assert expected==Counter((r['category'],r['key'],r['step']) for r in ss)
        for i,r in enumerate(rr):
            assert r['ordinal']==i+1 and 100<=r['step']<300 and all(math.isfinite(a) and a>=0 and b==a*.25 for a,b in zip(r['base'],r['effective']))
        for i,s in enumerate(ss):
            assert s['ordinal']==i+1 and s['selected']==sum(s['classes'])==sum(b for a,b in s['image_budgets']) and s['eligible']==sum(a for a,b in s['image_budgets'])
            assert all(b==a//2 for a,b in s['image_budgets']) and 0<=s['tiles']<=16*s['images']
        rates+=len(rr);selections+=len(ss)
    assert set(jobnames)=={'qualification',*[f'collect{s}_{p}' for s in (601,602) for p in range(4)]}
    assert joint==Counter(json.dumps(r,sort_keys=True) for r in events(root/'PHYSICAL_LEDGER.jsonl'))
    assert total==Counter(costs['counts']) and rates==12808 and selections==9604 and queries==896
    assert total['query_images']==3584 and total['query_Q_train_new']==2304 and total['query_Q_train_old']==1280
    assert costs['native_updates']==12808 and costs['actor_optimizer_updates']==costs['linear_solves']==costs['Q_dev_images']==0
    rows=read(root/'RESULTS.json');assert len(rows)==len({(r['seed'],r['group'],r['method']) for r in rows})==256
    for r in rows:
        assert r['context']==r['group']%8 and r['cycle']==r['group']//8 and r['entry_step']==100+50*r['cycle']
        assert abs(r['gain']-(.25*(r['mid_new']-r['new_entry'])+.75*(r['new']-r['new_entry'])))<1e-14
        assert abs(r['final_new_gain']-(r['new']-r['new_entry']))<1e-14 and abs(r['old_gain']-(r['old']-r['old_entry']))<1e-14 and abs(r['reward']-(r['gain']+r['old_gain']))<1e-14
    spec=importlib.util.spec_from_file_location('readout',root/'analysis.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    for name,value in a.analyze(rows,read(root/'FEATURES.private.json')).items():assert value==read(root/(name+'.json'))
    out=dict(status='PASS',jobs=9,native_optimizer_pairs=12808,rate_records=12808,selector_records=9604,query_pairs=896,query_images=3584,Q_dev_images=0,all_root_child_keys_exact=True,all256_rewards_exact=True,all_pixel_budgets_exact=True,readout_recomputed=True,campaign_success=False)
    with (root/'COMPLETION_AUDIT.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out))


if __name__=='__main__':main(Path(os.environ['EXEC_RUN']))
