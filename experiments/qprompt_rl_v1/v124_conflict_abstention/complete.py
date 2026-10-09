"""Once-only, zero-model audit of the conflict-abstention protocol."""
import fcntl
import importlib.util
import json
import math
import os
from collections import Counter
from pathlib import Path


def read(path):return json.loads(path.read_text())
def events(path):return [json.loads(v) for v in path.read_text().splitlines()]


def main(root):
    lock=(root/'COORDINATOR.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(root/'FINAL.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    spec=importlib.util.spec_from_file_location('analysis',root/'analysis.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    cost=read(root/'COSTS.json');expected=dict(qualification=24,prior=32000)
    assert cost['physical']==cost['success']==expected and not cost['failures'] and cost['native_updates']==sum(expected.values())
    joint=Counter();counts=Counter();pairs=Counter();rates=selections=0
    kinds=('train','evaluate')
    jobs={'qualification',*[f'{k}{i}_{s}' for k in kinds for i in range(16) for s in (3,4)]}
    assert {p.name for p in (root/'jobs').iterdir()}==jobs
    entrytime=read(root/'ENTRY_REUSE.json')['time'];qualificationtime=read(root/'jobs/qualification/FINAL.json')['time']
    assert read(root/'ENTRY_REUSE.json')['entries']==32
    for name in sorted(jobs):
        job=root/'jobs'/name;assert read(job/'FINAL.json')['status']=='COMPLETE' and read(job/'PROCESS_EXIT.json')['exit_code']==0 and not (job/'FAILURE.json').exists()
        counts.update(read(job/'COUNTS.json'));physical=0
        for label in ('PHYSICAL','FORWARD','QUERY'):
            path=job/(label+'_LEDGER.jsonl')
            if not path.exists():continue
            rows=events(path);key=lambda r:json.dumps({k:v for k,v in r.items() if k!='event'},sort_keys=True)
            attempts=Counter(key(r) for r in rows if r['event']=='attempt');success=Counter(key(r) for r in rows if r['event']=='success')
            assert attempts==success and all(v==1 for v in attempts.values()) and len(rows)==2*len(attempts)
            pairs[label]+=len(attempts)
            if label=='PHYSICAL':joint.update(json.dumps(dict(r,job=name),sort_keys=True) for r in rows);physical=len(attempts)
            if label=='QUERY':assert all(r['role'] in ('Q_train_new','Q_train_old') for r in rows)
        rr=events(job/'RATE_LEDGER.jsonl') if (job/'RATE_LEDGER.jsonl').exists() else []
        ss=events(job/'SELECTION_LEDGER.jsonl') if (job/'SELECTION_LEDGER.jsonl').exists() else []
        assert len(rr)==physical and len(ss)==(0 if name.startswith('evaluate') else physical)
        for j,r in enumerate(rr):
            scale=1. if r['step']<100 else .25
            assert r['ordinal']==j+1 and 0<=r['step']<300 and r['scale']==scale and all(math.isfinite(x) and x>=0 and y==x*scale for x,y in zip(r['base'],r['effective']))
        for j,s in enumerate(ss):
            assert s['ordinal']==j+1 and all(y==x//2 for x,y in s['image_budgets']) and s['selected']==sum(y for x,y in s['image_budgets'])
            assert 0<=s['conflict']<=s['selected'] and 0<=s['removed_raw_weight']<=s['selected_raw_weight']
            if s['mode'] in ('BASE','GOLD'):assert s['removed_raw_weight']==0
            if s['mode']=='OFF':assert s['removed_raw_weight']==s['selected_raw_weight']
            assert 0<=s['conflict_raw_weight']<=s['selected_raw_weight']
            assert sum(sum(v) for v in s['conflict_class_counts'])==s['conflict']
            if s['mode'] in ('IGNORE_C','RANDOM_MATCHED'):
                assert s['removed_class_counts']==s['conflict_class_counts'] and s['removed_raw_weight']==s['conflict_raw_weight'] and s['removed_pixels']==s['conflict']
            if s['mode']=='IGNORE_C':assert s['removed_conflict_overlap']==s['conflict']
            if s['mode']=='UNIFORM_MATCHED':
                assert s['removed_raw_weight']==s['conflict_raw_weight'] and s['removed_pixels']==0
                assert s['uniform_scale']==1.-s['conflict_raw_weight']/max(s['selected_raw_weight'],1e-8)
        rates+=len(rr);selections+=len(ss)
        if name.startswith('train'):
            assert read(job/'STARTED.json')['time']>=qualificationtime>=entrytime
            for m in a.METHODS:
                assert read(job/(m+'_TRAINING.json'))['time']<=read(root/'ENDPOINT_LOCK.json')['time']
                assert all((job/f'{m}_{s}.private.pt').is_file() for s in (150,300))
        if name.startswith('evaluate'):assert read(job/'STARTED.json')['time']>=read(root/'ENDPOINT_LOCK.json')['time']
    assert joint==Counter(json.dumps(r,sort_keys=True) for r in events(root/'PHYSICAL_LEDGER.jsonl')) and counts==Counter(cost['counts'])
    assert rates==pairs['PHYSICAL']==sum(expected.values()) and selections==32024
    assert pairs['FORWARD']==0 and pairs['QUERY']==704 and counts.get('query_images',0)==2816
    assert counts['image_attempts_train_labeled']==counts['image_success_train_labeled'] and counts['image_attempts_train_unlabeled']==counts['image_success_train_unlabeled']
    assert read(root/'jobs/qualification/QUALIFICATION.json')['status']=='PASS'
    for name,value in a.training(read(root/'RESULTS.json')).items():assert value==read(root/(name+'.json'))
    cfg=read(root/'CONFIG.private.json');old=Path(cfg['v123'])
    assert read(old/'DECISION.json')['training_stage']=='NOT_RUN_GATE_FAILED'
    for ctx in a.contexts():
        for stream in a.STREAMS:
            source=old/f"jobs/entry{ctx['context']}_{stream}";partition=read(source/'PARTITION.private.json')
            assert (source/'ENTRY100.private.pt').is_file() and len(partition['held'])==4 and len(partition['fit'])==ctx['labeled_images']
            assert not set(partition['held'])&set(partition['fit']+partition['source'])
    result=dict(status='PASS',jobs=len(jobs),native_pairs=rates,selector_records=selections,diagnostic_forward_pairs=0,performance_query_pairs=pairs['QUERY'],all_root_child_pairs_exact=True,all_partition_exclusions=True,all_endpoints_sealed_before_readout=True,all_removal_controls_valid=True,readout_recomputed=True,independent_confirmation=False,campaign_success=False)
    with (root/'COMPLETION_AUDIT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main(Path(os.environ['EXEC_RUN']))
