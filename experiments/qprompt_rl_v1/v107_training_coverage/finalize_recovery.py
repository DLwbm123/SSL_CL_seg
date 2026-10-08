"""Finalize the sealed V107 dataset after its cost-file creation conflict; no training."""
import fcntl
import json
import os
import time
from collections import Counter
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def events(path):
    return [json.loads(v) for v in path.read_text().splitlines()]


def main(root):
    lock=(root/'COORDINATOR.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'FINAL.json').exists()
    failure=read(root/'FAILURE.json');assert failure['error'].startswith('FileExistsError') and "COSTS.json" in failure['traceback']
    assert read(root/'EXIT.json')['exit_code']==1 and read(root/'DATASET_LOCK.json')['status']=='SEALED'
    allrows=events(root/'PHYSICAL_LEDGER.jsonl');physical=Counter(r['category'] for r in allrows if r['event']=='attempt')
    assert physical==Counter(qualification=8,entries=1200,audit=84000)
    assert Counter(r['category'] for r in allrows if r['event']=='success')==physical and not any(r['event']=='failure' for r in allrows)
    aggregate=Counter((r['job'],r['event'],r['category'],r['key'],r['ordinal']) for r in allrows)
    counts=Counter();checks=[];joined=Counter();query_pairs=probe_pairs=0
    for job in sorted((root/'jobs').iterdir()):
        assert read(job/'FINAL.json')['status']=='COMPLETE' and read(job/'PROCESS_EXIT.json')['exit_code']==0
        ledger=events(job/'PHYSICAL_LEDGER.jsonl');joined.update((job.name,r['event'],r['category'],r['key'],r['ordinal']) for r in ledger)
        for label in ('PHYSICAL','QUERY','EXTRACTION'):
            path=job/(label+'_LEDGER.jsonl')
            if not path.exists():continue
            records=events(path);pair=Counter((r['event'],json.dumps({k:v for k,v in r.items() if k!='event'},sort_keys=True)) for r in records)
            attempted=Counter({key:count for (event,key),count in pair.items() if event=='attempt'})
            succeeded=Counter({key:count for (event,key),count in pair.items() if event=='success'})
            assert attempted==succeeded and all(v==1 for v in attempted.values()) and not any(r['event']=='failure' for r in records)
            if label=='QUERY':query_pairs+=sum(attempted.values())
            if label=='EXTRACTION':probe_pairs+=sum(attempted.values())
        counts.update(read(job/'COUNTS.json'));checks.append(dict(job=job.name,exit_code=0,physical=read(job/'FINAL.json')['physical']))
    assert len(checks)==37 and aggregate==joined
    assert query_pairs==1416 and probe_pairs==208 and counts['query_images']==5664 and counts['probe_extractions']==208 and counts['behavior_extractions']==340
    data=np.load(root/'DATASET.private.npz');keys=read(root/'KEYS.private.json');assert len(keys)==80 and data['returns'].shape==(80,12) and data['probes'].shape==(80,4,24)
    assert all(np.isfinite(data[k]).all() for k in ('original','probes','returns')) and np.array_equal(data['keys'],np.array(keys,dtype=str))
    cfg=read(root/'CONFIG.private.json');old=np.load(Path(cfg['v103'])/'FEATURES.private.npz')
    assert np.array_equal(data['original'][:32],old['original']) and np.array_equal(data['probes'][:32],old['probes'])
    oldrows=sum((read(Path(cfg['v101'])/f'jobs/collect{i}_{s}/TRAINING_TABLE.private.json') for i in range(8) for s in (1,2)),[])
    rows=read(root/'NEW_RETURN_TABLE.json');assert len(rows)==576
    lookup={(r['context'],r['stream'],r['step'],r['action']):r for r in oldrows+rows};assert len(lookup)==960
    for i,key in enumerate(keys):
        for action in range(12):
            r=lookup[(*key,action)];assert abs(r['dense_reward']-(r['gain']+r['old_final']-r['old_entry']))<1e-12
            assert data['returns'][i,action]==r['dense_reward']
    audit=dict(status='PASS',jobs=checks,all_root_child_optimizer_keys_exact=True,optimizer_attempt_success_pairs=sum(physical.values()),query_attempt_success_pairs=query_pairs,probe_attempt_success_pairs=probe_pairs,all960_return_keys_and_formulas=True,finite_dataset_shapes=True,old_features_and_returns_preserved=True)
    costs=dict(native_updates=sum(physical.values()),actor_optimizer_updates=0,linear_solves=0,physical=dict(physical),success=dict(physical),failures={},counts=dict(counts),new_annotation_cases=0)
    receipt=dict(status='METADATA_RECOVERY_COMPLETE',original_exit_code=1,original_error='FileExistsError while creating existing COSTS.json',all37_workers_exit_zero=True,original_failure_and_exit_preserved=True,original_costs_preserved=True,extra_native_updates=0,extra_actor_updates=0,extra_queries=0,extra_probes=0,time=time.time())
    write(root/'COMPLETION_AUDIT.json',audit);write(root/'FINAL_COSTS.json',costs);write(root/'RECOVERY.json',receipt)
    final=dict(status='COMPLETE',decision='TRAINING_COVERAGE_DATASET_ONLY_NO_PERFORMANCE_CLAIM',physical=dict(physical),final_costs='FINAL_COSTS.json',metadata_recovery='RECOVERY.json',original_exit_code=1,publication='PENDING',time=time.time())
    write(root/'FINAL.json',final)
    (root/'STATUS.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(dict(final=final,recovery=receipt,costs=costs,audit='PASS')))


if __name__=='__main__':main(Path(os.environ['EXEC_RUN']))
