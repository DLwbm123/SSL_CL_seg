"""Validate all OFF-prefix returns and physical costs without additional model calls."""
import fcntl
import json
import os
import statistics as st
from collections import Counter
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def events(p):return [json.loads(v) for v in p.read_text().splitlines()]
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def main(root):
    lock=(root/'COORDINATOR.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(root/'FINAL.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    cfg=read(root/'CONFIG.private.json');cost=read(root/'COSTS.json');joint=Counter();counts=Counter();queries=0;probes=0;jobs=[]
    key=lambda r:(r['event'],r['category'],r['key'],r['ordinal'])
    for job in sorted((root/'jobs').iterdir()):
        assert read(job/'FINAL.json')['status']=='COMPLETE' and read(job/'PROCESS_EXIT.json')['exit_code']==0
        counts.update(read(job/'COUNTS.json'));physical={}
        for name in ('PHYSICAL','QUERY','EXTRACTION'):
            p=job/(name+'_LEDGER.jsonl')
            if not p.exists():continue
            rows=events(p);ident=lambda r:json.dumps({k:v for k,v in r.items() if k!='event'},sort_keys=True)
            a=Counter(ident(r) for r in rows if r['event']=='attempt');s=Counter(ident(r) for r in rows if r['event']=='success')
            assert a==s and all(v==1 for v in a.values()) and not any(r['event']=='failure' for r in rows)
            if name=='PHYSICAL':joint.update((job.name,*key(r)) for r in rows);physical=dict(Counter(r['category'] for r in rows if r['event']=='success'))
            elif name=='QUERY':queries+=len(a)
            else:probes+=len(a)
        jobs.append(dict(job=job.name,physical=physical,exit_code=0))
    ledger=events(root/'PHYSICAL_LEDGER.jsonl')
    assert len(jobs)==41 and joint==Counter((r['job'],*key(r)) for r in ledger)
    assert Counter(r['category'] for r in ledger if r['event']=='success')==Counter(qualification=8,audit=48000)
    assert counts==Counter(cost['counts']) and queries==960 and probes==176 and counts['query_images']==3840
    assert counts['off_updates']==4008 and counts['original_action_updates']==44000 and counts['behavior_extractions']==40
    oldkeys=read(Path(cfg['v107'])/'KEYS.private.json');old=np.load(Path(cfg['v107'])/'DATASET.private.npz');oldreturns=np.load(Path(cfg['v109'])/'RETURNS13.private.npz')['returns'];new=np.load(root/'DATASET.private.npz');keys=read(root/'KEYS.private.json')
    assert len(keys)==len({tuple(k) for k in keys})==120 and keys[:80]==[list(k)+['ORIGINAL'] for k in oldkeys]
    assert np.array_equal(new['keys'],np.array(keys,dtype=str)) and new['returns'].shape==(120,13) and new['probes'].shape==(120,4,24) and new['original'].shape==(120,24)
    assert np.array_equal(new['returns'][:80],oldreturns) and np.array_equal(new['original'][:80],old['original']) and np.array_equal(new['probes'][:80],old['probes'])
    assert all(np.isfinite(new[n]).all() for n in ('returns','probes','original'))
    rows=read(root/'NEW_RETURN_TABLE.json');lookup={(r['context'],r['stream'],r['step'],r['prefix'],r['action']):r for r in rows};assert len(rows)==len(lookup)==520
    reused=0
    for i in range(20):
        for stream in (1,2):
            job=root/f'jobs/collect{i}_{stream}';source=Path(cfg['v109'])/f'jobs/collect{i}_{stream}';ref=next(r for r in read(source/'RESULTS.json') if r['step']==100);trace=read(source/'OFF_CONTINUATION.private.json');features=np.load(job/'FEATURES.private.npz');receipt=read(job/'COLLECTION_RECEIPT.json')
            current=read(job/'RESULTS.json');assert [r['action'] for r in current]==list(range(13)) and receipt['original_state_exact'] and receipt['reused_return_exact']
            k=[ref['context'],stream,200,'OFF'];idx=keys.index(k);assert np.array_equal(features['original'],trace['state']) and np.array_equal(new['original'][idx],features['original']) and np.array_equal(new['probes'][idx],features['probes'])
            for r in current:
                assert r==lookup[(*k,r['action'])]
                gain=ref['gain']+.75*(r['new_final']-ref['new_final']);assert abs(gain-r['gain'])<1e-14 and abs(gain+r['old_final']-r['old_entry']-r['dense_reward'])<1e-14
                assert r['dense_reward']==new['returns'][idx,r['action']] and r['old_entry']==ref['old_entry']
                if r['reused']:
                    reused+=1;assert r['action']==ref['continuation_action']==trace['action'] and r['dense_reward']==ref['dense_reward'] and r['old_final']==ref['old_final'] and r['new_final']==ref['new_final']
    assert reused==40
    summaries=[]
    for label,indices in [('ALL120',list(range(120))),('ORIGINAL80',list(range(80))),('OFF_PREFIX40',list(range(80,120))),('STEP100',[i for i,k in enumerate(keys) if k[2]==100]),('STEP200',[i for i,k in enumerate(keys) if k[2]==200])]:
        r=new['returns'][indices];margin=r[:,12]-r[:,:12].max(1)
        summaries.append(dict(group=label,states=len(indices),action_mean_returns=r.mean(0).tolist(),global_best_action=int(r.mean(0).argmax()),best_action_counts=dict(Counter(map(str,r.argmax(1)))),mean_OFF_minus_best_original=float(margin.mean()),positive_OFF_margins=int((margin>0).sum()),negative_OFF_margins=int((margin<0).sum())))
    write(root/'TRAINING_SUMMARY.json',summaries)
    write(root/'COMPLETION_AUDIT.json',dict(status='PASS',jobs=jobs,optimizer_attempt_success_pairs=48008,all_root_child_keys_exact=True,query_attempt_success_pairs=960,probe_attempt_success_pairs=176,all520_gain_reward_formulas_recomputed=True,all40_original_states_exact=True,all40_reused_returns_exact=True,old80_features1040returns_exact=True,all120_keys_unique_complete=True,root_exit_code=0,additional_model_calls=0,prior_metadata_launch_failures=1,prior_failed_launch_model_calls=0))
    print(json.dumps(summaries))


if __name__=='__main__':main(Path(os.environ['EXEC_RUN']))
