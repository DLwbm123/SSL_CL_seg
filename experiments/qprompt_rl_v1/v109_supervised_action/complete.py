"""Validate all OFF returns and physical costs without additional model calls."""
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
    cfg=read(root/'CONFIG.private.json');cost=read(root/'COSTS.json');joint=Counter();counts=Counter();queries=0;jobs=[]
    key=lambda r:(r['event'],r['category'],r['key'],r['ordinal'])
    for job in sorted((root/'jobs').iterdir()):
        assert read(job/'FINAL.json')['status']=='COMPLETE' and read(job/'PROCESS_EXIT.json')['exit_code']==0
        counts.update(read(job/'COUNTS.json'));physical={}
        for name in ('PHYSICAL','QUERY'):
            p=job/(name+'_LEDGER.jsonl')
            if not p.exists():continue
            rows=events(p);ident=lambda r:json.dumps({k:v for k,v in r.items() if k!='event'},sort_keys=True)
            a=Counter(ident(r) for r in rows if r['event']=='attempt');s=Counter(ident(r) for r in rows if r['event']=='success')
            assert a==s and all(v==1 for v in a.values()) and not any(r['event']=='failure' for r in rows)
            if name=='PHYSICAL':joint.update((job.name,*key(r)) for r in rows);physical=dict(Counter(r['category'] for r in rows if r['event']=='success'))
            else:queries+=len(a)
        jobs.append(dict(job=job.name,physical=physical,exit_code=0))
    ledger=events(root/'PHYSICAL_LEDGER.jsonl')
    assert len(jobs)==41 and joint==Counter((r['job'],*key(r)) for r in ledger)
    assert Counter(r['category'] for r in ledger if r['event']=='success')==Counter(qualification=12,audit=12000)
    assert counts==Counter(cost['counts']) and queries==240 and counts['query_images']==960
    assert counts['off_updates']==8008 and counts['original_action_updates']==4004 and counts['behavior_extractions']==120
    keys=read(Path(cfg['v107'])/'KEYS.private.json');old=np.load(Path(cfg['v107'])/'DATASET.private.npz')['returns'];new=np.load(root/'RETURNS13.private.npz')
    assert new['returns'].shape==(80,13) and np.isfinite(new['returns']).all() and np.array_equal(new['returns'][:,:12],old) and np.array_equal(new['keys'],np.array(keys,dtype=str))
    rows=read(root/'NEW_RETURN_TABLE.json');margins=read(root/'ACTION_MARGINS.json');lookup={(r['context'],r['stream'],r['step']):r for r in rows};mlookup={(r['context'],r['stream'],r['step']):r for r in margins}
    assert len(rows)==len(margins)==len(lookup)==len(mlookup)==80
    for i in range(20):
        for stream in (1,2):
            job=root/f'jobs/collect{i}_{stream}';replay=read(job/'REFERENCE_REPLAY.json');current=read(job/'RESULTS.json')
            src=Path(cfg['v101'])/f'jobs/collect{i}_{stream}' if i<8 else Path(cfg['v107'])/f'jobs/collect{i-8}_{stream}'
            previous=read(src/'TRAINING_TABLE.private.json');context=current[0]['context'];assert replay['first_selection_exact'] and replay['retained_state_exact']
            if i<8:
                ref=[r for r in events(Path(cfg['original'])/'jobs/audit_1/ACTION_ROWS.jsonl') if r['context']==context and r['entry_step']==100];assert len(ref)==9 and len({r['new_entry'] for r in ref})==1;entry=ref[0]['new_entry']
            else:entry=read(Path(cfg['v107'])/f'jobs/entry{i-8}/BASELINES.json')['new_entry']
            chosen=[r for r in previous if r['step']==200 and r['reused']];assert len(chosen)==1 and chosen[0]['gain']==replay['baseline_gain']
            for row in current:
                k=(context,stream,row['step']);assert row==lookup[k];idx=keys.index(list(k));m=mlookup[k]
                gain=.25*(row['mid_new']-entry)+.75*(row['new_final']-entry) if row['step']==100 else replay['baseline_gain']+.75*(row['new_final']-replay['baseline_final_new'])
                assert abs(gain-row['gain'])<1e-14 and abs(gain+row['old_final']-row['old_entry']-row['dense_reward'])<1e-14
                assert row['dense_reward']==new['returns'][idx,12] and m['best_old_action']==int(old[idx].argmax())
                best=next(r for r in previous if r['step']==row['step'] and r['action']==m['best_old_action'])
                assert abs(m['off_minus_best']-(row['dense_reward']-old[idx].max()))<1e-14 and abs(m['off_minus_mean']-(row['dense_reward']-old[idx].mean()))<1e-14
                assert abs(m['gain_difference']-(gain-best['gain']))<1e-14 and abs(m['old_retention_difference']-(row['old_final']-best['old_final']))<1e-14
    groups={'all':margins}
    for field in ('step','stream','context'):
        for value in sorted({r[field] for r in margins}):groups[f'{field}={value}']=[r for r in margins if r[field]==value]
    summary=[dict(group=k,n=len(rs),positive=sum(r['off_minus_best']>0 for r in rs),negative=sum(r['off_minus_best']<0 for r in rs),**{m:st.mean(r[m] for r in rs) for m in ('off_minus_best','off_minus_mean','gain_difference','old_retention_difference')}) for k,rs in groups.items()]
    write(root/'MARGIN_SUMMARY.json',summary)
    write(root/'COMPLETION_AUDIT.json',dict(status='PASS',jobs=jobs,optimizer_attempt_success_pairs=12012,all_root_child_keys_exact=True,query_attempt_success_pairs=240,all80_gain_reward_margin_formulas_recomputed=True,old960_return_columns_exact=True,off_updates=8008,original_updates=4004,root_exit_code=0,additional_model_calls=0))
    print(json.dumps(summary))


if __name__=='__main__':main(Path(os.environ['EXEC_RUN']))
