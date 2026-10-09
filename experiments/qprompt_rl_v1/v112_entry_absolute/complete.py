"""Audit the sixteen-image entry query and all reused scalar rows; no model calls."""
import fcntl
import json
import os
from collections import Counter
from pathlib import Path


def main(root):
    read=lambda p:json.loads(p.read_text())
    lock=(root/'COORDINATOR.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(root/'FINAL.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    events=[json.loads(v) for v in (root/'QUERY_LEDGER.jsonl').read_text().splitlines()]
    keys=lambda event:Counter(json.dumps({k:v for k,v in r.items() if k!='event'},sort_keys=True) for r in events if r['event']==event)
    assert len(events)==8 and len(keys('attempt'))==4 and keys('attempt')==keys('success') and not keys('failure')
    assert all(r['role']=='Q_dev_old' and r['step']==100 and r['images']==4 for r in events)
    entries={r['context']:r for r in read(root/'ENTRY_BASELINES.json')};assert len(entries)==4 and all(r['snapshot_RNG_preserved'] for r in entries.values())
    cfg=read(root/'CONFIG.private.json');original={(stage,r['context'],r['stream'],r['method']):r for stage in ('V111','V108') for r in read(Path(cfg[stage.lower()])/'DEVELOPMENT_RESULTS.json')['rows']}
    rows=read(root/'ABSOLUTE_RESULTS.json');assert len(rows)==len(original)==504
    for r in rows:
        p=original[(r['stage'],r['context'],r['stream'],r['method'])];e=entries[r['context']]
        for k in ('mid_new','new','old','utility','new_entry','old_memory_reference'):assert r[k]==p[k]
        expected=dict(final_new_minus_entry=p['new']-p['new_entry'],mid_new_minus_entry=p['mid_new']-p['new_entry'],weighted_new_gain=.25*(p['mid_new']-p['new_entry'])+.75*(p['new']-p['new_entry']),old_final_minus_entry=p['old']-e['old_entry'],old_final_minus_memory=p['old']-p['old_memory_reference'],frozen_entry_utility=-max(0.,p['old_memory_reference']-e['old_entry']-.005))
        expected['utility_minus_frozen']=p['utility']-expected['frozen_entry_utility']
        assert all(abs(r[k]-v)<1e-12 for k,v in expected.items())
    assert not (root/'PHYSICAL_LEDGER.jsonl').exists() or not (root/'PHYSICAL_LEDGER.jsonl').read_text().strip()
    result=dict(status='PASS',query_attempt_success_pairs=4,query_images=16,native_updates=0,actor_updates=0,all504_source_rows_and_absolute_formulas_exact=True,all4_fullsnapshot_RNG_checks=True)
    with (root/'COMPLETION_AUDIT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main(Path(os.environ['EXEC_RUN']))
