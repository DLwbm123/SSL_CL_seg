"""One terminal audit of receipts and physical/query/selector ledgers; no models."""
import fcntl
import json
import os
import time
from collections import Counter
from pathlib import Path


def read(p):return json.loads(p.read_text())

def audit(root):
    assert read(root/'FINAL.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0
    lock=(root/'COORDINATOR.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'COMPLETION_AUDIT.json').exists()
    q=read(root/'jobs/qualification/QUALIFICATION.json');assert q['historical_entry100_exact'] and q['both_new_checkpoint_replays_exact'] and q['frozen_memory'] and q['teacher_gradients_none']
    seal=read(root/'ENDPOINT_LOCK.json');assert seal['snapshots']==192 and seal['trajectories']==64
    jobs=list((root/'jobs').iterdir());assert len(jobs)==65
    counts=Counter();events={e:Counter() for e in ('attempt','success','failure')};rates=0;selectors=0;queries=Counter()
    for job in jobs:
        assert read(job/'PROCESS_EXIT.json')['exit_code']==0 and read(job/'FINAL.json')['status']=='COMPLETE'
        counts.update(read(job/'COUNTS.json'))
        if (job/'PHYSICAL_LEDGER.jsonl').exists():
            for line in (job/'PHYSICAL_LEDGER.jsonl').read_text().splitlines():
                r=json.loads(line);events[r['event']][r['category']]+=1
        if (job/'RATE_LEDGER.jsonl').exists():
            for line in (job/'RATE_LEDGER.jsonl').read_text().splitlines():
                r=json.loads(line);assert r['scale']==(1. if r['step']<100 else .25) and r['effective']==[a*r['scale'] for a in r['base']];rates+=1
        if (job/'SELECTION_LEDGER.jsonl').exists():
            for line in (job/'SELECTION_LEDGER.jsonl').read_text().splitlines():
                r=json.loads(line);assert r['target']==('EMA' if r['mode']=='EARLY_SELECTED_EMA' and r['step']<100 else 'MIXED_MEMORY');selectors+=1
        if job.name.startswith('train'):
            for m in ('EARLY_MEMORY','EARLY_SELECTED_EMA'):
                assert read(job/(m+'_TRAINING.json'))['time']<=seal['time']
                assert all((job/f'{m}_{t}.private.pt').is_file() for t in (100,150,300))
        if job.name.startswith('evaluate'):
            assert read(job/'STARTED.json')['time']>=seal['time']
            for line in (job/'QUERY_LEDGER.jsonl').read_text().splitlines():queries[json.loads(line)['event']]+=1
    assert dict(events['attempt'])==dict(events['success'])==dict(qualification=108,prior=19200) and not events['failure']
    root_events={e:Counter() for e in events}
    for line in (root/'PHYSICAL_LEDGER.jsonl').read_text().splitlines():
        r=json.loads(line);root_events[r['event']][r['category']]+=1
    assert root_events==events
    assert rates==19308 and selectors==19208 and queries==Counter(attempt=384,success=384)
    cost=read(root/'COSTS.json');assert cost['counts']==dict(counts) and cost['physical']==dict(events['attempt']) and counts['clean_student_forward_success']==counts['clean_student_forward_attempts']==1536
    result=dict(status='PASS',jobs_exit0=65,native_attempt_success=19308,selector_records=19208,rate_records=19308,query_calls=384,clean_query_images=1536,all_snapshots_sealed_before_readout=True,new_model_calls=0,time=time.time())
    with (root/'COMPLETION_AUDIT.json').open('x') as f:json.dump(result,f,indent=2)
    return result


if __name__=='__main__':print(json.dumps(audit(Path(os.environ['EXEC_RUN']))))
