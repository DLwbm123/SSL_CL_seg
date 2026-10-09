"""Finish metadata only after the recorded COSTS create-only collision."""
import fcntl
import json
import os
import time
from collections import Counter
from pathlib import Path


def main(root):
    read=lambda p:json.loads(p.read_text())
    def write(name,value):
        with (root/name).open('x') as f:json.dump(value,f,indent=2)
    lock=(root/'COORDINATOR.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(root/'EXIT.json')['exit_code']==1 and not (root/'FINAL.json').exists()
    failure=read(root/'FAILURE.json');assert failure['error']=="FileExistsError(17, 'File exists')" and "N.D.write(root/'COSTS.json'" in failure['traceback']
    jobs=[p for p in (root/'jobs').iterdir() if not p.is_symlink()];assert len(jobs)==25
    counts=Counter()
    for p in jobs:
        assert read(p/'PROCESS_EXIT.json')['exit_code']==0 and read(p/'FINAL.json')['status']=='COMPLETE' and not (p/'FAILURE.json').exists()
        counts.update(read(p/'COUNTS.json'))
    ledger=[json.loads(v) for v in (root/'PHYSICAL_LEDGER.jsonl').read_text().splitlines()]
    by={event:Counter(r['category'] for r in ledger if r['event']==event) for event in ('attempt','success','failure')}
    assert by['attempt']==by['success']==dict(qualification=8,development=7200) and not by['failure']
    rows=read(root/'DEVELOPMENT_RESULTS.json')['rows'];assert len(rows)==len({(r['context'],r['stream'],r['method']) for r in rows})==36 and counts['query_images']==432 and counts['probe_extractions']==304
    costs=dict(native_updates=7208,actor_optimizer_updates=0,linear_solves=0,physical=dict(by['attempt']),success=dict(by['success']),failures={},counts=dict(counts),new_annotation_cases=0)
    write('COMPLETED_COSTS.json',costs)
    write('METADATA_RECOVERY.json',dict(status='COMPLETE',original_coordinator_exit_code=1,reason='COSTS.json create-only collision after all model operations and scalar rows completed',new_native_updates=0,new_actor_updates=0,new_queries=0,original_failure_exit_costs_preserved=True,jobs=25,time=time.time()))
    print('PASS metadata recovery; zero additional model operations')


if __name__=='__main__':main(Path(os.environ['EXEC_RUN']))
