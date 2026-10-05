"""User-authorized early 20% scoring; reuse the registered evaluation jobs."""
import csv
import fcntl
import os
import subprocess
import time
from pathlib import Path

from experiments.qprompt_rl_v1.v6_low_label_grpo.run import read, write, matrix, jobroot, start


def main():
    root = Path(os.environ['EXEC_CAMPAIGN'])
    output = Path(os.environ['EXEC_INTERIM'])
    lock = (output/'EXECUTOR.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    c = read(root/'CONFIG.private.json')
    jobs = [j for j in matrix(c) if j['label_percent'] == 20]
    assert len(jobs) == 42
    assert all((jobroot(root, j)/'FINAL.json').exists() for j in jobs)
    assert read(root/'status.json')['phase'] == 'TRAIN'
    amendment = read(root/'EARLY_EVALUATION_AMENDMENT.json')
    assert amendment['label_percent'] == 20 and amendment['user_authorized']
    for scope in {j['scope'] for j in jobs}:
        write(root/'scopes'/scope/'V6_ENDPOINT_LOCK.json', dict(
            time=time.time(), all_training_complete=False,
            scope_training_complete=True, label_percent=20,
            authorization='EARLY_EVALUATION_AMENDMENT.json'))
    rows, audits, resources = [], [], []
    for j in jobs:
        if j['job'] != 'endpoint':
            continue
        evaluation = dict(j, id=j['scope']+'__eval_'+j['name'],
                          name='eval_'+j['name'], job='evaluate', deps=[], caps={},
                          target=str(jobroot(root, j)))
        dest = jobroot(root, evaluation)
        if not (dest/'FINAL.json').exists():
            free = int(subprocess.check_output([
                'nvidia-smi', '--id=5', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            assert free >= 6144, 'insufficient GPU memory; do not disturb training'
            assert not dest.exists(), 'inspect existing evaluation attempt before retry'
            item = start(root, c, evaluation, 5)
            write(output/'status.json', dict(status='RUNNING', time=time.time(),
                  completed=len(rows), expected=18, active=item['info']))
            rc = item['process'].wait()
            write(dest/'PROCESS_EXIT.json', dict(time=time.time(), exit_code=rc))
            assert rc == 0 and (dest/'FINAL.json').exists(), 'evaluation failed'
        result = read(dest/'RESULTS.json')
        assert len(result) == 1
        rows.append(dict(result[0], label_percent=20, subset_seed=j['subset_seed'],
                         variant=j['variant']))
        audits.append(dict(job=evaluation['id'], **read(dest/'COMPLETION_AUDIT.json')))
        for f in (dest/'evaluation_costs').glob('*/RESOURCE.json'):
            resources.append(dict(job=evaluation['id'], endpoint=f.parent.name, **read(f)))
    assert len(rows) == 18 and all(a['status'] == 'PASS' for a in audits)
    write(output/'RESULTS.json', rows)
    with (output/'RESULTS.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write(output/'COMPLETION_AUDIT.json', dict(status='PASS', endpoints=18,
        scope_training_complete_before_val=True, all_campaign_training_complete_before_val=False,
        amendment=amendment, jobs=audits, low_label_gate='NOT_EVALUATED'))
    write(output/'COSTS.json', dict(additional_student_updates=0, evaluators=resources,
        note='Registered evaluation jobs reused by the full campaign; no duplicate scoring.'))
    write(output/'status.json', dict(status='COMPLETE_PRIVATE', time=time.time(),
                                     completed=18, expected=18, publication='pending'))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        import traceback
        write(Path(os.environ['EXEC_INTERIM'])/'status.json', dict(
            status='FAILED', time=time.time(), error=repr(exc), traceback=traceback.format_exc()))
        raise
