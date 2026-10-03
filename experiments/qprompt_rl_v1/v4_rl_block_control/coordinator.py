"""Finite three-GPU dependency queue. No deadlines, retries, or score-driven expansion."""
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from statistics import mean
from . import protocol as p

ROOT = Path(os.environ['EXEC_RUN'])
C = json.loads(Path(os.environ['EXEC_CONFIG']).read_text())
COMMAND = [sys.executable, '-c', 'import os,runpy;runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")']


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def read(path):
    return json.loads(path.read_text())


def jobs():
    phases = [[dict(id='smoke', job='smoke', caps=dict(smoke=390))]]
    preparations = []
    for s in p.CONFIRMATION_SEEDS:
        preparations.append(dict(id=f'source_{s}', job='source', seed=s, caps=dict(source=8000)))
    for s in p.DEVELOPMENT_SEEDS:
        for d in p.DOMAINS:
            h = p.HORIZONS[d]
            preparations.append(dict(id=f'panel_{s}_{d}', job='panel', seed=s, domain=d,
                                     caps=dict(panel=h+(h//p.PANEL_STRIDE)*3*p.BLOCK)))
    phases.append(preparations)
    phases.append([dict(id='fit_'+d, job='fit', domain=d, caps=dict(controller=1280)) for d in p.DOMAINS])
    phases.append([dict(id=f'learn_{d}_{m}', job='learn', domain=d, method=m,
                        caps=dict(development=p.EPISODES*p.HORIZONS[d], controller=p.EPISODES*4))
                   for d in p.DOMAINS for m in p.LEARNERS])
    phases.append([dict(id=f'main_{s}_{d}', job='main', seed=s, domain=d,
                        caps=dict(main=len(p.METHODS)*p.HORIZONS[d]))
                   for s in p.CONFIRMATION_SEEDS for d in p.DOMAINS])
    return phases


def run_phase(index, queue):
    pending = list(queue)
    active = {}
    failed = []
    while pending or active:
        for gpu in p.GPUS:
            if gpu in active or not pending or failed:
                continue
            available = int(subprocess.check_output(['nvidia-smi', '--id='+str(gpu),
                        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            if available < 6144:
                continue
            job = pending.pop(0)
            root = ROOT/'jobs'/job['id']
            root.mkdir(exist_ok=False)
            config = dict(C, **job, campaign=str(ROOT), run_root=str(root), gpu=gpu)
            write(root/'CONFIG.private.json', config)
            write(root/'SESSION.json', dict(start=time.time(), optimizer_deadline=None, hard_deadline=None,
                                            fixed_job_caps=job['caps'], no_automatic_retry=True))
            env = dict(os.environ, EXEC_RUN=str(root), EXEC_CONFIG=str(root/'CONFIG.private.json'),
                       EXEC_MODULE='experiments.qprompt_rl_v1.v4_rl_block_control.worker', CUDA_VISIBLE_DEVICES=str(gpu))
            log = (root/'worker.log').open('a')
            command = ['bash', './with_nas_storage.sh'] + COMMAND
            process = subprocess.Popen(command, cwd=Path(C['code'])/'experiments/lcrseg/scripts', env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            info = dict(job=job['id'], gpu=gpu, pid=process.pid, time=time.time(),
                        process_start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21],
                        commit=C['commit'], command=command, initial_free_mib=available)
            write(root/'LAUNCH.json', info)
            active[gpu] = process, log, root, info
        for gpu, (process, log, root, info) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            write(root/'PROCESS_EXIT.json', dict(exit_code=code, time=time.time(), pid=process.pid))
            if code != 0 or not (root/'FINAL.json').exists():
                failed.append(info['job'])
            del active[gpu]
        write(ROOT/'QUEUE_STATUS.json', dict(phase=index, time=time.time(),
             active=[v[3] for v in active.values()], pending=[j['id'] for j in pending], failed=failed))
        if failed and not active:
            raise RuntimeError('Failed jobs; no retries or dependent launches: '+', '.join(failed))
        if pending or active:
            time.sleep(3)
    if index == 2:
        write(ROOT/'OFFLINE_POLICY_LOCK.json', dict(commit=C['commit'], time=time.time(),
              paths=[str(ROOT/'jobs'/('fit_'+d)) for d in p.DOMAINS], selection='fixed h5/h25, all development seeds'))
    if index == 3:
        write(ROOT/'FINAL_POLICY_LOCK.json', dict(commit=C['commit'], time=time.time(),
              policies=[str(ROOT/'jobs'/f'learn_{d}_{m}'/'FROZEN_POLICY.pt') for d in p.DOMAINS for m in p.LEARNERS],
              confirmation_started=False, selection='last episode; no val or audit selection'))


def report():
    rows = []
    counts = Counter()
    audits = []
    for phase in jobs():
        for job in phase:
            root = ROOT/'jobs'/job['id']
            audit = read(root/'COMPLETION_AUDIT.json')
            assert audit['status'] == 'PASS'
            counts.update(audit['physical_calls'])
            audits.append(dict(job=job['id'], **audit))
            if job['job'] == 'main':
                rows.extend(read(root/'RESULTS.json'))
    expected = p.expected_calls()
    assert all(counts[k] == expected[k] for k in ('source', 'panel', 'development', 'main'))
    assert counts['smoke'] == 390 and counts['controller'] == 2848
    assert len(rows) == len(p.CONFIRMATION_SEEDS)*len(p.DOMAINS)*len(p.METHODS)
    with (ROOT/'ENDPOINTS.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    means = {m: {k: mean(r[k] for r in rows if r['method'] == m)
                 for k in ('macro_Dice', 'old_REFUGE', 'old_change')} for m in p.METHODS}
    comparisons = {ref: p.compare(rows, 'PPO_25', ref, .002 if ref in ('BANDIT_25', 'REWARD_SHUFFLE') else .003 if ref == 'FINE_05' else 0.)
                   for ref in p.METHODS if ref != 'PPO_25'}
    decision = dict(rl_specific_supported=all(comparisons[m]['passed'] for m in ('BANDIT_25', 'REWARD_SHUFFLE', 'PHASE_SHUFFLE', 'EFFECT_RULE')),
                    practical_candidate=all(comparisons[m]['passed'] for m in ('ORIGINAL', 'FINE_0125', 'FINE_05', 'EARLY_05')),
                    descriptive_only=True, automatic_followup=False)
    summary = dict(means=means, paired_comparisons=comparisons, decision=decision,
                   units='Dice fractions; multiply by 100 for percent or percentage-point differences',
                   limitation='New optimization seeds, reused development val patients; two separate stage-1 adaptations, not full-sequence BWT.')
    write(ROOT/'SUMMARY.json', summary)
    write(ROOT/'DECISION.json', decision)
    write(ROOT/'ALL_COSTS.json', dict(physical_optimizer_calls=dict(counts), synthetic_controller_calls=268,
          total_optimizer_calls=sum(counts.values())+268, wall_seconds=time.time()-read(ROOT/'SESSION.json')['start'],
          reused_source_history=24000, no_time_limit=True, compute_matching='same retained student horizons; controller/development/feature overhead differs'))
    write(ROOT/'COMPLETION_AUDIT.json', dict(status='PASS', jobs=audits, endpoints=len(rows), physical_calls=dict(counts),
          main_controller_updates=0, automatic_retries=0, final_policy_lock_before_confirmation=True))
    text = '# V4 blockwise RL control: complete results\n\n'
    text += '110 final student endpoints; five new optimization seeds, two domains, eleven methods. '
    text += 'Native horizons: RIM-ONE 3200, Drishti 2100. Validation cohorts are reused development cohorts.\n\n'
    text += '| Method | New Dice (%) | Old REFUGE (%) | Old change (pp) |\n|---|---:|---:|---:|\n'
    for m, v in means.items():
        text += f'| {m} | {100*v["macro_Dice"]:.4f} | {100*v["old_REFUGE"]:.4f} | {100*v["old_change"]:.4f} |\n'
    text += '\nAll paired seed/domain comparisons, adverse cells, sample SD and fixed criteria are in SUMMARY.json. '
    text += 'No best-epoch or best-seed selection. Audit feedback was never used to select a controller. '
    text += 'Old-domain retention is evaluated, not an online reward. No claim of independent patient confirmation or full BWT.\n\n'
    text += 'Frozen decision: `'+json.dumps(decision)+'`. Negative results do not trigger new experiments.\n'
    (ROOT/'FINAL_INTERPRETATION.md').write_text(text)


def main():
    p.check()
    handle = (ROOT/'COORDINATOR.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (ROOT/'RUN_LOCK.json').exists(), 'no automatic resume'
    (ROOT/'jobs').mkdir(exist_ok=False)
    write(ROOT/'RUN_LOCK.json', dict(protocol='V4-RL-BLOCK-CONTROL', commit=C['commit'], gpus=p.GPUS,
          development_seeds=p.DEVELOPMENT_SEEDS, confirmation_seeds=p.CONFIRMATION_SEEDS,
          domains=p.DOMAINS, horizons=p.HORIZONS, methods=p.METHODS, block=p.BLOCK,
          episodes=p.EPISODES, expected_student_calls=p.expected_calls(), smoke_calls=390,
          expected_controller_calls=2848, synthetic_controller_calls=268, jobs=jobs(),
          optimizer_deadline=None, hard_deadline=None, retries=0))
    write(ROOT/'PROCESS.json', dict(pid=os.getpid(), time=time.time(),
          process_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    try:
        for index, queue in enumerate(jobs()):
            write(ROOT/'status.json', dict(status='RUNNING', phase=index, time=time.time()))
            run_phase(index, queue)
        report()
        write(ROOT/'FINAL.json', dict(status='COMPLETE_PRIVATE', time=time.time(), commit=C['commit'],
                                      publication='pending local verification and GitHub delivery'))
        write(ROOT/'status.json', dict(status='COMPLETE_PRIVATE', time=time.time()))
    except BaseException as exc:
        write(ROOT/'status.json', dict(status='FAILED', error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        raise


if __name__ == '__main__':
    main()
