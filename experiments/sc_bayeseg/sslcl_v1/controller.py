"""Detached four-device queue with shared wall-clock and disk limits."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from sc_bayeseg.window_v1.r_start import write


def run(s):
    root = Path(s['root']); active = {}; finished = {}; attempted = set()
    def interrupted(signum, frame):
        raise RuntimeError('controller termination signal')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    assert json.loads((root / 'INTEGRATION.json').read_text())['status'] == 'PASS'
    sources = [('source_B', 'B', 'source', True, False), ('source_U', 'U', 'source', True, False)]
    targets = [(name, name[0], 'target', name[1] == '1', name[2] == '1') for name in ('B00', 'B10', 'B01', 'B11', 'U10', 'U11')]
    tasks = s.get('tasks', sources + targets)
    external = s.get('source_paths', {})
    def stop(reason):
        write(root / 'STOP.json', dict(reason=reason, time=time.time()))
        for p, _ in active.values():
            if p.poll() is None: os.killpg(p.pid, signal.SIGTERM)
        time.sleep(5)
        for p, _ in active.values():
            if p.poll() is None: os.killpg(p.pid, signal.SIGKILL)
    try:
        while True:
            elapsed = time.time() - s['original_started_unix']
            total = sum(int(subprocess.check_output(['du', '-s', '-B1', p]).decode().split()[0]) for p in (s['root'], s['ops']))
            if elapsed >= 85500 or total >= 49_000_000_000:
                stop('wall_clock_limit' if elapsed >= 85500 else 'storage_limit'); return
            for name, (p, gpu) in list(active.items()):
                code = p.poll()
                if code is not None:
                    finished[name] = dict(exit_code=code, gpu=gpu)
                    del active[name]
            available = [g for g in s['gpus'] if g not in [v[1] for v in active.values()]]
            for name, family, stage, use_u, history in tasks:
                if name in attempted or not available: continue
                if stage == 'target' and family not in external and finished.get('source_' + family, {}).get('exit_code') != 0: continue
                gpu = available.pop(0); config = dict(s, output=str(root/name), family=family, stage=stage, domain='Siemens' if stage == 'source' else 'Philips', use_u=use_u, history=history)
                if stage == 'target': config['source'] = external.get(family, str(root/('source_'+family)/'S5000.pt'))
                if name in s.get('training_seeds', {}): config['training_seed'] = s['training_seeds'][name]
                settings = Path(s['ops']) / (name + '.json'); write(settings, config)
                env = dict(os.environ, TASK_SETTINGS=str(settings), CUDA_VISIBLE_DEVICES=gpu)
                log = open(root/(name+'.log'), 'ab', buffering=0)
                p = subprocess.Popen(['python', 'worker.py'], cwd=s['ops'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True); log.close()
                active[name] = (p, gpu); attempted.add(name)
            write(root/'CONTROLLER.json', dict(status='RUNNING' if active else 'COMPLETE', active={k:dict(pid=p.pid,gpu=g) for k,(p,g) in active.items()}, finished=finished, attempted=sorted(attempted), elapsed=elapsed, allocated_bytes=total))
            if not active:
                blocked = [v[0] for v in tasks if v[0] not in attempted]
                write(root/'CONTROLLER_EXIT.json', dict(exit_code=0 if not blocked and all(v['exit_code']==0 for v in finished.values()) else 1, blocked=blocked, finished=finished)); return
            time.sleep(30)
    except BaseException:
        stop('controller_exception'); raise


if __name__ == '__main__': run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
