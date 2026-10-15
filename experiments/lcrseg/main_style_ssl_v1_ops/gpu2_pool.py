"""Authorized GPU2 pooling; reuse frozen workers and exact-boundary handoffs."""
import importlib.util
import json
import os
import runpy
import signal
import subprocess
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location('_pool_resource', Path(__file__).with_name('resource.py'))
resource = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resource)


def slot_available(running, watchers, capacity=2):
    occupied = sum(e['gpu'] == 2 for e in running.values())
    return occupied + len(watchers) < capacity


def has_progress(status, entry):
    return (status.get('status') == 'RUNNING' and status.get('step', 0) > (entry.get('resumed_from') or 0)) or status.get('status') == 'COMPLETE'


def card_available(running, colocated=()):
    try:
        cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                         '--format=csv,noheader,nounits'], text=True, timeout=8)
        card = next(row.split(',') for row in cards.splitlines() if row.split(',')[0].strip() == '2')
        clients = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                           '--format=csv,noheader,nounits'], text=True, timeout=8)
        allowed = {e['pid'] for e in running.values() if e['gpu'] == 2 and resource.alive(e)}
        allowed.update(e['pid'] for e in colocated if resource.alive(e))
        foreign = [int(row.split(',')[1]) for row in clients.splitlines()
                   if row.split(',')[0].strip() == card[1].strip() and int(row.split(',')[1]) not in allowed]
        return not foreign and int(card[2]) >= 10000
    except (OSError, subprocess.SubprocessError, StopIteration, ValueError):
        return False


def launch(config, action, payload, name, gpu=None, code=None):
    change = Path(config['change'])
    path = change / (name + '.config.json')
    resource.write(path, payload)
    env = dict(os.environ, POOL_SCRIPT=config['script'], POOL_ACTION=action, POOL_CONFIG=str(path))
    env.update(config.get('runtime_environment', {}))
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if code is not None:
        env['PYTHONPATH'] = code
    log = (change / (name + '.log')).open('x')
    process = subprocess.Popen(['bash', './s', config['python'], './n.py'], cwd=config['ops'],
                               env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    return process


def worker_config(config, node):
    root = Path(config['root'])
    return dict(node=node, output=str(root / 'jobs' / node['id']), data=config['data'],
                reference=config['reference'], predecessor=None if node['predecessor'] is None
                else str(root / 'jobs' / node['predecessor'] / 'sealed.pt'))


def run(config):
    from experiments.lcrseg.main_style_ssl_v1.protocol import nodes
    from experiments.lcrseg.main_style_ssl_v1.controller import summarize
    root = Path(config['root']); change = Path(config['change']); plan = nodes()
    frozen = json.loads((root / 'PLAN.json').read_text())
    assert config['authorized_gpu2'] is True and config['gpus'] == [2] and config['capacity'] == 2
    assert frozen['nodes'] == plan and frozen['total_optimizer_updates'] == 511600
    running = config['inherited'].copy(); states = config['states'].copy()
    watchers = {}; disabled = set(); start = time.monotonic()
    old = config['old_controller']
    if resource.alive(old):
        assert resource.process(old['pid'])['state'] == 'T'
        os.kill(old['pid'], signal.SIGTERM); os.kill(old['pid'], signal.SIGCONT)
        resource.write(change / 'OLD_CONTROLLER_RETIRED.json', dict(old=old, requested=time.time()))
    else:
        assert config.get('failed_controller_recovery') is True
        assert not list(change.glob('*.ADMITTED.json'))
    while len(states) < len(plan):
        for key, entry in list(running.items()):
            status_path = root / 'jobs' / key / 'STATUS.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            retire = entry.get('retire_old')
            progressed = has_progress(status, entry)
            if retire and progressed:
                physical = sum(1 for _ in (root / 'jobs' / key / 'physical_updates.jsonl').open('rb'))
                assert physical >= status['step'] > entry['resumed_from']
                if resource.alive(retire):
                    os.kill(retire['pid'], signal.SIGTERM); os.kill(retire['pid'], signal.SIGCONT)
                resource.write(change / (key + '.migrated.json'), dict(old=retire, new={k:v for k,v in entry.items() if k != 'retire_old'},
                               verified_step=status['step'], physical_updates=physical, repeated_optimizer_updates=0,
                               old_gpu_checkpoint_hold_seconds=time.time() - entry['stopped']))
                del entry['retire_old']
            if resource.alive(entry):
                continue
            ex = root / 'jobs' / key / 'EXIT.json'
            exit_value = json.loads(ex.read_text()) if ex.exists() else {}
            states[key] = 'COMPLETE' if status.get('status') == 'COMPLETE' and exit_value.get('exit_code') == 0 else 'FAILED'
            if retire:
                resource.write(change / (key + '.TARGET_FAILED.json'), dict(old_checkpoint_process=retire,
                               target=entry, automatic_retry=False, exit=exit_value))
            del running[key]
        for key, watcher in list(watchers.items()):
            ready = change / 'watches' / key / (key + '.ready.json')
            if not ready.exists():
                if not resource.alive(watcher):
                    del watchers[key]; disabled.add(key)
                continue
            entry = running.get(key)
            if entry is None:
                del watchers[key]; continue
            handoff = json.loads(ready.read_text())
            assert handoff['old']['pid'] == entry['pid'] and handoff['old']['birth'] == entry['birth']
            assert resource.alive(entry) and resource.process(entry['pid'])['state'] == 'T'
            payload = worker_config(config, next(n for n in plan if n['id'] == key))
            code = config['worker_code_roots'].get(key, config['default_code_root'])
            admitted = {}; success = False
            if card_available(running, config.get('colocated_gpu2', [])):
                result_path = change / (key + '.preflight.json')
                check = dict(worker=payload, checkpoint=handoff['checkpoint'], compatibility=True, result=str(result_path))
                process = launch(config, 'preflight', check, key + '.preflight', 2, code)
                returncode = process.wait()
                admitted = json.loads(result_path.read_text()) if result_path.exists() else {}
                success = returncode == 0 and admitted.get('status') == 'PASS' and admitted.get('formal_optimizer_updates') == 0
            if not success or not card_available(running, config.get('colocated_gpu2', [])):
                if resource.alive(entry):os.kill(entry['pid'], signal.SIGCONT)
                resource.write(change / (key + '.ADMISSION_BLOCKED.json'), dict(preflight=admitted,
                               old_worker_continues=True, automatic_training_retry=False))
                del watchers[key]; disabled.add(key); continue
            assert sum(e['gpu'] == 2 for e in running.values()) < 2
            handoff['preflight_status'] = 'PASS'
            process = launch(config, 'resume', dict(worker=payload, handoff=handoff), key + '.resume', 2, code)
            running[key] = dict(pid=process.pid, birth=resource.process(process.pid)['birth'], gpu=2,
                                resumed_from=handoff['step'], retire_old=entry, stopped=handoff['stopped'])
            resource.write(change / (key + '.ADMITTED.json'), handoff)
            del watchers[key]
        for node in plan:
            key, predecessor = node['id'], node['predecessor']
            if key in states or key in running:continue
            if predecessor and states.get(predecessor) in ('FAILED', 'BLOCKED'):
                states[key] = 'BLOCKED'; out = root / 'jobs' / key; out.mkdir(exist_ok=False)
                resource.write(out / 'STATUS.json', dict(status='BLOCKED', node=node, reason='predecessor failed; no automatic retry'))
                continue
            if predecessor and states.get(predecessor) != 'COMPLETE':continue
            if not slot_available(running, watchers) or not card_available(running, config.get('colocated_gpu2', [])):continue
            payload = worker_config(config, node); code = config['worker_code_roots'].get(key, config['default_code_root'])
            process = launch(config, 'worker', payload, key + '.new', 2, code)
            running[key] = dict(pid=process.pid, birth=resource.process(process.pid)['birth'], gpu=2)
        candidates = []
        for key, entry in running.items():
            if entry['gpu'] == 2 or key in watchers or key in disabled:continue
            physical = sum(1 for _ in (root / 'jobs' / key / 'physical_updates.jsonl').open('rb'))
            candidates.append(((2000 - physical % 2000), key, entry))
        for _, key, entry in sorted(candidates):
            if not slot_available(running, watchers) or not card_available(running, config.get('colocated_gpu2', [])):break
            watch_dir = change / 'watches' / key
            watch_dir.mkdir(parents=True, exist_ok=False)
            watch_cfg = dict(root=str(root), change=str(watch_dir), inherited={key:entry}, gpus=[2])
            process = launch(config, 'watch', watch_cfg, key + '.watch')
            watchers[key] = dict(pid=process.pid, birth=resource.process(process.pid)['birth'])
        resource.write(root / 'STATUS.json', dict(status='RUNNING', completed=states, running=running,
                       elapsed_seconds=config['previous_elapsed'] + time.monotonic() - start, total_nodes=30, active_host='bridge-gpu',
                       scheduling=dict(gpus=[2], capacity_per_gpu=2, legacy_draining=[k for k,e in running.items() if e['gpu'] != 2],
                                       pending_retirements=[k for k,e in running.items() if 'retire_old' in e]),
                       checkpoint_watchers=watchers, resource_admission_blocked=sorted(disabled)))
        resource.write(change / 'POOL_STATUS.json', dict(status='RUNNING', watchers=watchers,
                       migrated=[k for k,e in running.items() if e['gpu'] == 2], blocked=sorted(disabled), training_retries=0))
        if not (change / 'CONTROLLER_ADMITTED.json').exists():
            resource.write(change / 'CONTROLLER_ADMITTED.json', dict(pid=os.getpid(), birth=resource.process(os.getpid())['birth'],
                           formal_optimizer_updates=0, inherited=config['inherited'], states=config['states'], first_loop_completed=True))
        if len(states) < len(plan):time.sleep(2)
    failed = {k:v for k,v in states.items() if v != 'COMPLETE'}
    resource.write(root / 'STATUS.json', dict(status='COMPLETE' if not failed else 'COMPLETE_WITH_FAILURES', nodes=states,
                   elapsed_seconds=config['previous_elapsed'] + time.monotonic() - start, active_host='bridge-gpu'))
    resource.write(root / 'FINAL.json', dict(completed=30, failures=failed, summary=summarize(root, plan),
                   public_delivery='PENDING_COMPLETION_DELIVERY', scientific_success='NOT_ASSUMED',
                   attempt_history=config['attempt_history'], host_migration=config['migration_receipt'], resource_amendment=str(change)))
    resource.write(change / 'POOL_STATUS.json', dict(status='COMPLETE', failures=failed, training_retries=0))


def main():
    config = json.loads(Path(os.environ['POOL_CONFIG']).read_text()); action = os.environ['POOL_ACTION']
    if action == 'watch':resource.watch(config); return
    if action == 'preflight':
        runpy.run_path(str(Path(__file__).with_name('host_preflight.py')))['check'](config); return
    runpy.run_path(str(Path(__file__).with_name('host_migration.py')))['install_compatibility']()
    if action == 'controller':run(config)
    elif action == 'resume':resource.resume(config)
    elif action == 'worker':
        from experiments.lcrseg.main_style_ssl_v1.runtime import worker
        worker(config)
    else:raise ValueError('unsupported pool action')


if __name__ == '__main__':main()
