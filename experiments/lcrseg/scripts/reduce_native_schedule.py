"""User-authorized subset dispatcher; workers use the unchanged qualified checkout.

This controller adopts live workers by PID/start time. It never resumes or retries
a previously launched node, and never changes training options or old receipts.
"""
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def process_state(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def reduced_nodes(plan, amendment):
    original = {n['id']: n for n in plan['nodes']}
    keep = set(amendment['retained_node_ids'])
    cancel = set(amendment['cancelled_node_ids'])
    if keep & cancel or keep | cancel != set(original):
        raise ValueError('reduction must partition the original DAG')
    for ident in cancel:
        n = original[ident]
        allowed = (n.get('phase') == 'D' and
                   (n['seed'] != 162 or n['family'] not in amendment['replication_families']))
        allowed |= (n.get('phase') == 'C' and n.get('family') == 'F5' and
                    n['candidate_id'] not in amendment['retained_F5_candidates'])
        if not allowed:
            raise ValueError('unauthorized cancellation: ' + ident)
    nodes = [copy.deepcopy(n) for n in plan['nodes'] if n['id'] in keep]
    for n in nodes:
        if n['id'] == 'SELECT_F5':
            n['dependencies'] = [d for d in n['dependencies'] if d in keep]
        if not set(n['dependencies']) <= keep:
            raise ValueError('missing prerequisite: ' + n['id'])
    return nodes


def summarize(root, receipts, amendment, write):
    families = amendment['replication_families']
    table = {}
    by_order = {}
    for family in families:
        rows = [r for n, r in receipts.items() if n.startswith('D__') and
                r.get('identity', {}).get('family') == family and r['identity']['stage'] == 2]
        if {(r['identity']['seed'], r['identity']['order']) for r in rows} != {(162, 1), (162, 2)} or len(rows) != 2:
            raise ValueError('incomplete single-seed paired orders: ' + family)
        table[family] = {k: sum(r[k] for r in rows) / 2 for k in ('Final', 'Old', 'Incoming', 'Forget')}
        table[family]['seed_Final'] = {'162': table[family]['Final']}
        table[family]['class_macro'] = {
            c: sum(r['scores'][d][c] for r in rows for d in r['seen']) / 6
            for c in ('rim', 'cup', 'disc_union')}
        by_order[family] = {r['identity']['order']: r for r in rows}
    comparisons = {}
    for family in ('F1', 'F2', 'F3', 'F4', 'F5'):
        for reference in amendment['comparison_references']:
            comparisons[family + '__vs__' + reference] = {
                'mean_delta': {k: table[family][k] - table[reference][k] for k in ('Final', 'Old', 'Incoming', 'Forget')},
                'order_Final_deltas': {str(o): by_order[family][o]['Final'] - by_order[reference][o]['Final'] for o in (1, 2)}}
    write(root / 'replication_summary_reduced.json', {
        'status': 'COMPLETE_REDUCED_PROTOCOL', 'amendment_id': amendment['amendment_id'],
        'table': table, 'paired_comparisons': comparisons,
        'claims': 'One replication seed on development patients; no multi-seed stability or independent-patient claim. F5 used two development candidates; other frameworks used eight.',
        'seed_confidence_intervals': None, 'phase_E_started': False})


def run():
    # Import only the frozen worker implementation, not a second training engine.
    config = read(os.environ['EXEC_CONFIG'])
    sys.path.insert(0, config['code'])
    from experiments.lcrseg.five_frameworks_v1.native_runner import admit, NativeRunner, write
    root = Path(config['run_root'])
    amendment = read(os.environ['SCHEDULE_AMENDMENT'])
    with (root / 'reduced_executor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, plan = admit(config)
        nodes = reduced_nodes(plan, amendment)
        for name in ('CPU_QUALIFICATION', 'CUDA_QUALIFICATION', 'SMOKE'):
            receipt = read(root / (name + '.json'))
            if receipt['status'] != 'PASS' or receipt['execution_commit'] != config['execution_commit']:
                raise ValueError('original worker qualification mismatch')
        old = amendment['old_executor']
        state = process_state(old['pid'])
        if state and state[0] != 'Z':
            raise ValueError('old dispatcher must have exited before replacement')
        for ident in amendment['cancelled_node_ids']:
            nr = root / ident
            if any((nr / x).exists() for x in ('launch.json', 'receipt.json', 'physical.jsonl', 'latest.pt', 'failure.json')):
                raise ValueError('cannot cancel a started node: ' + ident)
        active = dict(amendment['adopted_workers'])
        children = {}
        while True:
            receipts = {n['id']: read(root / n['id'] / 'receipt.json') for n in nodes if (root / n['id'] / 'receipt.json').exists()}
            failures = [n['id'] for n in nodes if (root / n['id'] / 'failure.json').exists()]
            for ident, v in list(active.items()):
                child = children.get(ident)
                rc = child.poll() if child else None
                state = process_state(v['pid'])
                alive = state and state[0] != 'Z' and state[1] == v['start_ticks']
                if not alive:
                    if ident not in receipts or (child and rc not in (None, 0)):
                        failures.append(ident)
                    del active[ident]
            status = {'status': 'EXECUTOR_ACTIVE', 'sealed': len(receipts), 'total': len(nodes),
                      'original_total': len(plan['nodes']), 'cancelled': len(amendment['cancelled_node_ids']),
                      'amendment_id': amendment['amendment_id'], 'executor_pid': os.getpid(), 'active': active}
            if failures:
                write(root / 'status.json', {**status, 'status': 'ENGINEERING_STOP', 'failed_nodes': failures})
                return
            if len(receipts) == len(nodes) and not active:
                summarize(root, receipts, amendment, write)
                write(root / 'aggregate_results_reduced.json', {
                    'status': 'COMPLETE_REDUCED_PROTOCOL', 'execution_commit': config['execution_commit'],
                    'controller_commit': os.environ['CONTROLLER_COMMIT'], 'amendment': amendment,
                    'receipts': list(receipts.values()), 'population': 'development patients; one new optimization seed'})
                write(root / 'status.json', {**status, 'status': 'COMPLETE'})
                return
            pending = [n for n in nodes if n['id'] not in receipts and n['id'] not in active and set(n['dependencies']) <= set(receipts)]
            for node in [n for n in pending if n['kind'] == 'selection']:
                receipt = NativeRunner.selection(node, receipts)
                receipt['schedule_amendment_id'] = amendment['amendment_id']
                write(root / node['id'] / 'receipt.json', receipt)
                receipts[node['id']] = receipt
            used = {v['gpu'] for v in active.values()}
            free = {int(a): int(b) for a, b in (line.split(',') for line in subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines())}
            for node in [n for n in pending if n['kind'] != 'selection']:
                gpu = next((g for g in (5, 6, 7) if g not in used and free.get(g, 0) >= 12000), None)
                if gpu is None:
                    break
                nr = root / node['id']
                if any((nr / x).exists() for x in ('launch.json', 'physical.jsonl', 'latest.pt', 'failure.json')):
                    raise RuntimeError('previously launched node cannot be retried: ' + node['id'])
                nr.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NODE_ID=node['id'],
                           EXEC_MODULE='experiments.lcrseg.five_frameworks_v1.native_runner')
                with (nr / 'worker.log').open('a') as log:
                    child = subprocess.Popen([sys.executable, '-c', 'import os,runpy;runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")'],
                                             cwd=config['code'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state = process_state(child.pid)
                if state is None:
                    raise RuntimeError('worker exited during launch: ' + node['id'])
                active[node['id']] = {'pid': child.pid, 'gpu': gpu, 'start_ticks': state[1]}
                children[node['id']] = child
                used.add(gpu)
                write(nr / 'launch.json', {**active[node['id']], 'execution_commit': config['execution_commit'],
                                          'node_id': node['id'], 'time': time.time(), 'schedule_amendment_id': amendment['amendment_id']})
            write(root / 'status.json', {**status, 'sealed': len(receipts), 'active': active})
            time.sleep(3)


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        import traceback
        # A dispatcher failure must be visible; no worker is killed or retried.
        config = read(os.environ['EXEC_CONFIG'])
        root = Path(config['run_root'])
        from experiments.lcrseg.five_frameworks_v1.native_runner import write
        write(root / 'reduced_executor_failure.json', {'status': 'ENGINEERING_STOP', 'traceback': traceback.format_exc()})
        raise
