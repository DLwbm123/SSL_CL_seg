"""Validate a completed reduced run and export only public aggregate evidence.

Usage: python export_reduced_native_results.py RUN_ROOT > PUBLIC_RESULTS.json
No patient-level file or model tensor is read; no training is launched.
"""
import collections
import datetime
import json
import math
from pathlib import Path
import sys


def export(root):
    root = Path(root)
    read = lambda path: json.loads(path.read_text())
    status = read(root / 'status.json')
    amendment = read(root / 'SCHEDULE_REDUCTION_20260915.json')
    summary = read(root / 'replication_summary_reduced.json')
    aggregate = read(root / 'aggregate_results_reduced.json')
    assert status['status'] == 'COMPLETE' and status['sealed'] == status['total'] == 328
    assert not status['active']
    assert aggregate['status'] == summary['status'] == 'COMPLETE_REDUCED_PROTOCOL'
    assert not list(root.glob('*/failure.json')) and not (root / 'reduced_executor_failure.json').exists()
    rows = {p.parent.name: read(p) for p in root.glob('*/receipt.json')}
    assert set(rows) == set(amendment['retained_node_ids'])
    assert len(aggregate['receipts']) == len(rows)
    assert {r['node_id'] for r in aggregate['receipts']} == set(rows)
    for ident in amendment['cancelled_node_ids']:
        assert not any((root / ident / name).exists() for name in ('launch.json', 'receipt.json', 'physical.jsonl', 'latest.pt', 'failure.json'))
    caps = {'REFUGE': 8000, 'RIM_ONE_r3': 3200, 'Drishti_GS': 2100}
    totals = collections.defaultdict(lambda: {'stages': 0, 'optimizer_updates': 0, 'summed_worker_seconds': 0., 'cost_counters': collections.Counter(), 'operation_counts': collections.Counter()})
    index = []
    selections = {}
    for ident, r in sorted(rows.items()):
        if r['status'] == 'RESOLVED':
            selections[ident] = {k: r[k] for k in ('candidate_id', 'strong_L_only', 'strong_SSL', 'Final', 'Old', 'compute', 'compute_unit', 'all_candidates', 'schedule_amendment_id') if k in r}
            index.append({'node_id': ident, 'kind': 'selection', 'status': 'RESOLVED'})
            continue
        assert r['status'] == 'SEALED'
        identity = r['identity']
        cap = caps[identity['domain']]
        assert identity['execution_commit'] == amendment['worker_execution_commit']
        assert r['step'] == r['physical_optimizer_calls'] == cap
        assert r['operation_counts']['optimizer_steps'] == r['operation_counts']['optimizer_steps_attempts'] == cap
        with (root / ident / 'physical.jsonl').open('rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 4096))
            last = json.loads(handle.read().decode().splitlines()[-1])
        assert last['invocation'] == cap
        assert (root / ident / 'student.pt').stat().st_size > 0
        phase = 'SOURCE' if identity.get('kind') == 'source' else ident.split('__')[0]
        family = identity.get('family', 'SOURCE')
        for key in ('ALL', phase, phase + '/' + family):
            group = totals[key]
            group['stages'] += 1
            group['optimizer_updates'] += cap
            group['summed_worker_seconds'] += r['seconds']
            group['cost_counters'].update(r.get('cost', {}))
            group['operation_counts'].update(r['operation_counts'])
        for domain in r['scores'].values():
            assert all(math.isfinite(v) for v in domain.values())
        safe = {k: r[k] for k in ('step', 'scores', 'timeline', 'Final', 'Old', 'Incoming', 'Forget', 'seconds', 'physical_optimizer_calls', 'cost', 'operation_counts', 'resolved_options', 'peak_cuda_allocated', 'peak_cuda_reserved') if k in r}
        safe.update(node_id=ident, kind='training', status='SEALED', phase=phase, identity=identity)
        index.append(safe)
    assert totals['SOURCE']['stages'] == 4 and totals['SOURCE']['optimizer_updates'] == 32000
    assert totals['C']['stages'] == 280 and totals['D']['stages'] == 32
    assert totals['C']['optimizer_updates'] + totals['D']['optimizer_updates'] == 826800
    assert totals['ALL']['optimizer_updates'] == 858800
    for family in amendment['replication_families']:
        final = [r for ident, r in rows.items() if ident.startswith('D__') and r['identity']['family'] == family and r['identity']['stage'] == 2]
        assert len(final) == 2 and {(r['identity']['seed'], r['identity']['order']) for r in final} == {(162, 1), (162, 2)}
        for r in final:
            seen = r['seen']
            expected = {'Final': sum(r['scores'][d]['macro_Dice'] for d in seen) / 3,
                        'Old': sum(r['scores'][d]['macro_Dice'] for d in seen[:2]) / 2,
                        'Incoming': r['scores'][seen[2]]['macro_Dice'],
                        'Forget': (r['timeline']['0'][seen[0]]['macro_Dice'] - r['scores'][seen[0]]['macro_Dice'] +
                                   r['timeline']['1'][seen[1]]['macro_Dice'] - r['scores'][seen[1]]['macro_Dice']) / 2}
            assert all(math.isclose(r[k], v, abs_tol=1e-12) for k, v in expected.items())
        for metric in ('Final', 'Old', 'Incoming', 'Forget'):
            assert math.isclose(summary['table'][family][metric], sum(r[metric] for r in final) / 2, abs_tol=1e-12)
    assert len(summary['paired_comparisons']) == 15
    for pair, result in summary['paired_comparisons'].items():
        family, reference = pair.split('__vs__')
        for metric, value in result['mean_delta'].items():
            assert math.isclose(value, summary['table'][family][metric] - summary['table'][reference][metric], abs_tol=1e-12)
        for order, delta in result['order_Final_deltas'].items():
            lookup = lambda f: next(r['Final'] for n, r in rows.items() if n.startswith('D__') and r['identity']['family'] == f and r['identity']['stage'] == 2 and r['identity']['order'] == int(order))
            assert math.isclose(delta, lookup(family) - lookup(reference), abs_tol=1e-12)
    started = read(root / 'executor_launch.json')['time']
    ended = (root / 'status.json').stat().st_mtime
    return {'schema': 'public_reduced_native_results_v1', 'run_id': 'native_lr_src_a_3domain_v1_20260914_01',
            'status': 'VERIFIED_COMPLETE', 'amendment_id': amendment['amendment_id'],
            'worker_commit': aggregate['execution_commit'], 'controller_commit': aggregate['controller_commit'],
            'completed_at_utc': datetime.datetime.fromtimestamp(ended, datetime.timezone.utc).isoformat(),
            'launch_to_completion_hours': (ended - started) / 3600,
            'checks': {'retained_receipts': 328, 'cancelled_unstarted': 112, 'failure_records': 0,
                       'physical_tail_and_optimizer_event_counts_match_per_node_caps': True,
                       'retained_student_files_nonempty': 316, 'summary_and_order_differences_recomputed': True,
                       'scope': 'Receipt and final ledger checks; no reread of every historical event, patient data or weight tensors.'},
            'summary': summary, 'costs': dict(totals), 'selections': selections, 'index': index,
            'cancelled_node_ids': amendment['cancelled_node_ids'],
            'cost_notes': ['Worker seconds include training and per-stage evaluation; sum is not exclusive GPU compute or elapsed wall time.',
                           'Concurrent shared GPU load affects timing. Qualification/smoke costs are separately recorded in COST_SUMMARY.json.',
                           'All four trained sources and all completed development candidates are counted, including discarded configurations.'],
            'privacy': 'No patient identifiers, per-case validation, images, labels, private configuration, raw logs or weights exported.'}


if __name__ == '__main__':
    print(json.dumps(export(sys.argv[1]), indent=2, allow_nan=False))
