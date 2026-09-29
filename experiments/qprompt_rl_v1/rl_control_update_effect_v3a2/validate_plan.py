"""Validate the proposed experiment's arithmetic and dependency graph only.

No medical data, model construction, network calls, or optimizer execution.
Usage: python validate_plan.py [path/to/EXECUTION_PLAN.json]
"""
from __future__ import annotations
import json
import math
import sys
from collections import Counter
from pathlib import Path


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(path: Path) -> dict:
    p = json.loads(path.read_text(encoding='utf-8'))
    nodes = p['nodes']
    ids = [x['id'] for x in nodes]
    check(len(ids) == len(set(ids)), 'Duplicate task id')
    universe = set(ids)
    for n in nodes:
        check(set(n['depends_on']) <= universe, 'Missing dependency for ' + n['id'])
    finished: set[str] = set()
    while len(finished) < len(nodes):
        ready = {n['id'] for n in nodes if n['id'] not in finished and set(n['depends_on']) <= finished}
        check(bool(ready), 'Cyclic task graph')
        finished |= ready
    panels = [x for x in nodes if x['kind'] == 'counterfactual_panel']
    features = [x for x in nodes if x['kind'] == 'feature_extraction']
    neural = [x for x in nodes if x['kind'] == 'neural_fit']
    analytic = [x for x in nodes if x['kind'] == 'analytic_fit']
    check(len(panels) == 12, 'Expected 12 prefix cells')
    check(sum(x['scenes'] for x in panels) == 192, 'Expected 192 scenes')
    check(sum(x['branches'] for x in panels) == 1728, 'Expected 1,728 branches')
    check(sum(x['real_student_optimizer_calls'] for x in nodes) == 8640, 'Student call arithmetic')
    check(len(neural) == 60, 'Expected 60 neural fits')
    check(sum(x['controller_optimizer_calls'] for x in neural) == 15360, 'Controller arithmetic')
    check(len(analytic) == 60, 'Expected 60 analytic/ridge fits')
    check(sum(x['virtual_adamw_previews'] for x in features) == 1728, 'Preview arithmetic')
    check(sum(x['feature_vjp_calls'] for x in features) == 2304, 'Feature VJP arithmetic')
    check(p['budgets']['main_real_student_optimizer_calls'] == 8640, 'Cap mismatch')
    check(p['budgets']['new_retained_student_updates'] == 0, 'Unexpected retained student training')
    check(p['budgets']['new_medical_endpoints'] == 0, 'Unexpected medical endpoints')
    check(len(p['features']['schema']) == 40, 'Feature schema width')
    check([x['index'] for x in p['features']['schema']] == list(range(40)), 'Feature ordering')
    check(len({x['name'] for x in p['features']['schema']}) == 40, 'Feature names not unique')
    check(p['learners']['parameter_count'] == 40*32 + 32 + 32*3 + 3 == 1411, 'Parameter count')
    check(p['learners']['epsilon_mix'] == 0, 'Unintended forced exploration')
    prior = p['learners']['prior']
    exps = [math.exp(math.log(x)) for x in prior]
    probs = [x/sum(exps) for x in exps]
    check(max(abs(a-b) for a,b in zip(probs, prior)) < 1e-14, 'Prior initialization')
    check(max(range(3), key=lambda j: probs[j]) == 2, 'Greedy FINE must be reachable')
    for j in range(16):
        origin = 20 * (60*j//16)
        check(origin + 4 < 1200, 'Scene exceeds frozen schedule')
        check(origin//20 == (origin+4)//20, 'Five-step branch crosses L role rotation')
    for a, actions in p['scenes']['branch_actions'].items():
        check(actions == [int(a),2,2,2,2], 'Only first action may vary')
    for n in panels:
        if n['seed'] != 261:
            check('SEAL_ALL_TRANSFER_PREDICTIONS' in n['depends_on'], 'Test rewards before prediction seal')
    # Compute transitive predecessors of test panels, proving all final fits are frozen first.
    by_id = {n['id']: n for n in nodes}
    def ancestors(task_id: str) -> set[str]:
        todo = list(by_id[task_id]['depends_on'])
        found: set[str] = set()
        while todo:
            x = todo.pop()
            if x not in found:
                found.add(x)
                todo.extend(by_id[x]['depends_on'])
        return found
    final_fits = {n['id'] for n in neural if n['phase'] == 'final_fit_seed261'}
    for n in panels:
        if n['seed'] != 261:
            check(final_fits <= ancestors(n['id']), 'Test panel missing prior final policy fit')
    return {
        'status': 'PASSED_PLANNING_ARITHMETIC_ONLY',
        'nodes': len(nodes), 'node_types': dict(Counter(n['kind'] for n in nodes)),
        'prefix_cells': 12, 'scenes': 192, 'branches': 1728,
        'planned_real_student_calls': 8640,
        'planned_controller_calls': 15360,
        'planned_virtual_adamw_previews': 1728,
        'planned_feature_vjps': 2304,
        'neural_fits': 60, 'analytic_fits': 60,
        'new_retained_student_updates': 0, 'new_medical_endpoints': 0,
        'actual_optimizer_calls_by_this_validator': 0,
        'actual_medical_file_reads_by_this_validator': 0,
        'prefix_payload_verified': False,
        'student_runner_implemented_or_tested': False,
        'remote_execution_started': False,
    }


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('EXECUTION_PLAN.json')
    try:
        result = validate(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'FAILED', 'error': str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
