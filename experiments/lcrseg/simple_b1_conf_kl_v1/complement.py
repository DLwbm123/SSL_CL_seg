"""Complete only the three unstarted B1 trajectories; reuse the sealed fourth."""
import copy
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if os.environ['CUBLAS_WORKSPACE_CONFIG'] != ':4096:8':
    raise RuntimeError('original deterministic CuBLAS workspace configuration required')
from . import protocol
from .run import construct, engine

PRIOR_JOB = 'SIMPLE_B1_S163_O1'
PRIOR_COMMIT = '3ba19c9f6c982427d98aa242c7b09816ba3fe1b0'
AUTHORITY = 'user 2026-10-13: 按原配置补齐 B1 的163/O2、164/O1、164/O2；已有163/O1不重跑；固定15900更新；原F5/B2双gate汇总推送；不做V27减权或新增层级变体'
CAPS = dict(formal_updates=15900, source_updates=0, cpu_optimizer_updates=0,
            synthetic_cuda_updates=0, real_smoke_updates=0, diagnostic_vjps=0)


def jobs():
    return [j for j in protocol.jobs() if j['id'] != PRIOR_JOB]


def plan():
    value = copy.deepcopy(protocol.plan())
    value.pop('plan_id')
    value.update(study_id='SIMPLE_B1_CONF_KL_V1_COMPLEMENT', authority=AUTHORITY,
                 caps=CAPS, jobs=jobs(), execution_batches=[[j['id'] for j in jobs()]],
                 inherited_job=PRIOR_JOB, previously_completed_formal_updates=5300,
                 combined_baseline_formal_updates=21200,
                 training_qualification_source=PRIOR_COMMIT,
                 evaluation='all six NEW stages and three EXIT0 before fifteen seen-domain evaluations; reuse prior163/O1 scores without training or evaluation',
                 qualification='reuse original native qualification of unchanged training code; zero-optimizer scope/merge/gate checks',
                 success='report full four-trajectory original F5/B2 gates and all negative cells; no automatic next experiment',
                 limitations=value['limitations']+['163/O1 was previously observed; only the other three trajectories are new'])
    value['plan_id'] = protocol.digest(value)
    return value


def inherited(config):
    root = Path(config['inherited_b1_root'])
    if root.resolve() == Path(config['run_root']).resolve():
        raise PermissionError('the completed baseline root must remain read-only')
    audit = protocol.read(root/'COMPLETION_AUDIT.json')
    qualification = protocol.read(root/'QUALIFICATION.json')
    if audit['status'] != 'PASS' or audit['formal_updates'] != 5300:
        raise ValueError('missing completed single-trajectory receipt')
    if qualification['status'] != 'PASS' or qualification['execution_commit'] != PRIOR_COMMIT:
        raise ValueError('wrong original B1 qualification')
    if protocol.read(root/'QUALIFICATION_EXIT.json')['exit_code'] != 0:
        raise ValueError('original qualification failed')
    if protocol.read(root/'jobs'/PRIOR_JOB/'EXIT.json')['exit_code'] != 0:
        raise ValueError('original training did not finish')
    result = protocol.read(root/'RESULTS.json')
    if [(r['arm'], r['seed'], r['order']) for r in result['rows']] != [(protocol.B1, 163, 1)]:
        raise ValueError('wrong inherited result coverage')
    if {(s['seed'], s['order'], s['stage']) for s in result['stages']} != {(163, 1, 1), (163, 1, 2)}:
        raise ValueError('wrong inherited stage coverage')
    return result


def merge_rows(rows, prior):
    selected = [r for r in rows if r['arm'] == protocol.B1]
    keys = [(r['seed'], r['order']) for r in selected]
    if len(keys) != 3 or set(keys) != {(163, 2), (164, 1), (164, 2)}:
        raise ValueError('new B1 rows must cover exactly the three missing trajectories')
    return rows + copy.deepcopy(prior['rows'])


def check():
    value = plan()
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert len(jobs()) == 3 and sum(s['steps'] for j in jobs() for s in j['stages']) == 15900
    assert value['options'] == protocol.OPTIONS and protocol.CAPS['formal_updates'] == 21200
    scores = {d: dict(rim=.5, cup=.5, macro_Dice=.5) for d in protocol.ORDERS[0]}
    row = dict(arm=protocol.B1, seed=163, order=1, scores=scores,
               metrics=dict(Final=.5, Old=.5, Incoming=.5, Forget=.1))
    prior = dict(rows=[copy.deepcopy(row)])
    rows = [{**copy.deepcopy(row), 'seed': j['seed'], 'order': j['order']} for j in jobs()]
    merged = merge_rows(rows, prior)
    controls = [{**copy.deepcopy(r), 'arm': 'CONTROL'} for r in merged]
    for r in merged:
        r['metrics']['Final'] += .01
    result = protocol.paired_compare(merged+controls, 'CONTROL', arms=protocol.ARMS)[protocol.B1]
    assert result['practical_development_gate'] and len(result['class_costs']) == 24
    assert prior['rows'][0]['metrics']['Final'] == .5
    merged[0]['metrics']['Incoming'] -= .02
    assert not protocol.paired_compare(merged+controls, 'CONTROL', arms=protocol.ARMS)[protocol.B1]['practical_development_gate']
    try:
        merge_rows(rows+prior['rows'], prior)
    except ValueError:
        pass
    else:
        raise AssertionError('rerun/duplicate inherited cell accepted')
    return dict(status='PASS', optimizer_updates=0, model_forwards=0, patient_access=0,
                checks=['exact three missing jobs and15900 cap', 'original configuration unmodified',
                        'four-row/24-class merge without modifying prior result',
                        'original positive and incoming-failure gates', 'duplicate prior cell rejected'])


def bind(config):
    prior = inherited(config)
    original_readout = engine.final_readout
    engine.plan = plan; engine.jobs = jobs; engine.CAPS = CAPS; engine.ARMS = protocol.ARMS
    engine.construct = construct
    engine.compare = lambda rows, reference: protocol.paired_compare(merge_rows(rows, prior), reference, arms=protocol.ARMS)

    def readout(current):
        original_readout(current)
        root = Path(current['run_root']); result = protocol.read(root/'RESULTS.json')
        result['new_rows'] = copy.deepcopy(result['rows'])
        result['rows'] = sorted(merge_rows(result['rows'], prior), key=lambda r: (r['seed'], r['order']))
        result['stages'] += copy.deepcopy(prior['stages'])
        result['stages'].sort(key=lambda r: (r['seed'], r['order'], r['stage']))
        result['inherited_result'] = dict(job=PRIOR_JOB, training_execution_commit=PRIOR_COMMIT,
                                          formal_updates=5300, new_training=0, new_evaluations=0,
                                          previously_observed=True)
        for version, arm in ((13, 'CLASS_REFERENCE'), (17, 'SARD'), (27, 'HIERARCHY_CONTRAST')):
            context = protocol.read(current[f'v{version}_comparison_results'])['rows']
            result[f'vs_V{version}'] = protocol.paired_compare(result['rows']+context, arm, arms=protocol.ARMS)
        protocol.write(root/'RESULTS.json', result)
        means = {m: sum(r['metrics'][m] for r in result['rows'])/4 for m in ('Final', 'Old', 'Incoming', 'Forget')}
        (root/'REPORT.md').write_text('\n'.join([
            '# B1 fixed matrix completed', '',
            'Three newly completed trajectories:163/O2,164/O1,164/O2;15900 formal updates, six new seals and fifteen new seen-domain evaluations.',
            'Previously observed163/O1 is reused without training or evaluation; full baseline total21200 updates. All new stages sealed before readout.',
            '', '| Final % | Old % | Incoming % | Forget pp |', '|---:|---:|---:|---:|',
            '|'+'|'.join(f'{100*means[m]:.4f}' for m in means)+'|', '',
            'Original F5 gate: '+str(result['vs_F5'][protocol.B1]['practical_development_gate']),
            'Original B2 gate: '+str(result['vs_B2'][protocol.B1]['practical_development_gate']),
            'Full paired positive/negative results and the three new rows are in RESULTS.json. No independent patient confirmation or automatic follow-up experiment.'
        ])+'\n')
        final = protocol.read(root/'FINAL.json')
        final.update(inherited_formal_updates=5300, combined_baseline_formal_updates=21200,
                     new_readout_domain_evaluations=15, inherited_readout_domain_evaluations=5)
        protocol.write(root/'FINAL.json', final)

    engine.final_readout = readout


def main():
    config = protocol.read(os.environ['EXEC_CONFIG']); bind(config)
    engine.main()


if __name__ == '__main__':
    main()
