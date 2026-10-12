"""User-reduced single-trajectory completion; never runs an optimizer."""
import copy
import json
import os
import time
from pathlib import Path
from . import protocol
from .run import engine

JOB = 'SIMPLE_B1_S163_O1'


def single_compare(rows, reference, arms=protocol.ARMS):
    table = {(r['arm'], r['seed'], r['order']): r for r in rows}
    result = {}
    for arm in arms:
        current, control = table[arm, 163, 1], table[reference, 163, 1]
        delta = {key: current['metrics'][key]-control['metrics'][key]
                 for key in ('Final', 'Old', 'Incoming', 'Forget')}
        cells = [dict(seed=163, order=1, domain=domain, cls=cls,
                      delta=current['scores'][domain][cls]-control['scores'][domain][cls])
                 for domain in current['scores'] for cls in ('rim', 'cup')]
        result[arm] = dict(pairs=[dict(seed=163, order=1, **delta)], mean=delta,
                           per_seed={'163':delta}, per_order={'1':delta}, class_costs=cells,
                           practical_development_gate=None, independent_confirmation=False,
                           scope='one preselected running seed/order; no two-seed practical gate')
    return result


def single_plan(original):
    value = copy.deepcopy(original)
    value.pop('plan_id')
    value.update(study_id='SIMPLE_B1_CONF_KL_V1_USER_REDUCED', seeds=[163], orders=[original['orders'][0]],
                 jobs=[j for j in original['jobs'] if j['id']==JOB],
                 execution_batches=[[JOB]],
                 authority='user 2026-10-12: 一个种子就行，一个顺序就行，我想尽快得到结果',
                 evaluation='both selected target stages sealed and selected job EXIT0 before five seen-domain evaluations',
                 amendment='original training and PLAN retained; cancel only unstarted trajectories before any readout',
                 comparison='only matching seed163/order1 of historical B0/B2 and matched F5/V13/V17; no multi-seed gate',
                 limitations=original['limitations']+['single seed and single order; descriptive result only'])
    value['caps']['formal_updates'] = 5300
    value['plan_id'] = protocol.digest(value)
    return value


def check():
    original = protocol.plan(); reduced = single_plan(original)
    assert len(original['jobs']) == 4 and original['caps']['formal_updates'] == 21200
    assert len(reduced['jobs']) == 1 and reduced['caps']['formal_updates'] == 5300
    assert sum(s['steps'] for s in reduced['jobs'][0]['stages']) == 5300
    scores = {d:dict(rim=.4, cup=.6, macro_Dice=.5) for d in original['orders'][0]}
    rows = [dict(arm=arm, seed=163, order=1, scores=copy.deepcopy(scores),
                 metrics=dict(Final=.5, Old=.5, Incoming=.5, Forget=.1))
            for arm in (protocol.B1, 'CONTROL')]
    rows[0]['metrics']['Final'] += .02
    compared = single_compare(rows, 'CONTROL')[protocol.B1]
    assert abs(compared['mean']['Final']-.02) < 1e-12
    assert compared['practical_development_gate'] is None and len(compared['class_costs']) == 6
    assert len(compared['pairs']) == 1 and set(compared['per_seed']) == {'163'}
    return dict(status='PASS', checks=['original matrix unmodified', 'single selected job and5300 cap',
                                      'paired single comparison and all six class cells', 'multi-seed gate explicitly unavailable'],
                optimizer_updates=0, model_forwards=0, patient_access=0)


def main():
    config = protocol.read(os.environ['EXEC_CONFIG']); root = Path(config['run_root'])
    original = protocol.read(root/'PLAN.json'); amendment = protocol.read(root/'SCOPE_AMENDMENT.json')
    assert original == json.loads(json.dumps(protocol.plan()))
    assert amendment['selected_job'] == JOB and amendment['effective_formal_cap'] == 5300
    assert amendment['training_execution_commit'] == config['execution_commit']
    assert (root/'CODE_COMMIT').read_text().strip() == config['execution_commit']
    assert (Path(config['code'])/'CODE_COMMIT').read_text().strip() == config['readout_execution_commit']
    qualification = protocol.read(root/'QUALIFICATION.json')
    assert qualification['status'] == 'PASS' and qualification['execution_commit'] == config['execution_commit']
    assert protocol.read(root/'QUALIFICATION_EXIT.json')['exit_code'] == 0
    assert sorted(p.name for p in (root/'jobs').iterdir()) == [JOB]
    if (root/'CONTROLLER_EXIT_SINGLE.json').exists() or (root/'RESULTS.json').exists():
        raise FileExistsError('single completion already attempted; no repeat evaluation')
    plan = single_plan(original)
    assert plan == protocol.read(root/'SINGLE_READOUT_PLAN.json')
    started = time.time(); job_root = root/'jobs'/JOB
    try:
        protocol.write(root/'STATUS.json', dict(status='RUNNING', phase='single_trajectory_train',
                                               arm=protocol.B1, effective_formal_cap=5300,
                                               plan_id=plan['plan_id']))
        worker = protocol.read(job_root/'STARTUP.json')['pid']
        while not (job_root/'EXIT.json').exists():
            os.kill(worker, 0)
            time.sleep(5)
        if protocol.read(job_root/'EXIT.json')['exit_code'] != 0:
            raise RuntimeError('selected worker failed; no retry')
        for stage in plan['jobs'][0]['stages']:
            seal = protocol.read(job_root/f'stage{stage["stage"]}'/'SEALED.json')
            assert seal['identity']['execution_commit'] == config['execution_commit']
            assert seal['identity']['job_id'] == JOB and seal['steps'] == stage['steps']
        engine.jobs = lambda: plan['jobs']
        engine.plan = lambda: plan
        engine.CAPS = plan['caps']; engine.ARMS = protocol.ARMS; engine.compare = single_compare
        protocol.write(root/'STATUS.json', dict(status='RUNNING', phase='single_sealed_readout', plan_id=plan['plan_id']))
        engine.final_readout(config)
        results = protocol.read(root/'RESULTS.json')
        for version, arm in ((13, 'CLASS_REFERENCE'), (17, 'SARD')):
            context = protocol.read(config[f'v{version}_comparison_results'])['rows']
            results[f'vs_V{version}'] = single_compare(results['rows']+context, arm)
        results.update(status='COMPLETE_SINGLE_DEVELOPMENT_ONLY', scope_amendment=amendment,
                       readout_execution_commit=config['readout_execution_commit'],
                       practical_two_seed_gate_available=False)
        protocol.write(root/'RESULTS.json', results)
        (root/'REPORT.md').write_text('\n'.join([
            '# Simple B1: user-reduced single-seed, single-order result', '',
            'Seed163, REFUGE→RIM→Drishti; two stages sealed before five scheduled domain evaluations.',
            'The user selected the already running first trajectory for speed before any efficacy readout.',
            'Only the same seed/order is compared with frozen references; no multi-seed gate or generalization claim.',
            '', '| Final % | Old % | Incoming % | Forget pp |', '|---:|---:|---:|---:|',
            '|'+ '|'.join(f'{100*results["rows"][0]["metrics"][m]:.4f}' for m in ('Final','Old','Incoming','Forget'))+'|',
            '', 'All paired positive/negative domain/class effects and the preserved scope amendment are in RESULTS.json.',
            'The other three trajectories were never started; original qualification and all physical costs remain.'
        ])+'\n')
        total = sum(sum(1 for line in (job_root/f'stage{s["stage"]}'/'physical.jsonl').open() if line.strip())
                    for s in plan['jobs'][0]['stages'])
        if total != 5300:
            raise RuntimeError('single physical ledger mismatch')
        protocol.write(root/'COSTS.json', dict(formal_updates=total, source_updates=0,
            qualification=qualification['costs'], original_formal_cap=21200, user_reduced_formal_cap=5300,
            cancelled_unstarted_trajectories=3, readout_domain_evaluations=5,
            elapsed_seconds=time.time()-started))
        protocol.write(root/'COMPLETION_AUDIT.json', dict(status='PASS', jobs=1, target_stages=2,
            formal_updates=total, readout_seal=protocol.read(root/'READOUT_SEAL.json'),
            original_plan_preserved=True, user_scope_amendment='SCOPE_AMENDMENT.json'))
        final = protocol.read(root/'FINAL.json')
        final.update(status='COMPLETE_SINGLE_DEVELOPMENT_ONLY', scope_amendment='SCOPE_AMENDMENT.json',
                     readout_execution_commit=config['readout_execution_commit'])
        protocol.write(root/'FINAL.json', final)
        protocol.write(root/'STATUS.json', dict(status='COMPLETE_SINGLE_PENDING_PUBLIC_DELIVERY',
                                               phase='complete', plan_id=plan['plan_id']))
        protocol.write(root/'CONTROLLER_EXIT_SINGLE.json', dict(exit_code=0, seconds=time.time()-started))
    except BaseException as error:
        protocol.write(root/'CONTROLLER_FAILURE_SINGLE.private.json', dict(error=repr(error),
            traceback=__import__('traceback').format_exc()))
        protocol.write(root/'STATUS.json', dict(status='STOPPED_SINGLE', phase='engineering', error_type=type(error).__name__))
        protocol.write(root/'CONTROLLER_EXIT_SINGLE.json', dict(exit_code=1, seconds=time.time()-started))
        raise


if __name__ == '__main__':
    main()
