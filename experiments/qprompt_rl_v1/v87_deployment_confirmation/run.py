"""Frozen actors, native student continuations, then one sealed D1 readout."""
import csv
import fcntl
import json
import os
import random
import statistics as st
import subprocess
import time
import traceback
from pathlib import Path

REUSED = ('NATIVE', 'FIXED_BEST', 'UNIFORM_ACTION')
LEARNED = tuple(f'{a}_{s}' for a in ('Z_64', 'RAW_1024', 'Z_1024') for s in (601, 602))
NEW = ('TIME_FIXED',) + LEARNED
CONTEXTS = ((2000, 2, ('brightness', 1.2), 'dev0'),
            (2000, 8, ('contrast', .8), 'dev1'),
            (8000, 2, ('contrast', .8), 'dev2'),
            (8000, 8, ('brightness', 1.2), 'dev3'))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path);tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n');tmp.replace(path)


def summarize(rows):
    keys = ('new', 'old', 'utility', 'gain', 'forget_penalty', 'absolute_forgetting')
    means = {m: {k: st.mean(r[k]['macro'] if k in ('new', 'old') else r[k]
                            for r in rows if r['method'] == m) for k in keys} for m in REUSED+NEW}
    for arm in ('Z_64', 'RAW_1024', 'Z_1024'):
        means[arm] = {k: st.mean(means[f'{arm}_{s}'][k] for s in (601, 602)) for k in keys}
    primary = means['Z_1024'];comparators = REUSED+('TIME_FIXED', 'Z_64', 'RAW_1024')
    delta = {m: {k: primary[k]-means[m][k] for k in ('new', 'old', 'utility')} for m in comparators}
    paired = {str(s): means[f'Z_1024_{s}']['utility']-means[f'Z_64_{s}']['utility'] for s in (601, 602)}
    tradeoff = {m: (delta[m]['new'] >= .002 and delta[m]['old'] >= -.0025)
                  or (delta[m]['old'] >= .005 and delta[m]['new'] >= -.0025)
               for m in ('UNIFORM_ACTION', 'Z_64')}
    passed = all(v['utility'] >= .0005 for v in delta.values()) and all(v > 0 for v in paired.values()) and all(tradeoff.values())
    return dict(status='PASS_V87_CURRENT_D1_DEV_ONLY' if passed else 'STOP_V87_NO_PRACTICAL_DEPLOYMENT_GAIN',
                means=means, primary_deltas=delta, paired_seed_utility=paired, tradeoff=tradeoff,
                independent_confirmation=False, V84='NOT_RUN')


def selfcheck():
    rows = [dict(method=m, context=f'dev{i}', new={'macro': .8+(.003 if m.startswith('Z_1024') else 0)},
                 old={'macro': .8}, utility=.003 if m.startswith('Z_1024') else 0.,
                 gain=0., forget_penalty=0., absolute_forgetting=.01) for i in range(4) for m in REUSED+NEW]
    assert summarize(rows)['status'].startswith('PASS')
    assert summarize([dict(r, utility=0.) for r in rows])['status'].startswith('STOP')
    assert summarize([dict(r, new={'macro': .8}) for r in rows])['status'].startswith('STOP')
    assert summarize([dict(r, utility=.004) if r['method'] == 'Z_64_602' else r for r in rows])['status'].startswith('STOP')


def job(root, cfg):
    import numpy as np
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c, stage_b as b
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    e = c.e
    torch.set_num_threads(2);torch.cuda.set_device(0)
    random.seed(168);np.random.seed(168);torch.manual_seed(168)
    with (root/'STARTED.json').open('x') as f:
        json.dump(dict(pid=os.getpid(), time=time.time(), commit=cfg['commit']), f)
    ledger = b.JobLedger(root, cfg['caps'])
    roles = c.split_roles(cfg['data']);assert roles == e.read(Path(cfg['action_root'])/'ROLES.private.json')
    source = Path(cfg['original'])/'jobs/development'
    historical = read(source/'DEVELOPMENT_RESULTS.json')['results']
    models = {}
    for method in LEARNED:
        payload = torch.load(cfg['actors'][method], map_location='cpu', weights_only=False)
        assert payload['heldout'] == 'FULL_TABLE' and payload['seed'] == int(method.rsplit('_', 1)[1])
        assert payload['updates'] == (64 if method.startswith('Z_64') else 1024)
        assert payload['arm'] == ('Z_T025' if method.startswith('Z_64') else method.rsplit('_', 1)[0])
        actor = c.Actor(int(method.rsplit('_', 1)[1]));actor.load_state_dict(payload['actor']);actor.eval()
        assert payload['mean'].shape == payload['scale'].shape == (24,)
        assert torch.isfinite(payload['mean']).all() and torch.isfinite(payload['scale']).all() and (payload['scale'] > 0).all()
        models[method] = (actor, payload['mean'], payload['scale'])

    def select(t, method, generator):
        before = c.snapshot(t);z = t.extract()
        if method == 'TIME_FIXED':
            action = {100: 6, 200: 8}[t.step];prob = torch.nn.functional.one_hot(torch.tensor(action), 9).float()
        else:
            actor, center, scale = models[method]
            with torch.no_grad():prob = actor((z-center)/scale).softmax(-1)
            assert torch.isfinite(prob).all() and abs(float(prob.sum())-1.) < 1e-6
            action = int(torch.multinomial(prob, 1, generator=generator))
        assert e.same(before, c.snapshot(t)), 'controller inference mutated student state/global RNG'
        return action, dict(step=t.step, state=z.tolist(), probabilities=prob.tolist(), action=action)

    def trainer(index):
        ctx = CONTEXTS[index];t = b.make(cfg, roles, ledger, ctx, 300)
        entry = torch.load(source/ctx[3]/'ENTRY.private.pt', map_location='cpu', weights_only=False)
        c.restore(t, entry);assert t.step == 100 and t.options['total_steps'] == 300
        return ctx, t, entry

    if cfg['job'] == 'qualification':
        for index in (0, 3):
            ctx, t, entry = trainer(index);t.category = 'qualification';t.key = ctx[3]
            action, _ = select(t, 'Z_1024_601', torch.Generator().manual_seed(860301+index))
            t.action = action
            for _ in range(2):t.update()
            continuous = c.snapshot(t);c.restore(t, entry)
            # Exercise every frozen transformation while leaving the physical ledger outside restore.
            for method in LEARNED:select(t, method, torch.Generator().manual_seed(860301+index))
            t.action = action
            for _ in range(2):t.update()
            assert e.same(continuous, c.snapshot(t)), 'paired native continuation mismatch'
            del t;torch.cuda.empty_cache()
        assert ledger.count['qualification'] == 8
        write(root/'QUALIFICATION.json', dict(status='PASS', updates=8, exact_full_state_pairs=2,
             contexts=['dev0', 'dev3'], controller_state_rng_isolation=True, role_identity=True,
             native_engine_qualification='REUSED_V83', actor_updates=0))
    elif cfg['job'] == 'train':
        assert read(Path(cfg['campaign'])/'jobs/qualification/QUALIFICATION.json')['status'] == 'PASS'
        index = cfg['context_index'];ctx, t, entry = trainer(index)
        for method in NEW:
            c.restore(t, entry);t.category = 'development';t.key = ctx[3]+'/'+method
            generator = torch.Generator().manual_seed(860301+index);decisions = []
            for offset in range(200):
                if offset % 100 == 0:
                    t.action, row = select(t, method, generator);decisions.append(row)
                t.update()
                if t.step == 150:e.atomic_save(c.snapshot(t), root/(method+'_MID.private.pt'))
                if t.step % 50 == 0:
                    write(root/'STATUS.json', dict(status='RUNNING', method=method, step=t.step,
                                                   physical=dict(ledger.count)))
            assert t.step == 300
            e.atomic_save(c.snapshot(t), root/(method+'_FINAL.private.pt'))
            write(root/(method+'_DECISIONS.private.json'), decisions)
            write(root/(method+'_TRAINING.json'), dict(status='SEALED', context=ctx[3], method=method,
                  updates=200, actions=[r['action'] for r in decisions], step=300))
        assert ledger.count['development'] == 1400
    elif cfg['job'] == 'evaluate':
        campaign = Path(cfg['campaign']);assert read(campaign/'ENDPOINT_LOCK.json')['endpoints'] == 40
        index = cfg['context_index'];ctx, t, _ = trainer(index);rows = []
        entry_values = [r for r in historical if r['context'] == ctx[3]]
        assert len({r['new_entry'] for r in entry_values}) == len({r['old_memory_reference'] for r in entry_values}) == 1
        entrynew = entry_values[0]['new_entry'];reference = entry_values[0]['old_memory_reference']
        for method in NEW:
            src = campaign/f'jobs/train{index}'
            c.restore(t, torch.load(src/(method+'_MID.private.pt'), map_location='cpu', weights_only=False))
            mid = scores(t, roles, 'Q_dev_new', ctx[2])
            c.restore(t, torch.load(src/(method+'_FINAL.private.pt'), map_location='cpu', weights_only=False))
            new = scores(t, roles, 'Q_dev_new', ctx[2]);old = scores(t, roles, 'Q_dev_old', ('identity', 1.))
            gain = .25*(mid['macro']-entrynew)+.75*(new['macro']-entrynew);penalty = max(0., reference-old['macro']-.005)
            rows.append(dict(context=ctx[3], method=method, new50=mid, new=new, old=old, new_entry=entrynew,
                old_memory_reference=reference, gain=gain, forget_penalty=penalty, utility=gain-penalty,
                absolute_forgetting=reference-old['macro'], source='NEW',
                actions=read(src/(method+'_TRAINING.json'))['actions']))
        assert not ledger.count
        write(root/'RESULTS.json', rows)
    else:raise ValueError('unknown job')
    write(root/'FINAL.json', dict(status='COMPLETE', physical=dict(ledger.count), time=time.time()))


def audit(root, cfg):
    """Admit only the exact existing controls; no regeneration or score-based selection."""
    import torch
    source = Path(cfg['original'])/'jobs/development';old = read(source/'CONFIG.private.json')
    for key in ('data', 'reference', 'action_root', 'training_horizon', 'code'):assert cfg[key] == old[key]
    assert old['training_horizon'] == 300 and read(source/'ENDPOINT_LOCK.json')['all_twenty_sealed']
    assert read(Path(cfg['original'])/'jobs/qualification/QUALIFICATION.json')['status'] == 'PASS'
    assert read(Path(cfg['original'])/'FIXED_BEST.json')['action'] == 8
    dense = [json.loads(line) for stream in (1, 2) for line in
             (Path(cfg['original'])/f'jobs/audit_{stream}/ACTION_ROWS.jsonl').read_text().splitlines()]
    assert len(dense) == 288
    for step, action in ((100, 6), (200, 8)):
        values = [st.mean(r['reward'] for r in dense if r['entry_step'] == step and r['action'] == a) for a in range(9)]
        assert max(range(9), key=lambda a: (values[a], -a)) == action
    registered = read(Path(cfg['code'])/'experiments/qprompt_rl_v1/v8_grpo_transfer_prior/ACTION_AUDIT_CONFIG.json')
    assert registered['dev_contexts'] == [[m, n, kind, factor] for m, n, (kind, factor), _ in CONTEXTS]
    rows = read(source/'DEVELOPMENT_RESULTS.json')['results'];admitted = []
    for _, _, _, context in CONTEXTS:
        entry = torch.load(source/context/'ENTRY.private.pt', map_location='cpu', weights_only=False)
        assert entry['step'] == 100 and entry['options']['total_steps'] == 300
        for method in REUSED:
            receipt = read(source/context/(method+'_TRAINING.json'))
            assert receipt['status'] == 'SEALED' and receipt['steps'] == 200
            if method in ('NATIVE', 'FIXED_BEST'):assert receipt['actions'] == ([0, 0] if method == 'NATIVE' else [8, 8])
            else:
                g = torch.Generator().manual_seed(860301+int(context[-1]))
                assert receipt['actions'] == [int(torch.multinomial(torch.ones(9)/9, 1, generator=g)) for _ in range(2)]
            for stage, step in (('MID', 150), ('FINAL', 300)):
                state = torch.load(source/context/(method+'_'+stage+'.private.pt'), map_location='cpu', weights_only=False)
                assert state['step'] == step and state['provider_identity'] == entry['provider_identity']
                assert state['condition'] == entry['condition'] and state['options'] == entry['options']
                del state
            row = next(r for r in rows if r['context'] == context and r['method'] == method)
            admitted.append(dict(row, source='REUSED_V83', actions=receipt['actions'],
                                 absolute_forgetting=row['old_memory_reference']-row['old']['macro']))
    assert len(admitted) == 12
    write(root/'REUSE_AUDIT.json', dict(status='PASS', admitted=12, same_native_source=True,
          same_entry_lifecycle=True, same_categorical_rng=True, regenerated_updates=0))
    write(root/'REUSED_RESULTS.json', admitted)


def schedule(root, cfg, jobs, accounting):
    pending = list(jobs);active = {};failed = [];done = []
    while pending or active:
        for gpu, (process, spec) in list(active.items()):
            if process.poll() is None:continue
            dest = root/'jobs'/spec['id'];write(dest/'PROCESS_EXIT.json', dict(exit_code=process.returncode, time=time.time()))
            (failed if process.returncode or not (dest/'FINAL.json').exists() else done).append(spec['id'])
            del active[gpu]
        if not failed:
            for gpu in (5, 6, 7):
                if not pending:break
                if gpu in active:continue
                free = int(subprocess.check_output(['nvidia-smi', '--id='+str(gpu), '--query-gpu=memory.free',
                                                   '--format=csv,noheader,nounits'], text=True).strip())
                if free < 12000:continue
                spec = pending.pop(0);dest = root/'jobs'/spec['id'];dest.mkdir(exist_ok=False)
                write(dest/'CONFIG.private.json', dict(cfg, **spec, campaign=str(root), gpu=gpu))
                env = dict(os.environ, EXEC_RUN=str(dest), EXEC_CONFIG=str(dest/'CONFIG.private.json'),
                           EXEC_MODE='job', EXEC_ENTRY=cfg['entry'], PYTHONPATH=cfg['code'],
                           CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8')
                with (dest/'worker.log').open('x') as log:
                    process = subprocess.Popen(['bash', './with_nas_storage.sh', cfg['python'], '-c',
                        'import os,runpy;runpy.run_path(os.environ["EXEC_ENTRY"],run_name="__main__")'],
                        cwd=cfg['code']+'/experiments/lcrseg/scripts', env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                write(dest/'LAUNCH.json', dict(pid=process.pid, gpu=gpu, time=time.time(), commit=cfg['commit']))
                active[gpu] = process, spec
        accounting.refresh()
        write(root/'COSTS.json', dict(attempts=dict(accounting.count), success=dict(accounting.success),
              failures=dict(accounting.failure), scope='V87_NEW_ONLY', student_cap=5608, actor_updates=0,
              historical_cost=read(root/'PREREGISTRATION.json')['historical_cost']))
        write(root/'STATUS.json', dict(status='ENGINEERING_STOP' if failed else 'RUNNING',
              active=[j['id'] for _, j in active.values()], pending=[j['id'] for j in pending],
              completed=done, failed=failed, physical=dict(accounting.count), time=time.time()))
        if failed and not active:raise RuntimeError('job failed: '+','.join(failed))
        if pending or active:time.sleep(10)


def coordinator(root, cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock = (root/'COORDINATOR.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(), time=time.time(), commit=cfg['commit']), f)
    assert read(Path(cfg['v86'])/'NEXT_STAGE_DECISION.json')['status'] == 'READY_FOR_SEPARATELY_PREREGISTERED_DEV_CONFIRMATION'
    audit(root, cfg)
    qualify = [dict(id='qualification', job='qualification', caps=dict(qualification=8))]
    train = [dict(id=f'train{i}', job='train', context_index=i, caps=dict(development=1400)) for i in range(4)]
    evaluate = [dict(id=f'eval{i}', job='evaluate', context_index=i, caps={}) for i in range(4)]
    (root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    account = Accounting(root, caps=dict(qualification=8, development=5600), jobs=qualify+train+evaluate)
    schedule(root, cfg, qualify, account);schedule(root, cfg, train, account)
    sealed = []
    for i in range(4):
        for method in NEW:
            dest = root/f'jobs/train{i}';receipt = read(dest/(method+'_TRAINING.json'))
            assert receipt['status'] == 'SEALED' and receipt['step'] == 300
            assert all((dest/(method+'_'+s+'.private.pt')).is_file() for s in ('MID', 'FINAL'))
            sealed.append(dict(context=f'dev{i}', method=method, status='SEALED'))
    assert dict(account.count) == dict(qualification=8, development=5600)
    write(root/'ENDPOINT_LOCK.json', dict(endpoints=40, new=sealed, reused=12, time=time.time(), all_before_new_evaluation=True))
    schedule(root, cfg, evaluate, account)
    rows = read(root/'REUSED_RESULTS.json')
    for i in range(4):rows += read(root/f'jobs/eval{i}/RESULTS.json')
    assert len(rows) == 40 and len({(r['context'], r['method']) for r in rows}) == 40
    result = summarize(rows);write(root/'DEVELOPMENT_RESULTS.json', dict(rows=rows, **result));write(root/'DECISION.json', result)
    context_deltas = []
    for _, _, _, context in CONTEXTS:
        selected = [r for r in rows if r['context'] == context]
        context_deltas.append(dict(context=context, **summarize(selected)))
    write(root/'CONTEXT_COMPARISONS.json', context_deltas)
    write(root/'WORST_CONTEXT_DELTAS.json', {method: {metric: min(r['primary_deltas'][method][metric]
          for r in context_deltas) for metric in ('new', 'old', 'utility')} for method in result['primary_deltas']})
    flat = [dict(context=r['context'], method=r['method'], source=r['source'], actions=str(r['actions']),
        **{k:r[k] for k in ('gain', 'forget_penalty', 'utility', 'absolute_forgetting')},
        **{f'{phase}_{k}': v for phase in ('new50', 'new', 'old') for k,v in r[phase].items()}) for r in rows]
    with (root/'ENDPOINT_RESULTS.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    write(root/'COSTS.json', dict(attempts=dict(account.count), success=dict(account.success), failures=dict(account.failure),
        student_total=5608, actor_updates=0, new_endpoints=28, reused_endpoints=12,
        new_query_image_evaluations=336, unique_dev_images=8, historical_cost=read(root/'PREREGISTRATION.json')['historical_cost']))
    report = ['# V8.7 current-D1 deployment confirmation', '', '**'+result['status']+'**', '',
              '| Method | New | Old | Utility | Absolute forgetting |', '|---|---:|---:|---:|---:|']
    for m,v in result['means'].items():report.append('| '+m+' | '+' | '.join(f'{v[k]:.9f}' for k in ('new','old','utility','absolute_forgetting'))+' |')
    report += ['', 'All28 new endpoints were sealed before any new development scoring; twelve historical controls were audited and reused.',
        'This repeatedly observed current-D1 screen is not independent confirmation; old is simulated memory retention, not a real preceding domain.',
        'Both seeds and every endpoint are retained. No new actor optimization or reward lookup occurs in deployment. V84 stays NOT_RUN.',
        'New student cost5,608 plus disclosed historical preparation; new Qdev image evaluations336 across eight images. See PREREGISTRATION.json for prior costs and label roles.']
    (root/'FINAL_INTERPRETATION.md').write_text('\n'.join(report)+'\n')
    write(root/'FINAL.json', dict(status='COMPLETE', decision=result['status'], physical=dict(account.count), publication='PENDING', time=time.time()))
    write(root/'STATUS.json', read(root/'FINAL.json'))


if __name__ == '__main__':
    selfcheck()
    if os.environ.get('EXEC_SELFCHECK') == '1':print('PASS: V87 primary gate self-check; no optimizer updates')
    else:
        run = Path(os.environ['EXEC_RUN']);config = read(os.environ['EXEC_CONFIG'])
        try:
            (job if os.environ.get('EXEC_MODE') == 'job' else coordinator)(run, config)
        except BaseException as exc:
            write(run/'STATUS.json', dict(status='ENGINEERING_STOP', error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
            raise
