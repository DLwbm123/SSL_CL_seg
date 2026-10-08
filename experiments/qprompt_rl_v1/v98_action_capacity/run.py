"""Add three memory-mixture actions to the sealed finite-grid diagnostic."""
import fcntl
import importlib.util
import json
import os
import random
import subprocess
import time
import traceback
from pathlib import Path

CAPS = dict(qualification=8, audit=26400)
WEIGHTS = ((1., 1., 1.), (.5, 1.25, 1.25), (.5, 1., 1.5))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def install_actions(c):
    original = c.transfer_loss
    assert c.ACTIONS == [(a, b) for a in (0., .25, .5) for b in WEIGHTS]

    def expanded(p, q, valid, action, qm=None, qflip=None):
        if action not in range(12):raise ValueError('unknown action')
        if action < 9:return original(p, q, valid, action, qm, qflip)
        if qm is None or qflip is None:raise PermissionError('memory-unavailable action')
        # Original actions dispatch unchanged; only the new alpha=.75 branch is added.
        q = q.detach();valid = valid.bool();mask = valid & (q.max(1).values > .7)
        gate = valid & (qm.detach().max(1).values > .7) & (qm.detach().argmax(1) == qflip.detach().argmax(1))
        target = (1-.75*gate[:, None])*q+.75*gate[:, None]*qm.detach()
        raw = q.new_tensor(WEIGHTS[action-9])[q.argmax(1)];weight = raw*mask
        weight = weight*mask.sum()/weight.sum().clamp_min(1e-8)
        loss = ((target*(target.clamp_min(1e-8).log()-p.float().clamp_min(1e-8).log())).sum(1)*weight).sum()/valid.sum().clamp_min(1)
        stats = dict(admitted=int(mask.sum()), valid=int(valid.sum()), gate=float(gate.sum()/valid.sum().clamp_min(1)),
                     weight_raw=float((raw*mask).sum()), weight_normalized=float(weight.sum()))
        return loss, stats, target

    c.transfer_loss = expanded
    return original


def action_check(c, original):
    import torch
    def field(values):return torch.tensor(values).reshape(1, 3, 1, 1).expand(1, 3, 2, 2).clone().requires_grad_()
    p, q, qm, qflip = field([.2, .3, .5]), field([.8, .1, .1]), field([.1, .8, .1]), field([.1, .8, .1])
    with torch.no_grad():
        q[0, :, 0, 1] = q.new_tensor([.1, .8, .1]);q[0, :, 1, 0] = q.new_tensor([.1, .1, .8])
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    for action in range(9):
        old = original(p, q, valid, action, qm, qflip);new = c.transfer_loss(p, q, valid, action, qm, qflip)
        assert torch.equal(old[0], new[0]) and old[1] == new[1] and torch.equal(old[2], new[2])
        assert torch.equal(torch.autograd.grad(old[0], p)[0], torch.autograd.grad(new[0], p)[0])
    for action in range(9, 12):
        loss, stats, target = c.transfer_loss(p, q, valid, action, qm, qflip)
        assert torch.allclose(target, .25*q.detach()+.75*qm.detach()) and not target.requires_grad
        assert torch.all(target >= 0) and torch.allclose(target.sum(1), torch.ones_like(target[:, 0]))
        assert torch.isfinite(loss) and loss >= 0 and abs(stats['weight_normalized']-4) < 1e-6
        gradients = torch.autograd.grad(loss, (p, q, qm, qflip), allow_unused=True)
        assert torch.isfinite(gradients[0]).all() and all(g is None for g in gradients[1:])
        off = c.transfer_loss(p, q, valid, action, qm, field([.1, .1, .8]))
        assert torch.equal(off[2], q.detach()) and off[1]['gate'] == 0
        empty = c.transfer_loss(p, q, ~valid, action, qm, qflip)
        assert torch.isfinite(empty[0]) and empty[0] == 0
    for action, memory in ((12, qm), (9, None)):
        try:c.transfer_loss(p, q, valid, action, memory, qflip)
        except (ValueError, PermissionError):pass
        else:raise AssertionError('invalid action or missing memory accepted')


def gpu_job(root, cfg):
    import numpy as np
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c, stage_b as b
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    torch.set_num_threads(2);torch.cuda.set_device(0)
    random.seed(168);np.random.seed(168);torch.manual_seed(168)
    original = install_actions(c);action_check(c, original)
    roles = c.split_roles(cfg['data']);assert roles == G.read(Path(cfg['action_root'])/'ROLES.private.json')
    ledger = b.JobLedger(root, cfg['caps']);queries = 0;campaign = Path(cfg['campaign'])
    if cfg['job'] == 'qualification':
        for index, action in ((0, 9), (7, 11)):
            ctx = b.contexts()[index];t = b.make(cfg, roles, ledger, ctx, 300)
            entry = torch.load(Path(cfg['original'])/f'jobs/decision_entries/{ctx[3]}/ENTRY100.private.pt', map_location='cpu', weights_only=False)
            assert entry['step'] == 100 and entry['options']['total_steps'] == 300
            c.restore(t, entry);t.category = 'qualification';t.key = ctx[3]
            before = c.snapshot(t);t.extract()
            torch.multinomial(torch.ones(12), 1, generator=torch.Generator().manual_seed(861000+index))
            assert c.e.same(before, c.snapshot(t)), 'inference altered student state/RNG'
            t.action = action
            for _ in range(2):t.update()
            expected = c.snapshot(t);c.restore(t, entry);t.action = action
            for _ in range(2):t.update()
            assert c.e.same(expected, c.snapshot(t)), 'new action paired continuation mismatch'
            del t;torch.cuda.empty_cache()
        assert dict(ledger.count) == {'qualification':8}
        G.write(root/'QUALIFICATION.json', dict(status='PASS', native_updates=8, exact_pairs=2, new_actions=[9, 11],
                all_three_new_actions_synthetic='PASS', original_nine_loss_gradient_exact=True, teacher_detach=True,
                policy_student_rng_isolation=True, query_accesses=0))
    else:
        i = cfg['context_index'];old = Path(cfg['v97']);previous = [r for r in G.read(old/'GRID_RESULTS.json')['rows'] if r['context'] == f'dev{i}']
        assert len(previous) == 81
        m, n, kind, factor = G.read(Path(cfg['code'])/'experiments/qprompt_rl_v1/v8_grpo_transfer_prior/ACTION_AUDIT_CONFIG.json')['dev_contexts'][i]
        ctx = (m, n, (kind, factor), f'dev{i}');t = b.make(cfg, roles, ledger, ctx, 300)
        def restore(path, step):
            value = torch.load(path, map_location='cpu', weights_only=False)
            assert value['step'] == step and value['options']['total_steps'] == 300
            c.restore(t, value);return value
        def query(role, condition):
            nonlocal queries
            queries += len(roles[role]);record = dict(role=role, step=t.step, image_calls_upper_bound=queries)
            c.e.append(root/'QUERY_LEDGER.jsonl', dict(event='attempt', **record))
            try:result = scores(t, roles, role, condition)
            except BaseException:
                c.e.append(root/'QUERY_LEDGER.jsonl', dict(event='failure', **record));raise
            c.e.append(root/'QUERY_LEDGER.jsonl', dict(event='success', **record));return result
        if cfg['job'] == 'train':
            assert G.read(campaign/'jobs/qualification/QUALIFICATION.json')['status'] == 'PASS'
            entry = restore(Path(cfg['original'])/f'jobs/development/dev{i}/ENTRY.private.pt', 100)
            t.category = 'audit';sealed = []
            for first in range(12):
                if first < 9:prefix = restore(old/f'jobs/train{i}/FIRST_{first}_PREFIX.private.pt', 200)
                else:
                    c.restore(t, entry);t.action = first;t.key = f'dev{i}/prefix{first}'
                    for _ in range(100):
                        t.update()
                        if t.step == 150:c.e.atomic_save(c.snapshot(t), root/f'FIRST_{first}_MID.private.pt')
                    assert t.step == 200;prefix = c.snapshot(t);c.e.atomic_save(prefix, root/f'FIRST_{first}_PREFIX.private.pt')
                for second in range(9 if first < 9 else 0, 12):
                    c.restore(t, prefix);t.action = second;t.key = f'dev{i}/{first}_{second}'
                    for _ in range(100):t.update()
                    assert t.step == 300;c.e.atomic_save(c.snapshot(t), root/f'PAIR_{first}_{second}_FINAL.private.pt')
                    sealed.append(dict(context=f'dev{i}', first=first, second=second))
                    G.write(root/'STATUS.json', dict(status='RUNNING', endpoints=len(sealed), physical=dict(ledger.count)))
            assert dict(ledger.count) == {'audit':6600} and queries == 0 and len(sealed) == 63
            G.write(root/'SEALED.json', dict(status='SEALED', endpoints=sealed, new_midpoints=3, physical=dict(ledger.count), time=time.time()))
        elif cfg['job'] == 'evaluate':
            assert G.read(campaign/'ENDPOINT_LOCK.json')['endpoints'] == 576
            assert len({r['new_entry'] for r in previous}) == len({r['old_memory_reference'] for r in previous}) == 1
            newentry, ref = previous[0]['new_entry'], previous[0]['old_memory_reference'];src = campaign/f'jobs/train{i}'
            rows = [dict(r, source='REUSED_V97') for r in previous]
            for first in range(12):
                if first < 9:
                    mids = [r['new50'] for r in previous if r['first'] == first]
                    assert len(mids) == 9 and all(m == mids[0] for m in mids);mid = mids[0]
                else:
                    restore(src/f'FIRST_{first}_MID.private.pt', 150);mid = query('Q_dev_new', ctx[2])
                for second in range(9 if first < 9 else 0, 12):
                    restore(src/f'PAIR_{first}_{second}_FINAL.private.pt', 300)
                    new = query('Q_dev_new', ctx[2]);oldscore = query('Q_dev_old', ('identity', 1.))
                    v = B.reward(mid['macro'], new['macro'], newentry, oldscore['macro'], ref)
                    rows.append(dict(context=f'dev{i}', first=first, second=second, new50=mid, new=new, old=oldscore,
                        new_entry=newentry, old_memory_reference=ref, gain=v['gain'], forget_penalty=v['forget_penalty'],
                        utility=v['reward'], absolute_forgetting=ref-oldscore['macro'], source='V98_NEW'))
            assert queries == 516 and not ledger.count and len(rows) == 144
            G.write(root/'RESULTS.json', rows)
        else:raise ValueError('unknown job')
    G.write(root/'FINAL.json', dict(status='COMPLETE', physical=dict(ledger.count), query_image_evaluations=queries, time=time.time()))


def schedule(root, cfg, jobs, account):
    pending = list(jobs);active = {};failed = [];done = []
    while pending or active:
        for slot, (p, spec) in list(active.items()):
            if p.poll() is None:continue
            dest = root/'jobs'/spec['id'];G.write(dest/'PROCESS_EXIT.json', dict(exit_code=p.returncode, time=time.time()))
            (failed if p.returncode or not (dest/'FINAL.json').exists() else done).append(spec['id']);del active[slot]
        if not failed:
            for slot in (5, 6, 7):
                if not pending:break
                if slot in active:continue
                if int(subprocess.check_output(['nvidia-smi', '--id='+str(slot), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) < 12000:continue
                spec = pending.pop(0);dest = root/'jobs'/spec['id'];dest.mkdir(exist_ok=False)
                G.write(dest/'CONFIG.private.json', dict(cfg, **spec, campaign=str(root), gpu=slot))
                env = dict(os.environ, EXEC_RUN=str(dest), EXEC_CONFIG=str(dest/'CONFIG.private.json'), EXEC_ENTRY=cfg['entry'], EXEC_MODE='job',
                           PYTHONPATH=cfg['code'], CUDA_VISIBLE_DEVICES=str(slot), CUBLAS_WORKSPACE_CONFIG=':4096:8')
                with (dest/'worker.log').open('x') as log:
                    p = subprocess.Popen(['bash', './with_nas_storage.sh', cfg['python'], '-c', 'import os,runpy;runpy.run_path(os.environ["EXEC_ENTRY"],run_name="__main__")'], cwd=cfg['code']+'/experiments/lcrseg/scripts', env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                G.write(dest/'LAUNCH.json', dict(pid=p.pid, gpu=slot, time=time.time(), commit=cfg['commit']));active[slot] = p, spec
        account.refresh()
        G.write(root/'COSTS.json', dict(scope='V98_NEW_ONLY', attempts=dict(account.count), success=dict(account.success), failures=dict(account.failure), student_cap=26408, actor_cap=0))
        G.write(root/'STATUS.json', dict(status='ENGINEERING_STOP' if failed else 'RUNNING', active=[s['id'] for _, s in active.values()], pending=[s['id'] for s in pending], completed=done, failed=failed, physical=dict(account.count), time=time.time()))
        if failed and not active:raise RuntimeError('failed jobs: '+','.join(failed))
        if pending or active:time.sleep(10)


def coordinator(root, cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock = (root/'COORDINATOR.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    old = Path(cfg['v97']);assert G.read(old/'FINAL.json')['status'] == 'COMPLETE'
    oldrows = G.read(old/'GRID_RESULTS.json')['rows'];olddecision = G.read(old/'DECISION.json')
    assert len(oldrows) == len({(r['context'], r['first'], r['second']) for r in oldrows}) == 324
    assert olddecision['status'] == 'NO_ACTION_SEQUENCE_MARGIN_ON_FROZEN_D1_STREAMS'
    assert G.read(old/'COSTS.json')['attempts'] == G.read(old/'COSTS.json')['success'] == dict(qualification=8, audit=36000)
    oldjobs = list((old/'jobs').iterdir());assert len(oldjobs) == 9
    assert all(G.read(j/'PROCESS_EXIT.json')['exit_code'] == 0 and G.read(j/'FINAL.json')['status'] == 'COMPLETE' for j in oldjobs)
    replay = G.read(old/'REPLAY_AUDIT.json');assert len(replay) == 168 and all(r['pass_tolerance'] for r in replay)
    G.write(root/'REUSE_AUDIT.json', dict(status='PASS', old_endpoints=324, old_midpoints=36, historical_replays=168, new_updates=0))
    quals = [dict(id='qualification', job='qualification', caps=dict(qualification=8))]
    trains = [dict(id=f'train{i}', job='train', context_index=i, caps=dict(audit=6600)) for i in range(4)]
    evals = [dict(id=f'eval{i}', job='evaluate', context_index=i, caps={}) for i in range(4)]
    (root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    account = Accounting(root, caps=CAPS, jobs=quals+trains+evals)
    schedule(root, cfg, quals, account);schedule(root, cfg, trains, account)
    assert dict(account.count) == CAPS;sealed = []
    for i in range(4):
        src = root/f'jobs/train{i}';receipt = G.read(src/'SEALED.json');assert len(receipt['endpoints']) == 63
        for row in receipt['endpoints']:assert (src/f"PAIR_{row['first']}_{row['second']}_FINAL.private.pt").is_file()
        for a in range(9, 12):
            assert (src/f'FIRST_{a}_MID.private.pt').is_file() and (src/f'FIRST_{a}_PREFIX.private.pt').is_file()
        sealed += receipt['endpoints']
    G.write(root/'ENDPOINT_LOCK.json', dict(endpoints=576, new_endpoints=252, reused_endpoints=324, midpoints=48, new_midpoints=12, sealed=sealed, time=time.time(), all_before_new_evaluation=True))
    schedule(root, cfg, evals, account)
    rows = sum((G.read(root/f'jobs/eval{i}/RESULTS.json') for i in range(4)), [])
    assert len(rows) == len({(r['context'], r['first'], r['second']) for r in rows}) == 576
    subgrid = [{k:v for k,v in r.items() if k != 'source'} for r in rows if r['source'] == 'REUSED_V97']
    assert subgrid == oldrows and G.feasibility(subgrid, olddecision['controls']) == olddecision
    decision = G.feasibility(rows, olddecision['controls']);assert decision['combinations'] == 144**4
    decision['status'] = 'EXPANDED_ACTION_MARGIN_EXISTS_ON_FROZEN_D1_ONLY' if decision['feasible_combinations'] else 'STOP_V98_NO_EXPANDED_ACTION_MARGIN'
    decision['old_subgrid_unchanged'] = True;decision['automatic_research_stop'] = False
    decision['marginal_bounds'] = {phase:sum(max(r[phase]['macro'] for r in rows if r['context'] == f'dev{i}') for i in range(4))/4 for phase in ('new', 'old')}
    G.write(root/'GRID_RESULTS.json', dict(rows=rows, decision=decision));G.write(root/'DECISION.json', decision)
    B.csvfile(root/'ACTION_GRID.csv', [dict(context=r['context'], first=r['first'], second=r['second'], source=r['source'], **{k:r[k] for k in ('gain', 'forget_penalty', 'utility', 'absolute_forgetting')}, **{f'{phase}_{k}':v for phase in ('new50', 'new', 'old') for k,v in r[phase].items()}) for r in rows])
    qd = sum(G.read(root/f'jobs/eval{i}/FINAL.json')['query_image_evaluations'] for i in range(4));assert qd == 2064
    G.write(root/'COSTS.json', dict(attempts=dict(account.count), success=dict(account.success), failures=dict(account.failure), student_total=26408, actor_total=0, training_query_image_evaluations=0, development_query_image_evaluations=qd, new_unique_images=0, new_endpoints=252, reused_endpoints=324))
    (root/'REPORT.md').write_text('# V98 expanded memory-mixture capacity\n\n'+decision['status']+'\n\n'+json.dumps(decision, indent=2)+'\n\nFull grid is repeated-D1 diagnostic only, not a deployable policy or independent confirmation. No development labels enter actor training. Historical verdicts retained; scientific criteria are comparative diagnostics, not automatic research stops.\n')
    G.write(root/'FINAL.json', dict(status='COMPLETE', decision=decision['status'], physical=dict(account.count), publication='PENDING', time=time.time()))
    G.write(root/'STATUS.json', G.read(root/'FINAL.json'))


if __name__ == '__main__':
    cfg = json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root = Path(os.environ['EXEC_RUN'])
    G = load('grid_base', cfg['grid_entry']);B = load('reward_base', cfg['base_entry']);G.selfcheck()
    if os.environ.get('EXEC_SELFCHECK') == '1':
        from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c
        original = install_actions(c);action_check(c, original)
        print('PASS original/new actions, detach, gates, gradients and feasibility; zero optimizer/query updates')
    else:
        with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(), time=time.time(), commit=cfg['commit']), f)
        try:
            if os.environ.get('EXEC_MODE') == 'job':gpu_job(root, cfg)
            else:coordinator(root, cfg)
        except BaseException as exc:
            G.write(root/'STATUS.json', dict(status='ENGINEERING_STOP', error=repr(exc), traceback=traceback.format_exc(), time=time.time()));raise
