"""One GPU job; reuse native source, student, data, snapshots and evaluator."""
import copy
import csv
import fcntl
import gc
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e, runner as io
from experiments.qprompt_rl_v1.v3c_rule_timing.runner import source
from experiments.lcrseg.five_frameworks_v1.native_operations import NativeOperations
from . import controller as policy, protocol as p

ROOT, C = io.ROOT, io.C
CAMPAIGN = Path(C['campaign'])


def status(name, **extra):
    e.write(ROOT/'status.json', dict(status=name, pid=os.getpid(), time=time.time(), **extra))


class Ledger(e.Ledger):
    """Finite update accounting, explicitly without a wall-clock deadline."""
    smoke_context = ''

    def call(self, category, key, fn):
        if category == 'smoke':
            key = self.smoke_context + '/' + key
        assert self.count[category] < self.caps[category], 'physical budget: ' + category
        identity = category, key
        assert identity not in self.seen, 'duplicate optimizer transaction'
        self.seen.add(identity)
        self.count[category] += 1
        row = dict(category=category, key=key, ordinal=self.count[category], time=time.time())
        e.append(ROOT/'PHYSICAL_LEDGER.jsonl', dict(event='attempt', **row))
        try:
            result = fn()
        except BaseException as exc:
            e.append(ROOT/'PHYSICAL_LEDGER.jsonl', dict(event='failure', error=repr(exc), **row))
            raise
        e.append(ROOT/'PHYSICAL_LEDGER.jsonl', dict(event='success', **row))
        return result


def create(seed, domain, development, ledger, reused=False):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    e.random.seed(seed); np.random.seed(seed)
    config = dict(C, source=C['reused_source'] if reused else C['source'])
    t = e.create(config, seed, domain, development, ledger)
    assert t.options['total_steps'] == p.HORIZONS[domain]
    if development:
        assert not (set(t.provider.roles['online']) & set(t.provider.roles['audit']))
        assert not (set(t.provider.roles['fit']) & (set(t.provider.roles['online']) | set(t.provider.roles['audit'])))
        e.write(ROOT/f'ROLES_{seed}_{domain}.private.json', t.provider.roles)
    return t


def features(t):
    """Same 40 feature equations, candidates now mean lambda 0/.125/.5."""
    started = time.time()
    with t.readonly():
        params = [v for v in t.model.parameters() if v.requires_grad]
        names = [n for n, v in t.model.named_parameters() if v.requires_grad]
        labeled, pred, teacher, valid = t.components()
        base, _ = e.state_vector(teacher, pred, valid, labeled, t.step,
                                t.optimizer.param_groups[0]['lr']/t.scheduler.base_lrs[0])
        base[0] = t.step / p.HORIZONS[t.provider.domain]
        u = e.u_loss(pred, teacher, valid, 2)[0]
        targets = [labeled + weight * u for weight in p.WEIGHTS]
        gradients = [list(torch.autograd.grad(loss, params, retain_graph=i < 2, allow_unused=True))
                     for i, loss in enumerate(targets)]
        x, y = t.provider.clean_fit(t.cursor)
        qgrad = torch.autograd.grad(-e.quality(t.clean(x), y), params, allow_unused=True)
        updates = [e.effect.preview(t.optimizer, params, g)[0] for g in gradients]
        z = base.cpu().tolist() + e.effect.features(params, names, gradients, qgrad, updates)
        assert len(z) == 40 and np.isfinite(z).all()
    e.append(ROOT/'FEATURE_COST.jsonl', dict(time=time.time(), seconds=time.time()-started,
                                           step=t.step, VJP=4, virtual_previews=3))
    return z


def advance(t, ledger, category, key, action, steps=p.BLOCK, observe=None):
    weight = p.WEIGHTS[action]
    t.options['lambda_U'] = weight
    for i in range(steps):
        ledger.update(t, category, key + '/' + str(i), -1 if action == 0 else 2)
        assert t.last['active_U'] == (weight != 0)
        if observe is not None:
            observe(i + 1)


def load_policy(path):
    value = torch.load(path, map_location='cpu', weights_only=False)
    model = policy.ActorCritic()
    model.load_state_dict(value['state'])
    model.eval()
    return model, value


def save_policy(model, path, **extra):
    e.atomic_save(dict(state=e.cpu(model.state_dict()), commit=C['commit'], **extra), path)


def choose(model, z, scale):
    with torch.no_grad():
        logits, _ = model(policy.normalize(z, scale))
    # Prefer the full-U reference on an exact tie, as in the historical controller.
    return next(a for a in (2, 1, 0) if float(logits.max()-logits[a]) <= 1e-8)


def smoke(ledger):
    e.write(ROOT/'CONTROLLER_SELF_CHECK.json', policy.self_check())
    checks = {}
    for domain in p.DOMAINS:
        t = create(168, domain, False, ledger, reused=True)
        initial = e.snapshot(t)
        for weight in p.WEIGHTS:
            e.restore(t, initial)
            t.options['lambda_U'] = weight
            ledger.smoke_context = domain + '/' + str(weight)
            checks[ledger.smoke_context] = io.check_native(t, ledger, {})
        e.restore(t, initial)
        root = e.snapshot(t)
        z = features(t)
        assert e.same(root, e.snapshot(t)), 'new feature extraction changed training state'
        branches = []
        for action, weight in ((-1, .5), (2, 0.)):
            e.restore(t, initial); t.options['lambda_U'] = weight
            ledger.smoke_context = domain + '/zero/' + str(action)
            ledger.update(t, 'smoke', 'alias', action)
            snap = e.snapshot(t); snap['action'] = -1
            branches.append(snap)
        assert e.same(*branches), 'zero weight is not the native L-only update'
        # Two fixed training cases: a smoke-only overfit check, never a reported endpoint.
        e.restore(t, initial)
        old = t.provider._l
        t.provider._l = copy.copy(old)
        t.provider._l.rows = old.rows[:2]
        t.provider._l.checked = set()
        x, y = t.provider.clean_fit(0)
        with t.readonly(), torch.no_grad():
            before = float(e.quality(t.clean(x), y))
        ledger.smoke_context = domain + '/two_case'
        for k in range(128):
            advance(t, ledger, 'smoke', str(k), 0, 1)
        with t.readonly(), torch.no_grad():
            after = float(e.quality(t.clean(x), y))
        assert np.isfinite(after) and after > before, ('two-case fit failed', before, after)
        checks[domain+'/two_case'] = dict(before=before, after=after, improved=True)
        t.provider._l = old
        e.restore(t, initial)
        # Verify a full action block and its exact snapshot continuation.
        block_root = e.snapshot(t)
        ledger.smoke_context = domain + '/block/a'
        advance(t, ledger, 'smoke', 'block', 1)
        expected = e.snapshot(t)
        e.restore(t, block_root)
        ledger.smoke_context = domain + '/block/b'
        advance(t, ledger, 'smoke', 'block', 1)
        assert e.same(expected, e.snapshot(t))
        checks[domain+'/block'] = dict(action_steps=25, exact_resume=True, feature_immutable=True)
        io.save_state(t, ROOT/f'{domain}_smoke.pt', synthetic_endpoint=True)
        del t, initial, root, branches, block_root, expected
        gc.collect(); torch.cuda.empty_cache()
    assert ledger.count['smoke'] == 390
    e.write(ROOT/'INTEGRATION_CHECK.json', dict(status='PASS', checks=checks, calls=390))


def panel(ledger):
    seed, domain = C['seed'], C['domain']
    t = create(seed, domain, True, ledger, reused=True)
    h = p.HORIZONS[domain]
    rows = []
    for step in range(h):
        if step % p.PANEL_STRIDE == 0:
            root = e.snapshot(t)
            z = features(t)
            assert e.same(root, e.snapshot(t))
            initial = {role: t.feedback_score(role) for role in ('online', 'audit')}
            rewards = []
            for action in range(3):
                e.restore(t, root)
                values = {}
                def observe(k):
                    if k in (5, 25):
                        values[str(k)] = {role: t.feedback_score(role) for role in ('online', 'audit')}
                advance(t, ledger, 'panel', f'{step}/branch/{action}', action, observe=observe)
                rewards.append(values)
            rows.append(dict(seed=seed, domain=domain, step=step, x=z, before=initial, outcomes=rewards))
            e.write(ROOT/'PANELS.private.json', rows)
            e.restore(t, root)
        advance(t, ledger, 'panel', f'{step}/retained', 2, 1)
        if t.step % 100 == 0 or t.step == h:
            io.save_state(t, ROOT/'latest.pt', phase='panel_retained', panel_rows=len(rows))
    assert len(rows) == h // p.PANEL_STRIDE
    assert ledger.count['panel'] == h + len(rows) * 3 * p.BLOCK
    e.write(ROOT/'PANEL_SUMMARY.json', dict(seed=seed, domain=domain, rows=len(rows),
                                          early_middle_late_covered=True, h=h, student_calls=ledger.count['panel']))


def fit(ledger):
    domain = C['domain']
    rows = []
    for seed in p.DEVELOPMENT_SEEDS:
        rows.extend(e.read(CAMPAIGN/'jobs'/f'panel_{seed}_{domain}'/'PANELS.private.json'))
    x = np.asarray([r['x'] for r in rows]); scale = policy.scaler(x)
    normalized = policy.normalize(x, scale)
    rewards = {}
    models = {}
    for horizon in (5, 25):
        raw = np.asarray([[a[str(horizon)]['online'] for a in r['outcomes']] for r in rows])
        advantage = raw - raw[:, 2:3]
        reward_scale = max(1e-4, float(np.sqrt(np.mean(advantage**2))))
        rewards[horizon] = dict(raw=raw, advantage=advantage, scale=reward_scale)
        name = 'OFFLINE_' + str(horizon)
        models[name] = policy.offline(normalized, advantage/reward_scale, e.stable('V4/fit', domain),
                                      lambda opt, i: ledger.step(opt, 'controller', name+'/'+str(i)))
        save_policy(models[name], ROOT/(name+'.pt'), scaler=scale, reward_scale=reward_scale, horizon=horizon)
    # Diagnostic transfer: fit on two development seeds, predict the third before using its audit targets.
    oof = []
    for heldout in p.DEVELOPMENT_SEEDS:
        train = np.asarray([r['seed'] != heldout for r in rows]); test = ~train
        s = policy.scaler(x[train])
        raw = rewards[25]['advantage'][train]
        rs = max(1e-4, float(np.sqrt(np.mean(raw**2))))
        model = policy.offline(policy.normalize(x[train], s), raw/rs, e.stable('V4/oof', domain, heldout),
                               lambda opt, i: ledger.step(opt, 'controller', f'OOF/{heldout}/{i}'))
        selected = [choose(model, z, s) for z in x[test]]
        for row, action in zip([r for r, keep in zip(rows, test) if keep], selected):
            oof.append(dict(seed=heldout, step=row['step'], action=action,
                            audit_delta=row['outcomes'][action]['25']['audit']-row['outcomes'][2]['25']['audit']))
    increments = rewards[25]['raw'] - np.asarray([r['before']['online'] for r in rows])[:, None]
    online_scale = max(1e-4, float(np.sqrt(np.mean(increments**2))))
    e.write(ROOT/'FROZEN_NORMALIZATION.json', dict(scaler=scale, online_reward_scale=online_scale,
                                                fitting_seeds=p.DEVELOPMENT_SEEDS, contexts=len(rows)))
    diagnostics = []
    for i, row in enumerate(rows):
        online5 = [a['5']['online'] for a in row['outcomes']]
        online25 = [a['25']['online'] for a in row['outcomes']]
        audit25 = [a['25']['audit'] for a in row['outcomes']]
        diagnostics.append(dict(seed=row['seed'], step=row['step'], best5=int(np.argmax(online5)),
                                best25=int(np.argmax(online25)), audit_best25=int(np.argmax(audit25)),
                                selected_online_audit_delta=audit25[int(np.argmax(online25))]-audit25[2]))
    e.write(ROOT/'REWARD_DIAGNOSTICS.json', dict(rows=diagnostics, heldout_seed_oof=oof,
                 limitation='Audit L is held out from gradient fitting, not an independent external patient cohort. No score gate or selection.'))


def train_policy(ledger):
    domain, method = C['domain'], C['method']
    fitroot = CAMPAIGN/'jobs'/('fit_'+domain)
    model, initial = load_policy(fitroot/'OFFLINE_25.pt')
    frozen = e.read(fitroot/'FROZEN_NORMALIZATION.json')
    scaler, reward_scale = frozen['scaler'], frozen['online_reward_scale']
    opt = torch.optim.Adam(model.parameters(), lr=.0003)
    summaries = []
    for episode in range(p.EPISODES):
        seed = p.DEVELOPMENT_SEEDS[episode % len(p.DEVELOPMENT_SEEDS)]
        t = create(seed, domain, True, ledger, reused=True)
        trajectory = []
        previous = t.feedback_score('online')
        beginning = previous
        audit_start = t.feedback_score('audit')
        generator = torch.Generator().manual_seed(e.stable('V4/explore', domain, episode))
        for block in range(p.HORIZONS[domain] // p.BLOCK):
            z = features(t); normalized = policy.normalize(z, scaler)
            with torch.no_grad():
                logits, value = model(normalized)
                probabilities = policy.distribution(logits).probs
                action = int(torch.multinomial(probabilities, 1, generator=generator))
                logp = float(probabilities[action].log())
            advance(t, ledger, 'development', f'{episode}/{block}', action)
            quality = t.feedback_score('online')
            trajectory.append(dict(x=normalized, action=action, logp=logp, value=float(value),
                                   reward=(quality-previous)/reward_scale))
            previous = quality
            if t.step % 100 == 0 or t.step == p.HORIZONS[domain]:
                io.save_state(t, ROOT/'student_latest.pt', episode=episode, method=method)
                e.atomic_save(dict(trajectory=trajectory, generator=generator.get_state(),
                              actor=e.cpu(model.state_dict()), optimizer=e.cpu(opt.state_dict()),
                              episode=episode, scaler=scaler, previous_quality=previous, commit=C['commit']),
                              ROOT/'controller_latest.private.pt')
        assert abs(sum(r['reward'] for r in trajectory)*reward_scale-(previous-beginning)) < 1e-8
        audit_end = t.feedback_score('audit')
        diag = policy.update(model, opt, trajectory, method, e.stable('V4/shuffle', domain, episode),
                             lambda o, k: ledger.step(o, 'controller', f'{episode}/{k}'))
        summaries.append(dict(episode=episode, seed=seed, online_delta=previous-beginning,
                              audit_delta=audit_end-audit_start, actions=dict(Counter(r['action'] for r in trajectory)),
                              controller_updates=diag))
        e.write(ROOT/'LEARNING_CURVES.json', summaries)
        save_policy(model, ROOT/'latest_policy.pt', scaler=scaler, reward_scale=reward_scale,
                    method=method, episode=episode, optimizer=e.cpu(opt.state_dict()))
        del t, trajectory; gc.collect(); torch.cuda.empty_cache()
    save_policy(model, ROOT/'FROZEN_POLICY.pt', scaler=scaler, reward_scale=reward_scale,
                method=method, episodes=p.EPISODES, selection='last episode; no audit/val selection')


def evaluate_file(path):
    env = dict(os.environ, EVAL_INPUT=str(path), EXEC_MODULE='experiments.qprompt_rl_v1.v3b_lrref_endpoint.evaluator')
    with (ROOT/'evaluator.log').open('a') as log:
        subprocess.run([sys.executable, '-c', 'import os,runpy;runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")'],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main_matrix(ledger):
    seed, domain = C['seed'], C['domain']
    t = create(seed, domain, False, ledger)
    initial = e.snapshot(t)
    io.save_state(t, ROOT/'entry_state.pt', phase='main_entry')
    io.export(t, ROOT/'entry.pt')
    models = {}
    for name in ('OFFLINE_5', 'OFFLINE_25'):
        models[name] = load_policy(CAMPAIGN/'jobs'/('fit_'+domain)/(name+'.pt'))
    for name in p.LEARNERS:
        models[name] = load_policy(CAMPAIGN/'jobs'/f'learn_{domain}_{name}'/'FROZEN_POLICY.pt')
    schedules = {}; endpoints = []
    for method in p.METHODS:
        e.restore(t, initial)
        actions = []; active = 0; outside = 0; feature_count = 0
        shuffled = None
        if method == 'PHASE_SHUFFLE':
            shuffled = list(schedules['PPO_25'])
            for left, right in p.phases(domain):
                order = np.random.RandomState(e.stable('V4/phase', seed, domain, left)).permutation(right-left)
                shuffled[left:right] = [schedules['PPO_25'][left+i] for i in order]
                assert Counter(shuffled[left:right]) == Counter(schedules['PPO_25'][left:right])
        for block in range(p.HORIZONS[domain] // p.BLOCK):
            if method in ('ORIGINAL', 'FINE_0125', 'FINE_05'):
                action = ('ORIGINAL', 'FINE_0125', 'FINE_05').index(method)
            elif method == 'EARLY_05':
                action = 2 if t.step < p.HORIZONS[domain] // 2 else 0
            elif method == 'PHASE_SHUFFLE':
                action = shuffled[block]
            else:
                z = features(t)
                if method == 'EFFECT_RULE':
                    scores = [z[16] + z[22], z[28] + z[34], 0.]
                    action = next(a for a in (2, 1, 0) if max(scores)-scores[a] <= 1e-12)
                else:
                    model, saved = models[method]
                    action = choose(model, z, saved['scaler'])
                    outside += int(((np.asarray(z) < saved['scaler']['minimum']) | (np.asarray(z) > saved['scaler']['maximum'])).sum())
                    feature_count += 40
            actions.append(action)
            advance(t, ledger, 'main', f'{method}/{block}', action)
            active += p.BLOCK * int(action != 0)
            e.append(ROOT/'ACTIONS.jsonl', dict(seed=seed, domain=domain, method=method,
                     block=block, step=t.step, action=action, weight=p.WEIGHTS[action]))
            if t.step % 100 == 0 or t.step == p.HORIZONS[domain]:
                io.save_state(t, ROOT/(method+'_latest.pt'), method=method, active_U_calls=active, actions=actions)
            if t.step == 1200 or t.step == p.HORIZONS[domain]:
                io.export(t, ROOT/f'{method}_{t.step}.pt')
            status('MAIN_MATRIX', seed=seed, domain=domain, method=method, step=t.step, H=p.HORIZONS[domain])
        schedules[method] = actions
        endpoints.append(dict(seed=seed, domain=domain, method=method, step=t.step, active_U_calls=active,
                              out_of_range_fraction=outside/feature_count if feature_count else None,
                              action_counts=[actions.count(a) for a in range(3)],
                              path=str(ROOT/f'{method}_{t.step}.pt')))
        e.write(ROOT/'ENDPOINT_REGISTRY.json', endpoints)
    assert ledger.count['main'] == len(p.METHODS)*p.HORIZONS[domain]
    del t, models, initial; gc.collect(); torch.cuda.empty_cache()
    # Confirmation scores are produced only after all methods in this cell are frozen.
    status('EVALUATION', seed=seed, domain=domain)
    evaluate_file(ROOT/'entry.pt')
    start = e.read(ROOT/'entry.scores.json')['scores']
    rows = []
    for endpoint in endpoints:
        path = Path(endpoint.pop('path'))
        evaluate_file(path)
        scores = e.read(path.with_suffix('.scores.json'))['scores']
        rows.append(dict(endpoint, macro_Dice=scores[domain]['macro_Dice'], rim=scores[domain]['rim'],
                         cup=scores[domain]['cup'], disc_union=scores[domain]['disc_union'],
                         old_REFUGE=scores['REFUGE']['macro_Dice'],
                         old_change=scores['REFUGE']['macro_Dice']-start['REFUGE']['macro_Dice']))
    e.write(ROOT/'RESULTS.json', rows)


def audit(ledger):
    events = [e.json.loads(line) for line in (ROOT/'PHYSICAL_LEDGER.jsonl').read_text().splitlines()]
    a = Counter((r['category'], r['key']) for r in events if r['event'] == 'attempt')
    b = Counter((r['category'], r['key']) for r in events if r['event'] == 'success')
    assert a == b and all(n == 1 for n in a.values())
    assert all(ledger.count[k] == v for k, v in C['caps'].items())
    e.write(ROOT/'COMPLETION_AUDIT.json', dict(status='PASS', physical_calls=dict(ledger.count),
                                             failures=0, duplicate_keys=0, commit=C['commit']))


def main():
    p.check(); torch.set_num_threads(2); torch.cuda.set_device(0)
    handle = (ROOT/'EXECUTOR.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (ROOT/'PHYSICAL_LEDGER.jsonl').exists(), 'explicit recovery required; no implicit retries'
    ledger = Ledger(ROOT); ledger.caps = dict(C['caps']); ledger.seen = set()
    status('STARTING', job=C['job'])
    try:
        if C['job'] == 'smoke': smoke(ledger)
        elif C['job'] == 'source': source(C['seed'], ledger, {})
        elif C['job'] == 'panel': panel(ledger)
        elif C['job'] == 'fit': fit(ledger)
        elif C['job'] == 'learn': train_policy(ledger)
        elif C['job'] == 'main': main_matrix(ledger)
        else: raise ValueError('unknown job')
        audit(ledger)
        e.write(ROOT/'FINAL.json', dict(status='COMPLETE', commit=C['commit'], time=time.time(),
                                       physical_calls=dict(ledger.count), peak_cuda_allocated=torch.cuda.max_memory_allocated()))
        status('COMPLETE', job=C['job'], physical_calls=dict(ledger.count))
    except BaseException as exc:
        status('FAILED', error=repr(exc), traceback=traceback.format_exc(), physical_calls=dict(ledger.count))
        raise


if __name__ == '__main__':
    with NativeOperations(ROOT/'operations'):
        main()
