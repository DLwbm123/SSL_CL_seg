"""Finite CPU attribution on the already observed reward table; no student API."""
import csv
import json
import os
import runpy
import statistics as st
import time
from pathlib import Path

ARMS = ('RAW_T1', 'RAW_T025', 'Z_T1', 'Z_T025')
NEW = ARMS[1:]
SEEDS = (601, 602)
FLOOR = .0007379373167760999


def mean(rows, key):
    return st.mean(st.mean(r[key] for r in rows if r['context'] == c)
                   for c in sorted({r['context'] for r in rows}))


def gate(rows):
    by_seed = {}
    for seed in SEEDS:
        primary = [r for r in rows if r['arm'] == 'Z_T025' and r['seed'] == seed]
        baseline = [r for r in rows if r['arm'] == 'RAW_T1' and r['seed'] == seed]
        by_seed[str(seed)] = {key: mean(primary, key) for key in
                             ('actor_minus_uniform', 'actor_minus_global', 'actor_minus_time')}
        by_seed[str(seed)]['primary_minus_RAW_T1'] = mean(primary, 'actor_reward')-mean(baseline, 'actor_reward')
    improvement = st.mean(r['primary_minus_RAW_T1'] for r in by_seed.values())
    controls_pass = all(v > 0 for r in by_seed.values() for key, v in r.items() if key != 'primary_minus_RAW_T1')
    return dict(status='READY_FOR_SEPARATELY_PREREGISTERED_DEV_CONFIRMATION'
                if controls_pass and improvement >= .0005 else 'STOP_V85_PRIMARY_MAPPING_GATE_FAILED',
                primary_arm='Z_T025', both_seeds_beat_all_controls=controls_pass,
                mean_gain_vs_RAW_T1=improvement, minimum_mean_gain=.0005, seeds=by_seed,
                no_independent_generalization=True, V84='NOT_RUN')


def selfcheck():
    rows = [dict(arm=a, seed=s, context='synthetic', actor_reward=.001 if a == 'RAW_T1' else .002,
                 actor_minus_uniform=.001, actor_minus_global=.001, actor_minus_time=.001)
            for a in ('RAW_T1', 'Z_T025') for s in SEEDS]
    assert gate(rows)['status'].startswith('READY')
    assert gate([dict(r, actor_minus_time=0.) for r in rows])['status'].startswith('STOP')
    assert gate([dict(r, actor_reward=.001) for r in rows])['status'].startswith('STOP')


def main():
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.core import Actor
    from experiments.qprompt_rl_v1.v83_dense_reward.round import dataset, preferences
    from experiments.qprompt_rl_v1.v83_dense_reward.worker import loss
    selfcheck();torch.set_num_threads(2)
    h = runpy.run_path(os.environ['EXEC_HELPER'])
    source = Path(os.environ['EXEC_SOURCE'])
    assert h['read'](source/'FINAL.json')['status'] == 'COMPLETE'
    assert h['read'](source/'NEXT_STAGE_DECISION.json')['status'] == 'STOP_V83_PRIMARY_GATE_FAILED'
    rows = [json.loads(line) for line in (source/'V83_ACTION_ROWS.jsonl').read_text().splitlines()]
    h['audit'](rows)
    table = dataset(rows, FLOOR)
    contexts = sorted({r['context'] for r in table})
    output = Path(os.environ['EXEC_RUN'])/'results';output.mkdir(exist_ok=False)
    h['save'](output/'STARTED.json', dict(time=time.time(), pid=os.getpid(), protocol='V85_CPU_TARGET_AND_CONDITIONING'))
    ledger = h['Budget'](output, dict(qualification=3, LOCO=3072, full_table=384))

    def targets(rewards, arm):
        p = preferences(rewards, FLOOR)
        if arm.endswith('T1'):return p
        sharp = [v**4 for v in p]
        return [v/sum(sharp) for v in sharp]

    def normalize(x, train, arm):
        if arm.startswith('Z_'):
            center = x[train].mean(0)
            sd = x[train].std(0, unbiased=False)
            scale = torch.where(sd >= 1e-6, sd, torch.ones_like(sd))
        else:
            center, scale = torch.zeros(24), torch.ones(24)
        return (x-center)/scale, center, scale

    x = torch.tensor([r['state'] for r in table])
    qualification = []
    for arm in NEW:
        z = torch.stack((torch.ones(24), -torch.ones(24)))
        z, _, _ = normalize(z, [0, 1], arm)
        actor = Actor(601);opt = torch.optim.Adam(actor.parameters(), lr=.001)
        target = torch.tensor([targets([1.]+[0.]*8, arm), targets([0.]*8+[1.], arm)])
        assert torch.allclose(target.sum(-1), torch.ones(2)) and torch.isfinite(target).all()
        before = float(loss(actor, z, target).detach())
        value = loss(actor, z, target);opt.zero_grad(set_to_none=True);value.backward()
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.))
        ledger.step('qualification', arm, opt)
        assert float(loss(actor, z, target).detach()) < before
        # Unseen holdout perturbations must not change training-fold statistics.
        altered = x.clone();altered[-2:] += 1000
        _, center, scale = normalize(x, list(range(14)), arm)
        _, center2, scale2 = normalize(altered, list(range(14)), arm)
        assert torch.equal(center, center2) and torch.equal(scale, scale2)
        constant, _, constant_scale = normalize(torch.ones(2, 24), [0, 1], arm)
        assert torch.isfinite(constant).all() and torch.all(constant_scale > 0)
        qualification.append(dict(arm=arm, finite_decreasing_objective=True, fold_isolation=True, zero_variance_safe=True))
    h['save'](output/'QUALIFICATION.json', dict(status='PASS', checks=qualification, student_updates=0, actor_updates=3))

    def baseline(filename):
        values = list(csv.DictReader((source/filename).open()))
        for row in values:
            for key in row:
                if key != 'context':row[key] = float(row[key])
            for key in ('seed', 'step', 'support', 'global_action', 'time_action'):row[key] = int(row[key])
            row['arm'] = 'RAW_T1'
        return values

    loco = baseline('LOCO_RESULTS.csv');formal = baseline('FORMAL_POLICY_FIT.csv')
    provenance = [];target_readouts = []
    for arm in NEW:
        for held in contexts + ['FULL_TABLE']:
            train = [i for i, s in enumerate(table) if s['context'] != held]
            test = [i for i, s in enumerate(table) if s['context'] == held] if held != 'FULL_TABLE' else list(range(16))
            assert (len(train), len(test)) == ((16, 16) if held == 'FULL_TABLE' else (14, 2))
            z, center, scale = normalize(x, train, arm)
            transformed = [dict(s, state=z[i].tolist(), preferences=targets(s['rewards'], arm)) for i, s in enumerate(table)]
            target = torch.tensor([transformed[i]['preferences'] for i in train])
            global_action = h['best']([st.mean(table[i]['rewards'][a] for i in train) for a in range(9)])
            time_actions = {t: h['best']([st.mean(table[i]['rewards'][a] for i in train if table[i]['step'] == t)
                                        for a in range(9)]) for t in (100, 200)}
            category = 'full_table' if held == 'FULL_TABLE' else 'LOCO'
            for seed in SEEDS:
                actor = Actor(seed);opt = torch.optim.Adam(actor.parameters(), lr=.001)
                for epoch in range(64):
                    value = loss(actor, z[train], target);assert torch.isfinite(value)
                    opt.zero_grad(set_to_none=True);value.backward()
                    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.))
                    ledger.step(category, f'{arm}/{held}/{seed}/{epoch}', opt)
                checkpoint = output/f'{arm}_{held}_{seed}.private.pt'
                with checkpoint.open('xb') as f:
                    torch.save(dict(actor=actor.state_dict(), mean=center, scale=scale, arm=arm,
                                    heldout=held, seed=seed, updates=64), f)
                provenance.append(dict(arm=arm, heldout=held, seed=seed, actor_updates=64,
                                       sha256=h['digest'](checkpoint)))
                values = h['policy_rows'](actor, transformed, test, global_action, time_actions, seed)
                for row in values:row['arm'] = arm
                (formal if held == 'FULL_TABLE' else loco).extend(values)
            if held == 'FULL_TABLE':
                for s in transformed:
                    target_readouts.append(dict(arm=arm, context=s['context'], step=s['step'],
                        exact_target_reward=sum(p*r for p, r in zip(s['preferences'], s['rewards'])),
                        global_reward=s['rewards'][global_action], time_reward=s['rewards'][time_actions[s['step']]]))
    assert dict(ledger.count) == dict(qualification=3, LOCO=3072, full_table=384)
    h['csvfile'](output/'LOCO_RESULTS.csv', loco)
    h['csvfile'](output/'FORMAL_POLICY_FIT.csv', formal)
    h['csvfile'](output/'EXACT_TARGET_EXPECTATION.csv', target_readouts)
    h['save'](output/'CHECKPOINT_PROVENANCE.json', provenance)
    contrasts = []
    for kind, values in [('LOCO', loco), ('in_table', formal)]:
        index = {(r['arm'], r['seed'], r['context'], r['step']): r['actor_reward'] for r in values}
        for seed in SEEDS:
            for c in contexts:
                for t in (100, 200):
                    r0, r1, z0, z1 = [index[a, seed, c, t] for a in ARMS]
                    contrasts.append(dict(evaluation=kind, seed=seed, context=c, step=t,
                        conditioning_at_T1=z0-r0, conditioning_at_T025=z1-r1,
                        sharpness_at_RAW=r1-r0, sharpness_at_Z=z1-z0,
                        interaction=z1-z0-r1+r0, primary_minus_RAW_T1=z1-r0))
    h['csvfile'](output/'FACTORIAL_CONTRASTS.csv', contrasts)
    decision = gate(loco)
    h['save'](output/'NEXT_STAGE_DECISION.json', decision)
    h['save'](output/'COSTS.json', dict(student_updates=0, actor_updates=dict(ledger.count),
        total_actor_updates=sum(ledger.count.values()), actor_cap=3459,
        query_image_accesses=0, GPU_training_seconds=0,
        reused_RAW_T1=True, historical_dense_information_student_updates=29600,
        generation_charged_once_physically_but_to_all_information_using_methods=True))
    h['save'](output/'PROVENANCE.json', dict(source_decision_sha256=h['digest'](source/'NEXT_STAGE_DECISION.json'),
        source_table_sha256=h['digest'](source/'V83_ACTION_ROWS.jsonl'),
        script_sha256=h['digest'](__file__), helper_sha256=h['digest'](os.environ['EXEC_HELPER']),
        preregistration_commit='d78fb5a', primary_arm='Z_T025', baseline_fits_reused=True))
    report = ['# V8.5 CPU attribution', '', '**'+decision['status']+'**', '',
        '| Evaluation | Arm | Seed | Expected reward | Minus uniform | Minus global | Minus time | Entropy |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for kind, values in [('LOCO', loco), ('in_table', formal)]:
        for arm in ARMS:
            for seed in SEEDS:
                selected = [r for r in values if r['arm'] == arm and r['seed'] == seed]
                report.append(f'| {kind} | {arm} | {seed} | '+' | '.join(f'{mean(selected,k):.9f}' for k in
                    ('actor_reward','actor_minus_uniform','actor_minus_global','actor_minus_time','entropy'))+' |')
    report += ['', 'All metrics are expected raw reward on saved tables. No new student endpoints or image queries were run.',
        'The primary combined arm and its gate were fixed before these fits. Secondary arms do not replace it.',
        'This is an adaptive diagnostic with shared D1 images and model sources, not independent confirmation or a claim of GRPO efficacy.',
        'Target temperature affects only supervised preferences; categorical deployment temperature remains 1.',
        'Physical new cost is 3,459 CPU actor updates; the 29,600 student-update information generation cost remains attributable to all four arms.',
        'RAW_T1 is reused from V8.3-R, not an additional repetition. All new arms retain both final seeds and 64 updates per fit.']
    (output/'FINAL_INTERPRETATION.md').write_text('\n'.join(report)+'\n')
    h['save'](output/'FINAL.json', dict(status='COMPLETE', decision=decision['status'],
        actor_updates=3459, student_updates=0, time=time.time(), publication='PENDING'))


if __name__ == '__main__':
    if os.environ.get('EXEC_SELFCHECK') == '1':
        selfcheck();print('PASS: V85 continuation self-check; zero optimizer updates')
    else:
        main()
