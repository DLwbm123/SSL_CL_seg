"""One fixed 64-vs-1024 CPU comparison, with exact prefix verification."""
import csv
import json
import os
import runpy
import statistics as st
import time
from pathlib import Path


def macro(rows, key):
    return st.mean(st.mean(r[key] for r in rows if r['context'] == c)
                   for c in sorted({r['context'] for r in rows}))


def decision(rows):
    seeds = {}
    for seed in (601, 602):
        long = [r for r in rows if r['arm'] == 'Z_1024' and r['seed'] == seed]
        short = [r for r in rows if r['arm'] == 'Z_64_REUSED' and r['seed'] == seed]
        seeds[str(seed)] = {k: macro(long, k) for k in
                           ('actor_minus_uniform', 'actor_minus_global', 'actor_minus_time')}
        seeds[str(seed)]['gain_vs_64'] = macro(long, 'actor_reward')-macro(short, 'actor_reward')
    gain = st.mean(r['gain_vs_64'] for r in seeds.values())
    passed = gain >= .0005 and all(v > 0 for r in seeds.values() for k, v in r.items() if k != 'gain_vs_64')
    return dict(status='READY_FOR_SEPARATELY_PREREGISTERED_DEV_CONFIRMATION' if passed
                else 'STOP_V86_OPTIMIZATION_BUDGET_INSUFFICIENT', seeds=seeds,
                mean_gain_vs_64=gain, primary='Z_1024', V84='NOT_RUN', independent_confirmation=False)


def selfcheck():
    rows = [dict(arm=a, seed=s, context='synthetic', actor_reward=.001 if a == 'Z_64_REUSED' else .002,
                 actor_minus_uniform=.001, actor_minus_global=.001, actor_minus_time=.001)
            for a in ('Z_64_REUSED', 'Z_1024') for s in (601, 602)]
    assert decision(rows)['status'].startswith('READY')
    assert decision([dict(r, actor_minus_time=0.) for r in rows])['status'].startswith('STOP')
    assert decision([dict(r, actor_reward=.001) for r in rows])['status'].startswith('STOP')


def main():
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.core import Actor
    from experiments.qprompt_rl_v1.v83_dense_reward.round import dataset, preferences
    from experiments.qprompt_rl_v1.v83_dense_reward.worker import loss
    selfcheck();torch.set_num_threads(2)
    h = runpy.run_path(os.environ['EXEC_HELPER'])
    previous = Path(os.environ['EXEC_SOURCE']);source = Path(os.environ['EXEC_TABLE_SOURCE'])
    assert h['read'](previous/'FINAL.json')['status'] == 'COMPLETE'
    assert h['read'](previous/'NEXT_STAGE_DECISION.json')['status'] == 'STOP_V85_PRIMARY_MAPPING_GATE_FAILED'
    rows = [json.loads(s) for s in (source/'V83_ACTION_ROWS.jsonl').read_text().splitlines()]
    h['audit'](rows);table = dataset(rows, .0007379373167760999)
    output = Path(os.environ['EXEC_RUN'])/'results';output.mkdir(exist_ok=False)
    h['save'](output/'STARTED.json', dict(time=time.time(), pid=os.getpid(), protocol='V86_CPU_OPTIMIZATION_BUDGET'))
    ledger = h['Budget'](output, dict(qualification=2, LOCO=32768, full_table=4096))
    x = torch.tensor([s['state'] for s in table]);contexts = sorted({r['context'] for r in table})
    targets = []
    for s in table:
        pref = preferences(s['rewards'], .0007379373167760999);sharp = [p**4 for p in pref]
        targets.append([p/sum(sharp) for p in sharp])
    targets = torch.tensor(targets)

    def baseline(name):
        out = []
        for row in csv.DictReader((previous/name).open()):
            if row['arm'] not in ('RAW_T025', 'Z_T025'):continue
            for k in row:
                if k not in ('arm', 'context'):row[k] = float(row[k])
            for k in ('seed', 'step', 'support', 'global_action', 'time_action'):row[k] = int(row[k])
            row['arm'] = row['arm'].replace('_T025', '_64_REUSED');out.append(row)
        return out

    loco, formal = baseline('LOCO_RESULTS.csv'), baseline('FORMAL_POLICY_FIT.csv')
    prefixes, provenance, losses = [], [], []
    for kind in ('RAW', 'Z'):
        actor = Actor(601);opt = torch.optim.Adam(actor.parameters(), lr=.001)
        z = torch.stack((torch.ones(24), -torch.ones(24)))
        target = torch.tensor([[1.]+[0.]*8, [0.]*8+[1.]])
        before = float(loss(actor, z, target).detach())
        opt.zero_grad(set_to_none=True);loss(actor, z, target).backward()
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.))
        ledger.step('qualification', kind, opt)
        assert float(loss(actor, z, target).detach()) < before
        for held in contexts + ['FULL_TABLE']:
            train = [i for i, s in enumerate(table) if s['context'] != held]
            test = [i for i, s in enumerate(table) if s['context'] == held] if held != 'FULL_TABLE' else list(range(16))
            if kind == 'Z':
                center = x[train].mean(0);sd = x[train].std(0, unbiased=False)
                scale = torch.where(sd >= 1e-6, sd, torch.ones_like(sd))
            else:center, scale = torch.zeros(24), torch.ones(24)
            z = (x-center)/scale
            transformed = [dict(s, state=z[i].tolist(), preferences=targets[i].tolist()) for i, s in enumerate(table)]
            global_action = h['best']([st.mean(table[i]['rewards'][a] for i in train) for a in range(9)])
            time_actions = {t: h['best']([st.mean(table[i]['rewards'][a] for i in train if table[i]['step'] == t)
                                        for a in range(9)]) for t in (100, 200)}
            category = 'full_table' if held == 'FULL_TABLE' else 'LOCO'
            for seed in (601, 602):
                actor = Actor(seed);opt = torch.optim.Adam(actor.parameters(), lr=.001)
                for epoch in range(1024):
                    value = loss(actor, z[train], targets[train]);assert torch.isfinite(value)
                    opt.zero_grad(set_to_none=True);value.backward()
                    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.))
                    ledger.step(category, f'{kind}/{held}/{seed}/{epoch}', opt)
                    if epoch+1 in (1, 64, 256, 1024):
                        losses.append(dict(arm=kind+'_1024', heldout=held, seed=seed, update=epoch+1,
                                           objective_before_update=float(value.detach())))
                    if epoch+1 == 64:
                        old = torch.load(previous/f'{kind}_T025_{held}_{seed}.private.pt', map_location='cpu', weights_only=False)
                        assert torch.equal(center, old['mean']) and torch.equal(scale, old['scale'])
                        assert all(torch.equal(v, old['actor'][k]) for k, v in actor.state_dict().items()), 'V85 prefix mismatch'
                        prefixes.append(dict(arm=kind, heldout=held, seed=seed, update=64, exact_tensor_match=True))
                checkpoint = output/f'{kind}_1024_{held}_{seed}.private.pt'
                with checkpoint.open('xb') as f:
                    torch.save(dict(actor=actor.state_dict(), mean=center, scale=scale, seed=seed,
                                    arm=kind+'_1024', updates=1024, heldout=held), f)
                provenance.append(dict(arm=kind+'_1024', heldout=held, seed=seed, sha256=h['digest'](checkpoint)))
                values = h['policy_rows'](actor, transformed, test, global_action, time_actions, seed)
                for row in values:row['arm'] = kind+'_1024'
                (formal if held == 'FULL_TABLE' else loco).extend(values)
    assert dict(ledger.count) == dict(qualification=2, LOCO=32768, full_table=4096)
    assert len(prefixes) == 36
    h['save'](output/'QUALIFICATION.json', dict(status='PASS', synthetic_actor_updates=2, prefix_matches=36,
                                              prefix_checks=prefixes, no_student_updates=True))
    h['csvfile'](output/'LOCO_RESULTS.csv', loco);h['csvfile'](output/'FORMAL_POLICY_FIT.csv', formal)
    h['csvfile'](output/'TRAINING_LOSS.csv', losses);h['save'](output/'CHECKPOINT_PROVENANCE.json', provenance)
    outcome = decision(loco);h['save'](output/'NEXT_STAGE_DECISION.json', outcome)
    h['save'](output/'COSTS.json', dict(student_updates=0, actor_updates=dict(ledger.count),
        actor_total=36866, actor_cap=36866, historical_teacher_student_cost=29600,
        query_image_accesses=0, GPU_training_seconds=0, reused_64_update_fits=True))
    h['save'](output/'PROVENANCE.json', dict(preregistration_commit='ccf4915', script_sha256=h['digest'](__file__),
        helper_sha256=h['digest'](os.environ['EXEC_HELPER']), table_sha256=h['digest'](source/'V83_ACTION_ROWS.jsonl'),
        baseline_results_sha256=h['digest'](previous/'LOCO_RESULTS.csv')))
    report = ['# V8.6 fixed optimization-budget diagnostic', '', '**'+outcome['status']+'**', '',
        '| Evaluation | Arm | Seed | Raw expected reward | Minus global | Minus time | Target KL |', '|---|---|---:|---:|---:|---:|---:|']
    for name, values in [('LOCO', loco), ('in_table', formal)]:
        for arm in ('RAW_64_REUSED', 'Z_64_REUSED', 'RAW_1024', 'Z_1024'):
            for seed in (601, 602):
                selected = [r for r in values if r['arm'] == arm and r['seed'] == seed]
                report.append(f'| {name} | {arm} | {seed} | '+' | '.join(f'{macro(selected,k):.9f}' for k in
                              ('actor_reward', 'actor_minus_global', 'actor_minus_time', 'target_KL'))+' |')
    report += ['', 'All 36 update64 parameter prefixes match their V8.5 counterparts exactly; optimizers then continue without reset.',
        'Only update1024 models are evaluated; loss logs at earlier updates are descriptive and are not used for selection.',
        'This is adaptive shared-source D1 context holdout, not independent patient/image validation. No new image read or student update occurred.',
        'New cost: 36,866 CPU actor updates. Historical dense information still costs 29,600 student updates and is not free.',
        'V8.3, V8.3-R and V8.5 STOPs remain. V8.4 is NOT_RUN. Any development experiment must be separately preregistered.']
    (output/'FINAL_INTERPRETATION.md').write_text('\n'.join(report)+'\n')
    h['save'](output/'FINAL.json', dict(status='COMPLETE', decision=outcome['status'],
                                     actor_updates=36866, student_updates=0, time=time.time(), publication='PENDING'))


if __name__ == '__main__':
    if os.environ.get('EXEC_SELFCHECK') == '1':
        selfcheck();print('PASS: V86 gate self-check; zero optimizer updates')
    else:
        main()
