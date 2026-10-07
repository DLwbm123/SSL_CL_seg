"""CPU-only, create-only readout of a completely sealed V8.3 campaign.

Inputs and output are passed in EXEC_SOURCE / EXEC_RUN. No student or data
provider is constructed. Existing Actor, preference and loss implementations
are reused lazily so the stdlib self-check needs no training environment.
"""
import csv
import hashlib
import json
import math
import os
import statistics as st
import time
from collections import Counter
from pathlib import Path


FLOOR = 0.0007379373167760999
SEEDS = (601, 602)
STEPS = (100, 200)
CONTEXTS = tuple(f'm{m}_n{n}_{c}' for m in (2000, 8000)
                 for n in (2, 8) for c in ('brightness', 'contrast'))
METHODS = ('NATIVE', 'FIXED_BEST', 'UNIFORM_ACTION', 'PRE_FROZEN_601', 'PRE_FROZEN_602')
PARTS = ('reward', 'gain', 'forget_penalty')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def csvfile(path, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def best(values):
    # Registered tie order: native, then low alpha / action ID.
    return min(range(9), key=lambda a: (-values[a], a != 0, a // 3, a))


def macro(rows, field):
    """Equal contexts; average selected times/streams within each context."""
    return st.mean(st.mean(r[field] for r in rows if r['context'] == c)
                   for c in sorted({r['context'] for r in rows}))


def ranks(values):
    return [1 + sum(w < v for w in values) + (sum(w == v for w in values) - 1) / 2
            for v in values]


def spearman(x, y):
    x, y = ranks(x), ranks(y)
    x, y = [v - st.mean(x) for v in x], [v - st.mean(y) for v in y]
    denominator = math.sqrt(sum(v*v for v in x) * sum(v*v for v in y))
    return sum(a*b for a, b in zip(x, y)) / denominator if denominator else None


def audit(rows):
    def finite(value):
        if isinstance(value, dict):return all(finite(v) for v in value.values())
        if isinstance(value, list):return all(finite(v) for v in value)
        return math.isfinite(value) if isinstance(value, (int, float)) else True
    assert finite(rows), 'nonfinite table value'
    table = {(r['context'], r['entry_step'], r['action'], r['stream']): r for r in rows}
    expected = {(c, t, a, k) for c in CONTEXTS for t in STEPS for a in range(9) for k in (1, 2)}
    assert len(rows) == len(table) == 288 and set(table) == expected, 'incomplete/duplicate keys'
    for c in CONTEXTS:
        references = []
        for t in STEPS:
            group = [table[c, t, a, k] for a in range(9) for k in (1, 2)]
            state = group[0]['state']
            assert len(state) == 24 and all(math.isfinite(v) for v in state)
            assert abs(state[0] - t/300) < 1e-6
            assert all(r['state'] == state for r in group), 'paired state mismatch'
            assert max(r['new_entry'] for r in group) - min(r['new_entry'] for r in group) < 1e-8
            for r in group:
                values = [r[key] for key in PARTS] + [r['new_entry'], r['old_frozen_memory_reference']]
                values += [r[key][channel] for key in ('new_short', 'new', 'old')
                           for channel in ('macro', 'rim', 'cup')]
                assert all(math.isfinite(v) for v in values)
                gain = .25*(r['new_short']['macro']-r['new_entry']) + .75*(r['new']['macro']-r['new_entry'])
                penalty = max(0., r['old_frozen_memory_reference']-r['old']['macro']-.005)
                assert abs(gain-r['gain']) < 1e-8 and abs(penalty-r['forget_penalty']) < 1e-8
                assert abs(gain-penalty-r['reward']) < 1e-8
                references.append(r['old_frozen_memory_reference'])
        assert max(references)-min(references) < 1e-8, 'memory reference varies within context'
    return table


def cross_stream(table):
    rows, rank_rows = [], []
    for c in CONTEXTS:
        for t in STEPS:
            a, b = ([table[c, t, i, k]['reward'] for i in range(9)] for k in (1, 2))
            rank_rows.append(dict(context=c, step=t, support=int(c.split('_')[1][1:]),
                                  spearman=spearman(a, b), best_agreement=int(best(a) == best(b)),
                                  best_stream1=best(a), best_stream2=best(b)))
    for train, test in ((1, 2), (2, 1)):
        global_action = best([st.mean(table[c, t, a, train]['reward'] for c in CONTEXTS for t in STEPS)
                              for a in range(9)])
        time_actions = {t: best([st.mean(table[c, t, a, train]['reward'] for c in CONTEXTS)
                                for a in range(9)]) for t in STEPS}
        for c in CONTEXTS:
            for t in STEPS:
                chosen = best([table[c, t, a, train]['reward'] for a in range(9)])
                row = dict(direction=f'{train}->{test}', context=c, step=t,
                           support=int(c.split('_')[1][1:]), state_action=chosen,
                           global_action=global_action, time_action=time_actions[t])
                for part in PARTS:
                    vals = {name: table[c, t, action, test][part] for name, action in
                            [('state', chosen), ('global', global_action), ('time', time_actions[t])]}
                    vals['uniform'] = st.mean(table[c, t, a, test][part] for a in range(9))
                    row.update({f'{name}_{part}': value for name, value in vals.items()})
                    for lhs, rhs in (('state', 'uniform'), ('state', 'global'), ('state', 'time'), ('global', 'uniform')):
                        row[f'{lhs}_minus_{rhs}_{part}'] = vals[lhs]-vals[rhs]
                # Same-stream maximum is only a biased, post-hoc table reference.
                row['same_table_posthoc_best_reward'] = table[c, t, chosen, train]['reward']
                rows.append(row)
    return rows, rank_rows


def aggregate(rows, fields, seed_field=None):
    output = []
    values = sorted({r[seed_field] for r in rows}) if seed_field else [None]
    for value in values:
        base = [r for r in rows if seed_field is None or r[seed_field] == value]
        slices = [('all', 'all', base)]
        for field in ('context', 'step', 'support'):
            slices += [(field, str(v), [r for r in base if r[field] == v]) for v in sorted({r[field] for r in base})]
        for kind, label, selected in slices:
            out = dict(group=value if seed_field else 'both', slice=kind, value=label, states=len(selected))
            out.update({field: macro(selected, field) for field in fields})
            output.append(out)
    return output


def decide(original_pass, cross, loco):
    directions = {d: {key: macro([r for r in cross if r['direction'] == d], key)
                       for key in ('state_minus_global_reward', 'state_minus_time_reward')}
                  for d in ('1->2', '2->1')}
    conditional = all(v > 0 for row in directions.values() for v in row.values()) and all(
        st.mean(row[key] for row in directions.values()) >= .0005
        for key in ('state_minus_global_reward', 'state_minus_time_reward'))
    by_seed = {str(seed): {key: macro([r for r in loco if r['seed'] == seed], key)
                           for key in ('actor_minus_global', 'actor_minus_time')}
               for seed in SEEDS}
    mapping = all(v > 0 for row in by_seed.values() for v in row.values())
    status = ('STOP_V83_PRIMARY_GATE_FAILED' if not original_pass else
              'STOP_CONDITIONAL_ACTION_VALUE_NOT_ESTABLISHED' if not conditional else
              'STOP_STATE_TO_ACTION_MAPPING_NOT_ESTABLISHED' if not mapping else
              'READY_FOR_V84_WARM_START_GRPO')
    return dict(status=status, original_V83_pass=original_pass,
                conditional_action_value=conditional, direction_deltas=directions,
                state_mapping=mapping, LOCO_seed_deltas=by_seed,
                engineering_criteria_not_significance=True, independent_generalization=False)


def sealed(source):
    """Read receipts before any reward values; reject partial development."""
    decision = read(source/'DECISION.json')
    assert decision['status'] in ('PASS_DENSE_REWARD_DIAGNOSTIC', 'STOP_DENSE_REWARD_NO_PRACTICAL_GAIN')
    assert read(source/'jobs/qualification/QUALIFICATION.json')['status'] == 'PASS'
    dev = source/'jobs/development'
    lock = read(dev/'ENDPOINT_LOCK.json')
    assert lock['all_twenty_sealed'] is True and (dev/'FINAL.json').exists()
    assert decision['time'] >= lock['time']
    endpoints = []
    for c in range(4):
        for method in METHODS:
            f = dev/f'dev{c}'/(method+'_TRAINING.json')
            row = read(f)
            assert row['status'] == 'SEALED' and row['steps'] == 200 and len(row['actions']) == 2
            assert all(a in range(9) for a in row['actions']) and f.stat().st_mtime <= lock['time']
            final = f.with_name(method+'_FINAL.private.pt')
            assert final.is_file() and final.stat().st_mtime <= lock['time']
            endpoints.append(dict(context=f'dev{c}', method=method, actions=row['actions'],
                                  status='SEALED', checkpoint_sha256=digest(final)))
    for seed in SEEDS:
        receipt = read(source/f'jobs/prior_{seed}/FINAL.json')
        assert receipt['status'] == 'COMPLETE' and receipt['actor_epochs'] == 64
    for k in (1, 2):
        receipt = read(source/f'jobs/audit_{k}/FINAL.json')
        assert receipt['status'] == 'COMPLETE' and receipt['rows'] == 144
    return decision, endpoints


def archive_original(source, output, original, endpoints, config):
    from experiments.qprompt_rl_v1.v83_dense_reward.round import screen
    dev = read(source/'jobs/development/DEVELOPMENT_RESULTS.json')
    baseline = read(Path(config['prior_campaign'])/'jobs/development/DEVELOPMENT_RESULTS.json')
    recomputed = screen(dev, baseline)
    assert recomputed['status'] == original['status'] and recomputed['gates'] == original['gates']
    assert len(dev['results']) == 20
    costs = read(source/'COSTS.json')
    previous = read(source/'FROZEN_EXECUTION.json')['previous_attempts']
    events = {event: Counter() for event in ('attempt', 'success', 'failure')}
    with (source/'PHYSICAL_LEDGER.jsonl').open() as f:
        for line in f:
            row = json.loads(line);events[row['event']][row['category']] += 1
    for event, key in [('attempt', 'attempts'), ('success', 'success'), ('failure', 'failures')]:
        assert dict(events[event]) == costs[key], 'cumulative physical ledger mismatch'
    delta = {k: n-previous.get(k, 0) for k, n in costs['attempts'].items()}
    assert all(n >= 0 for n in delta.values())
    assert delta['entries'] == 1200 and delta['audit'] == 28800 and delta['development'] == 4000
    assert delta['qualification'] == 8 and delta['actor_prior'] == 128 and delta['actor_qualification'] <= 9
    receipts = {}
    for job in ('qualification', 'decision_entries', 'audit_1', 'audit_2', 'prior_601', 'prior_602', 'development'):
        root = source/'jobs'/job
        started, final = read(root/'STARTED.json'), read(root/'FINAL.json')
        ended = read(root/'PROCESS_EXIT.json')
        assert ended['exit_code'] == 0
        receipts[job] = dict(physical=final['physical'], commit=started['commit'],
                             process_elapsed_seconds=ended['time']-started['time'], exit_code=0)
    save(output/'V83_ORIGINAL_DECISION.json', original)
    save(output/'V83_ENDPOINT_MANIFEST.json', endpoints)
    save(output/'V83_DEVELOPMENT_RESULTS.json', dev)
    save(output/'V83_COSTS.json', costs)
    save(output/'V83_PHYSICAL_ACCOUNTING.json', dict(
        historical_prefix=previous, new_attempts=delta, cumulative_attempts=costs['attempts'],
        cumulative_success=costs['success'], cumulative_failures=costs['failures'],
        common_historical_source_student_updates=8000, common_source_not_double_counted=True,
        teacher_information_generation=dict(native_entry200=800, action_branches=28800, total=29600),
        jobs=receipts, query_calls='not physically instrumented; query roles and schedule preserved in code',
        teacher_forward_calls='NOT_INSTRUMENTED', GPU_kernel_time='NOT_INSTRUMENTED',
        peak_GPU_memory='NOT_INSTRUMENTED', process_elapsed_is_not_GPU_kernel_time=True))
    result_rows = []
    for r in dev['results']:
        row = dict(context=r['context'], method=r['method'], new_Dice=r['new']['macro'],
                   old_Dice=r['old']['macro'], utility=r['utility'], utility_times100=r['utility']*100,
                   absolute_forgetting_pp=100*(r['old_memory_reference']-r['old']['macro']),
                   gain=r['gain'], forget_penalty=r['forget_penalty'], memory_reference=r['old_memory_reference'])
        row.update({f'{role}_{channel}': r[role][channel] for role in ('new', 'old') for channel in ('rim', 'cup')})
        result_rows.append(row)
    csvfile(output/'V83_ENDPOINT_RESULTS.csv', result_rows)
    save(output/'V83_ARCHIVE_RECEIPT.json', dict(status='SEALED_FOR_READOUT', original_screen_reproduced=True,
         endpoint_count=20, all_endpoints_before_readout=True, cumulative_ledger_checked=True,
         publication='PENDING', time=time.time()))
    return dev


def policy_rows(actor, table, indices, global_action, time_actions, seed):
    import torch
    out = []
    with torch.no_grad():
        for i in indices:
            s = table[i]
            log = actor(torch.tensor(s['state'])).log_softmax(-1)
            probs = log.exp().double().tolist()
            # PyTorch float32 softmax can differ from unit sum by a few ulps.
            assert abs(sum(probs)-1.) < 1e-6
            reward = sum(p*r for p, r in zip(probs, s['rewards']))
            uniform = st.mean(s['rewards']);glob = s['rewards'][global_action]
            temporal = s['rewards'][time_actions[s['step']]]
            target = s['preferences'];entropy = -sum(p*float(l) for p, l in zip(probs, log))
            ce = -sum(p*float(l) for p, l in zip(target, log))
            out.append(dict(seed=seed, context=s['context'], step=s['step'],
                            support=int(s['context'].split('_')[1][1:]), actor_reward=reward,
                            uniform_reward=uniform, global_reward=glob, time_reward=temporal,
                            actor_minus_uniform=reward-uniform, actor_minus_global=reward-glob,
                            actor_minus_time=reward-temporal, global_action=global_action,
                            time_action=time_actions[s['step']], entropy=entropy,
                            target_cross_entropy=ce, target_KL=ce+sum(p*math.log(p) for p in target),
                            **{f'p{a}': probs[a] for a in range(9)}))
    return out


class Budget:
    def __init__(self, root):
        self.path = root/'PHYSICAL_LEDGER.jsonl'
        assert not self.path.exists(), 'no implicit retry or ledger reset'
        self.count = Counter()

    def step(self, category, key, optimizer):
        cap = {'qualification': 16, 'LOCO': 1024}[category]
        assert self.count[category] < cap, 'actor budget exhausted'
        self.count[category] += 1
        row = dict(category=category, key=key, ordinal=self.count[category])
        with self.path.open('a') as f:
            f.write(json.dumps(dict(row, event='attempt'))+'\n');f.flush()
        try:
            optimizer.step()
        except BaseException:
            with self.path.open('a') as f:f.write(json.dumps(dict(row, event='failure'))+'\n')
            raise
        with self.path.open('a') as f:f.write(json.dumps(dict(row, event='success'))+'\n')


def controllers(source, output, table, ledger):
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.core import Actor, actor_update
    from experiments.qprompt_rl_v1.v83_dense_reward.worker import loss
    from experiments.qprompt_rl_v1.v83_dense_reward.round import preferences
    torch.set_num_threads(2)
    rng = torch.get_rng_state().clone()
    actor = Actor(601);opt = torch.optim.Adam(actor.parameters(), lr=.001)
    assert torch.equal(rng, torch.get_rng_state()), 'actor construction polluted RNG'
    z = torch.stack((torch.ones(24), -torch.ones(24)))
    target = torch.tensor([preferences([1.]+[0.]*8, FLOOR), preferences([0.]*8+[1.], FLOOR)])
    initial = float(loss(actor, z, target).detach())
    opt.zero_grad(set_to_none=True);loss(actor, z, target).backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters())
    ledger.step('qualification', 'synthetic', opt)
    assert float(loss(actor, z, target).detach()) < initial
    assert actor_update(actor, opt, z[0], [0, 1, 2, 3], [0.]*4, FLOOR, None, '', '')['skipped']
    save(output/'QUALIFICATION.json', dict(status='PASS', student_updates=0, actor_updates=1,
         checks=['stdlib self-check', 'CPU finite gradient and objective decrease', 'private RNG isolation',
                 'unchanged floor skip', 'sealed source prerequisite', 'create-only copies'],
         reused_native_qualification='V83 and V8 full restore, query isolation, branch0, fixed reference'))
    all_indices = list(range(16));loco = [];formal = []
    for held in CONTEXTS:
        train = [i for i, s in enumerate(table) if s['context'] != held]
        test = [i for i, s in enumerate(table) if s['context'] == held]
        assert len(train) == 14 and len(test) == 2
        global_action = best([st.mean(table[i]['rewards'][a] for i in train) for a in range(9)])
        time_actions = {t: best([st.mean(table[i]['rewards'][a] for i in train if table[i]['step'] == t)
                                for a in range(9)]) for t in STEPS}
        # Targets are constructed afresh using training states only. There is no fitted scaler.
        z = torch.tensor([table[i]['state'] for i in train])
        targets = torch.tensor([preferences(table[i]['rewards'], FLOOR) for i in train])
        for seed in SEEDS:
            actor = Actor(seed);opt = torch.optim.Adam(actor.parameters(), lr=.001)
            for epoch in range(64):
                value = loss(actor, z, targets);assert torch.isfinite(value)
                opt.zero_grad(set_to_none=True);value.backward()
                norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
                assert torch.isfinite(norm)
                ledger.step('LOCO', f'{held}/{seed}/{epoch}', opt)
            checkpoint = output/f'LOCO_{held}_{seed}.private.pt'
            assert not checkpoint.exists()
            torch.save(dict(actor=actor.state_dict(), seed=seed, heldout=held, updates=64), checkpoint)
            loco += policy_rows(actor, table, test, global_action, time_actions, seed)
    assert ledger.count['LOCO'] == 1024
    global_action = best([st.mean(s['rewards'][a] for s in table) for a in range(9)])
    time_actions = {t: best([st.mean(s['rewards'][a] for s in table if s['step'] == t)
                            for a in range(9)]) for t in STEPS}
    for seed in SEEDS:
        payload = torch.load(source/f'jobs/prior_{seed}/actor_final.pt', map_location='cpu', weights_only=False)
        assert payload['controller'] == seed and payload['epochs'] == 64
        actor = Actor(seed);actor.load_state_dict(payload['actor'])
        formal += policy_rows(actor, table, all_indices, global_action, time_actions, seed)
    return loco, formal


def selfcheck():
    assert best([0.]*9) == 0 and best([0., 2., 2.]+[0.]*6) == 1
    assert spearman([1, 2, 2, 4], [4, 2, 2, 1]) == -1.
    assert spearman([0.]*9, list(range(9))) is None
    rows = []
    for i, c in enumerate(CONTEXTS):
        for t in STEPS:
            for a in range(9):
                for k in (1, 2):
                    r = .01 if a == i+1 else 0.
                    metric = dict(macro=.5+r, rim=.5+r, cup=.5+r)
                    rows.append(dict(context=c, entry_step=t, action=a, stream=k,
                        state=[t/300]+[0.]*23, reward=r, gain=r, forget_penalty=0.,
                        new_entry=.5, old_frozen_memory_reference=.5,
                        new_short=metric, new=metric, old=dict(macro=.5, rim=.5, cup=.5)))
    table = audit(rows);cross, _ = cross_stream(table)
    assert all(r['state_minus_uniform_reward'] > 0 for r in cross)
    loco = [dict(seed=s, context=c, actor_minus_global=.001, actor_minus_time=.001)
            for s in SEEDS for c in CONTEXTS]
    assert decide(True, cross, loco)['status'] == 'READY_FOR_V84_WARM_START_GRPO'
    assert decide(False, cross, loco)['status'] == 'STOP_V83_PRIMARY_GATE_FAILED'
    flat = [dict(r, state_minus_global_reward=0.) for r in cross]
    assert decide(True, flat, loco)['status'] == 'STOP_CONDITIONAL_ACTION_VALUE_NOT_ESTABLISHED'
    assert decide(True, cross, [dict(r, actor_minus_time=0.) for r in loco])['status'] == 'STOP_STATE_TO_ACTION_MAPPING_NOT_ESTABLISHED'
    for broken in (rows[:-1], rows+[rows[0]]):
        try:audit(broken)
        except AssertionError:pass
        else:raise AssertionError('bad table accepted')
    # Unequal row counts must not give one context more macro weight.
    assert macro([dict(context='a', v=1.)]*3+[dict(context='b', v=0.)], 'v') == .5
    # No optimization call is possible at an exhausted budget.
    budget = Budget.__new__(Budget);budget.count = Counter(LOCO=1024)
    try:budget.step('LOCO', 'overrun', None)
    except AssertionError:pass
    else:raise AssertionError('exhausted budget accepted')


def main():
    selfcheck()
    source = Path(os.environ['EXEC_SOURCE'])
    original, endpoints = sealed(source)
    output = Path(os.environ['EXEC_RUN'])/'results'
    output.mkdir(exist_ok=False)
    save(output/'STARTED.json', dict(time=time.time(), pid=os.getpid(), source_run=source.name))
    config = read(source/'CONFIG.private.json')
    frozen = read(Path(__file__).with_name('READOUT_MANIFEST.json'))
    assert config['protocol'] == 'V83_DENSE_REWARD' and config['training_horizon'] == 300
    assert config['sigma_floor_override'] == FLOOR
    assert digest(Path(config['action_root'])/'ROLES.private.json') == frozen['private_role_file_sha256']
    for name, expected in frozen['code_sha256'].items():
        assert digest(source/'code/experiments/qprompt_rl_v1'/name) == expected
    # Seal and reproduce the original result before performing new analyses.
    dev = archive_original(source, output, original, endpoints, config)
    rows = [json.loads(line) for k in (1, 2)
            for line in (source/f'jobs/audit_{k}/ACTION_ROWS.jsonl').read_text().splitlines()]
    table = audit(rows)
    save(output/'ACTION_TABLE_AUDIT.json', dict(status='PASS', rows=288, states=16,
         numeric_checks=['unique complete finite keys', 'paired 24D states and progress',
                         'fixed memory reference', 'gain/penalty/reward recomputation'],
         implementation_evidence={'paired_entry':'same complete ENTRY restored before each action',
          'streams':'provider 10168/20168 and global RNG 860101/860102',
          'candidates':'all discarded; native ENTRY200 prepared separately',
          'features':'fit/U only; core.Trainer.extract, no query or reward input'},
         evidence_limit='Execution code and qualification plus stored results; no per-action full-state digest trace.',
         branch_readout='entry+25 and entry+100', endpoint_readout='150 and 300 relative to ENTRY100'))
    with (output/'V83_ACTION_ROWS.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row, allow_nan=False)+'\n')
    hashes = {f'actor_{s}': digest(source/f'jobs/prior_{s}/actor_final.pt') for s in SEEDS}
    save(output/'SEALED_INPUTS.json', dict(source_run=source.name, actor_sha256=hashes,
         table_sha256={str(k): digest(source/f'jobs/audit_{k}/ACTION_ROWS.jsonl') for k in (1, 2)},
         decision_sha256=digest(source/'DECISION.json'), plan_commit='11731b5',
         execution_commit=config['commit'], readout_script_sha256=digest(__file__), time=time.time()))
    cross, rank_rows = cross_stream(table)
    csvfile(output/'CROSS_STREAM_VALUE.csv', cross)
    csvfile(output/'ACTION_RANK_STABILITY.csv', rank_rows)
    fields = [k for k in cross[0] if '_minus_' in k]
    csvfile(output/'CROSS_STREAM_SUMMARY.csv', aggregate(cross, fields, 'direction')+aggregate(cross, fields))
    from experiments.qprompt_rl_v1.v83_dense_reward.round import dataset
    data = dataset(rows, FLOOR)
    ledger = Budget(output)
    loco, formal = controllers(source, output, data, ledger)
    csvfile(output/'LOCO_RESULTS.csv', loco)
    csvfile(output/'FORMAL_POLICY_FIT.csv', formal)
    metrics = ('actor_reward', 'uniform_reward', 'global_reward', 'time_reward',
               'actor_minus_uniform', 'actor_minus_global', 'actor_minus_time', 'entropy', 'target_KL')
    csvfile(output/'POLICY_SUMMARY.csv',
            [dict(kind=kind, **r) for kind, values in [('LOCO', loco), ('in_table', formal)]
             for r in aggregate(values, metrics, 'seed')])
    decision = decide(original['status'] == 'PASS_DENSE_REWARD_DIAGNOSTIC', cross, loco)
    save(output/'NEXT_STAGE_DECISION.json', decision)
    save(output/'COSTS.json', dict(student_updates=0, actor_updates=dict(ledger.count),
         actor_attempts=sum(ledger.count.values()), student_cap=0, actor_cap=1040,
         historical_V83_costs='V83_COSTS.json; shared generation counted once',
         GPU_training_time_seconds=0, query_image_accesses=0))
    reports(output, original, decision, cross, rank_rows, loco, formal, dev, endpoints)
    save(output/'FINAL.json', dict(status='COMPLETE', decision=decision['status'],
         student_updates=0, actor_updates=sum(ledger.count.values()), time=time.time(), publication='PENDING'))


def reports(output, original, decision, cross, ranking, loco, formal, dev, endpoints):
    fit = ['# Policy fit and held-context diagnostics', '',
           'Raw reward units; entropy in nats. These are expected categorical values, not argmax deployment.', '',
           '| Evaluation | Seed | Reward | Actor − uniform | Actor − global | Actor − time | Entropy | Target KL |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, values in [('Full training table', formal), ('LOCO held contexts', loco)]:
        for seed in SEEDS:
            rows = [r for r in values if r['seed'] == seed]
            keys = ('actor_reward', 'actor_minus_uniform', 'actor_minus_global', 'actor_minus_time', 'entropy', 'target_KL')
            fit.append(f'| {name} | {seed} | '+' | '.join(f'{macro(rows,k):.9f}' for k in keys)+' |')
    fit += ['', 'All action probabilities and fold-specific control choices are in the CSVs. Formal actors were never updated.',
            'LOCO used 1,024 CPU updates and one synthetic qualification update. No student updates or query-image reads.',
            'The held contexts share D1 image pools and auxiliary model sources. No patient-independent or cross-domain inference.']
    (output/'POLICY_FIT_REPORT.md').write_text('\n'.join(fit)+'\n')
    coverage = ['# Saved trajectory and state coverage', '', 'DEPLOYMENT_STATE_200_NOT_AVAILABLE', '',
                'The frozen deployment writer saves action sequences and step150/300 snapshots, not step200 feature vectors. No replacement state was reconstructed.',
                'Training-branch reward uses entry+25/+100; endpoint utility uses step150/300 from ENTRY100. This is an observed timing difference, not a proved failure mechanism.', '',
                '| Method | Decision step | Action counts across four development contexts |', '|---|---:|---|']
    for method in METHODS:
        for i, step in enumerate(STEPS):
            counts = Counter(r['actions'][i] for r in endpoints if r['method'] == method)
            coverage.append(f'| {method} | {step} | {dict(sorted(counts.items()))} |')
    coverage += ['', 'Formal-policy variation on native training states:', '',
                 '| Seed | Step | Mean L1 distance from within-step mean probability | Maximum distance |', '|---|---:|---:|---:|']
    for seed in SEEDS:
        for step in STEPS:
            group = [r for r in formal if r['seed'] == seed and r['step'] == step]
            mean = [st.mean(r[f'p{a}'] for r in group) for a in range(9)]
            distances = [sum(abs(r[f'p{a}']-mean[a]) for a in range(9)) for r in group]
            coverage.append(f'| {seed} | {step} | {st.mean(distances):.9f} | {max(distances):.9f} |')
    coverage += ['', 'Small within-step variation is descriptive only. It cannot establish that a network uses only progress, nor prove deployment distribution shift.']
    (output/'STATE_COVERAGE_REPORT.md').write_text('\n'.join(coverage)+'\n')
    report = ['# V8.3 and V8.3-R interpretation', '',
        f'1. Original V8.3 gate: **{original["status"]}**; original screen reproduced without changing thresholds.',
        f'2–3. Cross-stream conditional value over global AND time controls: **{decision["conditional_action_value"]}** under the prespecified engineering gate.',
        f'4. Both LOCO seeds exceed both fold-trained controls: **{decision["state_mapping"]}**.',
        '5–6. GRPO increment and warm-start total-cost benefit: NOT_TESTED by this readout; do not infer them from dense supervision.',
        '7. New/old tradeoffs and absolute forgetting are listed below for all methods; utility×100 is not a Dice change.',
        '8. Cross-stream and LOCO findings locate evidence gaps within fixed D1 contexts. Reward timing and deployment distribution shift remain hypotheses, not established causes.', '',
        f'Continuation status: **{decision["status"]}**.', '',
        '| Method | New Dice | Old Dice | Utility | Utility ×100 | Mean absolute forgetting (pp) |',
        '|---|---:|---:|---:|---:|---:|']
    # stage_b stores its endpoint list as rows; fail loudly on an unexpected format.
    results = dev['rows'] if 'rows' in dev else dev['results']
    for method in METHODS:
        mean = dev['means'][method]
        selected = [r for r in results if r['method'] == method]
        forgetting = st.mean(r['old_memory_reference']-r['old']['macro'] for r in selected)*100
        report.append(f'| {method} | {mean["new"]:.9f} | {mean["old"]:.9f} | {mean["utility"]:.9f} | {mean["utility"]*100:.6f} | {forgetting:.6f} |')
    report += ['', 'Absolute forgetting is memory-reference Dice minus endpoint Dice (signed; no deadband). The reward penalty applies a separate 0.005 deadband.',
        '', 'Cross-stream deltas (raw reward):', '', '| Direction | State − uniform | State − global | State − time | Global − uniform |', '|---|---:|---:|---:|---:|']
    for direction in ('1->2', '2->1', 'both'):
        selected = [r for r in cross if direction == 'both' or r['direction'] == direction]
        report.append('| '+direction+' | '+' | '.join(f'{macro(selected,k):.9f}' for k in
             ('state_minus_uniform_reward','state_minus_global_reward','state_minus_time_reward','global_minus_uniform_reward'))+' |')
    valid = [r['spearman'] for r in ranking if r['spearman'] is not None]
    report += ['', f'Rank agreement: mean Spearman {st.mean(valid) if valid else None}; best-action agreement {st.mean(r["best_agreement"] for r in ranking):.4f}. Constant rank vectors are missing, not zero.',
        '', 'Evidence limits: repeated D1 development, image-level roles, shared query images across random streams, shared image/model sources across LOCO folds. The 288 rows, 16 states and two actor seeds are not independent patients or independent student repetitions.',
        'Label budget: 16 M_fit + 8 A_fit + 4 Q_train_old + 4 Q_train_new + 4 Q_dev_old + 4 Q_dev_new = 40 labeled D1 images; support 2/8 is only the adaptation subset. U_memory/U_adapt contain 80 images each, with hidden labels unused.',
        'Any dense-supervision PASS supports only complete-action reward-supervised development diagnostics, not GRPO efficacy or independent generalization.']
    (output/'FINAL_INTERPRETATION.md').write_text('\n'.join(report)+'\n')


if __name__ == '__main__':
    if os.environ.get('EXEC_SELFCHECK') == '1':
        selfcheck();print('PASS: stdlib readout checks; zero optimizer updates')
    else:
        main()
