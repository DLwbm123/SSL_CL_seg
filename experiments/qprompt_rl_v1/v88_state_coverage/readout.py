"""Query-free scalar diagnostics of already saved controller inputs."""
import csv
import json
import math
import os
import statistics as st
import time
from pathlib import Path


def rms(a, b):
    assert len(a) == len(b) and len(a) > 0
    return math.sqrt(st.mean((x-y)**2 for x, y in zip(a, b)))


def selfcheck():
    assert rms([1., 2.], [1., 2.]) == 0.
    assert rms([1., 2.], [2., 3.]) == 1.


def main():
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.core import Actor
    from experiments.qprompt_rl_v1.v83_dense_reward.round import dataset
    torch.set_num_threads(2)
    root = Path(os.environ['EXEC_RUN']);out = root/'results';out.mkdir(exist_ok=False)
    source = Path(os.environ['EXEC_DENSE']);deployment = Path(os.environ['EXEC_DEPLOY'])
    assert json.loads((deployment/'FINAL.json').read_text())['status'] == 'COMPLETE'
    cfg = json.loads((deployment/'CONFIG.private.json').read_text())

    def save(name, value):
        with (out/name).open('x') as f:json.dump(value, f, indent=2, allow_nan=False);f.write('\n')

    def csvfile(name, rows):
        with (out/name).open('x') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    save('STARTED.json', dict(pid=os.getpid(), time=time.time(), protocol='V88_ZERO_UPDATE_STATE_COVERAGE'))
    raw = [json.loads(line) for stream in (1, 2) for line in
           (source/f'jobs/audit_{stream}/ACTION_ROWS.jsonl').read_text().splitlines()]
    table = dataset(raw, .0007379373167760999);assert len(table) == 16
    models = {}
    for name, path in cfg['actors'].items():
        value = torch.load(path, map_location='cpu', weights_only=False)
        assert value['heldout'] == 'FULL_TABLE'
        actor = Actor(value['seed']);actor.load_state_dict(value['actor']);actor.eval()
        models[name] = actor, value['mean'], value['scale']
    center, scale = models['Z_1024_601'][1:]
    x = torch.tensor([r['state'] for r in table]);z = (x-center)/scale
    assert torch.allclose(center, x.mean(0))
    constant = x.std(0, unbiased=False) < 1e-6
    reference = []
    for i, row in enumerate(table):
        others = [j for j,r in enumerate(table) if r['step'] == row['step'] and r['context'] != row['context']]
        reference.append(dict(context=row['context'], step=row['step'],
                              nearest_other_context=rms(z[i].tolist(), z[min(others, key=lambda j:rms(z[i].tolist(), z[j].tolist()))].tolist())))
    limits = {step:max(r['nearest_other_context'] for r in reference if r['step'] == step) for step in (100, 200)}

    def policy(state, method):
        actor, mean, sd = models[method]
        with torch.no_grad():
            hidden = actor.net[1](actor.net[0]((state-mean)/sd));prob = actor(state.sub(mean).div(sd)).softmax(-1)
        return prob, dict(hidden_saturation=float((hidden.abs() >= .99).float().mean()),
                         entropy=float(-(prob*prob.clamp_min(1e-30).log()).sum()/math.log(9)),
                         maximum_probability=float(prob.max()))

    policies = [];coverage = [];saved = [];prob_checks = 0
    for i in range(4):
        for method in ('TIME_FIXED', *models):
            rows = json.loads((deployment/f'jobs/train{i}/{method}_DECISIONS.private.json').read_text())
            assert [r['step'] for r in rows] == [100, 200]
            for row in rows:saved.append(dict(row, context=f'dev{i}', method=method))
    assert len(saved) == 56
    for context in range(4):
        entries = [r['state'] for r in saved if r['context'] == f'dev{context}' and r['step'] == 100]
        assert all(v == entries[0] for v in entries), 'method-specific entry100 feature mismatch'
    for row in saved:
        state = torch.tensor(row['state']);standard = (state-center)/scale
        indices = [i for i,r in enumerate(table) if r['step'] == row['step']]
        nearest = min(indices, key=lambda i:rms(standard.tolist(), z[i].tolist()))
        distance = rms(standard.tolist(), z[nearest].tolist())
        outside = (state < x[indices].min(0).values-1e-7) | (state > x[indices].max(0).values+1e-7)
        base = dict(context=row['context'], method=row['method'], step=row['step'], action=row['action'])
        coverage.append(dict(base, nearest_train_context=table[nearest]['context'], distance=distance,
            training_loco_max=limits[row['step']], coverage_flag=distance > limits[row['step']],
            outside_dimensions=int(outside.sum()), extreme_dimensions=int((standard.abs() > 5).sum()),
            max_abs_standardized=float(standard.abs().max()),
            shifted_constant_dimensions=int((constant & ((state-center).abs() > 1e-6)).sum())))
        for method in models:
            prob, stats = policy(state, method)
            policies.append(dict(base, evaluated_actor=method, **stats, sampled_action_probability=float(prob[row['action']])))
            if method == row['method']:
                assert (prob-torch.tensor(row['probabilities'])).abs().max() < 1e-6
                prob_checks += 1
    assert prob_checks == 48
    training = []
    for row in table:
        for method in models:
            _,stats = policy(torch.tensor(row['state']), method)
            training.append(dict(context=row['context'], step=row['step'], actor=method, **stats))
    primary = [r for r in saved if r['method'].startswith('Z_1024') and r['step'] == 200]
    unique = {(r['context'], tuple(r['state'])):r for r in primary}
    flags = [next(s['coverage_flag'] for s in coverage if (s['context'],s['method'],s['step']) ==
                  (r['context'],r['method'],r['step'])) for r in unique.values()]
    own = [r for r in policies if r['method'] == r['evaluated_actor'] and r['method'].startswith('Z_1024') and r['step'] == 200]
    train = [r for r in training if r['actor'].startswith('Z_1024') and r['step'] == 200]
    shift_fraction = st.mean(flags);sat_delta = st.median(r['hidden_saturation'] for r in own)-st.median(r['hidden_saturation'] for r in train)
    summary = dict(unique_primary_step200_states=len(unique), primary_step200_records=len(primary),
        primary_coverage_fraction=shift_fraction, coverage_priority=shift_fraction >= .5,
        saturation_median_increase=sat_delta, saturation_priority=sat_delta >= .25,
        training_distance_limits=limits, probability_replay_checks=prob_checks,
        raw_private_features_published=False, causal_claim=False, new_queries=0,
        actor_updates=0, student_updates=0, V84='NOT_RUN')
    csvfile('STATE_COVERAGE.csv', coverage);csvfile('TRAINING_DISTANCE_REFERENCE.csv', reference)
    csvfile('POLICY_DIAGNOSTICS.csv', policies);csvfile('TRAINING_POLICY_REFERENCE.csv', training)
    save('SUMMARY.json', summary)
    save('QUALIFICATION.json', dict(status='PASS', saved_probability_replay_checks=48,
                                  shared_entry_feature_checks=4, student_or_query_created=False))
    save('COSTS.json', dict(student_updates=0, actor_updates=0, new_queries=0, new_images=0))
    (out/'REPORT.md').write_text('# V8.8 query-free deployed-state diagnosis\n\n'+json.dumps(summary,indent=2)+'\n\nAdaptive descriptive analysis of repeatedly observed D1 development. Distances use the frozen full-table scaler. The threshold is a coverage heuristic, not a calibrated OOD test. Duplicate states are collapsed for the primary coverage fraction. Saturation compares both seeds and is descriptive. No counterfactual development reward is inferred. Raw features and checkpoints stay private.\n')
    save('FINAL.json', dict(status='COMPLETE', time=time.time(), publication='PENDING',
         student_updates=0, actor_updates=0, new_queries=0))


if __name__ == '__main__':
    selfcheck()
    if os.environ.get('EXEC_SELFCHECK') == '1':print('PASS: standardized distance self-check')
    else:main()
