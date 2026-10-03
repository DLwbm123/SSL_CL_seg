"""Scientific scope shared by the coordinator, workers and report."""
DOMAINS = ('RIM_ONE_r3', 'Drishti_GS')
HORIZONS = dict(zip(DOMAINS, (3200, 2100)))
DEVELOPMENT_SEEDS = (168, 169, 170)
CONFIRMATION_SEEDS = (173, 174, 175, 176, 177)
GPUS = (5, 6, 7)
WEIGHTS = (0., .125, .5)
BLOCK = 25
PANEL_STRIDE = 50
EPISODES = 12
METHODS = ('ORIGINAL', 'FINE_0125', 'FINE_05', 'EARLY_05', 'EFFECT_RULE',
           'OFFLINE_5', 'OFFLINE_25', 'BANDIT_25', 'PPO_25',
           'REWARD_SHUFFLE', 'PHASE_SHUFFLE')
LEARNERS = ('BANDIT_25', 'PPO_25', 'REWARD_SHUFFLE')


def phases(domain):
    n = HORIZONS[domain] // BLOCK
    return ((0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n))


def expected_calls():
    # Panel: one full retained trajectory plus three 25-step branches every 50 steps.
    panel = sum(h + (h // PANEL_STRIDE) * 3 * BLOCK for h in HORIZONS.values()) * len(DEVELOPMENT_SEEDS)
    development = sum(HORIZONS.values()) * EPISODES * len(LEARNERS)
    main = sum(HORIZONS.values()) * len(CONFIRMATION_SEEDS) * len(METHODS)
    return dict(source=8000 * len(CONFIRMATION_SEEDS), panel=panel,
                development=development, main=main, student_without_smoke=40000 + panel + development + main)


def compare(rows, candidate, reference, minimum_gain=0.):
    from statistics import mean, stdev
    lookup = {(r['seed'], r['domain'], r['method']): r for r in rows}
    cells = [dict(seed=s, domain=d, **{k: lookup[s, d, candidate][k] - lookup[s, d, reference][k]
             for k in ('macro_Dice', 'old_REFUGE')}) for s in CONFIRMATION_SEEDS for d in DOMAINS]
    seeds = [{k: mean(r[k] for r in cells if r['seed'] == s) for k in ('macro_Dice', 'old_REFUGE')}
             for s in CONFIRMATION_SEEDS]
    averages = {k: mean(r[k] for r in cells) for k in ('macro_Dice', 'old_REFUGE')}
    domains = {d: {k: mean(r[k] for r in cells if r['domain'] == d) for k in averages} for d in DOMAINS}
    joint = sum(r['macro_Dice'] > 0 and r['old_REFUGE'] >= 0 for r in seeds)
    passed = (averages['macro_Dice'] >= minimum_gain and averages['macro_Dice'] > 0
              and averages['old_REFUGE'] >= 0 and joint >= 4
              and all(v[k] >= 0 for v in domains.values() for k in averages))
    return dict(candidate=candidate, reference=reference, mean=averages,
                sample_sd={k: stdev(r[k] for r in seeds) for k in averages},
                seed_deltas=seeds, domain_deltas=domains, all_cells=cells,
                joint_positive_seeds=joint, minimum_new_gain=minimum_gain, passed=passed)


def check():
    assert not set(DEVELOPMENT_SEEDS) & set(CONFIRMATION_SEEDS)
    assert all(h % PANEL_STRIDE == 0 and h % BLOCK == 0 for h in HORIZONS.values())
    assert expected_calls()['student_without_smoke'] == 562050
    for d in DOMAINS:
        assert [i for a, b in phases(d) for i in range(a, b)] == list(range(HORIZONS[d] // BLOCK))


if __name__ == '__main__':
    check()
