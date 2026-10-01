"""Frozen paired, phase-matched action permutation; no torch or data dependency."""
import hashlib
import random
from collections import Counter

SEEDS = (165, 166, 167)
METHODS = ('ORIGINAL', 'FINE', 'EFFECT_RULE', 'PHASE_SHUFFLE')
DOMAINS = ('RIM_ONE_r3', 'Drishti_GS')
H = 1200
CAPS = dict(source=24000, main=28800, smoke=80, development=0, controller=0, replay=2644)


def phases(domain):
    switch = {'RIM_ONE_r3': 640, 'Drishti_GS': 420}[domain]
    edges = sorted({0, 300, 600, 900, H, switch})
    return list(zip(edges[:-1], edges[1:]))


def permute(actions, seed, domain, start, end):
    assert (start, end) in phases(domain)
    assert len(actions) == (end - start) // 5
    assert all(type(a) is int and a in (0, 1, 2) for a in actions)
    token = f'V3C-RULE-TIMING/{seed}/{domain}/{start}/{end}'
    rng = random.Random(int.from_bytes(hashlib.sha256(token.encode()).digest(), 'big'))
    indices = list(range(len(actions)))
    rng.shuffle(indices)
    shuffled = [actions[i] for i in indices]
    assert Counter(shuffled) == Counter(actions)
    return dict(start=start, end=end, permutation=indices, rule_actions=list(actions),
                actions=shuffled, counts=[actions.count(a) for a in range(3)],
                changed_positions=sum(a != b for a, b in zip(actions, shuffled)),
                randomization=token, score_access=False)


def check():
    state = random.getstate()
    for domain in DOMAINS:
        covered = []
        for start, end in phases(domain):
            covered.extend(range(start, end, 5))
            for actions in ([2] * ((end-start)//5), [i % 3 for i in range((end-start)//5)]):
                first = permute(actions, 165, domain, start, end)
                assert first == permute(actions, 165, domain, start, end)
                assert sorted(first['permutation']) == list(range(len(actions)))
                assert first['counts'] == [first['actions'].count(a) for a in range(3)]
        assert covered == list(range(0, H, 5))
    assert random.getstate() == state
    assert len(SEEDS)*len(DOMAINS)*len(METHODS)*H == CAPS['main']
    assert sum(8000 for _ in SEEDS) == CAPS['source']
    print('PASS: deterministic bijection, exact phase counts, phase coverage, constant-action phases, RNG isolation, budgets')


if __name__ == '__main__':
    check()
