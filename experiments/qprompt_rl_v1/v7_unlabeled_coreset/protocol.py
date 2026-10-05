"""Finite V7 image-selection pilot; freeze before any native performance runs."""
import math

DOMAINS = {'RIM_ONE_r3': 3200, 'Drishti_GS': 2100}
UNLABELED = {'RIM_ONE_r3': 63, 'Drishti_GS': 41}
REWARD_LABELS = {'RIM_ONE_r3': 4, 'Drishti_GS': 2}
LABELS = {'RIM_ONE_r3': 16, 'Drishti_GS': 10}
SEEDS = (601, 602)
BLOCK = 100
GROUP = 4
ROLE_SEED = 7101
FIXED = ('NO_U', 'ALL_U', 'CONFIDENCE', 'KCENTER', 'RETRIEVE_FO')


def k(domain):
    return math.ceil(.25 * UNLABELED[domain])


def matrix():
    jobs = []
    quals = ['qualification_' + d for d in DOMAINS]
    def add(name, kind, domain, caps, deps=(), **extra):
        jobs.append(dict(id='pilot__'+name, scope='pilot', name=name, job=kind,
            domain=domain, seed=168, caps=caps, deps=['pilot__'+v for v in deps], **extra))
    for d in DOMAINS:
        add('qualification_'+d, 'qualification', d, dict(smoke=12))
    for d, horizon in DOMAINS.items():
        for seed in SEEDS:
            name=f'learn_{d}_{seed}'
            add(name, 'learn', d, dict(development=GROUP*horizon, actor=16*(horizon//BLOCK)), quals, controller=seed)
        for method, seed in [(m, None) for m in FIXED] + [(m, s) for s in SEEDS for m in ('RANDOM', 'GRPO')]:
            name=f'endpoint_{d}_{method}'+('' if seed is None else f'_{seed}')
            deps=list(quals)
            if method=='GRPO':deps.append(f'learn_{d}_{seed}')
            add(name, 'endpoint', d, dict(endpoint=horizon), deps, method=method, controller=seed)
    return jobs


def scope():
    jobs=matrix()
    counts={}
    for j in jobs:
        for key,value in j['caps'].items():counts[key]=counts.get(key,0)+value
    assert len(jobs)==24 and sum(j['job']=='endpoint' for j in jobs)==18
    assert counts==dict(smoke=24,development=42400,actor=1696,endpoint=47700)
    return dict(protocol='V7_UNLABELED_CORESET_PILOT', source_seed=168, role_seed=ROLE_SEED,
        nominal_label_percent=20, labels=LABELS, reward_labels=REWARD_LABELS,
        unlabeled_images=UNLABELED, selected_images={d:k(d) for d in DOMAINS},
        horizons=DOMAINS, selection_interval=BLOCK, grpo_group_size=GROUP,
        controller_seeds=list(SEEDS), grpo_lr=.001, clip=.2, entropy=.01,
        group_reward_normalization='population std, floor 1e-6', source_KL_penalty=.1,
        source_confidence_threshold=.7, unlabeled_source_anchors=8,
        fixed_lambda_U=.5, endpoint_jobs=18, preparation_training_jobs=24,
        calls=counts, student_calls=90124, synthetic_actor_calls=16,
        wall_clock_limit=None, gpus=[5,6,7], sealed_test_access=False,
        validation='all campaign training complete before any val',
        prototype_RL='deferred; not implemented in this pilot')


if __name__=='__main__':
    import json
    print(json.dumps(scope(),indent=2))
