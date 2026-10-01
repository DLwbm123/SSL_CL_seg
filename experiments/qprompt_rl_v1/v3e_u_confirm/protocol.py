"""Frozen comparison and descriptive stability rule, independent of torch."""
import math
import statistics

SEEDS = (168,169,170,171,172)
DOMAINS = ('RIM_ONE_r3','Drishti_GS')
WEIGHTS = {'ORIGINAL':0.,'FINE_0125':.125,'FINE_05':.5}
H = 1200
CAPS = dict(source=40000, main=36000, smoke=80, development=0, controller=0, replay=3804)
METRICS = ('macro_Dice','old_REFUGE')


def decision(rows, candidate='FINE_0125', references=('ORIGINAL','FINE_05'), weights=WEIGHTS):
    lookup = {(r['seed'],r['domain'],r['method']):r for r in rows}
    expected = {(s,d,m) for s in SEEDS for d in DOMAINS for m in weights}
    assert len(rows)==len(lookup)==len(expected) and set(lookup)==expected
    assert all(math.isfinite(float(r[k])) for r in rows for k in METRICS)
    comparisons = {}
    for ref in references:
        cells = [dict(seed=s,domain=d,**{k:float(lookup[s,d,candidate][k])-float(lookup[s,d,ref][k]) for k in METRICS}) for s in SEEDS for d in DOMAINS]
        seeds = {str(s):{k:statistics.mean(r[k] for r in cells if r['seed']==s) for k in METRICS} for s in SEEDS}
        domains = {d:{k:statistics.mean(r[k] for r in cells if r['domain']==d) for k in METRICS} for d in DOMAINS}
        stats = {}
        for k in METRICS:
            values = [v[k] for v in seeds.values()];mean=statistics.mean(values);sd=statistics.stdev(values)
            margin = 2.7764451051977987*sd/math.sqrt(5)
            stats[k] = dict(mean=mean,sample_std=sd,descriptive_t95=[mean-margin,mean+margin])
        comparisons[ref] = dict(cells=cells,seeds=seeds,domains=domains,statistics=stats,
            jointly_positive_seeds=sum(all(v[k]>=0 for k in METRICS) and any(v[k]>0 for k in METRICS) for v in seeds.values()))
    primary = comparisons[references[0]]
    gates = dict(joint_mean_gain=all(primary['statistics'][k]['mean']>0 for k in METRICS),
        four_of_five_joint_seeds=primary['jointly_positive_seeds']>=4,
        no_domain_mean_regression=all(v[k]>=0 for v in primary['domains'].values() for k in METRICS))
    return dict(stable_candidate=all(gates.values()),gates=gates,comparisons=comparisons,
        interpretation='Descriptive optimization-seed stability on reused validation patients; not independent clinical confirmation or a significance test.')


def check():
    assert len(SEEDS)*len(DOMAINS)*len(WEIGHTS)*H==CAPS['main']
    assert 8000*len(SEEDS)==CAPS['source']
    assert CAPS['replay']==math.ceil(.05*(CAPS['main']+CAPS['source']+CAPS['smoke']))
    def rows(bad=False):
        return [dict(seed=s,domain=d,method=m,macro_Dice=(.61 if m=='FINE_0125' else .6)-(.02 if bad and m=='FINE_0125' and d=='Drishti_GS' else 0),old_REFUGE=.71 if m=='FINE_0125' else .7) for s in SEEDS for d in DOMAINS for m in WEIGHTS]
    assert decision(rows())['stable_candidate']
    assert not decision(rows(True))['gates']['no_domain_mean_regression']
    try:decision(rows()[:-1])
    except AssertionError:pass
    else:raise AssertionError('missing cells accepted')
    print('PASS fixed budgets, positive candidate, domain regression rejection and complete paired coverage')


if __name__=='__main__':check()
