"""Frozen V5 matrix and paired decision rules; no performance-dependent tuning."""
from statistics import mean, stdev

BASELINE = '90eacbacca00d3c03e1296d5c7e6ac22a047df3b'
DOMAINS = ('RIM_ONE_r3', 'Drishti_GS')
HORIZONS = dict(zip(DOMAINS, (3200, 2100)))
DEVELOPMENT_SEEDS = (168, 169, 170)
CONFIRMATION_SEEDS = (183, 184, 185, 186, 187)
CONTROLLERS = (401, 402, 403)
GPUS = (5, 6, 7)
WEIGHTS = (0., .125, .5)
BLOCK, GROUP_SIZE, GROUPS = 25, 4, 6
P0_CALLS = 690
LEARNERS = ('PPO_MATCHED', 'BANDIT_MATCHED', 'GRPO_STD', 'GRPO_FS', 'GRPO_FS_SHUFFLE')
BASELINES = ('ORIGINAL', 'FINE_05', 'OFFLINE_25')
CONTROLLED = LEARNERS + ('PHASE_SHUFFLE_FS',)

def phases(domain):
    n = HORIZONS[domain] // BLOCK
    return ((0,n//3),(n//3,2*n//3),(2*n//3,n))

def learning_jobs(controllers, methods=LEARNERS):
    return [dict(id=f'learn_{c}_{d}_{m}', job='learn', controller=c, domain=d, method=m,
                 caps=dict(development=GROUPS*GROUP_SIZE*HORIZONS[d], actor=96,
                           **({'critic':96} if m in LEARNERS[:2] else {})))
            for c in controllers for d in DOMAINS for m in methods]

def endpoint_methods(stage):
    if stage == 'pilot': return [(m, None) for m in BASELINES]+[(m,401) for m in CONTROLLED]
    if stage == 'cliphi': return [('GRPO_FS_CLIPHI',401)]
    return [(m,None) for m in BASELINES]+[(m,c) for c in CONTROLLERS for m in CONTROLLED]

def endpoint_jobs(stage):
    seeds = CONFIRMATION_SEEDS if stage == 'confirmation' else DEVELOPMENT_SEEDS
    return [dict(id=f'{stage}_{s}_{d}', job='endpoint', stage=stage, seed=s, domain=d,
                 caps=dict(endpoint=len(endpoint_methods(stage))*HORIZONS[d])) for s in seeds for d in DOMAINS]

def jobs():
    return dict(qualification=[dict(id='qualification',job='qualification',caps=dict(smoke=P0_CALLS))],
        initialize=[dict(id='init_'+d,job='initialize',domain=d,caps={}) for d in DOMAINS],
        calibration=[dict(id=f'cal_{s}_{d}',job='calibration',seed=s,domain=d,caps=dict(calibration=5*HORIZONS[d]))
                     for s in DEVELOPMENT_SEEDS for d in DOMAINS],
        references=[dict(id=f'ref_{g}_{d}',job='reference',group=g,seed=DEVELOPMENT_SEEDS[g%3],domain=d,
                         caps=dict(reference=HORIZONS[d])) for g in range(GROUPS) for d in DOMAINS],
        scale=[dict(id='scale_'+d,job='scale',domain=d,caps=dict(critic_warmup=128)) for d in DOMAINS],
        learn401=learning_jobs((401,)), pilot=endpoint_jobs('pilot'),
        extension=learning_jobs((402,403)),
        sources=[dict(id=f'source_{s}',job='source',seed=s,caps=dict(source=8000)) for s in CONFIRMATION_SEEDS],
        confirmation=endpoint_jobs('confirmation'),
        cliphi_learn=learning_jobs((401,),('GRPO_FS_CLIPHI',)), cliphi=endpoint_jobs('cliphi'))

def budget():
    student_categories={'smoke','calibration','reference','development','endpoint','source'}
    counts={stage:sum(v for j in queue for k,v in j['caps'].items() if k in student_categories)
            for stage,queue in jobs().items()}
    pilot=sum(counts[k] for k in ('calibration','references','learn401','pilot'))
    full=pilot+sum(counts[k] for k in ('extension','sources','confirmation'))
    extra=counts['cliphi_learn']+counts['cliphi']
    assert (pilot,full,extra)==(890400,2758900,143100)
    assert P0_CALLS<=1500
    return dict(stages=counts,pilot_without_P0=pilot,full_without_P0=full,conditional_extra=extra,
                P0=P0_CALLS,maximum_all=full+extra+P0_CALLS,reused_source_updates=24000,
                shared_reference_entries=12,reference_sharing='exact entry state + provider state + source receipt + code commit + domain + group; verified before use',
                source_seed_history='Not used in V5 development. Project-wide novelty not established; inaccessible historical directory exists.',
                jobs=jobs())

def compare(rows, reference, seeds, controllers, minimum=0., joint_required=4):
    lookup={(r['seed'],r['domain'],r['method'],r['controller']):r for r in rows}
    cells=[]
    for s in seeds:
        for d in DOMAINS:
            for c in controllers:
                a=lookup[s,d,'GRPO_FS',c]
                b=lookup[s,d,reference,None if reference in BASELINES else c]
                cells.append(dict(seed=s,domain=d,controller=c,**{k:a[k]-b[k] for k in ('macro_Dice','old_REFUGE')}))
    keys=('macro_Dice','old_REFUGE')
    seedrows=[dict(seed=s,**{k:mean(r[k] for r in cells if r['seed']==s) for k in keys}) for s in seeds]
    avg={k:mean(r[k] for r in seedrows) for k in keys}
    domains={d:{k:mean(r[k] for r in cells if r['domain']==d) for k in keys} for d in DOMAINS}
    joint=sum(r['macro_Dice']>0 and r['old_REFUGE']>=0 for r in seedrows)
    passed=avg['macro_Dice']>0 and avg['macro_Dice']>=minimum and avg['old_REFUGE']>=0 and joint>=joint_required and all(v[k]>=0 for v in domains.values() for k in keys)
    return dict(reference=reference,minimum=minimum,mean=avg,sample_sd={k:stdev(r[k] for r in seedrows) for k in keys},seed_deltas=seedrows,domain_deltas=domains,all_cells=cells,joint=joint,passed=passed,worst_cells={k:min(cells,key=lambda r:r[k]) for k in keys})

def pilot_decision(rows, engineering):
    # B asks aggregate mean comparisons only; per-seed/domain conditions belong to A.
    comparisons={m:compare(rows,m,DEVELOPMENT_SEEDS,(401,),.001 if m=='ORIGINAL' else 0.,2) for m in ('ORIGINAL','PPO_MATCHED','BANDIT_MATCHED','GRPO_FS_SHUFFLE')}
    passed=engineering and comparisons['ORIGINAL']['passed'] and all(v['mean']['macro_Dice']>0 and v['mean']['old_REFUGE']>=0 for m,v in comparisons.items() if m!='ORIGINAL')
    return dict(promoted=passed,engineering_pass=engineering,comparisons=comparisons,primary='GRPO_FS',uses_development_val=True,audit_used=False)

def confirmation_decision(rows):
    thresholds={'ORIGINAL':.002,'FINE_05':.003,'OFFLINE_25':0.,'PPO_MATCHED':.002,'BANDIT_MATCHED':.002,'GRPO_FS_SHUFFLE':.002,'PHASE_SHUFFLE_FS':0.}
    comparisons={m:compare(rows,m,CONFIRMATION_SEEDS,CONTROLLERS,t) for m,t in thresholds.items()}
    individual={c:compare(rows,'ORIGINAL',CONFIRMATION_SEEDS,(c,),.002) for c in CONTROLLERS}
    reproducible=sum(v['mean']['macro_Dice']>=.002 and v['mean']['old_REFUGE']>=0 for v in individual.values())>=2
    return dict(practical_candidate=all(comparisons[m]['passed'] for m in BASELINES),grpo_advantage_supported=comparisons['PPO_MATCHED']['passed'],multistep_control_supported=all(comparisons[m]['passed'] for m in ('BANDIT_MATCHED','GRPO_FS_SHUFFLE','PHASE_SHUFFLE_FS')),controller_reproduction_supported=reproducible,comparisons=comparisons,individual_controllers=individual)

if __name__=='__main__':
    import json
    print(json.dumps(budget(),indent=2))
