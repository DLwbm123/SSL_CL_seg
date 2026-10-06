"""V7.1 finite frozen-policy deployment diagnosis."""
from experiments.qprompt_rl_v1.v7_unlabeled_coreset import protocol as old
BASELINE='03224a49cef09a3325abb7d32ffd8762e678e880'
TRAINING='37ac930652dc2da6585342a437d014ae3b4b995d'
METHODS=('GRPO_SAMPLE','UNIFORM_MATCHED')
def matrix():
    jobs=[]
    def add(name,kind,d,caps,**kw):
        jobs.append(dict(id='v71__'+name,scope='v71',name=name,job=kind,domain=d,seed=168,caps=caps,deps=[],**kw))
    for d in old.DOMAINS:add('diagnosis_'+d,'diagnosis',d,{})
    for d in old.DOMAINS:add('qualification_'+d,'qualification',d,dict(smoke=16))
    for d,h in old.DOMAINS.items():
        for s in old.SEEDS:
            for m in METHODS:add(f'endpoint_{d}_{m}_{s}','endpoint',d,dict(endpoint=h),controller=s,method=m)
    return jobs

def scope():
    return dict(protocol='V7.1 Frozen-policy deployment diagnosis',baseline_commit=BASELINE,training_commit=TRAINING,
        source_seed=168,source_updates=8000,roles_seed=7101,label_percent=20,domains=old.DOMAINS,
        U=old.UNLABELED,labels=old.LABELS,reward_labels=old.REWARD_LABELS,K={d:old.k(d) for d in old.DOMAINS},
        interval=100,lambda_U=.5,confidence=.7,temperature=1,methods=METHODS,controllers=old.SEEDS,
        qualification_updates=32,endpoint_updates=21200,student_budget=21232,controller_updates=0,
        virtual_updates=0,replay_updates=0,gpus=[5,6,7],automatic_retry=False,sealed_test=False,
        rng='stable("V7.1/deploy", domain, controller_seed, block); private CPU torch.Generator; no method key',
        evaluation='all eight final checkpoints frozen before any new historical-development validation',
        D1_descriptive_recovery='overall equal-pair mean new-domain delta > 0; mixed cells always disclosed; no significance claim',
        D2=dict(mean_new_pp=.5,positive_pairs=3,each_seed_mean_new_positive=True,mean_old_pp=-.5,min_old_pp=-1),
        D3=dict(mean_best_nonRL_pp=.5,mean_random_pp=.5,mean_old_allU_pp=-.5,min_old_allU_pp=-1,min_best_nonRL_pp=-.25,each_seed_mean_best_positive=True),
        native='NATIVE_LR_SRC_A_3DOMAIN_V1; A-only projection; not exact original KI reproduction',jobs=matrix())
