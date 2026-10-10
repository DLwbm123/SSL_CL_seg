import json
from pathlib import Path
from ..five_frameworks_v1.gate import digest
from ..five_frameworks_v1.native_data import ORDERS
from ..single_teacher_scd_v0_1.data import MANIFEST_SHA, SPLIT_SHA
from .modules import ARMS, SPEC

SEEDS=(163,164)
CAPS=dict(formal_updates=84800,source_updates=0,real_smoke_updates=16,
          synthetic_cuda_updates=64,cpu_optimizer_updates=768,diagnostic_vjps=272)
OPTIONS=dict(lr=.001,weight_decay=4e-5,parent_lr_multiplier=.5,lr_B_over_A=1.,
             feature_lr_multiplier=1.,feature_weight_decay_multiplier=1.,rank_ratio=.25,
             warmup_fraction=.2,U_ramp_fraction=.2,lambda_U=.25,lambda_SWD=.2,
             PAS_confidence=.7,PAS_cosine=.5)
STEPS={'RIM_ONE_r3':3200,'Drishti_GS':2100}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();__import__('os').fsync(f.fileno())
    tmp.replace(path)


def read(path):return json.loads(Path(path).read_text())


def jobs():
    return [dict(id=f'{arm}_S{seed}_O{order}',arm=arm,seed=seed,order=order,
                 stages=[dict(stage=s,domain=ORDERS[order-1][s],steps=STEPS[ORDERS[order-1][s]]) for s in (1,2)])
            for arm in ARMS for seed in SEEDS for order in (1,2)]


def plan():
    result=dict(study_id='F5_MODULE_PILOTS_V1',parent='F5_C02',seeds=SEEDS,orders=ORDERS,
                arms=ARMS,module_spec=SPEC,options=OPTIONS,caps=CAPS,jobs=jobs(),
                sequence='fresh F5 control, boundary/interior, OT triplet, loss-indexed linear adapters',
                evaluation='seal all 32 target students before 80 seen-domain evaluations; reuse historical B0/B2 aggregates',
                retries=0,hyperparameter_search=False,combined_module_arm=False,
                train_access='current L image/GT and current U image/geometry only; no historical images',
                authority='user 2026-10-09: 按照优先级，对这几个方法的模块都进行尝试，设置每小时监测',
                boundary='probability-derived 2D edge BCE plus x/y marginal interior CDF; 2px band and mix-seam exclusion; CDF after original warmup',
                ot='two current-L teacher centroids per foreground class; uniform empirical Sinkhorn triplet; no persistent bank; U gradients remain R only',
                adapter='R_eff=R+(1/3)sum_k up_k@down_k, three rank1 linear branches; 0.001 down-vector orthogonality; shared-output losses; effective-matrix EMA',
                limitations=['development cohort reused','two optimization seeds only','linear adapter is an adaptation, not nonlinear PINN reproduction',
                             'OT centroids are current-batch fixed references, not the paper learned prototypes','spatial marginal CDF is not topology preservation'])
    result['plan_id']=digest(result);return result


def metrics(stages,source):
    first,last=stages;seen=last['seen'];values=[last['scores'][d]['macro_Dice'] for d in seen]
    return dict(Final=sum(values)/3,Old=sum(values[:2])/2,Incoming=values[2],
                Forget=(source['scores'][seen[0]]['macro_Dice']-values[0]+first['scores'][seen[1]]['macro_Dice']-values[1])/2)


def compare(rows,reference,arms=ARMS[1:]):
    table={(r['arm'],r['seed'],r['order']):r for r in rows};output={}
    for arm in arms:
        pairs=[dict(seed=s,order=o,**{m:table[arm,s,o]['metrics'][m]-table[reference,s,o]['metrics'][m]
                                    for m in ('Final','Old','Incoming','Forget')}) for s in SEEDS for o in (1,2)]
        per_seed={str(s):{m:sum(r[m] for r in pairs if r['seed']==s)/2 for m in ('Final','Old','Incoming','Forget')} for s in SEEDS}
        mean={m:sum(v[m] for v in per_seed.values())/2 for m in ('Final','Old','Incoming','Forget')}
        order_means={str(o):{m:sum(r[m] for r in pairs if r['order']==o)/2 for m in ('Final','Old','Incoming','Forget')} for o in (1,2)}
        gate=all(v['Final']>0 for v in per_seed.values()) and mean['Final']>=.005
        gate=gate and all(v['Old']>=-.005 and v['Incoming']>=-.005 for v in order_means.values())
        class_costs=[]
        for s in SEEDS:
            for o in (1,2):
                a,b=table[arm,s,o],table[reference,s,o]
                for domain in a['scores']:
                    for c in ('rim','cup'):
                        class_costs.append(dict(seed=s,order=o,domain=domain,cls=c,delta=a['scores'][domain][c]-b['scores'][domain][c]))
        gate=gate and all(v['delta']>=-.05 for v in class_costs)
        output[arm]=dict(pairs=pairs,per_seed=per_seed,mean=mean,per_order=order_means,class_costs=class_costs,
                         practical_development_gate=gate,independent_confirmation=False)
    return output
