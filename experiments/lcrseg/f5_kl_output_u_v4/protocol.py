from ..f5_module_pilots_v1.protocol import OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,write,read,metrics,compare as paired_compare
from ..five_frameworks_v1.native_data import ORDERS
from ..five_frameworks_v1.gate import digest

ARMS=('F5','KL_OUTPUT_U')
CAPS=dict(formal_updates=21200,source_updates=0,real_smoke_updates=4,
          synthetic_cuda_updates=32,cpu_optimizer_updates=96,diagnostic_vjps=64)


def jobs():
    return [dict(id=f'KL_OUTPUT_U_S{s}_O{o}',arm='KL_OUTPUT_U',seed=s,order=o,
                 stages=[dict(stage=k,domain=ORDERS[o-1][k],steps=STEPS[ORDERS[o-1][k]]) for k in (1,2)])
            for s in (163,164) for o in (1,2)]


def plan():
    j=jobs()
    v=dict(study_id='F5_KL_OUTPUT_U_V4',parent='F5_C02',authority='user 2026-10-10: 继续分析 F5 和 B2 的差距，选一个最有依据的改进直接开展下一轮实验，保留每小时监测，直到达到成功门槛。',
           jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,seeds=[163,164],orders=ORDERS,
           execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],
           equation='gA=gL; gB=gL+gU_KL; gR=gL+gU_KL+gU_SWD; SWD detached from parent',
           hypothesis='isolate whether exposing B to SWD contributed to V3 retention failure; KL keeps B/R and SWD returns to R only',
           source_updates=0,retries=0,hyperparameter_search=False,
           comparison='reuse matched V1 F5 after disabled-output-U archived state parity; also qualify unchanged B2 and V3-total-U state parity; retain historical B0/B2',
           evaluation='seal all8 targets before20 seen-domain evaluations; no validation-based diagnostic pruning',
           diagnostic_first='S163/O1 runs first; continue remaining3 only on engineering completion, not efficacy selection',
           success='same paired practical gate against both matched F5 and historical B2; development only',
           limitations=['B2 differs also in U weight and PAS thresholds; permission effect is a hypothesis',
                        'B-only U may increase forgetting despite unchanged hard input constraints','development patients reused'])
    v['plan_id']=digest(v);return v


def compare(rows,reference):return paired_compare(rows,reference,arms=('KL_OUTPUT_U',))
