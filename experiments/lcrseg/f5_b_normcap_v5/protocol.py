from ..f5_module_pilots_v1.protocol import OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,write,read,metrics,compare as paired_compare
from ..five_frameworks_v1.native_data import ORDERS
from ..five_frameworks_v1.gate import digest

ARMS=('F5','B_NORMCAP')
CAPS=dict(formal_updates=21200,source_updates=0,real_smoke_updates=4,
          synthetic_cuda_updates=32,cpu_optimizer_updates=96,diagnostic_vjps=64)


def jobs():
    return [dict(id=f'B_NORMCAP_S{s}_O{o}',arm='B_NORMCAP',seed=s,order=o,
                 stages=[dict(stage=k,domain=ORDERS[o-1][k],steps=STEPS[ORDERS[o-1][k]]) for k in (1,2)])
            for s in (163,164) for o in (1,2)]


def plan():
    j=jobs()
    v=dict(study_id='F5_B_NORMCAP_V5',parent='F5_C02',authority='user 2026-10-10: 继续分析 F5 和 B2 的差距，选一个最有依据的改进直接开展下一轮实验，保留每小时监测，直到达到成功门槛。',
           jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,seeds=[163,164],orders=ORDERS,
           execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],
           equation='gA=gL; gB=gL+min(1,norm(gL_B)/norm(gU_B))*gU_KL; gR remains V4; zero L suppresses B U',
           hypothesis='bound B KL-U magnitude by same-step supervised B gradient; V4 retention failure persisted after SWD B exclusion',
           source_updates=0,retries=0,hyperparameter_search=False,
           comparison='reuse matched V1 F5 after disabled-output-U archived state parity; also qualify unchanged B2, restored V3-total-U, and cap-disabled archived V4 state parity; retain historical B0/B2 and report paired V5-V4',
           evaluation='seal all8 targets before20 seen-domain evaluations; no validation-based diagnostic pruning',
           diagnostic_first='S163/O1 runs first; continue remaining3 only on engineering completion, not efficacy selection',
           success='same paired practical gate against both matched F5 and historical B2; development only',
           limitations=['B2 differs also in U weight and PAS thresholds; magnitude interference is an unproven hypothesis',
                        'raw gradient norm bound does not bound Adam displacement or guarantee retention','development patients reused'])
    v['plan_id']=digest(v);return v


def compare(rows,reference):return paired_compare(rows,reference,arms=('B_NORMCAP',))
