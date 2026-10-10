from ..f5_module_pilots_v1.protocol import OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,write,read,metrics,compare as paired_compare
from ..five_frameworks_v1.native_data import ORDERS
from ..five_frameworks_v1.gate import digest

ARMS=('F5','PROJECT')
CAPS=dict(formal_updates=21200,source_updates=0,real_smoke_updates=4,
          synthetic_cuda_updates=32,cpu_optimizer_updates=96,diagnostic_vjps=64)


def jobs():
    return [dict(id=f'PROJECT_S{s}_O{o}',arm='PROJECT',seed=s,order=o,
                 stages=[dict(stage=k,domain=ORDERS[o-1][k],steps=STEPS[ORDERS[o-1][k]]) for k in (1,2)])
            for s in (163,164) for o in (1,2)]


def plan():
    j=jobs()
    v=dict(study_id='F5_PROJECTION_V2',parent='F5_C02',authority='user 2026-10-09: 如果结果出来不满意，你可以自己分析并提升，直至成功',
           jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,seeds=[163,164],orders=ORDERS,
           execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],
           equation='R gradient: gL + gU - min(0,dot(gL,gU))/norm(gL)^2*gL; zero gL leaves gU unchanged',
           hypothesis='filter observed conflicting total-U components without changing F5 architecture/objectives/data/permissions',
           source_updates=0,retries=0,hyperparameter_search=False,
           comparison='reuse matched V1 F5 after disabled-projection historical state parity; retain historical B0/B2',
           evaluation='seal all8 targets before20 seen-domain evaluations; no validation-based diagnostic pruning',
           diagnostic_first='S163/O1 runs first; continue remaining3 only on engineering completion, not efficacy selection',
           success='same paired practical gate against both matched F5 and historical B2; development only',
           limitations=['sparse V1 conflict samples do not establish cause','one-sided adaptation is not symmetric PCGrad',
                        'raw gradient projection does not guarantee an Adam step or old-domain improvement','development patients reused'])
    v['plan_id']=digest(v);return v


def compare(rows,reference):return paired_compare(rows,reference,arms=('PROJECT',))
