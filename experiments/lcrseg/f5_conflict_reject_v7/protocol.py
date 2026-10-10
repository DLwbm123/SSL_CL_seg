from ..f5_module_pilots_v1.protocol import OPTIONS as ORIGINAL_OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,read,write,metrics,compare as paired_compare
from ..f5_b_normcap_v5.protocol import jobs as old_jobs,plan as old_plan
from ..five_frameworks_v1.gate import digest
OPTIONS=dict(ORIGINAL_OPTIONS)
ARMS=('F5','U_REJECT')
CAPS=dict(formal_updates=21200,source_updates=0,cpu_optimizer_updates=96,synthetic_cuda_updates=32,real_smoke_updates=4,diagnostic_vjps=64)
def jobs():return [{**j,'id':j['id'].replace('B_NORMCAP','U_REJECT'),'arm':'U_REJECT'} for j in old_jobs()]
def plan():
    j=jobs();v=old_plan();v.pop('plan_id')
    v.update(study_id='F5_CONFLICT_REJECT_V7',authority='user 2026-10-10: 继续分析 F5 和 B2 的差距，选一个最有依据的改进直接开展下一轮实验，保留每小时监测，直到达到成功门槛。',jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,
             execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],equation='original F5: A/B L-only; R L+U if dot(gL_R,gU_R)>=0 else L; lambda_U0.25',
             hypothesis='reject entire R U gradient on negative supervised dot; V6 strength worsened retention and V2 projection gain was insufficient',
             comparison='archived matched F5 lambda_U0.25 and historical B0/B2; disabled-rejection F5 state parity; archived V2 paired context',
             limitations=['B2 also differs in PAS and U permissions; conflict causality unproven','rejecting U can reduce adaptation; current L alignment does not guarantee retention','reused development; prior V5 had disclosed cross-runtime recovery'])
    v['plan_id']=digest(v);return v
def compare(rows,reference):return paired_compare(rows,reference,arms=('U_REJECT',))
