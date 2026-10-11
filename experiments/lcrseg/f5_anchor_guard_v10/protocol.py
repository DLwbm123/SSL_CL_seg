from ..f5_module_pilots_v1.protocol import OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,read,write,metrics,compare as paired_compare
from ..f5_conflict_reject_v7.protocol import jobs as previous_jobs,plan as previous_plan
from ..five_frameworks_v1.gate import digest

ARMS=('F5','ANCHOR_GUARD')
CAPS=dict(formal_updates=21200,source_updates=0,cpu_optimizer_updates=96,synthetic_cuda_updates=32,real_smoke_updates=4,diagnostic_vjps=64)

def jobs():
    return [{**j,'id':j['id'].replace('U_REJECT','ANCHOR_GUARD'),'arm':'ANCHOR_GUARD'} for j in previous_jobs()]

def plan():
    j=jobs();v=previous_plan();v.pop('plan_id')
    v.update(study_id='F5_ANCHOR_GUARD_V10',authority='user 2026-10-10: 下一轮优先做有依据的方法模块改进，不要只调整系数，继续每小时监测。',
             jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],
             equation='positive V9 joint KL; reject PAS KL pixels only if frozen stage-entry confidence>=original PAS0.7 and class disagrees with current EMA; original SWD/PAS/coefficients unchanged',
             hypothesis='V9 passes F5 gate but seed164 Old remains5.9283pp below B2; immutable entry agreement could limit drift-amplifying KL feedback; incorrect pseudo-label causality unproven',
             comparison='disabled guard archived V9 full native parity; original F5 plus historical B0/B2 reuse; additionally paired vs V9; lambda_U0.25/PAS0.7,0.5/SWD0.2 unchanged',
             limitations=['entry teacher can be confidently wrong on new domain; rejection can reduce adaptation','entry predictions on current U are not old-domain supervision or retention guarantee','reused development; LwF-inspired rejection is not paper distillation reproduction or independent generalization'])
    v['plan_id']=digest(v);return v

def compare(rows,reference):
    return paired_compare(rows,reference,arms=('ANCHOR_GUARD',))
