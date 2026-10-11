from ..f5_module_pilots_v1.protocol import OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,read,write,metrics,compare as paired_compare
from ..f5_anchor_guard_v10.protocol import jobs as previous_jobs,plan as previous_plan
from ..five_frameworks_v1.gate import digest

ARMS=('F5','ANCHOR_RESTORE')
CAPS=dict(formal_updates=21200,source_updates=0,cpu_optimizer_updates=96,synthetic_cuda_updates=32,real_smoke_updates=4,diagnostic_vjps=64)

def jobs():
    return [{**j,'id':j['id'].replace('ANCHOR_GUARD','ANCHOR_RESTORE'),'arm':'ANCHOR_RESTORE'} for j in previous_jobs()]

def plan():
    j=jobs();v=previous_plan();v.pop('plan_id')
    v.update(study_id='F5_ANCHOR_RESTORE_V11',authority='user 2026-10-11: 继续分析种子164旧域退化的原因，以V10为底座选一个最有依据的方法模块直接开展下一轮，不要只调系数，继续每小时监测直到超过B2并达到成功门槛。',
             jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],
             equation='V10 entry guard conflicts on original PAS support now use immutable entry soft target; other pixels use EMA; one KL normalized by original PAS support; SWD/L unchanged',
             hypothesis='V10 rejects6.7872percent PAS support but seed164 Drishti cup still loses22.5312pp; restoration could provide missing corrective knowledge gradient; causality not established',
             comparison='disabled restoration archived V10 native full-state parity; paired vs V10 plus matched F5 and historical B0/B2; original coefficients/PAS unchanged',
             limitations=['entry teacher may be confidently wrong on current domain','restoring original PAS denominator also dilutes retained EMA pixels relative to V10; not isolated target-only causal attribution','current-U preservation does not constrain all supervised updates or guarantee old-domain retention','development reuse; LwF-inspired soft target routing, not independent confirmation or full paper reproduction'])
    v['plan_id']=digest(v);return v

def compare(rows,reference):
    return paired_compare(rows,reference,arms=('ANCHOR_RESTORE',))
