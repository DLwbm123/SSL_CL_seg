from ..f5_module_pilots_v1.protocol import OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,read,write,metrics,compare as paired_compare
from ..f5_conflict_reject_v7.protocol import jobs as previous_jobs,plan as previous_plan
from ..five_frameworks_v1.gate import digest

ARMS=('F5','JOINT_KL')
CAPS=dict(formal_updates=21200,source_updates=0,cpu_optimizer_updates=96,synthetic_cuda_updates=32,real_smoke_updates=4,diagnostic_vjps=64)

def jobs():
    return [{**j,'id':j['id'].replace('U_REJECT','JOINT_KL'),'arm':'JOINT_KL'} for j in previous_jobs()]

def plan():
    j=jobs();v=previous_plan();v.pop('plan_id')
    v.update(study_id='F5_JOINT_KL_BRIDGE_V9',authority='user 2026-10-10: 下一轮优先做有依据的方法模块改进，不要只调整系数，继续每小时监测。',
             jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],
             equation='original linear F5; KL jointly reaches all trainable A/B/R; SWD reaches R only; original coefficients and hard input projection',
             hypothesis='V8 orthogonality achieved but Old dropped0.7446pp vs F5; B2 all-layer joint A/B U feedback remains untested by prior B-only V3–V5',
             comparison='matched archived F5 via disabled joint bridge native parity; historical B0/B2; original lambda_U0.25/PAS0.7,0.5/SWD0.2 retained',
             limitations=['joint feedback may harm Old; capacity hypothesis is not causal proof','B2 differs in PAS/weight and has no R; this is not full B2 reproduction','reused development; not LoRA-Pro optimization, independent generalization or SOTA'])
    v['plan_id']=digest(v);return v

def compare(rows,reference):
    return paired_compare(rows,reference,arms=('JOINT_KL',))
