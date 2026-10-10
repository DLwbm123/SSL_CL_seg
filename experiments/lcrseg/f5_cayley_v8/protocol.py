from ..f5_module_pilots_v1.protocol import OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,read,write,metrics,compare as paired_compare
from ..f5_conflict_reject_v7.protocol import jobs as previous_jobs,plan as previous_plan
from ..five_frameworks_v1.gate import digest

ARMS=('F5','CAYLEY')
CAPS=dict(formal_updates=21200,source_updates=0,cpu_optimizer_updates=96,synthetic_cuda_updates=32,real_smoke_updates=4,diagnostic_vjps=64)

def jobs():
    return [{**j,'id':j['id'].replace('U_REJECT','CAYLEY'),'arm':'CAYLEY'} for j in previous_jobs()]

def plan():
    j=jobs();v=previous_plan();v.pop('plan_id')
    v.update(study_id='F5_CAYLEY_V8',authority='user 2026-10-10: 下一轮优先做有依据的方法模块改进，不要只调整系数，继续每小时监测。',
             jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],
             equation='original F5 A/B L-only, R-only U; S=(R-R.T)/2, O=solve(I-S,I+S), G=I+Q(O-I)Q.T, F_next=F_prev G; EMA raw R',
             hypothesis='remove adapter-induced cumulative feature scaling; original F5 sealed second-stage condition numbers1.93–2.36; V7 rejection gain0.0185pp insufficient',
             comparison='matched archived F5 via disabled-Cayley native parity; historical B0/B2; original lambda_U0.25/PAS0.7,0.5/SWD0.2 retained',
             limitations=['orthogonality does not preserve classifier logits or backbone retention; scaling causality unproven','effective degrees of freedom28 instead of64 may restrict adaptation','B2 joint A/B U capacity and PAS differences persist','reused development; original OFT was not medical continual segmentation'])
    v['plan_id']=digest(v);return v

def compare(rows,reference):
    return paired_compare(rows,reference,arms=('CAYLEY',))
