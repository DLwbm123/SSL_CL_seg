from ..f5_module_pilots_v1.protocol import OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,read,write,metrics,compare as paired_compare
from ..f5_anchor_guard_v10.protocol import jobs as previous_jobs,plan as previous_plan
from ..five_frameworks_v1.gate import digest
ARMS=('F5','ENTRY_GRADIENT')
CAPS=dict(formal_updates=21200,source_updates=0,cpu_optimizer_updates=96,synthetic_cuda_updates=32,real_smoke_updates=4,diagnostic_vjps=64,entry_reference_vjps=16960)
def jobs():
    return [{**j,'id':j['id'].replace('ANCHOR_GUARD','ENTRY_GRADIENT'),'arm':'ENTRY_GRADIENT'} for j in previous_jobs()]
def plan():
    j=jobs();v=previous_plan();v.pop('plan_id')
    v.update(study_id='F5_ENTRY_GRADIENT_V12',authority='user autonomous follow-up: retain positive V10 base, evidence-based module rather than coefficients, hourly monitoring until original F5/B2 gate and public delivery',jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],equation='V10 KL rejection unchanged; on active U project concatenated merged L+U A/B/R raw gradient if negative dot with confident original-PAS entry soft-KL reference; zero reference retains merged gradient',hypothesis='V11 average improves but164 Drishti cup loses27.7202pp; constrain combined supervised/unsupervised update direction rather than force entry targets',comparison='disabled guard archived V10 full native parity; paired V12-V10 and V12-V11 plus matched F5/historical B0/B2',limitations=['entry-reference current-U proxy may not cover old domain and may be wrong','raw-gradient half-space is not an Adam update or finite-loss retention guarantee','warmup no U/reference constraint; no old images/replay; no extra loss or coefficient','16960 extra active-U VJPs counted separately from64 fixed diagnostics; reused development only'])
    v['plan_id']=digest(v);return v
def compare(rows,reference):return paired_compare(rows,reference,arms=('ENTRY_GRADIENT',))
