from ..f5_module_pilots_v1.protocol import OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,read,write,metrics,compare as paired_compare
from ..f5_anchor_guard_v10.protocol import jobs as previous_jobs,plan as previous_plan
from ..five_frameworks_v1.gate import digest
ARMS=('F5','CLASS_CONE')
CAPS=dict(formal_updates=21200,source_updates=0,cpu_optimizer_updates=96,synthetic_cuda_updates=32,real_smoke_updates=4,diagnostic_vjps=64,entry_reference_vjps=50880)
def jobs():
    return [{**j,'id':j['id'].replace('ANCHOR_GUARD','CLASS_CONE'),'arm':'CLASS_CONE'} for j in previous_jobs()]
def plan():
    j=jobs();v=previous_plan();v.pop('plan_id')
    v.update(study_id='F5_CLASS_CONE_V15',authority='user autonomous method-module follow-up on V13 base, no coefficient search, hourly until original F5/B2 gates and public delivery',jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],equation='V13 losses unchanged; compute each present confident-PAS entry class KL gradient h_c; nearest raw merged L+U gradient with all dot(d,h_c)>=0, no current-L constraint',hypothesis='V14 current-L constraint never binds and all anonymous results equalV13; balanced aggregate entry direction does not individually constrain rim/cup. Test separate present-class directions; formal classwise conflict was unmeasured, not established cause',comparison='disabled class cone archived V13 complete native state/loss parity; paired V15-V13/V12/V10 and matched F5/historical B0/B2; V14 identical-result negative background retained',limitations=['current-U proxy and entry predictions may miss old domains or be wrong','classwise raw constraints do not constrain Adam update or guarantee finite losses/old domain','opposing class directions can suppress learning, possibly worsening Incoming','up to3 reference VJPs per active step,50880 cap,16960 additional-call-independent guard count, fixed R diagnostic64','no new forward/loss/coefficient/teacher/replay; reused development only'])
    v['plan_id']=digest(v);return v
def compare(rows,reference):return paired_compare(rows,reference,arms=('CLASS_CONE',))
