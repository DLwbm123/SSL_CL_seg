from ..f5_module_pilots_v1.protocol import OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,read,write,metrics,compare as paired_compare
from ..f5_anchor_guard_v10.protocol import jobs as previous_jobs,plan as previous_plan
from ..five_frameworks_v1.gate import digest
ARMS=('F5','DUAL_REFERENCE')
CAPS=dict(formal_updates=21200,source_updates=0,cpu_optimizer_updates=96,synthetic_cuda_updates=32,real_smoke_updates=4,diagnostic_vjps=64,entry_reference_vjps=16960)
def jobs():
    return [{**j,'id':j['id'].replace('ANCHOR_GUARD','DUAL_REFERENCE'),'arm':'DUAL_REFERENCE'} for j in previous_jobs()]
def plan():
    j=jobs();v=previous_plan();v.pop('plan_id')
    v.update(study_id='F5_DUAL_REFERENCE_V14',authority='user explicitly requests autonomous next method module on V13 base for164 RIM rim and Incoming, hourly until original F5/B2 gates and public delivery',jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],equation='retain V13 class-balanced entry KL reference h and all losses; nearest raw merged g in intersection d dot h >=0 and d dot current labeled gradient >=0; enumerate active sets for two-halfspace QP',hypothesis='V13 restores163 cup and preserves164 Old but Incoming drops; a single entry-reference constraint need not preserve current supervised direction. Test simultaneous protection without dropping balanced entry constraint; actual V13 labeled-direction conflict frequency was unmeasured',comparison='disabled dual guard archived V13 complete native state/loss parity; paired V14-V13/V12/V10 and matched F5/historical B0/B2',limitations=['current-U entry proxy does not cover old domains or separately guarantee rim','current-L includes original supervised constraint, not only task CE','raw-gradient constraints do not constrain Adam updates or guarantee finite losses','same one reference VJP, no added forward/VJP/loss/coefficient; double solve and cast numerical tolerances','reused development only; reference or current L may conflict, yielding zero gradient'])
    v['plan_id']=digest(v);return v
def compare(rows,reference):return paired_compare(rows,reference,arms=('DUAL_REFERENCE',))
