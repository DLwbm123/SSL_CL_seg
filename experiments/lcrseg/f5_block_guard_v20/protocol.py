from ..f5_module_pilots_v1.protocol import OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,read,write,metrics,compare as paired_compare
from ..f5_anchor_guard_v10.protocol import jobs as previous_jobs,plan as previous_plan
from ..five_frameworks_v1.gate import digest
ARMS=('F5','BLOCK_GUARD')
CAPS=dict(formal_updates=21200,source_updates=0,cpu_optimizer_updates=96,synthetic_cuda_updates=32,real_smoke_updates=4,diagnostic_vjps=64,entry_reference_vjps=16960)
def jobs():
    return [{**j,'id':j['id'].replace('ANCHOR_GUARD','BLOCK_GUARD'),'arm':'BLOCK_GUARD'} for j in previous_jobs()]
def plan():
    j=jobs();v=previous_plan();v.pop('plan_id')
    v.update(study_id='F5_BLOCK_GUARD_V20',authority='autonomous evidence-based independent follow-up after complete four-literature queue delivery and failed original dual gates; retain common V13',jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],equation='same V13 class-balanced h, partition A/B/R; independent nearest raw halfspace gX dot hX>=0 for each parameter block; native Adam unchanged',hypothesis='a global entry constraint can trade a negative A/B/R block contribution against a positive one; block constraints remove that cancellation without changing classes or reference divergence',comparison='disabled archived V13 full native state/loss/update parity; V13 matched base and V15/V17/V19/V10 context, matched F5 and historical B0/B2',qualification_advancement='all engineering checks PASS and at least one of exactly two unforced generated native active updates has a V13 counterfactual block cosine below -1e-6; otherwise preserve qualification-only negative without more probes/formal',limitations=['blockwise first-order Euclidean constraint only, not actual Adam or effective BA/finite-step guarantee','no new per-class constraints, GUA displacement, SARD or CaRE component','current-U teacher may be wrong and cannot cover all historical inputs','two generated native checks do not establish real patient incidence','no formal retry or coefficient search; repeated development data'])
    v['plan_id']=digest(v);return v
def compare(rows,reference):return paired_compare(rows,reference,arms=('BLOCK_GUARD',))
