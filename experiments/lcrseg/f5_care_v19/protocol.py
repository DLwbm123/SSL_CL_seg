from ..f5_module_pilots_v1.protocol import OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,read,write,metrics,compare as paired_compare
from ..f5_anchor_guard_v10.protocol import jobs as previous_jobs,plan as previous_plan
from ..five_frameworks_v1.gate import digest
ARMS=('F5','CARE')
CAPS=dict(formal_updates=21200,source_updates=0,cpu_optimizer_updates=96,synthetic_cuda_updates=32,real_smoke_updates=4,diagnostic_vjps=64,entry_reference_vjps=16960)
def jobs():
    return [{**j,'id':j['id'].replace('ANCHOR_GUARD','CARE'),'arm':'CARE'} for j in previous_jobs()]
def plan():
    j=jobs();v=previous_plan();v.pop('plan_id')
    v.update(study_id='F5_CARE_V19',authority='explicit independent four-paper queue and autonomous follow-up after complete SARD and REDistill delivery; common V13 base',jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],equation='reference-only hard CaRE: detached H(p)<=H(q) selects forward KL, else reverse KL; teacher reverse log floor log(dtype.tiny); unchanged V13 support/class means/raw halfspace and main losses',hypothesis='relative teacher/student entropy selects reference geometry without changing main supervision or spatial/class constraints; confidence is not correctness',comparison='disabled archived V13 full native state and old loss/update parity; V13 base plus V15/V17/V10 context, matched F5 and historical B0/B2',qualification_advancement='all engineering checks PASS including independent branch gradients, empty/invalid/extreme handling, generated-native reverse activation and nonzero gradient difference from V13; no patient efficacy selection',limitations=['hard CaRE divergence component adapted to segmentation reference, not full LLM CaRE-KD','no Revival or BALD; confidence can be confidently wrong','one aggregate raw-gradient constraint does not protect every class, actual Adam displacement or finite-step functions','fixed numerical floor is an explicit adaptation choice, not an original-paper constant','no formal retry; repeated development patients'])
    v['plan_id']=digest(v);return v
def compare(rows,reference):return paired_compare(rows,reference,arms=('CARE',))
