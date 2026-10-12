from ..f5_module_pilots_v1.protocol import OPTIONS,STEPS,MANIFEST_SHA,SPLIT_SHA,read,write,metrics,compare as paired_compare
from ..f5_anchor_guard_v10.protocol import jobs as previous_jobs,plan as previous_plan
from ..five_frameworks_v1.gate import digest
ARMS=('F5','GUA')
CAPS=dict(formal_updates=21200,source_updates=0,cpu_optimizer_updates=96,synthetic_cuda_updates=32,real_smoke_updates=4,diagnostic_vjps=64,entry_reference_vjps=16960)
def jobs():
    return [{**j,'id':j['id'].replace('ANCHOR_GUARD','GUA'),'arm':'GUA'} for j in previous_jobs()]
def plan():
    j=jobs();v=previous_plan();v.pop('plan_id')
    v.update(study_id='F5_GUA_V16',authority='explicit user authorization for independent GUA/SARD/REDistill/CaRE-KD queue after V15 public delivery; common V13 base, bounded methods, hourly original F5/B2 gates',jobs=j,arms=ARMS,options=OPTIONS,caps=CAPS,execution_batches=[[j[0]['id']],[x['id'] for x in j[1:]]],equation='retain V13 raw-gradient halfspace and all losses; after Adam project actual descent displacement u=theta_before-theta_proposed against the same balanced entry gradient h in metric M=sqrt(vhat)+eps; keep Adam states; commit before constraints/scheduler/EMA',hypothesis='classwise raw-gradient guard improved164 but lost163 cup; test whether Adam proposal violates balanced reference despite V13 raw projection, then repair displacement without changing losses/coefficients',comparison='disabled full native archived V13 update/loss/state parity; V13 matched base, V15/V10 background, matched F5 and historical B0/B2',qualification_advancement='all engineering checks PASS and at least one unforced generated-native active update has proposal dot<0 while raw-gradient dot is nonnegative within 1e-6 normalized tolerance; no patient performance gate',limitations=['direction-only GUA adaptation; no momentum/variance alignment or rho search','same current-U entry reference may be wrong or fail to cover old domains','finite precision parameter copy is reported separately from projected mathematical displacement','not finite-step loss or effective BA-function guarantee; no extra formal VJP/forward','formal failure not retried; development patients reused'])
    v['plan_id']=digest(v);return v
def compare(rows,reference):return paired_compare(rows,reference,arms=('GUA',))
