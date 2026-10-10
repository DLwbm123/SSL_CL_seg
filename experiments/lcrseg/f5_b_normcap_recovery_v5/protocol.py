"""A user-approved finite continuation; unchanged V5 science."""
from ..f5_b_normcap_v5 import protocol as original
from ..five_frameworks_v1.gate import digest
from ..f5_module_pilots_v1.protocol import read,write
OPTIONS=original.OPTIONS;STEPS=original.STEPS;ARMS=original.ARMS
CAPS={**original.CAPS,'formal_updates':21873,'synthetic_cuda_updates':4,'cpu_optimizer_updates':0,'real_smoke_updates':0}
PRIOR_COMMIT='3b099335a021572695e42b1b43306628ea991ec9'
RESUME={'B_NORMCAP_S163_O2':1396,'B_NORMCAP_S164_O1':1482,'B_NORMCAP_S164_O2':1395}
AUTHORITY='user 2026-10-10: 授权按此方案恢复; new formal12300, including673 checkpoint replay; generated CUDA qualification<=4, CPU0, realL0; preserve original costs and sealed trajectory'
jobs=original.jobs;compare=original.compare

def plan():
    v=original.plan();v.pop('plan_id')
    v.update(study_id='F5_B_NORMCAP_RECOVERY_V5',authority=AUTHORITY,caps=CAPS,
             execution_batches=[list(RESUME)],diagnostic_first='reuse previously completed S163/O1; resume remaining three without efficacy selection',
             inherited_formal_updates=9573,inherited_unsealed_physical_updates=4273,additional_formal_updates=12300,
             scientific_updates=21200,cumulative_formal_cap=170273,recovery_steps={j:1200 for j in RESUME},
             recovery_prior_calls=RESUME,prior_execution_commit=PRIOR_COMMIT,recovery_exception='single approved host-reboot continuation; no further formal retry',
             diagnostic_limit='retain inherited counts; missing pre-checkpoint gradient rows are NA, never rerun',
             qualification='four generated CUDA optimizer calls, native checkpoint semantic/identity validation without data reads')
    v['plan_id']=digest(v);return v
