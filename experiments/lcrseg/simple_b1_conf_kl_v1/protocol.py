from ..f5_module_pilots_v1.protocol import STEPS, MANIFEST_SHA, SPLIT_SHA, read, write, metrics, compare as paired_compare
from ..five_frameworks_v1.native_data import ORDERS
from ..five_frameworks_v1.gate import digest

B1 = 'B1_PARENT_CONF_KL'
ARMS = (B1,)
OPTIONS = dict(lr=.001, weight_decay=4e-5, parent_lr_multiplier=.5, lr_B_over_A=1.,
               feature_lr_multiplier=1., feature_weight_decay_multiplier=1.,
               warmup_fraction=.2, U_ramp_fraction=.2, lambda_U=1.,
               PAS_confidence=.6, PAS_cosine=.7)
CAPS = dict(formal_updates=21200, source_updates=0, cpu_optimizer_updates=96,
            synthetic_cuda_updates=32, real_smoke_updates=4, diagnostic_vjps=0)


def jobs():
    return [dict(id=f'SIMPLE_B1_S{seed}_O{order}', arm=B1, seed=seed, order=order,
                 stages=[dict(stage=stage, domain=ORDERS[order-1][stage],
                              steps=STEPS[ORDERS[order-1][stage]]) for stage in (1, 2)])
            for seed in (163, 164) for order in (1, 2)]


def plan():
    matrix = jobs()
    value = dict(study_id='SIMPLE_B1_CONF_KL_V1', parent='NATIVE_LR_SRC_A_3DOMAIN_V1',
                 authority='user 2026-10-12: 好的，你补一个更简单的基线',
                 seeds=(163, 164), orders=ORDERS, arms=ARMS, options=OPTIONS, caps=CAPS,
                 jobs=matrix, execution_batches=[[matrix[0]['id']], [j['id'] for j in matrix[1:]]],
                 equation='existing B1: current EMA soft KL, geometry & max(q)>0.6; no PAS prototype-similarity filter',
                 comparison='B2_C06 options unchanged except confidence-only eligibility; frozen historical B0/B2 and matched F5/V13/V17 aggregates',
                 evaluation='all 8 stages sealed and all 4 EXIT0 before 20 unified seen-domain evaluations',
                 train_access='current L image/GT and current U image/geometry; same source checkpoints and frozen splits; no old images',
                 retries=0, hyperparameter_search=False, automatic_followup=False,
                 qualification='built-in archived B1 full-state/loss/update parity; confidence-only golden mask/gradient; native EMA/checkpoint/deploy/own-prefix/failure ledger; current-L zero-U smoke; two-case KL fit',
                 success='complete and report the whole fixed baseline regardless of whether its score is lower or higher',
                 limitations=['simpler eligibility does not guarantee a weaker score',
                              'B1 still retains the common parent input projection and LCTX',
                              'shared engine retains prototype computation but discards its mask for B1; no speed claim',
                              'two seeds and repeatedly reused development patients; no independent generalization',
                              'B2 remains the strong historical reference; no replacement or revised original method gate'])
    value['plan_id'] = digest(value)
    return value


def compare(rows, reference):
    return paired_compare(rows, reference, arms=ARMS)
