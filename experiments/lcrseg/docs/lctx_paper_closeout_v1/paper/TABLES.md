# Tables and insertion templates

## Historical training recipes — separate cohort/space

Full-parameter old-seed development experiment only. Source: `ams_seq_transfer_v0_1/FINAL_REPORT.md` at 671d0e57.

| Arm | Final | Incoming | Old | Forget |
|---|---:|---:|---:|---:|
| T_CE | 0.549103 | 0.697653 | 0.400553 | 0.284266 |
| T_CED | 0.544644 | 0.686664 | 0.402625 | 0.282194 |
| T_LCTX | 0.600145 | 0.718077 | 0.482212 | 0.202606 |
| T_UCTX | 0.587655 | 0.725457 | 0.449854 | 0.234965 |
| T_AMS | 0.588958 | 0.719420 | 0.458495 | 0.226324 |

## Matched standard target labels — F_CONV, seeds 71–73

| Arm | Status | Final | Incoming | Old | Forget |
|---|---|---:|---:|---:|---:|
| C_CE | PENDING | — | — | — | — |
| C_CED | PENDING | — | — | — | — |
| C_FULLMIX | PENDING | — | — | — | — |
| C_LCTX | REUSED, e2ac3943 | 0.5957328953 | 0.6990033101 | 0.4924624805 | 0.1760217797 |

## Fixed-source low target annotation block

| Arm | Target labels | New tasks | Updates | Final / Incoming / Old / Forget |
|---|---|---:|---:|---|
| C_LCTX_LOW | RIM8, Drishti5 | 6 | 15,900 | PENDING |
| C_FULLMIX_LOW | RIM8, Drishti5 | 6 | 15,900 | PENDING |

## Five frozen contrasts

<!-- NEW_RESULTS_START -->
PENDING: fixed matrix has not completed.
<!-- NEW_RESULTS_END -->

Each contrast also requires two order means, three order-averaged seed differences, paired seed SD, physical-domain shared-draw patient CI, rim/cup effects and absolute forgetting. Budgets are never pooled into one effect.

## SIGN exploration and replication — never combined

| Analysis | Seeds | Role | Final SIGN−F_CONV | Conditional patient 95% interval |
|---|---|---|---:|---|
| Old saved-score analysis | 61–63 | Post-hoc control discovery | +0.00473157 | [-0.00104338,+0.01051551] |
| Frozen replication | 71–73 | New-seed comparison | -0.00018359 | [-0.00650457,+0.00600149] |

## Resource table

| Block | New source training | New target tasks | Formal updates | Extra physical work | Response VJP | Response forwards | Actual seconds / peak memory |
|---|---:|---:|---:|---|---:|---:|---|
| Historical SIGN replication, excluded from new budget | 6 | 12 | 47,700 | 15 relocation updates; 40 qualification updates separately | 25,440 | 38,160 | Existing RESOURCE_ACCOUNTING |
| New standard-label block | 0 | 18 | 47,700 planned | Report actual failures/recovery separately | 0 planned | 0 planned | PENDING |
| New low-target-label block | 0 | 12 | 31,800 planned | Report actual failures/recovery separately | 0 planned | 0 planned | PENDING |
| New qualification | 0 formal | 0 formal | Excluded | Cap 128 synthetic + 12 discarded real-L updates | 0 | 0 | PENDING |

Two full training models (student + current EMA); one final student for deployment. Report labeled sample reads, model/image forwards, valid supervision occurrences, Dice group counts, patient exposure min/max/mean, optimizer/backward/EMA operations and actual shared-GPU timings. Reused sources and LCTX are indexed with zero new cost.
