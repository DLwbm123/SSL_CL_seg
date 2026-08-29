# V0.4a freeze for SR-GAS V0.1

**Status:** `V0_4A_FROZEN_FOR_SRGAS`  
**Hypothesis ID:** `SRGAS_V0_1_H1_STABLE_RELATION_CONDITIONED_GAS`

## Branch declarations

- LCR-Seg routing/SRA branch = frozen
- V0.4a failure = research failure, not engineering failure
- SR-GAS = new optimization hypothesis
- RGM/GPM = not implemented
- PAS = not part of proposed

## Authoritative V0.4a outcome

`FUNDUS_V0_4A_INTERNAL_GATE_FAILED`. Engineering and mechanism gates passed;
the research gate failed. External baselines and Prostate were not run.

| Seed | Final | BWT | Incoming | Previous |
|---:|---:|---:|---:|---:|
| 0 | 0.656107 | -0.122683 | 0.737895 | 0.673729 |
| 1 | 0.647889 | -0.141904 | 0.742492 | 0.635473 |
| 2 | 0.650476 | -0.175526 | 0.767494 | 0.631010 |

Mean SRA-minus-R0 deltas were Final `-0.030569`, BWT `-0.040102`, Incoming
`-0.003835`, and Previous `-0.026327`. Only the registered incoming metric
threshold passed. Both foreground class-mean checks, the site-class safety
check, and both stability checks failed.

All three formal runs passed exact step count, finite numeric logging, zero AMP
skips, zero hidden-GT training use, zero old-model gradient, zero historical
anchor mutation, and complete checkpoint checks. All four registered mechanism
checks passed.

## Frozen evidence

The machine-readable companion records every run path, seedwise manifest/split
hash, complete-run checksum-manifest digest, run summary hash, train log hash,
and site checkpoint hash:

`reports/experiment_status/V0_4A_FREEZE_FOR_SRGAS.json`

The authoritative source report is:

`/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg/reports/experiment_status/V0_4A_FUNDUS_COMPLETION.json`

No V0.4a run may be overwritten or rerun under this branch. Routing and SRA
must not be tuned or reinterpreted as SR-GAS.
