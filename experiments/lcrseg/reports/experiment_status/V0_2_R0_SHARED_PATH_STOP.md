# LCR-Seg V0.2 R0 shared-path stop

**Date:** 2026-08-20  
**Status:** `hard_stop_pending_protocol_clarification`

## Decision

The literal preregistered Fundus seed-0 R0 completed successfully, but its
comparison with the stated legacy uniform-relation reference exceeds any
normal deterministic tolerance. Per the supplied prompt, this requires a stop
and investigation before R1, R2, or R3 may be compared.

This is **not** the `FUNDUS_V0_2_RESEARCH_GATE_NOT_MET` result: the complete
R0--R3 gate was not run, and must not be inferred from R0 alone.

## Completed R0 artifact

- Remote run: `/home/jiangsuiyang/SSL_CL/runs/fundus_seed0_lcrseg_v0_2_r0_uniform_full200e`
- Method: `lcrseg_v0_2`; seed `0`; validation role; 13,400 / 13,400 steps.
- Frozen manifest SHA-256:
  `0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3`
- Frozen split SHA-256:
  `f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88`
- The literal R0 configuration had progressive admission, compatibility
  calibration, and compatibility rejection disabled. Its branch report shows
  unit assimilation for every valid pseudo-label and uniform relation KD.

## Required reference comparison

| Metric | Legacy V0.1 uniform-relation run | Literal V0.2 R0 | R0 minus legacy |
|---|---:|---:|---:|
| Final average Dice | 0.6551054533 | 0.6309953259 | -0.0241101275 |
| BWT | -0.1184621104 | -0.1111381492 | +0.0073239612 |
| Incoming Dice | 0.7340801936 | 0.7050874253 | -0.0289927683 |
| Previous-site Dice | 0.6759458886 | 0.6687593930 | -0.0071864956 |

Both runs use the same seed, Fundus site order, validation role, frozen
manifest, frozen split, and 13,400-step budget. The size of the Final and
Incoming differences is far outside a deterministic rerun tolerance.

## Investigation finding

The legacy artifact named in the plan as the R0 reference is not semantically
identical to the literal V0.2 R0 definition:

- Legacy run:
  `/home/jiangsuiyang/SSL_CL/runs/fundus_seed0_lcrseg_uniform_relation_kd_full200e`,
  `lcrseg_v0_1`, `use_learnability=true`, and `use_compatibility=false`.
  Its assimilation path therefore retains continuous V0.1 learnability
  weighting.
- Literal V0.2 R0: `progressive_admission=false`, with all valid pseudo-labels
  admitted at unit weight, plus uniform relation KD.

Consequently, the R0 comparison exposes an inconsistency between the stated
``unit assimilation + uniform relation KD`` reference and the actual legacy
reference configuration. It cannot be treated as evidence that a supposedly
identical shared path passed deterministic equivalence, nor as a valid
method-performance comparison.

The supplied prompt explicitly states: if R0 differs from the old
uniform-relation run beyond normal deterministic tolerance, stop and
investigate rather than continue comparisons. That condition has occurred.

## Frozen supplemental diagnostics

Post-hoc diagnostics were run in a separate process after the R0 checkpoint;
they did not expose hidden ground truth to training. The local frozen output
is `reports/analysis/v0_2_r0/`, mirrored under the remote code root.

- `checkpoint_inventory.csv` SHA-256
  `70848a288de905f35b89e7fdd113e4a3173b20a857b819436273cd83202c72af`
- `branch_coverage.csv` SHA-256
  `39c5163960c6a4f6a9164853876159b7ea1fdf560774254527f02927ea09164c`
- `effective_sample_size.csv` SHA-256
  `9874b2609b05fd2312b6123190dad39989c33002ac87cfe3d09ee5e1a910754b`
- `gradient_diagnostics.csv` SHA-256
  `e1cf856dc9d484777d5ef81c48959c8c44f74f91ec6a39841717aeecba683697`
- `routing_analysis_summary.json` SHA-256
  `f3ec57e1e360b2692763583c02032b348df59e9d21d7d9c8f0888e5acf2bafbb`

## Work explicitly not performed

- Fundus R1, R2, and R3 were not launched.
- The aggregate V0.2 Fundus gate was not executed.
- The real R3 checkpoint-based golden artifact was not created; the prior
  synthetic create-and-independent-verify test remains only an engineering
  test, not a formal R3 result.
- The conditional Prostate A->B pilot and all later experiments were not
  launched.

## Required next decision

An explicit protocol amendment or direction is required to resolve the R0
reference-definition mismatch. Until then, the original plan remains
preserved unchanged and all downstream formal runs stay blocked.
