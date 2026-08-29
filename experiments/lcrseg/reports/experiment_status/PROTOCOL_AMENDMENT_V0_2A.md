# LCR-Seg V0.2a protocol amendment

**Date:** 2026-08-27  
**Protocol:** `lcrseg_v0_2a`  
**Status before bridge:** `not_evaluated`

## Amendment decision

The literal V0.2 R0 completed all 13,400 optimizer steps, but it is not
semantically equivalent to the legacy uniform-relation reference. The former
uses unit weight for every valid pseudo-label, while the latter retains the
continuous V0.1 learnability weighting. The original aggregate V0.2 gate was
therefore not run.

V0.2a resolves the mismatch as follows:

- Formal R0 is the frozen legacy run
  `/home/jiangsuiyang/SSL_CL/runs/fundus_seed0_lcrseg_uniform_relation_kd_full200e`,
  with `legacy_continuous_v01 + uniform_relation` semantics.
- The completed literal R0 is reclassified, without deletion or overwrite, as
  auxiliary U0 at
  `/home/jiangsuiyang/SSL_CL/runs/fundus_seed0_lcrseg_v0_2_r0_uniform_full200e`,
  with `unit_all + uniform_relation` semantics.
- U0 is not a formal R0 and is not part of the formal 2x2 factorial design.
- The formal Fundus gate remains `not_evaluated` until the shared-path bridge,
  pilots, and R1-R3 full runs complete.

## Registered variants

| Variant | Assimilation | Consolidation |
|---|---|---|
| R0 | `legacy_continuous_v01` | `uniform_relation` |
| R1 | `progressive_admission` | `uniform_relation` |
| R2 | `legacy_continuous_v01` | `calibrated_teacher_rejection` |
| R3 | `progressive_admission` | `calibrated_teacher_rejection` |
| U0 auxiliary | `unit_all` | `uniform_relation` |

No R1-R3 run is authorized before both the golden bridge and the 500-step
paired bridge pass their preregistered tolerances.

## Analysis operationalization fixed before full-run results

- Admission coverage is aggregated by site, epoch, and predicted class, as
  required by the experiment plan. This preserves the mandated global-threshold
  fallback when a batch contains fewer than 32 pixels for a present class.
- A "large" foreground-class sacrifice is fixed as an absolute final mean
  class-Dice decrease greater than 0.01 versus formal R0.
- Hidden GT is available only to the independent post-hoc diagnostics process;
  the training runner and calibrator do not import or read it.
