# LCR-Seg V0.2 preregistration acknowledgement

**Version:** V0.2 — Asymmetric Reliability Routing  
**Date:** 2026-08-20  
**Status:** frozen before V0.2 code changes or V0.2 training

## Fact sources and boundary

This acknowledgement records the user-supplied V0.2 prompt and experiment
plan before implementation.  It is read together with:

- `METHOD_SPEC_V0_1.md`;
- `IMPLEMENTATION_CONTRACT_V0_1.md`;
- `METHOD_ACCEPTANCE_TESTS_V0_1.md`;
- `reports/implementation/BASELINE_AND_V0_V3_COMPLETION.md`; and
- `LCRSEG_V0_2_ASYMMETRIC_ROUTING_EXPERIMENT_PLAN.md`.

The V0.2 prompt names the prior completion report under
`reports/experiment_status/`.  The checked workspace contains the unchanged
prior report at `reports/implementation/BASELINE_AND_V0_V3_COMPLETION.md`;
this acknowledgement records that path correction without copying, changing,
or reinterpreting the V0.1 artifact.

V0.2 is a new method/version and new run family only.  V0.1 mathematical
behavior, its checkpoint schema, its golden baseline, its completed run
artifacts, and its historical conclusion remain frozen.  This workspace is
not a Git worktree; changed-files manifests and diffs will be reported rather
than a fabricated branch, commit, or push.

## Fixed data, model, and budget

- Formal data root: `/home/jiangsuiyang/SSL_CL`.
- Frozen read-only inputs: `h5/v1`, `manifests`, `splits`, and
  `checksums` beneath that root.  They must not be written, regenerated, or
  replaced.
- Fundus protocol: seed 0; sites `REFUGE -> RIM_ONE_r3 -> Drishti_GS`;
  frozen manifest SHA-256
  `0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3`;
  frozen split SHA-256
  `f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88`.
- Model: unchanged 2D U-Net plus the existing projection head, one semantic
  anchor per class (`K=1`).
- Training: unchanged weak/strong augmentation and geometry alignment,
  optimizer, scheduler, visible-label fraction, 200 epochs per site, and
  exactly 13,400 optimizer steps.  The final checkpoint and validation role
  remain the primary comparison rule.
- Training may use only the current site's `train_labeled` visible labels
  and label-free `train_unlabeled` batches.  Hidden labels, diagnostics,
  validation, and test data must not influence training or calibration.

## Permitted and prohibited changes

Permitted V0.2 work is limited to a new `lcrseg_v0_2` method/configuration,
class-wise progressive learnability admission, a labeled-only compatibility
calibrator, rejection-only relation-KD weighting, and their diagnostics,
logging, regression tests, and gates.  The shared trainer and existing
relation field, anchor bank, and pseudo-label generator must be reused.

The following remain prohibited: changing `lcrseg_v0_1` behavior; K>1 or
multi-agent methods; RIC, EWC, MAS, replay, diffusion, VAE, an EMA/third
teacher, channel splitting, contrastive/triplet/new auxiliary losses; changes
to data, splits, backbone, augmentations, optimizer, scheduler, epochs, or
budget; hidden-GT leakage; unregistered hyperparameter sweeps; and full
Prostate or M&Ms runs before the Fundus gate.

## Pre-registered V0.2 variants

All variants preserve the V0.1 loss coefficients and use valid pseudo-labels
with unit assimilation weight unless admission is enabled.

| Variant | Learnability assimilation | Compatibility consolidation |
| --- | --- | --- |
| R0 | No admission; all valid pseudo-label pixels | Uniform relation KD |
| R1 | Per predicted class, retain top learnability fraction `0.4 + 0.4*rho` | Uniform relation KD |
| R2 | No admission; all valid pseudo-label pixels | Labeled-only calibrated rejection-only KD |
| R3 | Same class-wise progressive admission as R1 | Same calibrated rejection-only KD as R2 |

The fixed parameters are: `pi_start=0.4`, `pi_end=0.8`,
`calibration_bins=10`, `calibration_min_pixels=500`,
`calibration_update_epochs=10`, calibrated probability threshold `0.7`,
per-old-predicted-class rejection cap `0.2`, and rejected weight `0.5`.
Before the end of epoch 9 and whenever a calibrator is unavailable,
consolidation is uniform.  Calibrators are buffer/state only, never
parameters or optimizer state.

## Mandatory gates and sequence

1. Complete the V0.1 class-wise, region-wise, ESS, and golden-batch gradient
   diagnosis in a separate post-hoc process, then update `STATUS.md`.
2. Implement V0.2 and pass all existing tests, V0.1 golden regression with
   zero recorded errors, V0.2 unit/resume tests, and an independently verified
   R3 golden batch before a full run.
3. Run Fundus only in the fixed order R0, R1, R2, R3.  New runs use their own
   V0.2 directories and must not overwrite any historical run.
4. Evaluate the automated R3 Fundus gate: Final >= 0.6551, BWT > -0.1185,
   Incoming >= 0.7241, Previous >= 0.6709; the specified admission,
   calibration, rejection-cap, ESS, no-leakage, R1/R2, and R3 mechanism gates
   must all pass as well.
5. On any Fundus-gate failure, record
   `FUNDUS_V0_2_RESEARCH_GATE_NOT_MET`, retain all artifacts, stop, and do
   not add a new method or run Prostate.

Only if every Fundus gate passes may the conditional Prostate A->B pilot run:
`RUNMC -> BMC`, seed 0, 20% labeled, 200 epochs/site, comparing
Sequential-SSL, V0.2 R0, and V0.2 R3.  Its gate requires R3 retained A Dice
and BWT above R0, B incoming no more than 0.01 below R0, no numerical/leakage
failure, and calibration/rejection behavior directionally consistent with
Fundus.  No A->E or M&Ms run is authorized by this acknowledgement.

## Current pre-registration state

No V0.2 model code, configuration, training, or outcome has been created or
claimed at the time this acknowledgement is written.  The next required
milestone is the prescribed V0.1 post-hoc routing diagnostic.
