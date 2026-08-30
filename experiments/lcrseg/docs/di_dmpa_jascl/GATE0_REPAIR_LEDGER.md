# Gate 0 repaired baseline ledger

Date: 2026-08-29
Target: LCRSeg fixed-class domain-incremental benchmarks
Upstream repository: `https://github.com/prinshul/JASCL`
Pinned upstream commit: `3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53`

## Boundary

2026-08-30 amendment: v1 is archived and its overall PASS withdrawn. The
current runner is semantic v2; the revised user protocol explicitly authorizes
G0-R11 through G0-R14 below. No DI-DMPA method is implemented.

The official JASCL files under
`Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT` remain unchanged.
Within `SSL_CL_seg`, that pinned repository is an ignored provenance dependency
at `experiments/lcrseg/third_party/JASCL_REFERENCE`; it is never vendored into
this source-only backup. The runner verifies the official origin, exact commit,
source path, and clean tracked diff before constructing the classifier.
Every repair is implemented in the separate `di_dmpa_jascl` package and
`scripts/run_gate0_repaired.py`. `method_registered` is `false`. No DI-DMPA
optimizer training, multi-prototype, transport, soft fusion, history gate,
multi-prototype loss, prototype inference, or full sweep is authorized here.

The upstream constant 3x3 prototype-patch term at
`train_step2.py:397-424` is not run. The revised protocol reserves that term
for a later, separately named ablation and requires
`use_constant_patch_classifier_regularization=false` during Gate 0. This is a
protocol exclusion, not a claim that a constant patch is equivalent to a
linear or 1x1 prototype replay operation.

## Architecture contract

Gate 0 uses the frozen LCRSeg medical `UNet2D` body: input-channel aware
encoder/decoder, channels 16/32/64/128, GroupNorm, three downsampling levels,
and skip-connected transposed-convolution decoding. Its ordinary 1x1
segmentation head is replaced by the official pinned JASCL 3x3 stochastic
`ProbabilisticClassifier`, which remains available as
`decoder.conv_logit` for GAS and complete stage-transition loading.

This is the explicit reconciliation of the two frozen requirements: use UNet
for LCRSeg medical segmentation and preserve the official JASCL 3x3
classifier. It does not modify the upstream source. Two prematurely launched
DeepLab runs (Fundus seeds 0 and 1, 120 steps each) were terminated, have no
`.complete` marker, and are excluded from every Gate 0 result.

The pinned classifier stores `mu`/`sigma` convolutions configured with
`padding=1`, but its released `forward` calls functional `conv2d` without a
padding argument. Gate 0 preserves that effective no-padding behavior and,
like the released DeepLab wrapper, interpolates logits back to the input
geometry. This upstream quirk is recorded rather than silently repaired.

## Repair ledger

| ID | Upstream defect / limitation | Upstream source | Gate 0 repair | Behavioral impact | Test or assertion |
|---|---|---|---|---|---|
| G0-R01 | The unlabeled loop calls `total_loss.backward()` but never calls `optimizer.step()`. | `train_step2.py:504-505`; the same pattern occurs in step 3/4. | `Gate0RepairedRunner._unsupervised_phase` performs exactly one optimizer step after the backward call. | The optimizer now executes the computed update. V1 still had zero consistency gradient; R11 supplies the separately authorized v2 semantic correction. | V2 actual-gradient tests in `test_pas_probability.py`, `test_resume_inside_unlabeled_phase`; per-step log field `optimizer_step_executed=true`. The old structural-only test was removed under R13. |
| G0-R02 | Teacher parameters remain trainable and are included when the wrapper is passed to an optimizer over `model.parameters()`. | `fix_seed.py:20-33`; `train_step2.py:359`. | Set all teacher parameters `requires_grad=false`; construct Adam from student parameters only; assert disjoint parameter identities. | Removes unused teacher gradients and prevents optimizer mutation. EMA remains the only parameter update path. | `test_teacher_is_frozen_excluded_and_no_grad`; runtime optimizer exclusion assertion. |
| G0-R03 | Teacher forward is not explicitly protected by `no_grad`. | `train_step2.py:479-481`. | Execute every teacher forward in `torch.no_grad()`. | No teacher autograd graph is created. Predictions and buffer mode are unchanged. | `test_teacher_is_frozen_excluded_and_no_grad`; log field `teacher_forward_no_grad=true`. |
| G0-R04 | Fixed-class stage transitions would inherit the upstream class-expansion logic that omits `conv_logit`. | `train_step2.py:913-920`; analogous code in step 3/4. | Load student and EMA teacher with `strict=true`; require the complete `decoder.conv_logit` key set before loading. | The 3x3 classifier and GAS state persist across every fixed-class domain transition. | `test_full_classifier_is_required`; stage completion field `classifier_load=strict_complete`. |
| G0-R05 | Checkpoints save only student, optimizer, epoch, and best score. | `train_step2.py:568-574`. | Save student, EMA teacher, optimizer, scheduler, stage/epoch/global step, GAS `grad_update`, Python/NumPy/CPU/CUDA RNG, sampler phase/offset, current PAS prototype, config hash, and evaluation matrices. | Interrupted runs can continue the same data order and state trajectory. | `test_checkpoint_roundtrip_restores_full_state_and_rng`. |
| G0-R06 | The released resume branch references `model` before the mean-teacher wrapper is constructed. | `train_step2.py:889-900`. | Construct the wrapper and optimizer first, then restore all checkpoint state through one validated loader. | Resume is runnable and rejects config drift or incomplete classifier state. | Legacy six-step smoke plus v2 `test_resume_mid_supervised`, `test_resume_inside_unlabeled_phase`, and both `test_resume_across_stage_boundary` cases. |
| G0-R07 | Separate step 2/3/4 scripts duplicate the training loop and encode stage state implicitly. | `train_step2.py`, `train_step3.py`, `train_step4.py`. | One config-driven state machine records stage index, domain, epoch, phase, next batch, and global step. | All stages share the same repaired behavior and resume semantics. | State-machine integration test and checkpoint schema checks. |
| G0-R08 | Evaluation concatenates seen domains and reports only an aggregate score. | `train_step2.py:639-713`; analogous step 3/4 evaluators. | After stage `t`, evaluate each seen domain separately with the test-role evaluator and write lower-triangular mean-IoU, mean-Dice, and foreground-Dice matrices. | Per-domain retention and current-domain adaptation remain identifiable. | Matrix schema checks; per-seed `stage_by_domain_*.csv/json`. |
| G0-R09 | Stage loaders are hard-coded to natural-image pickle lists and global paths. | `train_step2.py:284-314`. | Read only frozen LCRSeg training manifests, require the current domain, and construct role-scoped HDF5 datasets. | Fundus, Prostate, and MnMS remain independent benchmarks; no historical-domain training image enters a current-stage loader. | `test_current_domain_only_and_hidden_gt_isolation`; `leakage_preflight.json`. |
| G0-R10 | The upstream benchmark has no hidden-GT boundary compatible with LCRSeg manifests. | Upstream dataset and pickle loaders. | Unlabeled manifest records are sanitized to omit label paths; unlabeled dataset objects have no label read path; val/test roles are rejected by training APIs. | Hidden train-unlabeled GT and final test GT cannot enter training objects. | `test_training_cannot_request_val_or_test`; `test_unlabeled_batch_has_no_label_key`. |

## Deliberately preserved upstream behavior

The loss/evaluation exceptions in R11/R12 supersede v1 claims of preservation.

- Official 3x3 stochastic classifier and its GAS behavior, attached to the
  frozen LCRSeg UNet2D medical body.
- Adam with learning rate `1e-3` and weight decay `4e-5`.
- Polynomial learning-rate factor with power `0.9` and 100 epochs per stage.
- JASCL PAS confidence and cosine-similarity thresholds, both `0.7`.
- Prototype start epoch 25 and pseudo-label interval 5 from the official
  launcher.
- EMA coefficient `0.99`.
- Teacher evaluation mode and its resulting BatchNorm-buffer behavior. The
  repair does not introduce a train/eval or buffer-EMA policy change.
- Labeled resize/horizontal-flip/at-most-two-pixel-translation transform and
  the non-augmented unlabeled view. No weak/strong augmentation is added.
- GAS is estimated only from the supervised classifier gradient. No
  constant-patch regularizer contributes to `grad_update`.
- cuDNN deterministic mode and deterministic CUBLAS workspace are enabled.
  PyTorch 2.2.1 reports that CUDA `nll_loss2d` has no declared deterministic
  implementation, so that kernel uses `warn_only`; interrupted/resumed
  equality is therefore tested against the preregistered `atol=rtol=1e-6`
  tolerance rather than claimed as universally bitwise deterministic.

## Necessary fixed-class LCRSeg adaptation, not method innovation

- The medical segmentation backbone/decoder is LCRSeg UNet2D rather than
  DeepLab/Xception. Fundus uses three input channels; Prostate and MnMS use one.
- Class counts are fixed at Fundus `C=3`, Prostate `C=2`, and MnMS `C=4`.
- All mapped classes are valid segmentation classes; padding uses ignore label
  255 instead of treating the last semantic class as void.
- Natural-dataset-specific class weights are not transferred; the same
  cross-entropy objective is used with uniform weights.
- Input tensors, stored resize/crop, normalization, role policy, and domain
  order are read from `DOMAIN_PROTOCOL.yaml` and checked against frozen asset
  hashes.
- Final test GT is instantiated only by the evaluator and is never used for
  training, checkpoint selection, or threshold selection.

## V2 semantic and evidence repairs

| ID | Defect / source | Repair | Behavioral impact | Evidence |
|---|---|---|---|---|
| G0-R11 | v1 `runner._unsupervised_phase`: detached hard-index MSE | `compute_pas_validity` plus `masked_probability_consistency_loss` on joint valid pixels | Student probabilities retain gradients; teacher/prototypes/masks do not. Per-pixel squared L2 is not divided by C. C0 lambda=0 still executes all forwards/sampling. | formula, zero-mask, nonzero-gradient, lambda-zero, total-minus-supervised tests and real-batch audit |
| G0-R12 | official classifier defaults to sampling even under eval | required explicit `stochastic_classifier` argument; true for training/teacher PAS, false for validation/test | Posterior-mean checkpoint selection and matrix; no classifier RNG consumed by formal evaluation | exact-repeat/RNG test and 20-draw vs 2-repeat v1 checkpoint audit |
| G0-R13 | v1 structural step test and misleading TinySegNet parity label | actual `autograd.grad(L_cons, student)` audit; rename deterministic supervised smoke | Rejects zero unlabeled gradients and teacher/prototype leakage; method parity is NOT_APPLICABLE_METHOD_NOT_IMPLEMENTED | real three-domain PAS_GRADIENT_AUDIT plus unit/integration tests |
| G0-R14 | compiler hard-coded resume/tests PASS | consumes real JSON/JUnit/transcript/checkpoint evidence and exact source/config hashes | Missing, stale, zero-gradient or detached-objective evidence blocks PASS. Stage-boundary phase is checkpointed before/after best load. | fail-closed compiler tests and four resume trajectories |

The official all-zero GAS initialization can suffer float32 cancellation in
its inverse-gradient noise scale. No repair to that source is authorized here;
the positive stochastic-sampling unit test uses a nondegenerate GAS state.
