# Fundus Memory Aware Synapses comparison, V1

Registration ID: `FUNDUS_MAS_V1`. Registered on 2026-09-01
(Asia/Shanghai), before any MAS real-data read, real-model forward, or test-role
access. The base commit is `afcec424f2b7efd1902185c0b35228e2f8e7fd5b`
and the branch is `codex/fundus-mas-v1`. This registration commit must precede
the execution-source commit; both must be published and byte-verified before
computation.

## Authority and independent method selection

The user's long-running authorization permits finite, prospectively registered
Fundus comparisons on shared GPUs 4/5/6/7 with generated storage on NAS. The
prototype-derived method line and Model-Fisher EWC V1/V2 remain closed. This is
a new external published baseline, not a rescue or amendment of a closed
study. No C0 regeneration, Gate2, Prostate, MnMS, replay, prototype, relation,
transport, parameter search, paid resource, or main merge is introduced.

MAS was selected from its ECCV 2018 paper and first-author implementation. The
selection and every method setting below are independent of prior test values.
No result from Model-Fisher EWC V2, LwF, or another closed test readout was used
to choose MAS, its coefficient, importance data, estimator, or budget.

## Frozen published method identity

The method is global Memory Aware Synapses (MAS), Aljundi et al., ECCV 2018.
For each current-site visible training image `x`, let `z(x; theta)` be the full
raw three-class U-Net logit tensor. The per-parameter importance contribution
is

`abs(d sum(z(x; theta)^2) / d theta_j)`.

The site importance is the arithmetic mean over every current-site
`train_labeled` image in manifest order, evaluated one image at a time without
augmentation or label access. Counts are 40, 16, and 10 images for REFUGE,
RIM_ONE_r3, and Drishti_GS. Batch size one follows the paper's per-data-point
definition and the author's explicit online `b1=True` path. After each stage,
including the terminal stage, add the new importance elementwise to the prior
running importance and snapshot the current parameters.

At later stages add exactly

`lambda * sum_j running_omega_j * (theta_j - reference_j)^2`

to the unchanged sequential-SSL loss. `lambda=1`, with no one-half factor, is
the paper's reported global-MAS setting for object recognition; the paper says
it was not tuned. There is no decay, normalization, clipping, damping,
threshold, class weighting, pixel sampling, label loss, probability transform,
prototype, replay, or old teacher in MAS importance. All trainable U-Net
parameters, including the fixed three-class head, participate.

This is a literal dense-segmentation adaptation of the published output-
sensitivity principle, not a reproduction of the paper's AlexNet multi-head
classification architecture or datasets. The full logit tensor is treated as
the learned vector-valued output function; its fixed 256 by 256 spatial scale
is not divided out.

## Fixed scientific comparison

Fresh `sequential_ssl` and `global_mas_v1` arms are paired within seeds 0/1/2.
The domain order is REFUGE -> RIM_ONE_r3 -> Drishti_GS. Both arms use fresh
initialization and identical input order; no historical checkpoint or prior
control artifact is reused.

Training is the frozen weak/strong sequential SSL control: CE plus foreground
soft Dice, confidence 0.95, SSL coefficient 1, a 1,000-step per-domain ramp,
Adam at 0.0005, weight decay 0.00001, FP32, labeled/unlabeled batches 2/4, and
200 epochs per domain. Every run receives 8,000 + 3,200 + 2,200 = 13,400
updates. Six formal runs permit at most 80,400 optimizer updates. Validation is
descriptive only and cannot select a checkpoint or setting.

Training reads only `train_labeled`, `train_unlabeled`, and validation roles.
MAS importance calls `image_at` only on the current site's labeled training
dataset, so no label or transform is read. No test role opens until all six
runs exit zero and pass source, budget, checkpoint, importance-state, and input
integrity gates.

## Prospective computation budgets

Each training update uses three current-model calls. Validation of all three
domains after each stage uses 126 calls per run. A control run permits 40,326
training-plus-validation calls; a MAS run permits those plus exactly 66
importance calls, or 40,392. Six formal runs permit at most 242,154 real-model
calls, including exactly 198 MAS importance calls and 198 MAS importance
backward calls. The separate test readout permits at most 612 model calls and
zero updates. Including the one real admission described below, the total
study limit is 242,788 real-model calls.

At most three formal workers run concurrently on GPU4/5/6. GPU7 is reserved
for the zero-real preflight, real admission, and eventual test readout. Each
formal child has a 12-hour limit. At most two infrastructure-only resumes per
run may use a verified same-source, same-config `checkpoint_last.pt`.
Scientific or code failure is never an infrastructure resume.

## Preflight and real engineering admission

After registration and exact source publication, one create-only zero-real
GPU7 preflight may instantiate the formal U-Net and method and exercise
synthetic tensors only. It may not read a dataset, call a real-data model,
take an optimizer step, or access validation/test roles. It must prove exact
penalty arithmetic, per-image absolute-gradient averaging, cumulative state,
checkpoint round-trip, rejection of mismatched state, and restoration of
parameters, buffers, modes, gradients, optimizer state, and RNG. Limits are
one invocation, 300 seconds, and zero real-model calls.

Only after that preflight passes may one create-only seed0 REFUGE admission
read the frozen visible training inputs. It uses no validation/test role or
hidden label. For both arms it performs two no-update golden calls and one
shared-Trainer update on the same labeled batch of two and unlabeled batch of
four. MAS then computes importance from exactly one manifest-first image via
`image_at`, verifies one forward and one backward, finite nonnegative nonzero
state, exact sample denominator, and immutable parameters/buffers/optimizer/RNG;
applies a deterministic 0.001 perturbation at a positive-importance entry; and
performs one positive-penalty shared-Trainer update. Limits are 22 real-model
calls, one MAS importance backward, three optimizer updates, one invocation,
and 900 seconds. Input bytes are rehashed afterward. Any failure closes V1;
there is no repair or changed-setting retry under this registration.

## Formal evaluation and success

The one fixed test readout evaluates all stage-final checkpoints on every seen
test domain, producing 36/36 cells. Patient-mean hard foreground Dice is
unchanged: empty/empty class Dice is one; average classes 1/2 within patient,
then patients within domain. For each seed/arm, F is the final-row three-domain
mean, I is the diagonal mean, and BWT is the first-two-domain final-minus-
diagonal mean. Pair MAS minus control within seed.

All five independently inherited external-baseline comparability conditions
are required: mean delta F >= 0.01; positive delta F in at least 2/3 seeds;
every seed delta F >= -0.01; mean delta BWT >= 0.01; and mean delta I >= -0.01.
A valid pass is `PASS_MAS_FEASIBILITY`; a valid metric failure is
`FAIL_MAS_FEASIBILITY`; incomplete, corrupt, nonfinite, over-budget, or
mismatched evidence is `FAIL_ENGINEERING`. Test results cannot trigger a
retry, coefficient change, estimator change, threshold change, added seed, or
checkpoint choice. Three seeds support no significance or clinical claim.

## Storage and failure exit

The create-only NAS root is
`/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/fundus_mas_v1_20260901`. Every
generated file uses the NAS storage wrapper; there is no HOME fallback. The
actual NFS mount and a create/write/read probe must pass before the root is
created. Frozen HDF5, manifests, splits, checksums, environments, HOME
compatibility symlinks, and closed evidence remain read-only.

Any failed preflight closes before formal training. A valid scientific failure
closes after the sole test readout. On closure, retain exact execution source,
configs, commands, parent-observed child exits, counters, checkpoints,
importance summaries, hashes, and failures; publish only sanitized source and
aggregate reports on this branch; perform a separate zero-model artifact and
arithmetic audit; and create an additive verified NAS archive. A same-NAS
archive is not an independent-device backup.
