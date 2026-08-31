# Fundus Model-Fisher EWC comparison, V2

Registration ID: `FUNDUS_MODEL_FISHER_EWC_V2`. Registered on 2026-09-01
(Asia/Shanghai), before any V2 real-data read or real-model forward. The base
commit is `91c9dd672182f6dcaa120f1f030c36583f2eaf63` and the branch is
`codex/fundus-model-fisher-ewc-v2`. This registration commit must precede the
V2 execution-source commit; both must be published and byte-verified before
V2 computation.

## Reason for V2 and immutable V1 evidence

V1 is closed as `FAIL_ENGINEERING_CLOSED`. Its single real admission used 18
training-path model calls and two optimizer updates, then rejected the valid
mapped CUDA device before Fisher estimation because unindexed `cuda` and live
`cuda:0` device objects were compared literally. V1 used zero Fisher model
calls and zero `autograd.grad` calls; it launched no formal run or test
evaluation. V1 is not retried or amended. Its execution source, failed child
exit, report, input audit, closure, and archive remain immutable.

V2 is a separately identified engineering correction. It changes only device
resolution at the Model-Fisher boundary:

- the live device is the first trainable parameter's exact device;
- every trainable parameter must share that exact live device;
- a requested device must have the same type as the live device;
- an omitted requested index is accepted as the runner's mapped default;
- a specified requested index must equal the live index; and
- the exact live device is passed to the already validated V1 estimator.

The V2 method identity is `model_fisher_ewc_v2`, version `2.0`. The tensor
state schema and Fisher mathematics are inherited unchanged, while the outer
method name/version prevents a V1 checkpoint from being restored as V2. V2
does not alter V1 source history or reinterpret its result.

## Fixed scientific comparison

The question, data, training, estimator, and success rule are identical to
V1. Fresh `sequential_ssl` and `model_fisher_ewc_v2` arms are paired within
seeds 0/1/2. Domain order is REFUGE -> RIM_ONE_r3 -> Drishti_GS. Both arms use
fresh initialization and input order; no V1, LwF, or historical checkpoint is
reused.

Training remains CE plus foreground soft Dice and current weak-to-strong SSL:
confidence 0.95, SSL coefficient 1, a 1,000-step per-domain ramp, Adam at
0.0005, weight decay 0.00001, FP32, labeled/unlabeled batches 2/4, and 200
epochs per domain. Every run receives 8,000 + 3,200 + 2,200 = 13,400 updates.
Six formal runs permit at most 80,400 optimizer updates. Validation is
descriptive only and cannot select a checkpoint or setting.

Model-Fisher settings remain `lambda=1.0` under the retained one-half
quadratic convention and `gamma=1.0`. Per stage, select at most 16 visible
training images and 16 output pixels per image without replacement, using the
optimization seed and registered site namespace. Use all three categorical
classes, detached model probabilities, individual scalar log-probability
gradient squares, and the exact actual selected-pixel denominator. Labels,
Dice, transforms, confidence masks, class balancing, prototypes, transport,
replay, and trace normalization do not enter Fisher. Consolidate after every
successful stage, including the terminal stage, before its final checkpoint.

The V1 synthetic coefficients 1.7/0.6 and closed LwF test values are not used
for V2 selection. The V2 real admission deliberately uses the same
deterministic seed0 REFUGE batch contract as V1 rather than selecting a new
batch after failure. No V1 loss value is exposed to or used by the protocol.

## Prospective computation budgets

Each training update uses three current-model calls. Validation of all three
domains after every stage uses 126 calls per run. A control run permits 40,326
training-plus-validation calls; a V2 run permits those plus at most 42 Fisher
image calls, or 40,368. Six formal runs permit at most 242,082 calls including
126 Fisher calls. V2 permits at most 2,016 Fisher `autograd.grad` calls per run
and 6,048 across three seeds. The single post-training test readout permits at
most 612 model calls and zero updates. Including the real admission, formal
runs, and test readout, V2 permits at most 242,731 real-model calls.

At most three formal workers run concurrently on GPU4/5/6. GPU7 is reserved
for the zero-real device preflight, the one real admission, and eventual test
readout. Each formal child has a 12-hour limit. At most two infrastructure-only
resumes per run may use a verified same-source, same-config
`checkpoint_last.pt`. Scientific or code failure is never an infrastructure
resume.

## Zero-real device preflight

After V2 registration and exact source publication, one create-only GPU7
preflight may instantiate the formal U-Net and method but may not read any
dataset, call the model, call `autograd.grad`, or take an optimizer step. It
must verify that unindexed `cuda` and explicit `cuda:0` reach the registered
empty-dataset boundary, while `cpu` and explicit `cuda:1` are rejected at the
device boundary. Parameters, buffers, gradients, modes, method state, and RNG
must remain exact. Limits are one invocation, 60 seconds, zero model calls,
zero gradient calls, and zero updates. Failure closes V2 before real access.

## One real engineering admission

Only after the zero-real preflight passes may V2 run one create-only seed0
REFUGE visible-training admission. It uses no validation/test role or hidden
label and the formal U-Net, FP32 settings, image cap, and point cap. For each
arm it performs two no-update golden calls and one shared-Trainer update on
the same labeled batch of two and unlabeled batch of four. V2 then performs
one full Fisher consolidation; checks exact 16-image, 256-point, 768-gradient
work, finite nonnegative nonzero state, and immutable model/optimizer/RNG
during estimation; applies the registered deterministic 0.001 positive-Fisher
perturbation; and performs one positive-penalty shared-Trainer update.

Limits are 37 real-model calls, 768 `autograd.grad` calls, three optimizer
updates, one invocation, and 900 seconds. Input bytes are rehashed afterward.
Any failure closes V2. There is no source, tolerance, data, or setting repair
under this registration.

## Formal evaluation and success

Formal training starts only after both preflights exit 0, source/config/input
hashes match, the real duration fits the 12-hour bound, and a sanitized
engineering-admission record is published. Training reads only train-labeled,
train-unlabeled, and validation roles. No test role opens until all six runs
exit 0 and pass source, budget, checkpoint, Fisher-state, and input-integrity
gates.

The one fixed test readout evaluates all stage-final checkpoints on every seen
test domain, producing 36/36 cells. Patient-mean hard foreground Dice is
unchanged: empty/empty class Dice is one; average classes 1/2 within patient,
then patients within domain. For each seed/arm, F is the final-row three-domain
mean, I is the diagonal mean, and BWT is the first-two-domain final-minus-
diagonal mean. Pair V2 minus control within seed.

All five conditions are required: mean delta F >= 0.01; positive delta F in
at least 2/3 seeds; every seed delta F >= -0.01; mean delta BWT >= 0.01; and
mean delta I >= -0.01. A valid pass is `PASS_EWC_FEASIBILITY`; a valid metric
failure is `FAIL_EWC_FEASIBILITY`; incomplete, corrupt, nonfinite, over-budget,
or mismatched evidence is `FAIL_ENGINEERING`. Test results cannot trigger a
retry, coefficient change, threshold change, added seed, or checkpoint choice.

## Storage and closure

The create-only NAS root is
`/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/fundus_model_fisher_ewc_v2_20260901`.
All generated files use NAS and the storage wrapper; there is no HOME fallback.
The actual NFS mount and a write/read probe must pass before writes. Frozen
HDF5, manifests, splits, checksums, existing environments, HOME compatibility
symlinks, and closed evidence are read-only.

Retain the exact execution checkout, configs, commands, parent-observed child
exits, counters, checkpoints, Fisher summaries, input hashes, and failures.
Publish sanitized source and aggregate reports only on this branch. Before
closure, run a separate zero-model artifact/arithmetic audit and create an
additive verified NAS archive. A same-NAS archive is not an independent-device
backup.
