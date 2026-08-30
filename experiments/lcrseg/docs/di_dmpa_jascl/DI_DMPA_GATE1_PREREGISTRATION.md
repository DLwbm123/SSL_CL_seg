# DI-DMPA Gate 1 preregistration — v1.0.0

Registration ID: `DI_DMPA_GATE1_V1_B0_EMA_PRIMARY`

Date: 2026-08-30 (Asia/Shanghai)

Repository: DLwbm123/SSL_CL_seg

Branch: `codex/di-dmpa-gate1-diagnostics`

This is an **offline mechanism-diagnostic protocol, not a training config or
method registration**. The user-approved primary-panel interpretation is
frozen here before any mechanism result. This turn ends after a separate
preregistration commit is pushed. No diagnostic implementation, geometry,
transport fitting, reliability, gradient-conflict, theory computation, model
training or Gate 2 starts in this turn.

The companion [JSON](DI_DMPA_GATE1_PREREGISTRATION.json) is the machine-readable
normative specification. It contains all exact input hashes, complete
case lists, five bootstrap case draws for every seed/domain, and all 72
gradient batch records with checkpoint SHA and forward seeds. This Markdown
explains the same rules and gives a readable checkpoint/batch index.
A disagreement between the files is a protocol blocker, not permission to
choose whichever interpretation passes.

## 1. Publication order, frozen baseline and scope

The unchanged prework commit
`39532af4898bd1ae13c76033c686ed7479389ae8` was pushed first and verified by
`git ls-remote` before these files were created. The preregistration must be
its own subsequent commit. Main remains at
`46e892960240543c946c570a9378d409b226384b`; do not merge main.

Frozen identities:

| Item | Identity |
| --- | --- |
| Gate 0 v2 report commit | `ea945382030e8eb2be070fa3d2ee20e5128f791d` |
| Six formal runs' training source | `fb55e8022bc379e2515a46214c6fdf45ea818de6` |
| Official JASCL | `3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53` |
| BASELINE_FREEZE.json SHA-256 | `e171a1d476ca626830541e80dbb1dff763ae02716a09463bb70ea5892da8231a` |
| C0 resolved/canonical config SHA-256 | `074b9afeef0d7acbbda9a9e03c4bc479248a200ca1e4cf6e1b0eccc30116b000` |
| B0 resolved/canonical config SHA-256 | `37876e4a5dae85a31ff8d8a211e975e745cffe4c395f89407585bb9ede682b4c` |
| C0 raw YAML file SHA-256 | `ed59c24d782ef9b475b35211a40bfc07303daca3b7a1f55b04eedfd8c026c315` |
| B0 raw YAML file SHA-256 | `0ed4e7b1a3db0540f91f3a59acc6aa7e76ac09277038afd7b3037700487ca632` |
| DOMAIN_PROTOCOL.yaml SHA-256 | `3e2da51b2aeb92a80f59fdf96b4a96a129cc937851230c2f920ad706ceb90359` |

Gate 0 v2 is accepted as **correctness PASS only**, not B0 superiority or
DI-DMPA admission. B0 is probability MSE on joint PAS validity with
`lambda_u=0.5`; C0 uses the same raw consistency computation but
`lambda_u=0`. Do not rerun or change either baseline. All six run logs,
completion hashes, configs, reports, matrices and 24 checkpoints remain
frozen in BASELINE_FREEZE.json. Only the 18 stage-best checkpoints in
Appendix A are mechanism inputs; six final `last.pt` files are provenance.

The historical checkpoint `config_hash` hashes parsed YAML as canonical
JSON (sort_keys=true, compact comma/colon separators, ensure_ascii=false),
not the raw YAML file bytes. Both identities are frozen separately here;
the raw YAML bytes are also checked against the original training commit.

The pre-confirmation GATE1_PREWORK_REPORT.md and GATE1_STATUS.json at
39532af are preserved historical snapshots. This preregistration resolves
their pending primary-panel question; it does **not** declare Gate 1 PASS.
The current publication lifecycle is `NOT_RUN_PREREGISTRATION_ONLY`, not
a final Gate 1 verdict.

All following flags remain false, even after a future diagnostic PASS:

- `method_registered=false`
- `di_dmpa_training_launched=false`
- `use_multi_prototype=false`
- `use_domain_indexed_bank=false`
- `use_transport=false`
- `use_soft_proto_fusion=false`
- `use_history_gate=false`
- `use_multi_proto_loss=false`
- `use_proto_inference=false`
- `use_constant_patch_classifier_regularization=false`

Segmentation-model optimizer steps must be zero. No student, EMA, classifier,
GAS or buffer update is allowed. Future offline transport fitting updates
only its own W/b and reports a separate optimizer-step count. Computing
diagnostic clusters/transforms does not turn on method switches.
Constant-patch classifier regularization remains unimplemented, excluded
from core and GAS, and must never be called historical 3x3 neighborhood
replay.

No Prostate/MnMS run, frozen asset edit, old LCR-Seg contract edit, overwritten
Gate 0 result, DI-DMPA config/registration, pilot, full sweep or automatic
Gate 2 is authorized. Future method-performance comparisons must include
**both B0 and C0**; B0-EMA is only the mechanism-admission panel.

## 2. Binding every future run to this registration

Before any future run/shard/resume starts, its metadata must record:

- `preregistration_id`
- `preregistration_version`
- `preregistration_git_commit`
- `preregistration_json_sha256`
- `preregistration_md_sha256`
- `preregistration_remote_verified_commit`
- `diagnostic_code_git_commit`
- `baseline_freeze_sha256`
- `input_checkpoint_sha256`
- `manifest_sha256`
- `sampling_plan_sha256`
- `panel_id`
- `primary_admission_panel`
- `primary_feature_source`
- `feature_source_selection_performed`
- `model_optimizer_steps`
- `transport_optimizer_steps`
- `test_gt_usage`
- `hidden_gt_training_usage`
- `method_registered`
- `di_dmpa_training_launched`

`preregistration_git_commit` is the full immutable commit that first adds
these exact two files, not the later diagnostic-code HEAD.
`preregistration_remote_verified_commit` must confirm that same commit
was pushed and is an ancestor of the execution revision. Hash both files'
**raw UTF-8 bytes including final newline**, and put both SHA-256 values in
every run metadata and every report, alongside code/config/input/output
hashes. Missing fields, placeholders, unpushed commits or mismatched bytes
block execution.

A commit cannot contain its own final commit SHA, and a file cannot contain
its own hash. Therefore these files freeze the binding rule; the actual
commit and both hashes are resolved after commit/push and written into
**future run metadata before any diagnostic work**. There are no Gate 1
run metadata files to backfill in this publication-only turn. Do not rewrite
historical Gate 0 metadata or claim runtime enforcement was already tested.

Future output namespace:
`/root/LCRSeg/runs/di_dmpa_gate1/<preregistration_git_commit>/<unique_attempt_id>/`.
Never overwrite attempts. Any protocol revision requires a new version and
commit before execution, preserving old bytes, outcomes and failures.
Changing the primary path to C0 or student requires such a new version.

## 3. Fundus protocol and data roles

Only Fundus seeds 0/1/2 are in scope. Fundus, Prostate and MnMS are independent
benchmarks, not a concatenated sequence. Canonical Fundus order is read from
the hashed DOMAIN_PROTOCOL.yaml:

`REFUGE -> RIM_ONE_r3 -> Drishti_GS` (stage indices 0, 1, 2).

Fundus has C=3: background=0, optic-disc rim=1, cup=2. The stored labels are
already mapped (raw 255->0, 128->1, 0->2); ignore label is 255 in the
**stored** label space. Input is 3-channel RGB, raw center crop 800x800,
stored 384x384, normalized by division by 255. Diagnostic inputs use stored
geometry without extra resize, crop, flip or translation.

| Domain | train_labeled | train_unlabeled | val |
| --- | --- | --- | --- |
| REFUGE | 40 | 160 | 100 |
| RIM_ONE_r3 | 16 | 63 | 40 |
| Drishti_GS | 10 | 41 | 25 |

Counts hold separately for each seed, not shared case identity across seeds.
JSON `benchmark.case_plans` enumerates every allowed case/role and role-plan
hash; `manifest_assets` freezes three CSV and three Fundus split hashes.
Data root is /root/LCRSeg; image/label root is /root/LCRSeg/h5/v1.
Use manifest `primary_20pct_split` and `site_or_vendor`, never infer a
role from a filename containing “test”.

- Current train_labeled: fit current prototypes; fixed supervised gradient
  batches. Historical-stage prototypes become immutable outputs of those
  fits.
- Current train_unlabeled: image-only transport fit/holdout and gradient
  batches. Both label path and label hash must be empty; no hidden GT lookup.
- Val: diagnostic evaluator only, including historical current-space oracle.
  No model/transport fitting, threshold search, checkpoint selection or
  training API.
- Test: no test-role dataset construction and no image/GT reads in Gate 1.
  Existing Gate 0 matrix artifacts are not sources of Gate 1 GT.

Only frozen CSV metadata was read to construct this registration. No HDF5
labels, checkpoint tensors, features or mechanism measurements were read or
computed. Label-dependent pixel plans will be materialized later under
separate authorization using the frozen rule below.

## 4. Four panels, one admission panel

| Panel | Role | Independent foreground units | Can determine admission/K? |
| --- | --- | --- | --- |
| B0-EMA | Primary admission panel | 18 | Yes |
| B0-student | Feature-source control | 18 | No |
| C0-EMA | SSL-objective control | 18 | No |
| C0-student | Joint control | 18 | No |

Each panel contains exactly 3 seeds x 3 domains x 2 foreground classes.
All four must be completely computed and separately reported. **Every
Gate 1A admission threshold applies only to B0-EMA's 18 units.**
Do not pool 72 admission units, average/vote across panels, select the best
panel after running, rescue B0-EMA using a control, or use controls to choose
K or feature source.

`primary_feature_source=ema_teacher`

`feature_source_selection_performed=false`

Controls share the exact cases, pixels, sample multiplicities, case/class
balancing, clustering seeds, bootstrap draws, K grid and metric definitions.
A control's poor performance does not change primary admission. Any panel's
provenance, leakage, numerical or sampling-protocol failure blocks **all**
of Gate 1. B0-EMA failure remains failure even if controls perform well.

## 5. Features, sampling, bootstrap and deterministic conventions

Use the medical LCRSeg UNet2D (16/32/64/128, GroupNorm), retaining the pinned
official stochastic **3x3 classifier**. Features are the 16-D
`decoder.dec1` output before `decoder.conv_logit`, at stored 384x384.
L2-normalize per pixel. Geometry and transport use eval/no_grad and
`stochastic_classifier=false` for all panels. Model forward is float32,
AMP off; clustering, fitting/metric reductions use float64. A nonfinite
feature or norm <=1e-12 blocks; never drop panel-specific coordinates.

The official classifier import/constructor reseeds RNG to 1024. Construct
and load first, then explicitly set the registered per-forward seeds.
Preserve this upstream fact; do not silently “repair” old baseline
initialization. No segmentation state may change in these diagnostics.

Define H(parts) as SHA-256 of compact UTF-8 JSON (separators comma/colon,
ensure_ascii=false, array order preserved). Define S(parts) as the first
8 hex digits interpreted as an integer, bitwise-AND 0x7fffffff.
For role-plan object hashes preserve key order
train_labeled, train_unlabeled, val.

Geometry sampling is model-independent:

1. Enumerate non-ignore pixels in every case/class within a seed/domain/role.
2. Set common quota to min(2048, minimum positive case-class pixel count).
3. Per nonempty case/class take that quota in ascending
   H(['geometry-pixel-v1', pixel_sampling_seed, role, case_id, class_id, y, x]),
   breaking ties by pixel UID. Pixel seed=20262830+100*seed+stage_index.
4. Fit classes independently, equal case weight within class; summaries
   weight classes equally. Record zero-count case/classes and actual counts.
   Missing a required foreground unit blocks sampling; do not shrink 18.
5. Materialize one coordinate/multiplicity plan before any panel extraction,
   store its IDs and SHA-256, then reuse and compare hashes for all panels.

Five training-case bootstraps are fixed in JSON, with replacement and the
same exact draws for every panel/K/class. Keep each case's original sampled
pixels/quota and use multiplicity weights. Draw seed =
S(['bootstrap-v1',seed,stage_index,replicate]); draw j indexes the sorted
training cases by integer H(['bootstrap-case-v1',bootstrap_seed,j]) modulo N.
Hungarian-match each bootstrap's prototypes to its original-fit prototypes.
No rerolling seeds or bootstrap samples based on outcomes.

## 6. Gate 1A — multi-prototype geometry

Fit on current-domain train_labeled; held-out geometry is on current-domain
val. Occupancy uses fit samples, stability uses the five bootstrap fits.
K=1 is reference; candidates are **2, 3, 5**, all tested independently.

Class-wise, case-weighted spherical K-means uses five restarts, CPU float64,
maximum 100 iterations, center angular-movement tolerance 1e-6.
Initialize with weighted spherical K-means++: first sample follows case/pixel
weights, later samples follow weight times nearest cosine distance (not
squared again). Use numpy Generator(PCG64) with
S(['kmeans-v1',seed,stage_index,class_id,K,replicate,restart]),
replicate=-1 for original fits and 0..4 for bootstrap; restart=0..4.
**Panel ID never enters the seed.**

Assign by maximal cosine (lowest cluster index on ties); update to normalized
weighted means. Choose the restart with lowest training Q, lowest restart
index on ties. Empty centers are reseeded at largest weighted quantization
residual, pixel-UID tie-break, without reusing a center; if distinct samples
are exhausted retain explicit inactive slots, never silently lower K.
K1 is the normalized weighted mean. Report degeneracy; only A1..A6 decide
candidate performance. Nonfinite calculations are global numerical blockers.
Unmatched/inactive bootstrap slots receive stability zero, not exclusion.

Metrics, per panel/class/domain/seed/K:

- Q_K: equal-case mean nearest-prototype cosine distance.
- Cosine-distance p95 and unit-sphere Euclidean
  R95 = p95(sqrt(max(0,2-2*maximum cosine))).
- Quantiles use the weighted ECDF: smallest value at cumulative weight >=q,
  with value/pixel-UID ordering and equal total mass per case.
- Foreground macro: equal rim/cup mean; background always separate.
- Original-fit cluster occupancy: case-weighted assignment mass divided by
  total class mass; active means positive assignment mass.
- Minimum within-class inter-prototype Euclidean/angular separation; K1
  has structurally null pairwise separation.
- Hungarian maximum-cosine matched bootstrap stability.
- Boundary/interior: per-class dilation minus erosion using a square 7x7
  element (Chebyshev radius 3), constant-false exterior, at stored
  resolution; ignore pixels excluded. Report sampled own-class pixels
  inside/outside the band. This never modifies training labels.

All conditions below must hold for a candidate K on **B0-EMA only**:

| ID | Frozen condition |
| --- | --- |
| A1 | number of B0-EMA foreground units with R95_K < R95_1 >= 12 (denominator 18) |
| A2 | median over 9 seed-domain pairs of (mean_fg(R95_1)-mean_fg(R95_K))/mean_fg(R95_1) >= 0.1 |
| A3 | fraction of active clusters with occupancy >= 0.05, pooled only across the 18 B0-EMA foreground units for this K >= 0.9 |
| A4 | foreground bootstrap/Hungarian matched cosine median >= 0.85 |
| A5 | number of domains with strictly lower mean-over-seeds foreground macro R95 than K1 >= 2 (denominator 3) |
| A6 | all preceding admission statistics use foreground only; background-only improvement cannot pass |

For A2, compute foreground macro R95 for each of the nine seed/domain pairs,
take its relative reduction from K1, then the median across nine pairs.
For A3, foreground active clusters are pooled only within this primary
panel and K. A4's median covers foreground units, bootstrap replicates and
matched slots. For A5, average foreground macro R95 equally over three seeds
within each domain. No background term enters admission.

Select the **smallest passing K**. If none passes, Gate1A =
`FAIL_MULTI_MODALITY_NOT_SUPPORTED`; downstream diagnostics may use K1
with explicit fallback labels. Multi-prototype main contribution/naming is
then forbidden. Control success cannot change this.

Outputs: `PROTOTYPE_GEOMETRY_DIAGNOSTIC.json`, `PROTOTYPE_GEOMETRY_DIAGNOSTIC.md`, `prototype_quantization.csv`, `prototype_occupancy.csv`, `prototype_stability.csv`, `prototype_boundary_interior.csv`.

## 7. Gate 1B — fixed B0 EMA transport

Primary source/target is **B0 previous-stage best EMA -> B0 current-stage
best EMA**, with the B0-EMA-selected K (or labeled K1 fallback).
Transitions: REFUGE->RIM_ONE_r3 and RIM_ONE_r3->Drishti_GS, each seed.

Use only current-domain train_unlabeled images. Hash-sort cases with
H(['transport-split-v1',split_seed,case_id]), split_seed=20261830+100*seed+stage.
Fit=floor(0.8*N), holdout=remaining: 50/13 RIM and 32/9 Drishti each seed.
Exact case lists and split hashes are in JSON. No historical image or val GT
may enter transport fit.

Per case select 2048 stored spatial positions by ascending
H(['transport-pixel-v1',seed,stage_index,case_id,y,x]), coordinate tie-break.
Old/current encoders and T0/T1/T2 use the same image and exact coordinates.
Record ordered coordinate IDs and their hash before fitting; labels are
never needed for these coordinates.

- T0: identity.
- T1: orthogonal Procrustes. For column vectors, M=mean(y*x^T), U,S,Vh=svd(M),
  R=U*Vh, T(x)=R*x. Reflection allowed, no bias, zero optimizer steps.
- T2: normalize(x+W*x+b), D=16, W/b zero initialized. Full-fit-set
  case-equal cosine error + 1e-4*(sum(W^2)+sum(b^2)).
  Float64 Adam, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0,
  amsgrad=false, foreach=false; exactly 1000 steps.
  No scheduler, clipping, early stopping or hyperparameter search.

Model optimizer steps=0. Each T2 fit=1000 transport-only steps; six complete
fits=6000 transport steps. Counts must be separate.

Historical prototypes are immutable outputs from the original historical
stage's B0-EMA labeled fit. Immediate previous-domain angular error is
reported for each transition; the two-step REFUGE chain is a separate
comparison, not counted twice in the immediate-transition macro.

A current-space historical oracle is built only by the diagnostic evaluator
from historical val images/labels under the current B0 EMA. Use the frozen
historical-domain validation sampling plan, same K and original domain-index
clustering seeds; Hungarian-match centers by angular distance.
`gt_consumer=diagnostic_evaluator_only`.
Oracles never enter fitted transport, pseudo-targets, threshold/checkpoint
selection, training, or the operational prototype bank.

Report T0/T1/T2 held-out paired cosine error, historical prototype angular
error and nearest-max-cosine prototype-only accuracy, all class-wise and
foreground macro; W Frobenius norm, spectral norm/condition number of I+W,
all singular values; and T_stage2(T_stage1(p_REFUGE)) versus the original
identity source, against stage2's oracle. Keep source prototypes immutable.

| ID | Frozen T2 admission condition |
| --- | --- |
| B1 | each transition mean-over-three-seeds heldout cosine error relative reduction versus T0 >= 0.15 |
| B2 | each transition seed count with strictly lower heldout error versus T0 >= 2 (denominator 3) |
| B3 | relative reduction of foreground macro prototype angular error, equally averaged over 2 transitions x 3 seeds x 2 foreground classes, versus T0 >= 0.1 |
| B4 | relative prototype angular-error worsening for every seed/transition/foreground-class unit <= 0.05 |
| B5 | historical prototype-only accuracy drop from T0 in every reported seed/source-domain/target-stage unit (including chain) <= 0.005; absolute fraction (0.5 percentage points) |
| B6 | REFUGE chained foreground prototype angular error relative worsening versus direct identity, each seed and foreground class <= 0.05 |
| B7 | all parameters, fit losses, feature/gradient values and singular values finite |

T2 must meet all conditions. Failure =
`FAIL_TRANSPORT_NOT_SUPPORTED`; downstream uses identity only.
T1 is a diagnostic comparator, not a rescue path. No drift-calibrated naming,
automatic MLP, nonlinear projector or search after failure.

Outputs: `TRANSPORT_FEASIBILITY_DIAGNOSTIC.json`, `TRANSPORT_FEASIBILITY_DIAGNOSTIC.md`, `transport_feature_error.csv`, `transport_prototype_error.csv`, `transport_chain_error.csv`, `transport_spectrum.csv`.

## 8. Gate 1C — B0 EMA reliability and offline PoE

Fix B0 EMA probability, B0 EMA features, B0-EMA prototypes, and B0 EMA
transport. Student is the gradient receiver/conflict object; only the
**original R1 joint PAS** control still needs its historical student-validity
check. That is not student feature-source selection.

Use selected K (or labeled K1 fallback); learned history only when Gate1B
passes, otherwise identity history and R4 unavailable.
Primary teacher probability is a fixed stochastic draw 0, preserving B0's
stochastic target semantics. All candidates share it. Posterior-mean teacher
is a separate control, never a silent replacement.

For validation case forwards use
S(['val-teacher-v1',seed,stage_index,case_id,0]) and, for R1 student validity,
S(['val-student-v1',seed,stage_index,case_id,0]).
No seed depends on candidate or GT.

Frozen equations:

```text
y_hat = argmax_c p_teacher(c)             # lowest class ID on ties
q = max_c p_teacher(c)
a_cur(c) = logmeanexp_k(cos(z, p_current[c,k]) / 0.07)
m_cur = a_cur(y_hat) - max_{c != y_hat} a_cur(c)
r_cur = q * sigmoid(m_cur / 0.10)

a_hist(c) = logmeanexp over valid historical prototypes of class c
s_hist = max same-predicted-class historical cosine
g_hist = sigmoid((s_hist - 0.30) / 0.10)
m_hist = a_hist(y_hat) - max_{c != y_hat} a_hist(c)
r_hist = r_cur * ((1-g_hist) + g_hist*sigmoid(m_hist / 0.10))

p_current = softmax(a_cur)
p_history = softmax(a_hist)
p_fused ∝ p_teacher * p_current^0.5 * p_history^(0.25*g_hist)
```

Logmeanexp subtracts log(valid prototype count) **within each class**.
Mask invalid/dormant prototypes. Missing current predicted-class prototype
or no competitor means weight zero and validity=false; no replacement
by fake zero-valued valid centers. Missing usable history means g_hist=0,
r_hist=r_cur and PoE history factor=1. Stage0 R3/R4, where available, reduce
to R2. Probabilities, features, masks, prototypes, reliability weights,
transport outputs and oracle targets are detached. Compute sigmoid/PoE
stably; no temperature/threshold search.

Candidates:

| Candidate | Frozen definition |
| --- | --- |
| R0 | teacher confidence q |
| R1 | original joint hard PAS validity: frozen checkpoint prototypes key 'prototypes', confidence >0.7 AND prototype similarity >0.7 separately for student and teacher, then joint AND. Do not substitute the new EMA multi-prototype bank into R1. |
| R2 | current B0-EMA r_cur |
| R3 | current + identity-history r_hist |
| R4 | current + admitted T2-history r_hist; unavailable if Gate1B fails |

R1 uses the frozen checkpoint's `prototypes`, strict confidence >0.7 and
similarity >0.7 in each original student/teacher validity calculation, then
AND. Do **not** swap the new EMA prototypes into this baseline control.

Validation-only metrics use all non-ignore pixels, equal case mass within
each stratum. Report overall, background, rim, cup, foreground macro,
accepted predicted/true class composition, AURC, ECE, Brier, and reliability
bin accuracy. Weight-only class strata use teacher-predicted class;
true-class recall/composition is additionally reported.

Precision coverage points are fixed to **5%, 10%, 20%, 30%, 40%, 50%**.
Rank by descending weight, GT-independent hash tie-break
H(['reliability-tie-v1',seed,stage_index,case_id,y,x]).
Only strictly positive weights are eligible; R1 support is exactly joint PAS.
Coverage is accepted case-balanced mass / full valid mass.

Compare each candidate and R1 on common positive-support range
[0,min(max_coverage_candidate,max_coverage_R1)]. AURC is the
right-continuous weighted risk step integral divided by this common upper
bound; also report full available-support AURC. Never extrapolate unsupported
coverage points. Matched-coverage precision is the equal mean difference at
the prelisted points supported by both in every required foreground unit;
publish the actual shared set, not the best point. Empty/unsupported primary
metrics cannot satisfy admission and must not be omitted or imputed.

The 18-unit macro is equally weighted. A unit improves if AURC decreases
strictly or shared-point mean precision increases strictly; publish both
deltas. Coverage guard uses the matched global operating point
min(0.50,max_global_coverage_R1,max_global_coverage_candidate) independently
for each seed/domain; each predicted foreground class must retain at least
0.8 times its R1 retained fraction. This point depends on scores/support,
not GT. ECE/reliability calibration use 15 fixed equal-width [0,1] bins;
empty bins are explicit. Brier sums squared error over all three classes.

| ID | Frozen reliability admission condition |
| --- | --- |
| C1 | foreground macro AURC relative reduction >=0.10 OR shared-point matched-coverage precision increase >=0.01 versus R1 |
| C2 | foreground units improving versus R1 >= 12 (denominator 18) |
| C3 | relative retained-coverage decrease for ANY foreground class at registered matched global operating point <= 0.2 |
| C4 | negative global gradient-cosine batch fraction relative reduction versus R1 over same 72 pairs >= 0.2 |
| C5 | global gradient cosine median increase over same 72 pairs versus R1 >= 0.05 |
| C6 | each domain median gradient cosine worsening versus R1 (24 pairs/domain) <= 0.05 |
| C7 | teacher/prototype/transport oracle receive no gradients |
| C8 | hidden-GT training usage=none and test GT usage=none |

Primary history-aware candidate is R4 if T2 admitted; otherwise R3 is an
identity fallback explanation, not evidence of learned transport.
Test pixel-normalized weight-only first. Class-balanced weight-only may be
proposed only if it independently meets every same condition when ordinary
normalization does not. All gradient admission deltas use original
**pixel-normalized R1** on the same 72 pairs; class-balanced R1 is an extra
reported control, never a replacement admission reference.
R0/R2/current-only or another favorable panel cannot rescue failed primary
history-aware reliability.

PoE is an independent offline target comparison with the same conditions,
fixed batches/support rules, and separately reported changed predictions
(stratified by its own predicted class). It cannot rescue a failed weight-only
core or become default automatically. If weight-only passes but PoE fails,
retain only a weight-only proposal. If only class balancing passes, disclose
both variants and propose class-balanced normalization. **No method is
registered this turn or automatically after Gate 1.**

Outputs: `RELIABILITY_DIAGNOSTIC.json`, `RELIABILITY_DIAGNOSTIC.md`, `reliability_precision_coverage.csv`, `reliability_classwise.csv`.

## 9. Fixed gradient-conflict and teacher-noise probes

There are **8 labeled/unlabeled pairs per seed/domain = 72 pairs** for B0,
batch size 2 on each side. Appendix B gives IDs and case lists; JSON rows
also contain full checkpoint SHA, explicit student forward seeds and eight
teacher draw seeds. No batch may be replaced after seeing its gradient.

For each role hash-order the cases by
H(['grad-order-v1',sampler_seed,role,cycle,case_id]), where sampler_seed =
20260830+100*seed+stage_index. Concatenate cycles until 16 IDs and form
consecutive pairs. Repeated labeled cases when a pool is exhausted are
predeclared, not a resampling repair.

Use stored RGB/255 with no probe augmentation. Load the unchanged complete
B0 student/EMA/classifier/GAS. Seed each forward, use common cached values:
stochastic unlabeled student, teacher draw0 under no_grad, stochastic labeled
student. Do not update GAS from diagnostic supervised gradients.

```text
L_sup = mean CE(student stochastic logits, labeled GT), ignore=255
loss_i = sum_c (p_student(i,c) - stopgrad(p_teacher(i,c)))^2
L_u(r) = sum_i r_i*loss_i / (sum_i r_i + 1e-12)
L_u_cb = mean over active predicted classes c:
         sum_{i:y_hat_i=c} r_i*loss_i / (sum_{i:y_hat_i=c} r_i + 1e-12)
g_sup = grad(L_sup); g_u = grad(L_u); g_u_cb = grad(L_u_cb)
norm_ratio = 0.5 * ||g_u|| / ||g_sup||
```

Active classes have positive weight sum. Zero weight returns graph-connected
`p_student.sum()*0.0`. Zero-norm cosine is undefined/null with a zero-gradient
flag, not fabricated 0; a required zero-norm comparison fails that candidate's
gradient condition, without dropping rows. Nonfinite gradients block all
Gate 1. Use autograd.grad only, no segmentation optimizer construction or
step. Parameters, buffers, classifier and GAS must match bitwise before/after.

Six complete active-parameter blocks:

| Block | Parameter prefixes |
| --- | --- |
| encoder | `enc1.`, `enc2.`, `enc3.` |
| bottleneck | `bottleneck.` |
| dec3 | `decoder.dec3.` |
| dec2 | `decoder.dec2.` |
| dec1 | `decoder.dec1.` |
| classifier.mu | `decoder.conv_logit.mu.` |

Inventory official sigma/grad_update and every other inactive registered
parameter explicitly, assert expected None gradients, and never silently
drop an active parameter from the partition.

Report global and block cosines, lambda-scaled norm ratios, negative-cosine
fraction, median/p10/p90, domain distributions, background/rim/cup component
norms and vector decomposition, current/history and pixel/class-balanced
comparisons. All 72 pairs have equal weight; each domain has 24 pairs.
Unweighted scalar quantiles use linear interpolation.

For every same pair, repeat **8 stochastic teacher classifier draws** using
the explicit registered seeds, keeping student forwards fixed and draws
common across candidates. Primary admission uses draw0, not a selected
or averaged favorable draw. Record target-probability variance and
gradient-cosine variance. Posterior-mean teacher is a separate control;
adopting it later requires an additional baseline, not changing formal B0.

Outputs: `GRADIENT_CONFLICT_DIAGNOSTIC.json`, `GRADIENT_CONFLICT_DIAGNOSTIC.md`, `gradient_alignment.csv`, `gradient_blockwise.csv`, `TEACHER_TARGET_STOCHASTICITY_DIAGNOSTIC.json`.

## 10. Empirical theory quantities

Use B0-EMA sources and the admitted K/transport, or explicitly labeled
K1/identity fallback, never a control-panel promotion.

- d(z,p)=sqrt(max(0,2-2*cosine)) on unit vectors.
- R_d,c,K: equal-case mean and weighted p95 nearest own-class radius.
- delta_d,c: p95 Hungarian-matched bootstrap-center Euclidean error;
  include mean/all replicates.
- E_d,t: p95 historical-prototype Euclidean error against current-space
  diagnostic oracle; class-wise and foreground macro. Current holdout paired
  feature error is additionally reported, not substituted.
- Delta_d: minimum inter-class mode Euclidean separation; additionally
  report foreground-only separation.
- S_d,c = Delta_d - 2*(R95_d,c,K + delta_d,c + E_d,t).
- Margin violation: fraction with nearest own-class pixel distance +
  delta + E >= Delta/2, case-equal.
- eta_c: reliability-weighted incorrect pseudo-label mass / reliability
  mass for predicted class c, using diagnostic val evaluator only.

Output `THEORY_QUANTITIES.json`. These are empirical assumption-direction
checks, **not a proved theorem**, and not an added admission gate.
Oracle quantities never feed fitting or training.

## 11. Fail-closed numerical rules and final state machine

Do not round before comparisons or loosen tolerances near a threshold.
Improvement is strict; only explicitly inclusive thresholds accept equality.
Relative reduction is (reference-candidate)/reference for reference >0.
If reference=0, equality yields zero improvement, not infinity; positive
candidate error cannot pass an improvement or bounded-relative-worsening
guard. In particular, if R1 already has zero negative-cosine fraction,
zero-to-zero does **not** demonstrate the required 20% reduction.

Defined structural nulls (K1 separation, empty boundary stratum, unsupported
coverage, zero-norm candidate failure, unavailable R4/stage0 history) are
explicit. Unexpected missing values cannot be silently dropped.

Any panel provenance, leakage or sampling failure =>
`BLOCKED_PROTOCOL_OR_LEAKAGE`. Otherwise any unexpected numerical failure
in any panel => `BLOCKED_NUMERICAL_FAILURE`. Preserve all errors if both.
All four panels must be complete before final admission; a primary PASS
cannot be published while a required control is missing.

Final decision order after valid complete diagnostics:

1. K>1 geometry + T2 transport + primary history-aware weight-only reliability
   all pass => `PASS_CORE_ADMISSION`.
2. Geometry fails but K1 T2/history-aware weight-only reliability pass =>
   `PASS_REDUCED_K1`. Gate1A itself remains
   `FAIL_MULTI_MODALITY_NOT_SUPPORTED`; no multi-prototype contribution/name.
3. Otherwise retain every component failure; overall first failing component
   is `FAIL_MULTI_MODALITY_NOT_SUPPORTED`, else
   `FAIL_TRANSPORT_NOT_SUPPORTED`, else
   `FAIL_RELIABILITY_NOT_SUPPORTED`.

Transport failure forbids drift-calibrated naming; reliability failure
forbids training. Controls never rescue primary. No K/source/threshold
switching after results. All final statuses retain method_registered=false
and di_dmpa_training_launched=false, even PASS. Next action is
`STOP_FOR_INDEPENDENT_REVIEW`, not a training launch.

GATE1_STATUS.json and GATE1_FINAL_REPORT.md must include all fields listed
in JSON `status_machine.required_report_fields`, including separate panel
statuses, the immutable registration identifiers, zero model optimizer
steps, separate transport counts, fixed EMA source and
feature_source_selection_performed=false.

## 12. Required future tests and evidence

- spherical K-means determinism
- class/case-balanced sampling
- empty cluster handling
- prototype normalization
- log-mean-exp prototype-count invariance
- bootstrap/Hungarian stability
- boundary/interior masks
- transport identity initialization
- transport old/current coordinate alignment
- model parameters unchanged
- transport-only optimizer isolation
- chain transport uses immutable source
- reliability bounded in [0,1]
- invalid prototype masking
- class-balanced loss
- zero-weight graph-connected zero
- gradient block partition completeness
- teacher/prototype no-grad
- no test-role construction
- diagnostic val GT cannot enter training API
- exact checkpoint SHA validation
- audit-on/off trajectory parity
- report compiler fail-closed behavior
- all four panels identical sampling hashes
- only B0-EMA threshold and K selection
- control failure cannot rescue primary
- any-panel protocol/numerical failure blocks whole Gate1
- required preregistration commit and both file hashes present in every run metadata

These are obligations for a later authorized implementation, **not tests
claimed to have run in this preregistration-only commit**. Historical audit
parity evidence remains under gate1_prework/; do not conflate its synthetic
fixture optimizer steps with frozen-baseline diagnostic steps.

The later delivery includes all listed reports, code/tests, exact commands,
pytest/JUnit, input/output hashes, branch/code/registration commits, and all
failures/warnings. Preserve unsupported-kernel and upstream warnings; no
silent rerun or altered tolerance.

Deterministic hashing, batch construction, clustering defaults, aggregation,
tie/support/zero rules and Adam defaults here are operational conventions
frozen **before results**, not claims about historical paper settings.
No measured mechanism outcome is used to set them.

## 13. Publication and metadata-only verification commands

Commands are run from the repository root. The first push must occur while
HEAD is still the unchanged prework commit; the later push publishes the
separate two-file preregistration commit.

```bash
git rev-parse HEAD
git push -u origin codex/di-dmpa-gate1-diagnostics
git ls-remote origin refs/heads/codex/di-dmpa-gate1-diagnostics refs/heads/main

python3 -m json.tool experiments/lcrseg/docs/di_dmpa_jascl/DI_DMPA_GATE1_PREREGISTRATION.json >/dev/null
git diff --check
git add experiments/lcrseg/docs/di_dmpa_jascl/DI_DMPA_GATE1_PREREGISTRATION.md experiments/lcrseg/docs/di_dmpa_jascl/DI_DMPA_GATE1_PREREGISTRATION.json
git commit -m "docs: preregister Gate 1 B0-EMA primary admission protocol"
git push origin codex/di-dmpa-gate1-diagnostics
git rev-parse HEAD
git ls-remote origin refs/heads/codex/di-dmpa-gate1-diagnostics refs/heads/main
shasum -a 256 experiments/lcrseg/docs/di_dmpa_jascl/DI_DMPA_GATE1_PREREGISTRATION.md experiments/lcrseg/docs/di_dmpa_jascl/DI_DMPA_GATE1_PREREGISTRATION.json
```

JSON/metadata consistency checks validate 18 checkpoint paths/hashes against
BASELINE_FREEZE; three manifest and split hashes; nine disjoint role plans;
72 fixed batch rows and memberships; six transport split plans; nine sets
of five exact bootstrap case resamples; false method flags and primary-only
selection. These checks import no model/data-loader code and read no HDF5.

Publication validation record (metadata only): PASS for 18 checkpoint
identities, six manifest/split hashes, nine disjoint role plans, 72 fixed
gradient pairs, six transport splits and 45 bootstrap draws. No model
imports or HDF5 reads occurred.

The first static validation stopped because it compared the historical
canonical-config hash with raw YAML bytes. This preregistration was corrected
**before publication** to record both hash domains; original configs and
BASELINE_FREEZE.json were unchanged. An optional local YAML reparse check
could not run because system Python lacks PyYAML; no packages were installed.
Canonical identities were checked against the accepted freeze, and raw YAML
bytes against the original training-source Git blobs. These are document
verification events, not mechanism outcomes or failed Gate 1 runs.

Exact reusable metadata-only validation (does not import Torch, model,
data-loader or YAML packages):

```bash
python3 - <<'PY'
import csv,hashlib,json,pathlib,subprocess
from collections import Counter
root=pathlib.Path('experiments/lcrseg')
doc=root/'docs/di_dmpa_jascl'
p=json.loads((doc/'DI_DMPA_GATE1_PREREGISTRATION.json').read_text())
freeze=json.loads((doc/'BASELINE_FREEZE.json').read_text())
H=lambda x:hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
S=lambda x:int(H(x)[:8],16)&0x7fffffff
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert p['immutable_baseline']['freeze_sha256']==sha(doc/'BASELINE_FREEZE.json')
assert p['benchmark']['domain_order_source']['sha256']==sha(doc/'DOMAIN_PROTOCOL.yaml')
subprocess.run(['git','merge-base','--is-ancestor',p['publication_scope']['prework_commit'],'HEAD'],check=True)
assert p['publication_scope']['current_turn_authorizes_diagnostics'] is False
assert p['publication_scope']['execution_state']=='NOT_RUN_PREREGISTRATION_ONLY'
assert all(v is False for v in p['method_flags'].values())
assert p['panels']['primary_admission_panel']=='B0-EMA'
assert p['panels']['primary_feature_source']=='ema_teacher'
assert p['panels']['feature_source_selection_performed'] is False
assert p['panels']['units_per_panel']==18
assert [v['panel_id'] for v in p['panels']['definitions'] if v['can_determine_admission']]==['B0-EMA']
assert p['gate1a']['K_candidates']==[2,3,5] and p['gate1a']['K_reference']==1
assert p['gate1a']['no_passing_K']['status']=='FAIL_MULTI_MODALITY_NOT_SUPPORTED'
assert p['gate1b']['primary_baseline']=='B0'
assert p['gate1c']['probability_source']==p['gate1c']['feature_source']=='ema_teacher'
assert p['gate1c']['prototype_source']=='B0-EMA'
assert p['gate1c']['evaluation']['coverage_points']==[.05,.10,.20,.30,.40,.50]
assert p['authorization_boundary']['test_gt_usage']==p['authorization_boundary']['hidden_gt_training_usage']=='none'
for baseline,cfg in p['immutable_baseline']['configs'].items():
 assert cfg['file_sha256']==sha(pathlib.Path(cfg['path']))
 assert cfg['resolved_config_sha256']==freeze['config_hashes'][baseline]
 original=subprocess.check_output(['git','show',p['immutable_baseline']['gate0_training_source_commit']+':'+cfg['path']])
 assert hashlib.sha256(original).hexdigest()==cfg['file_sha256']
checkpoints={c['checkpoint_id']:c for c in p['immutable_baseline']['checkpoint_inputs']}
assert len(checkpoints)==18
for c in checkpoints.values():
 assert c['sha256']==freeze['runs'][c['baseline']][str(c['seed'])]['checkpoints'][c['path']]
 assert c['domain']==p['benchmark']['domain_order'][c['stage_index']]
 assert c['path'].endswith(f"/stage_{c['stage_index']}_{c['domain']}/best.pt")
roleplans={}
for asset in p['benchmark']['manifest_assets']:
 seed=asset['seed']; path=pathlib.Path(asset['repository_path'])
 assert sha(path)==asset['sha256']
 assert sha(pathlib.Path(asset['fundus_split_path']))==asset['fundus_split_sha256']
 with path.open(newline='') as f:
  allowed=[r for r in csv.DictReader(f) if r['dataset']=='fundus' and r['primary_20pct_split'] in ['train_labeled','train_unlabeled','val']]
 for stage,domain in enumerate(p['benchmark']['domain_order']):
  plan=next(q for q in p['benchmark']['case_plans'] if q['seed']==seed and q['stage_index']==stage)
  pools={role:sorted(r['case_id'] for r in allowed if r['site_or_vendor']==domain and r['primary_20pct_split']==role) for role in ['train_labeled','train_unlabeled','val']}
  assert plan['roles']==pools and plan['case_plan_sha256']==H(pools)
  assert plan['counts']=={k:len(v) for k,v in pools.items()}
  assert all(len(v)==len(set(v)) for v in pools.values())
  assert len(set().union(*[set(v) for v in pools.values()]))==sum(map(len,pools.values()))
  assert all(not r['label_h5_relpath'] and not r['label_sha256'] for r in allowed if r['site_or_vendor']==domain and r['primary_20pct_split']=='train_unlabeled')
  roleplans[(seed,stage)]=plan
assert len(roleplans)==9
pairs=p['gradient_diagnostic']['batch_pairs']
assert len(pairs)==len({q['batch_id'] for q in pairs})==72
assert Counter((q['seed'],q['stage_index']) for q in pairs)==Counter({k:8 for k in roleplans})
for q in pairs:
 seed,stage=q['seed'],q['stage_index']; plan=roleplans[(seed,stage)]
 assert q['domain']==plan['domain'] and q['sampler_seed']==20260830+100*seed+stage
 assert checkpoints[q['checkpoint_id']]['baseline']=='B0'
 assert q['checkpoint_sha256']==checkpoints[q['checkpoint_id']]['sha256']
 for role,key in [('train_labeled','labeled_case_ids'),('train_unlabeled','unlabeled_case_ids')]:
  pool=plan['roles'][role]; ids=[]; cycle=0
  while len(ids)<16:
   ids.extend(sorted(pool,key=lambda cid:(H(['grad-order-v1',q['sampler_seed'],role,cycle,cid]),cid)))
   cycle+=1
  assert q[key]==ids[q['pair_index']*2:q['pair_index']*2+2]
  assert len(set(q[key]))==2
 for name,seedvalue in q['forward_seeds'].items():
  assert seedvalue==S(['grad-forward-v1',q['batch_id'],name])
 assert q['teacher_draw_seeds']==[S(['teacher-draw-v1',q['batch_id'],i]) for i in range(8)]
splits=p['gate1b']['split_plans']; assert len(splits)==6
for q in splits:
 seed,stage=q['seed'],q['stage_index']
 pool=roleplans[(seed,stage)]['roles']['train_unlabeled']
 order=sorted(pool,key=lambda cid:(H(['transport-split-v1',q['split_seed'],cid]),cid))
 nfit=4*len(order)//5
 assert q['fit_case_ids']==order[:nfit] and q['holdout_case_ids']==order[nfit:]
 assert q['split_hash']==H([order[:nfit],order[nfit:]])
 assert not set(q['fit_case_ids'])&set(q['holdout_case_ids'])
 assert q['source_checkpoint_id']==f'B0/seed{seed}/stage{stage-1}'
 assert q['target_checkpoint_id']==f'B0/seed{seed}/stage{stage}'
assert len(p['shared_sampling']['plans'])==9
for plan in p['shared_sampling']['plans']:
 seed,stage=plan['seed'],plan['stage_index']
 pool=roleplans[(seed,stage)]['roles']['train_labeled']
 assert len(plan['bootstrap'])==5
 for b in plan['bootstrap']:
  bs=S(['bootstrap-v1',seed,stage,b['replicate']])
  draws=[pool[int(H(['bootstrap-case-v1',bs,j]),16)%len(pool)] for j in range(len(pool))]
  assert b['seed']==bs and b['case_ids_with_replacement']==draws and b['case_draw_sha256']==H(draws)
required=['preregistration_git_commit','preregistration_json_sha256','preregistration_md_sha256']
assert all(k in p['runtime_binding']['required_in_every_run_metadata'] for k in required)
assert p['gate1b']['models']['T2']['steps']==1000
assert p['gate1b']['optimizer_accounting']['model_optimizer_steps']==0
assert p['gradient_diagnostic']['model_optimizer_steps']==0
md=(doc/'DI_DMPA_GATE1_PREREGISTRATION.md').read_text()
assert p['preregistration_id'] in md
assert all(c['sha256'] in md for c in checkpoints.values())
assert all(q['batch_id'] in md and all(c in md for c in q['labeled_case_ids']+q['unlabeled_case_ids']) for q in pairs)
assert not any(line.strip().startswith(('TODO:', 'TBD:', 'PLACEHOLDER:')) for line in md.splitlines())
print(json.dumps({'status':'PASS_STATIC_METADATA_ONLY','checkpoints':18,'manifest_and_split_hashes':6,'disjoint_seed_domain_role_plans':9,'fixed_gradient_pairs':72,'transport_split_plans':6,'bootstrap_draw_plans':45,'primary_panel':'B0-EMA','diagnostics_launched':False,'model_imports':0,'HDF5_reads':0},indent=2))
PY
```

After remote SHA verification, **stop**. This registration does not authorize
starting any mechanism diagnostic in the current turn.

## Appendix A. Exact stage-best input checkpoints

Both student and ema_teacher are read from each checkpoint; legacy R1 PAS
prototypes use its `prototypes` key. JSON also freezes config/source identity.
Each hash below is the frozen prework hash, not a claim that model tensors
were loaded during this preregistration.

| Checkpoint ID | Absolute path | SHA-256 |
| --- | --- | --- |
| B0/seed0/stage0 | `/root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed0/stage_0_REFUGE/best.pt` | `2bd834c0e0bb4183bfa5cc9a0319c9b796b67e5f2b7e2d746a5f5942596b3a3c` |
| B0/seed0/stage1 | `/root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed0/stage_1_RIM_ONE_r3/best.pt` | `a079413ebd0534de964fed034011cc559b59a937470b504c9fbc5fbad4d00bb0` |
| B0/seed0/stage2 | `/root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed0/stage_2_Drishti_GS/best.pt` | `daa7129a8807dfe5880eca480107d3613f448d55a0079cede9e0282ac7d4d244` |
| B0/seed1/stage0 | `/root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed1/stage_0_REFUGE/best.pt` | `1e3c99ab3fe39de9755401a31779b5670c624064a73e772938ae57cbb2c3a1b8` |
| B0/seed1/stage1 | `/root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed1/stage_1_RIM_ONE_r3/best.pt` | `fd61eb6c8c6b1b4e13ce16f9b442572f7c951e03b9403925ad5d898011201b11` |
| B0/seed1/stage2 | `/root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed1/stage_2_Drishti_GS/best.pt` | `f8915002599e8fae7cee4fdff0bc5212607010326c255e2f62e25962faf577d7` |
| B0/seed2/stage0 | `/root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed2/stage_0_REFUGE/best.pt` | `244c87368f252a660bf0d1934bf0ccf512790dc698d04ede01196d14c34064ac` |
| B0/seed2/stage1 | `/root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed2/stage_1_RIM_ONE_r3/best.pt` | `d7ca2081913a99acb94aeb6794b8cedd55958c2fd6d016ec6fd17e6f063e1be5` |
| B0/seed2/stage2 | `/root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed2/stage_2_Drishti_GS/best.pt` | `c7936dcf201076c66e2069f54044f59f8adbeed5326daf86fc94e5e0057fcab2` |
| C0/seed0/stage0 | `/root/LCRSeg/runs/gate0_v2_lambda0_fundus_seed0/stage_0_REFUGE/best.pt` | `79b3dfcfe494245e61cccbd94d6b9b06481a5093d7ba928bfee76449e50b263e` |
| C0/seed0/stage1 | `/root/LCRSeg/runs/gate0_v2_lambda0_fundus_seed0/stage_1_RIM_ONE_r3/best.pt` | `fcf5cd3041ece5caf7203954abe497883cd1d8626cd747dfc17ee0e364d2f2b2` |
| C0/seed0/stage2 | `/root/LCRSeg/runs/gate0_v2_lambda0_fundus_seed0/stage_2_Drishti_GS/best.pt` | `9f39ddc0931c792a89fee8709d43e6179dbe077b6b255cbe572f76ecc14a21f4` |
| C0/seed1/stage0 | `/root/LCRSeg/runs/gate0_v2_lambda0_fundus_seed1/stage_0_REFUGE/best.pt` | `117a5982085488fe9bcc9b11f9f4592b929a9ba129a33f2afb0b39daaf142900` |
| C0/seed1/stage1 | `/root/LCRSeg/runs/gate0_v2_lambda0_fundus_seed1/stage_1_RIM_ONE_r3/best.pt` | `63561c0e9e913fc96baee6afd6dfd7cdf64a07e21b0f75abccf7e84e22cdb37c` |
| C0/seed1/stage2 | `/root/LCRSeg/runs/gate0_v2_lambda0_fundus_seed1/stage_2_Drishti_GS/best.pt` | `0532926be6724f99b89f8da969da8acfd007e6cd870266a5ba94f4cca30f6cf9` |
| C0/seed2/stage0 | `/root/LCRSeg/runs/gate0_v2_lambda0_fundus_seed2/stage_0_REFUGE/best.pt` | `c2c29697aeb4f27258184db019aa99d3bcf1f888c56f0bfd65ff340f01ce348a` |
| C0/seed2/stage1 | `/root/LCRSeg/runs/gate0_v2_lambda0_fundus_seed2/stage_1_RIM_ONE_r3/best.pt` | `1de3299d9bf0d2b95da264e1af2a8970c2e10a769f2b9f1c8a81ecda5b7d0857` |
| C0/seed2/stage2 | `/root/LCRSeg/runs/gate0_v2_lambda0_fundus_seed2/stage_2_Drishti_GS/best.pt` | `b6135dd905b1ec4649685f5c7eeee975e371561bb817b01f567a55fa28a456f7` |

## Appendix B. Fixed gradient batch case IDs

Rows reference Appendix A for the full checkpoint path/SHA; JSON repeats the
checkpoint SHA in **every batch row** and records explicit student/teacher
forward seeds. These are B0 gradient probes, not 72 pooled geometry units.

| Batch ID | Sampler seed | Labeled case IDs | Unlabeled case IDs | Checkpoint ID |
| --- | --- | --- | --- | --- |
| B0/seed0/stage0/REFUGE/pair00 | 20260830 | REFUGE_train_g0039, REFUGE_train_n0007 | REFUGE_train_n0188, REFUGE_test_n0144 | B0/seed0/stage0 |
| B0/seed0/stage0/REFUGE/pair01 | 20260830 | REFUGE_train_n0086, REFUGE_train_g0028 | REFUGE_test_n0312, REFUGE_train_n0343 | B0/seed0/stage0 |
| B0/seed0/stage0/REFUGE/pair02 | 20260830 | REFUGE_train_n0202, REFUGE_train_n0296 | REFUGE_train_n0088, REFUGE_train_n0335 | B0/seed0/stage0 |
| B0/seed0/stage0/REFUGE/pair03 | 20260830 | REFUGE_test_n0101, REFUGE_train_n0323 | REFUGE_train_n0040, REFUGE_test_g0031 | B0/seed0/stage0 |
| B0/seed0/stage0/REFUGE/pair04 | 20260830 | REFUGE_train_n0153, REFUGE_train_n0161 | REFUGE_train_g0009, REFUGE_train_n0234 | B0/seed0/stage0 |
| B0/seed0/stage0/REFUGE/pair05 | 20260830 | REFUGE_train_n0285, REFUGE_train_n0190 | REFUGE_test_n0199, REFUGE_train_n0315 | B0/seed0/stage0 |
| B0/seed0/stage0/REFUGE/pair06 | 20260830 | REFUGE_train_n0302, REFUGE_train_n0206 | REFUGE_train_n0194, REFUGE_train_n0071 | B0/seed0/stage0 |
| B0/seed0/stage0/REFUGE/pair07 | 20260830 | REFUGE_train_n0235, REFUGE_train_n0020 | REFUGE_train_n0078, REFUGE_test_n0222 | B0/seed0/stage0 |
| B0/seed0/stage1/RIM_ONE_r3/pair00 | 20260831 | RIM_ONE_r3_test_S-33-R, RIM_ONE_r3_train_S-17-L | RIM_ONE_r3_train_G-32-L, RIM_ONE_r3_train_N-74-L | B0/seed0/stage1 |
| B0/seed0/stage1/RIM_ONE_r3/pair01 | 20260831 | RIM_ONE_r3_train_N-81-L, RIM_ONE_r3_test_N-47-L | RIM_ONE_r3_train_N-66-R, RIM_ONE_r3_train_N-72-R | B0/seed0/stage1 |
| B0/seed0/stage1/RIM_ONE_r3/pair02 | 20260831 | RIM_ONE_r3_train_N-62-R, RIM_ONE_r3_test_N-42-R | RIM_ONE_r3_train_N-26-R, RIM_ONE_r3_test_N-38-R | B0/seed0/stage1 |
| B0/seed0/stage1/RIM_ONE_r3/pair03 | 20260831 | RIM_ONE_r3_test_S-8-R, RIM_ONE_r3_test_N-20-R | RIM_ONE_r3_test_N-34-R, RIM_ONE_r3_test_N-3-L | B0/seed0/stage1 |
| B0/seed0/stage1/RIM_ONE_r3/pair04 | 20260831 | RIM_ONE_r3_train_G-33-R, RIM_ONE_r3_test_S-4-L | RIM_ONE_r3_train_G-3-R, RIM_ONE_r3_train_N-69-L | B0/seed0/stage1 |
| B0/seed0/stage1/RIM_ONE_r3/pair05 | 20260831 | RIM_ONE_r3_train_S-16-R, RIM_ONE_r3_test_G-23-R | RIM_ONE_r3_test_N-28-R, RIM_ONE_r3_train_N-82-R | B0/seed0/stage1 |
| B0/seed0/stage1/RIM_ONE_r3/pair06 | 20260831 | RIM_ONE_r3_train_N-10-R, RIM_ONE_r3_train_N-75-R | RIM_ONE_r3_test_S-30-L, RIM_ONE_r3_train_S-1-L | B0/seed0/stage1 |
| B0/seed0/stage1/RIM_ONE_r3/pair07 | 20260831 | RIM_ONE_r3_train_N-12-R, RIM_ONE_r3_test_N-30-R | RIM_ONE_r3_train_N-70-R, RIM_ONE_r3_test_N-36-R | B0/seed0/stage1 |
| B0/seed0/stage2/Drishti_GS/pair00 | 20260832 | Drishti_GS_train_gdrishtiGS_058, Drishti_GS_train_ndrishtiGS_046 | Drishti_GS_train_ndrishtiGS_092, Drishti_GS_test_gdrishtiGS_014 | B0/seed0/stage2 |
| B0/seed0/stage2/Drishti_GS/pair01 | 20260832 | Drishti_GS_train_ndrishtiGS_041, Drishti_GS_train_gdrishtiGS_040 | Drishti_GS_test_gdrishtiGS_073, Drishti_GS_train_gdrishtiGS_004 | B0/seed0/stage2 |
| B0/seed0/stage2/Drishti_GS/pair02 | 20260832 | Drishti_GS_train_ndrishtiGS_094, Drishti_GS_test_gdrishtiGS_074 | Drishti_GS_test_gdrishtiGS_006, Drishti_GS_test_gdrishtiGS_011 | B0/seed0/stage2 |
| B0/seed0/stage2/Drishti_GS/pair03 | 20260832 | Drishti_GS_test_gdrishtiGS_056, Drishti_GS_train_gdrishtiGS_060 | Drishti_GS_test_gdrishtiGS_027, Drishti_GS_train_gdrishtiGS_012 | B0/seed0/stage2 |
| B0/seed0/stage2/Drishti_GS/pair04 | 20260832 | Drishti_GS_test_gdrishtiGS_001, Drishti_GS_test_ndrishtiGS_100 | Drishti_GS_test_gdrishtiGS_020, Drishti_GS_train_gdrishtiGS_032 | B0/seed0/stage2 |
| B0/seed0/stage2/Drishti_GS/pair05 | 20260832 | Drishti_GS_train_ndrishtiGS_094, Drishti_GS_test_gdrishtiGS_074 | Drishti_GS_test_gdrishtiGS_071, Drishti_GS_test_ndrishtiGS_009 | B0/seed0/stage2 |
| B0/seed0/stage2/Drishti_GS/pair06 | 20260832 | Drishti_GS_test_gdrishtiGS_001, Drishti_GS_train_gdrishtiGS_058 | Drishti_GS_test_gdrishtiGS_053, Drishti_GS_test_ndrishtiGS_095 | B0/seed0/stage2 |
| B0/seed0/stage2/Drishti_GS/pair07 | 20260832 | Drishti_GS_test_ndrishtiGS_100, Drishti_GS_train_gdrishtiGS_060 | Drishti_GS_test_ndrishtiGS_099, Drishti_GS_test_ndrishtiGS_007 | B0/seed0/stage2 |
| B0/seed1/stage0/REFUGE/pair00 | 20260930 | REFUGE_train_n0309, REFUGE_train_n0002 | REFUGE_train_n0132, REFUGE_test_g0002 | B0/seed1/stage0 |
| B0/seed1/stage0/REFUGE/pair01 | 20260930 | REFUGE_test_n0050, REFUGE_train_n0105 | REFUGE_train_n0017, REFUGE_train_n0100 | B0/seed1/stage0 |
| B0/seed1/stage0/REFUGE/pair02 | 20260930 | REFUGE_test_n0200, REFUGE_test_n0128 | REFUGE_train_n0191, REFUGE_train_g0012 | B0/seed1/stage0 |
| B0/seed1/stage0/REFUGE/pair03 | 20260930 | REFUGE_train_n0330, REFUGE_train_g0013 | REFUGE_train_n0106, REFUGE_train_n0052 | B0/seed1/stage0 |
| B0/seed1/stage0/REFUGE/pair04 | 20260930 | REFUGE_train_n0143, REFUGE_train_n0320 | REFUGE_test_n0136, REFUGE_train_n0355 | B0/seed1/stage0 |
| B0/seed1/stage0/REFUGE/pair05 | 20260930 | REFUGE_train_n0126, REFUGE_train_n0219 | REFUGE_train_n0011, REFUGE_train_n0196 | B0/seed1/stage0 |
| B0/seed1/stage0/REFUGE/pair06 | 20260930 | REFUGE_train_n0168, REFUGE_test_n0064 | REFUGE_test_n0184, REFUGE_train_n0003 | B0/seed1/stage0 |
| B0/seed1/stage0/REFUGE/pair07 | 20260930 | REFUGE_train_n0167, REFUGE_test_n0291 | REFUGE_train_g0032, REFUGE_train_n0075 | B0/seed1/stage0 |
| B0/seed1/stage1/RIM_ONE_r3/pair00 | 20260931 | RIM_ONE_r3_test_G-27-R, RIM_ONE_r3_test_N-30-R | RIM_ONE_r3_test_N-2-R, RIM_ONE_r3_train_N-52-R | B0/seed1/stage1 |
| B0/seed1/stage1/RIM_ONE_r3/pair01 | 20260931 | RIM_ONE_r3_test_N-40-R, RIM_ONE_r3_train_S-11-L | RIM_ONE_r3_train_G-1-L, RIM_ONE_r3_train_N-13-L | B0/seed1/stage1 |
| B0/seed1/stage1/RIM_ONE_r3/pair02 | 20260931 | RIM_ONE_r3_train_N-7-L, RIM_ONE_r3_test_S-7-L | RIM_ONE_r3_train_S-14-R, RIM_ONE_r3_train_N-58-R | B0/seed1/stage1 |
| B0/seed1/stage1/RIM_ONE_r3/pair03 | 20260931 | RIM_ONE_r3_test_N-1-L, RIM_ONE_r3_test_S-29-R | RIM_ONE_r3_train_S-28-R, RIM_ONE_r3_train_N-64-R | B0/seed1/stage1 |
| B0/seed1/stage1/RIM_ONE_r3/pair04 | 20260931 | RIM_ONE_r3_train_S-15-L, RIM_ONE_r3_test_N-47-L | RIM_ONE_r3_train_G-4-L, RIM_ONE_r3_test_N-37-L | B0/seed1/stage1 |
| B0/seed1/stage1/RIM_ONE_r3/pair05 | 20260931 | RIM_ONE_r3_train_N-25-L, RIM_ONE_r3_train_N-10-R | RIM_ONE_r3_train_N-88-R, RIM_ONE_r3_train_G-32-L | B0/seed1/stage1 |
| B0/seed1/stage1/RIM_ONE_r3/pair06 | 20260931 | RIM_ONE_r3_test_N-50-R, RIM_ONE_r3_test_S-4-L | RIM_ONE_r3_test_N-49-L, RIM_ONE_r3_train_N-9-L | B0/seed1/stage1 |
| B0/seed1/stage1/RIM_ONE_r3/pair07 | 20260931 | RIM_ONE_r3_test_S-9-L, RIM_ONE_r3_train_N-56-R | RIM_ONE_r3_train_G-10-L, RIM_ONE_r3_train_G-33-R | B0/seed1/stage1 |
| B0/seed1/stage2/Drishti_GS/pair00 | 20260932 | Drishti_GS_train_ndrishtiGS_008, Drishti_GS_test_gdrishtiGS_082 | Drishti_GS_test_gdrishtiGS_054, Drishti_GS_test_gdrishtiGS_029 | B0/seed1/stage2 |
| B0/seed1/stage2/Drishti_GS/pair01 | 20260932 | Drishti_GS_test_gdrishtiGS_065, Drishti_GS_test_ndrishtiGS_013 | Drishti_GS_train_gdrishtiGS_012, Drishti_GS_train_gdrishtiGS_058 | B0/seed1/stage2 |
| B0/seed1/stage2/Drishti_GS/pair02 | 20260932 | Drishti_GS_test_gdrishtiGS_059, Drishti_GS_train_gdrishtiGS_040 | Drishti_GS_test_ndrishtiGS_096, Drishti_GS_train_gdrishtiGS_088 | B0/seed1/stage2 |
| B0/seed1/stage2/Drishti_GS/pair03 | 20260932 | Drishti_GS_test_gdrishtiGS_034, Drishti_GS_test_gdrishtiGS_030 | Drishti_GS_test_ndrishtiGS_072, Drishti_GS_train_gdrishtiGS_032 | B0/seed1/stage2 |
| B0/seed1/stage2/Drishti_GS/pair04 | 20260932 | Drishti_GS_test_gdrishtiGS_027, Drishti_GS_train_gdrishtiGS_069 | Drishti_GS_test_gdrishtiGS_006, Drishti_GS_train_gdrishtiGS_022 | B0/seed1/stage2 |
| B0/seed1/stage2/Drishti_GS/pair05 | 20260932 | Drishti_GS_test_gdrishtiGS_082, Drishti_GS_test_gdrishtiGS_027 | Drishti_GS_train_ndrishtiGS_098, Drishti_GS_train_gdrishtiGS_063 | B0/seed1/stage2 |
| B0/seed1/stage2/Drishti_GS/pair06 | 20260932 | Drishti_GS_test_gdrishtiGS_030, Drishti_GS_test_gdrishtiGS_059 | Drishti_GS_train_ndrishtiGS_092, Drishti_GS_test_gdrishtiGS_055 | B0/seed1/stage2 |
| B0/seed1/stage2/Drishti_GS/pair07 | 20260932 | Drishti_GS_train_gdrishtiGS_069, Drishti_GS_test_gdrishtiGS_065 | Drishti_GS_test_gdrishtiGS_071, Drishti_GS_test_gdrishtiGS_067 | B0/seed1/stage2 |
| B0/seed2/stage0/REFUGE/pair00 | 20261030 | REFUGE_test_n0278, REFUGE_train_n0359 | REFUGE_train_n0319, REFUGE_train_n0096 | B0/seed2/stage0 |
| B0/seed2/stage0/REFUGE/pair01 | 20261030 | REFUGE_train_g0006, REFUGE_train_n0130 | REFUGE_train_n0084, REFUGE_train_n0246 | B0/seed2/stage0 |
| B0/seed2/stage0/REFUGE/pair02 | 20261030 | REFUGE_train_n0192, REFUGE_train_n0205 | REFUGE_train_n0060, REFUGE_test_n0313 | B0/seed2/stage0 |
| B0/seed2/stage0/REFUGE/pair03 | 20261030 | REFUGE_test_n0328, REFUGE_train_n0234 | REFUGE_test_n0250, REFUGE_train_n0231 | B0/seed2/stage0 |
| B0/seed2/stage0/REFUGE/pair04 | 20261030 | REFUGE_train_n0204, REFUGE_train_n0263 | REFUGE_train_n0142, REFUGE_train_n0324 | B0/seed2/stage0 |
| B0/seed2/stage0/REFUGE/pair05 | 20261030 | REFUGE_train_g0014, REFUGE_train_n0261 | REFUGE_train_n0119, REFUGE_train_n0215 | B0/seed2/stage0 |
| B0/seed2/stage0/REFUGE/pair06 | 20261030 | REFUGE_train_n0163, REFUGE_train_n0342 | REFUGE_train_n0156, REFUGE_train_n0089 | B0/seed2/stage0 |
| B0/seed2/stage0/REFUGE/pair07 | 20261030 | REFUGE_train_g0028, REFUGE_train_n0270 | REFUGE_train_n0224, REFUGE_train_n0068 | B0/seed2/stage0 |
| B0/seed2/stage1/RIM_ONE_r3/pair00 | 20261031 | RIM_ONE_r3_train_N-72-R, RIM_ONE_r3_train_G-13-R | RIM_ONE_r3_train_S-21-R, RIM_ONE_r3_train_G-21-R | B0/seed2/stage1 |
| B0/seed2/stage1/RIM_ONE_r3/pair01 | 20261031 | RIM_ONE_r3_train_N-59-L, RIM_ONE_r3_train_N-61-L | RIM_ONE_r3_train_G-36-R, RIM_ONE_r3_train_G-11-R | B0/seed2/stage1 |
| B0/seed2/stage1/RIM_ONE_r3/pair02 | 20261031 | RIM_ONE_r3_train_N-8-L, RIM_ONE_r3_train_N-74-L | RIM_ONE_r3_test_N-28-R, RIM_ONE_r3_train_G-17-L | B0/seed2/stage1 |
| B0/seed2/stage1/RIM_ONE_r3/pair03 | 20261031 | RIM_ONE_r3_train_N-92-R, RIM_ONE_r3_test_N-39-L | RIM_ONE_r3_test_S-31-L, RIM_ONE_r3_train_N-26-R | B0/seed2/stage1 |
| B0/seed2/stage1/RIM_ONE_r3/pair04 | 20261031 | RIM_ONE_r3_test_G-2-R, RIM_ONE_r3_train_G-15-L | RIM_ONE_r3_train_S-12-L, RIM_ONE_r3_test_S-2-L | B0/seed2/stage1 |
| B0/seed2/stage1/RIM_ONE_r3/pair05 | 20261031 | RIM_ONE_r3_train_G-39-L, RIM_ONE_r3_train_G-10-L | RIM_ONE_r3_test_N-47-L, RIM_ONE_r3_train_G-37-R | B0/seed2/stage1 |
| B0/seed2/stage1/RIM_ONE_r3/pair06 | 20261031 | RIM_ONE_r3_train_N-68-R, RIM_ONE_r3_test_N-35-L | RIM_ONE_r3_test_G-22-L, RIM_ONE_r3_train_N-56-R | B0/seed2/stage1 |
| B0/seed2/stage1/RIM_ONE_r3/pair07 | 20261031 | RIM_ONE_r3_train_S-17-L, RIM_ONE_r3_test_N-1-L | RIM_ONE_r3_train_N-85-L, RIM_ONE_r3_train_S-28-R | B0/seed2/stage1 |
| B0/seed2/stage2/Drishti_GS/pair00 | 20261032 | Drishti_GS_test_gdrishtiGS_074, Drishti_GS_test_ndrishtiGS_085 | Drishti_GS_train_gdrishtiGS_045, Drishti_GS_test_ndrishtiGS_097 | B0/seed2/stage2 |
| B0/seed2/stage2/Drishti_GS/pair01 | 20261032 | Drishti_GS_train_gdrishtiGS_004, Drishti_GS_train_ndrishtiGS_018 | Drishti_GS_test_gdrishtiGS_087, Drishti_GS_test_gdrishtiGS_077 | B0/seed2/stage2 |
| B0/seed2/stage2/Drishti_GS/pair02 | 20261032 | Drishti_GS_train_gdrishtiGS_076, Drishti_GS_train_ndrishtiGS_101 | Drishti_GS_test_gdrishtiGS_059, Drishti_GS_train_ndrishtiGS_090 | B0/seed2/stage2 |
| B0/seed2/stage2/Drishti_GS/pair03 | 20261032 | Drishti_GS_test_ndrishtiGS_009, Drishti_GS_train_gdrishtiGS_012 | Drishti_GS_test_gdrishtiGS_025, Drishti_GS_test_ndrishtiGS_091 | B0/seed2/stage2 |
| B0/seed2/stage2/Drishti_GS/pair04 | 20261032 | Drishti_GS_train_ndrishtiGS_098, Drishti_GS_train_gdrishtiGS_022 | Drishti_GS_test_ndrishtiGS_078, Drishti_GS_train_gdrishtiGS_058 | B0/seed2/stage2 |
| B0/seed2/stage2/Drishti_GS/pair05 | 20261032 | Drishti_GS_test_gdrishtiGS_074, Drishti_GS_train_gdrishtiGS_004 | Drishti_GS_train_gdrishtiGS_081, Drishti_GS_test_gdrishtiGS_054 | B0/seed2/stage2 |
| B0/seed2/stage2/Drishti_GS/pair06 | 20261032 | Drishti_GS_train_ndrishtiGS_101, Drishti_GS_test_ndrishtiGS_009 | Drishti_GS_train_gdrishtiGS_084, Drishti_GS_test_gdrishtiGS_001 | B0/seed2/stage2 |
| B0/seed2/stage2/Drishti_GS/pair07 | 20261032 | Drishti_GS_test_ndrishtiGS_085, Drishti_GS_train_gdrishtiGS_012 | Drishti_GS_test_gdrishtiGS_056, Drishti_GS_train_gdrishtiGS_068 | B0/seed2/stage2 |

The full shared geometry case lists, bootstrap multiplicities and transport
fit/holdout lists are in the companion JSON. No hidden or test-role label
array is part of any of these plans.
