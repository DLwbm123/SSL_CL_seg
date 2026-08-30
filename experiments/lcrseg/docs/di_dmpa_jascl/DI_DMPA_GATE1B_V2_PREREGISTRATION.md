# Gate 1B v2 preregistration — null-aware B0-EMA transport

Registration ID: `DI_DMPA_GATE1B_V2_NULL_AWARE_B0_EMA_TRANSPORT`. Date:2026-08-30, Asia/Shanghai. The companion JSON is the exact machine contract, including all inherited case lists and hashes. This is a Gate1B-only preregistration, not a training configuration or a new method registration.

## Publication and frozen inputs

Branch `codex/gate1b-v2-null-aware-transport` starts at Gate1A v2 report `9b2ffd04c7a8e9da73f08edb0760be3f269065d8`, not main. No main merge.

Gate1A v2 freeze was separately pushed and remotely verified at `58f19e968700bd7708ec00e44a11759b48ce756f`. `GATE1A_V2_FREEZE.json` SHA256 is `9208473833f68731c0dd0856696c7bb34047aebd106872977fa1ce9f7598de05`.

The freeze accepts `PASS_MULTI_MODALITY_SUPPORTED`, passing K=[2,3,5], selected K=2. It pins27 B0-EMA K2 train_labeled original-fit records /54 centers, every record converged with exactly two finite unit-norm active centers. Source-file hashes, original values, restart IDs, training roles and checkpoint identities are frozen. No operational source prototype may be refit. Gate1A must not be rerun; K5 iterations, selected K and feature source must not change.

Frozen Gate1A identities:

- Preregistration `eaae37bbaa7546679d9e6893023afbeeef0ab5c6`.
- Code `8ae5d7532f90aee5d53c0d966706ef64c18a19ac`.
- Report `9b2ffd04c7a8e9da73f08edb0760be3f269065d8`.
- Geometry sampling-plan SHA256 `96277c841165d8cd84a63d3d4a9301b27c50dcdd87d0b0d215c162d95c345e24`.
- Formal artifact-manifest SHA256 `a483757ccd6918d06df83cf91ba89c8641a90a3eb4a40fb8749eaf84d7ee617d`.

The original complete Gate1 preregistration is `cfb62554f1e6a2a36850547485b1857dc9a28a20`: JSON SHA256 `6f50bd9df404d987aa70e2035a5c3f3853aa59ce49d21ffface34172cf754cbf`, MD SHA256 `32acdc5c24bcc5763daa6cb3650fea91f46da7ae3845b1fd0615c781619fbf0a`. Its exact transitions, split plans/seeds, coordinate rule, optimizer, original solver settings, oracle isolation and numerical B1–B7 thresholds are inherited. The sole methodological change is null-plus-sphere support and its corresponding null-aware metrics.

This preregistration is independently committed, pushed and remotely verified. Execution authorization is published in a subsequent independent commit. Before both are verified: no new checkpoint tensor read, model forward, transport coordinate materialization, support census or transform fit. Then implement/synthetic-test, publish exact code and verify its remote SHA before real integration.

## Primary path and six datasets

Primary baseline=B0, feature source=EMA teacher, K=2, feature tap=`decoder.dec1` post-ReLU16-D. Models are frozen existing UNets; no segmentation optimizer.

| Transition | Source checkpoint | Target checkpoint | Current-domain images | Fit / holdout per seed |
| --- | --- | --- | --- | --- |
| 01 | stage0 REFUGE best B0 EMA | stage1 RIM_ONE_r3 best B0 EMA | RIM_ONE_r3 train_unlabeled | 50 /13 |
| 12 | stage1 RIM_ONE_r3 best B0 EMA | stage2 Drishti_GS best B0 EMA | Drishti_GS train_unlabeled | 32 /9 |

Seeds0/1/2 run independently: six maps. The companion JSON copies all six original split plans without editing. Split seed=`20261830+100*seed+stage_index`; the original fit/holdout case-list order is retained and verified against hash-ranked case order. Both partitions are current-domain train_unlabeled and disjoint. No hidden label path or label hash may be constructed for them.

Every case uses2048 stored-resolution coordinates. Rank all384×384 integer coordinates by `(H(['transport-pixel-v1',seed,stage_index,case_id,y,x]),y,x)` and take the first2048 without replacement. H is the inherited compact-JSON UTF8 SHA256, preserving list order. No label or feature is read to generate this plan. All encoders and T0/T1/T2 use exactly the same rows. Each case has equal total weight and each of its2048 coordinates equal within-case weight; no AA renormalization defines the full objective.

The read-only integration uses the first registered fit case of each seed0 transition (listed in JSON), with all2048 coordinates computed in memory after exact-code publication. It records counts and coordinate hashes but writes no full shared plan. The full312-case transport plan is materialized only after integration PASS. Actual null counts are reported even if0 for these predetermined cases; integration cases are never replaced to find a null or a favorable result.

## Null-aware paired support

For each raw source/target vector, active iff float64 L2 norm>1e-12. Normalize active rows only. Preserve null rows with zero placeholders, false masks, original UIDs and weights. No epsilon denominator, fake direction, dropping, substitution, neighborhood/class-mean replacement or resampling.

`T_bar(NULL)=NULL`, including biased T2. Pair states are AA=0, A_NULL=1, NULL_A=2, NULL_NULL=3.

| Pair state | Full-support cosine error |
| --- | --- |
| AA | `1-clip(dot(T(u_source),u_target),-1,1)` |
| A_NULL | 2 |
| NULL_A | 2 |
| NULL_NULL | 0 |

All original rows enter the full-support weighted mean. AA-conditional error is descriptive only. Check the identity `full_error=AA_mass*AA_conditional+2*(A_NULL_mass+NULL_A_mass)` within float64 roundoff, without relaxing admission thresholds. Case-level conditional values with AA mass0 are explicitly undefined (null/defined=false), not deleted or reported as zero error.

If ANY seed-transition fit OR holdout whole-unit AA mass=0, return `FAIL_DIRECTIONAL_PAIR_SUPPORT_NOT_SUPPORTED` after the complete census, and stop before fitting any T0/T1/T2. Finite null is not a numerical block. Full-map or registered NaN/Inf is `BLOCKED_NONFINITE_FEATURE`.

## Extraction, caches and hard barrier

Source and target use the same image tensor, batch8, model.eval(), torch.no_grad(), AMP off, stochastic_classifier=false, float32 forward, float64 norms/statistics, TF32 off and deterministic algorithms. The original stored image normalization is float32 image/255. Re-seed after model construction using the inherited split seed; oracle extraction uses the inherited source-domain pixel seed.

Each pair cache retains N rows of source/target directions `[N,16]` float64, active masks `[N]` bool, raw norms `[N]` float64, and pair_state `[N]` uint8. Record ordered coordinate UID hash, original float64 little-endian weight hash, array shape/dtype/SHA and input/checkpoint identities.

First finish all12 paired units:3 seeds×2 transitions×(fit+holdout). Record case/transition/seed support counts and masses, source/target null mass, positive norm minima, full-map and registered nonfinite counts, maximum case mismatch fraction and coordinate/cache hashes. No T0/T1/T2 fit precedes the barrier.

The barrier requires12 unique expected keys, exact registered counts/weights/coordinates, all cache hashes, exact input/split hashes, all source/current model parameter/buffer/classifier/GAS hashes unchanged and all9 B0 checkpoint disk hashes unchanged. `transport_optimizer_steps=0` until all checks pass. Segmentation optimizer construction is forbidden independently of step counters.

## T0, T1 and T2

T0: identity,0 optimizer steps.

T1: AA-only weighted orthogonal Procrustes with ORIGINAL AA weights, not conditionally rescaled: `M=(Y_AA*w_AA).T@X_AA`, `U,S,Vh=svd(M)`, `R=U@Vh`, output `R@u`. Reflection allowed; no determinant correction, bias or optimizer. The orthogonal output is checked for unit norm (atol1e-12), not given an extra epsilon normalization.

T2: `normalize(u+W@u+b)`, W16×16, b16, initialized exactly zero. Step0 agrees with T0 active output to absolute1e-12 and relative0 (only float64 normalization roundoff). Null input always stays null; bias does not create a null direction.

Use CPU float64 Adam, lr1e-3, betas(0.9,0.999), eps1e-8, weight_decay0, amsgrad=false, foreach=false. Exactly1000 full-fit-set steps per map; no scheduler, clipping, early stopping, minibatching or hyperparameter search. Objective is the equal-case full-support error plus `1e-4*(sum(W^2)+sum(b^2))`. Cross-support penalties remain in every objective/trace even though their derivatives are zero. Only the272 W/b elements are optimizer parameters; features/models/operational prototypes are detached.

Trace step0 and post-update steps1..1000: full objective, full-support error, AA-conditional/weighted term, support constant, regularization, W/b norms, minimum raw output norm, gradient/finite checks. Exactly6000 transport optimizer updates on successful completion,0 model updates. Before and after updates check source-active fit rows and immutable operational-prototype sentinels; they contribute no loss. Holdout and every immediate/chain application are also guarded. Raw active output norm≤1e-12 or nonfinite output/parameter/loss/gradient/spectrum blocks as `BLOCKED_INVALID_TRANSPORT_OUTPUT`; never add epsilon.

Six CPU workers fit the six independent maps, single-threaded BLAS. Two GPUs serve the read-only model extraction. No seed result is previewed to decide whether to execute the others; final selection waits for all fits/evaluations.

## Operational prototypes, oracle and accuracy

Operational historical prototypes come ONLY from the freeze's B0-EMA K2 train_labeled ORIGINAL fits. They are immutable; no source refit, no per-class/per-prototype transport fitting.

For each seed, evaluator-only oracle units are: REFUGE val under stage1 EMA (immediate01), RIM_ONE_r3 val under stage2 EMA (immediate12), and REFUGE val under stage2 EMA (chain02). Reuse the original historical-val coordinate/class membership/weights from the raw frozen Gate1A sampling plan. This is diagnostic GT (`gt_consumer=diagnostic_evaluator_only`); it never enters a transform fit, step/hyperparameter/threshold selection, checkpoint choice or operational bank.

Fit each oracle class with the existing null-aware K2 spherical solver: source-domain stage-index clustering seeds, original replicate=-1, five restarts, max100 iterations, original tie rules. Preserve/disclose nonconvergence and inactive slots without extra attempts. Minimum-total-angular-cost Hungarian matching uses radians and both K slots. An unmatched/inactive oracle slot has conservative cost pi; all-null oracle support stays explicit, not a numerical exception or a silently reduced K.

Chain starts from a fresh immutable REFUGE stage0 prototype: `T12(T01(p))`. Direct identity is that original prototype, never a refit or an already-transported vector. T0/T1/T2 use the same queries/oracles.

Historical-val prototype-only accuracy: active queries score every class by max-over-two cosine, ties choose lowest class ID. Null queries are incorrect. Original case-equal weights are used within each true class, then equal three-class macro (primary), equal foreground1/2 macro and per-class accuracy. Directional-conditional accuracy additionally divides active-correct original mass by active original mass per class; undefined no-active classes have null/defined=false, never manufactured perfect/zero accuracy.

Report individual-map W Frobenius norm, b norm, all singular values, spectral norm and condition number of I+W. For T0, W=0/b=0; for T1, W=R-I/b=0 is a reporting representation, not an optimized residual map. No linear approximation of biased normalized T2 chains is substituted. Nonfinite/singular condition-number results block.

## Unified B1–B7 admission

Use unrounded float64 values. T2 must pass every condition; T1 is descriptive and cannot rescue T2.

| Gate | Exact rule |
| --- | --- |
| B1 | Each transition's mean-over-three-seeds heldout FULL support error relative reduction vs T0≥0.15 |
| B2 | Each transition has at least2/3 seeds with T2 full support error strictly below T0 |
| B3 | Relative reduction of immediate foreground angular error, equal mean over2 transitions×3 seeds×2 classes,≥0.10 |
| B4 | Every immediate seed/transition/foreground-class RELATIVE angular-error worsening vs T0≤0.05 |
| B5 | Every immediate/chain seed/source-domain/target-stage unit's absolute T0-minus-T2 three-class macro accuracy drop≤0.005 |
| B6 | Every seed/foreground-class REFUGE chain RELATIVE angular-error worsening vs direct identity≤0.05 |
| B7 | All defined features, gradients, parameters, losses, outputs, SVD, singular values and metrics finite |

B4/B6 are relative, as explicitly specified in the inherited preregistration. For positive reference, use ordinary ratios. For reference0/candidate0, reduction/worsening=0; for reference0/candidate>0, the relative ratio is structurally undefined and the corresponding bounded/improvement comparison fails. Record null plus defined=false instead of Inf; never use epsilon. Undefined conditional support statistics do not remove full-support units or relax any gate.

PASS selects `T2_residual_full_linear`. Scientific FAIL selects `T0_identity` as explicit downstream fallback only; downstream does not run. Engineering block selects null. Missing evidence never becomes a partial pass.

## Tests, outputs and stop

The companion JSON enumerates all44 requested test categories. Include exact split/coordinate/row/hash checks, all support states, null retention, Procrustes reflection,1000-step accounting and isolated optimizer, immutable models/chain, evaluator isolation, source-stage oracle seeds, null-query accuracy, exact B1–B7 boundaries and fail-closed report compilation. Exact-code real integration covers both transitions without fitting a transform.

Output namespace: `/root/LCRSeg/runs/di_dmpa_gate1b_v2/<prereg_commit>/gate1b_v2_<exact_code_commit>_attempt1`. Create-only outputs; no silent retry/overwrite. Interrupted extraction uses the existing cooperative stop mechanism with600-second guard-completion timeout. Preserve first failures and every warning. All run metadata binds freeze/prereg/auth/exact-code commits and file hashes; a report SHA is resolved from the commit first adding the exact report bytes, avoiding a self-reference.

Deliver the freeze, preregistration, authorization, coordinate plan/audit, support census/CSVs, paired-cache and model-immutability manifests, transport models and traces, all feature/prototype/chain/accuracy/spectrum metrics, raw B1–B7 status, test outputs, failures/warnings, exact commands and artifact hashes. Raw feature tensors stay remote; public report copies retain their descriptors/hashes.

Legal results: `PASS_TRANSPORT_SUPPORTED`, `FAIL_TRANSPORT_NOT_SUPPORTED`, `FAIL_DIRECTIONAL_PAIR_SUPPORT_NOT_SUPPORTED`, `BLOCKED_PROTOCOL_OR_LEAKAGE`, `BLOCKED_NONFINITE_FEATURE`, `BLOCKED_INVALID_TRANSPORT_OUTPUT`, `BLOCKED_MODEL_MUTATION`, `BLOCKED_INCOMPLETE_EVIDENCE`.

Always `method_registered=false`, `di_dmpa_training_launched=false`, `model_optimizer_steps=0`, hidden-GT training usage=none, test-GT usage=none. No Gate1C, reliability, gradient-conflict, teacher-noise, theory final, training, Prostate, MnMS, Gate2, full sweep or main merge. After report commit/push: **STOP_FOR_INDEPENDENT_REVIEW**.
