# DI-DMPA Gate 1A v2 preregistration — null-aware sphere

Registration: **DI_DMPA_GATE1A_V2_NULL_AWARE_SPHERE**, version2.0.0, 2026-08-30 Asia/Shanghai. The accompanying JSON is normative; any disagreement blocks. This is an offline diagnostic, not a training method/config.

## Publication and immutable ancestry

Branch `codex/gate1a-v2-null-aware-sphere` starts at `606a5c53a37d0e4c9605415e8b38a1f177d1604f`. Closure commit: `b61f6db0ca9e746d005937e7dfc51c45078e1d80`. Both v1 attempts remain BLOCKED_NUMERICAL_FAILURE with0 geometry jobs, uncomputed A1–A6 and selected_K=null; no v1 attempt3.

V1 normative JSON at `cfb62554f1e6a2a36850547485b1857dc9a28a20` is inherited through the exact JSON pointers listed in v2 JSON. Its raw SHA256 is `6f50bd9df404d987aa70e2035a5c3f3853aa59ce49d21ffface34172cf754cbf`; MD raw SHA is `32acdc5c24bcc5763daa6cb3650fea91f46da7ae3845b1fd0615c781619fbf0a`. All18 stage-best checkpoint IDs/SHAs, B0/C0 config identities, domain order, roles, class mapping, image preprocessing, cases, fixed bootstrap draws, H/S seeds, solver restarts/iterations/ties and numerical gate thresholds remain unchanged. No v1 permission or nonzero-feature support assumption is inherited.

Publish and remotely verify this preregistration, then separately publish and verify execution authorization. **No new checkpoint-tensor read or model forward before both barriers.** Publish exact diagnostic code before the real integration and unique formal v2 attempt. Historical documents and artifacts are never modified.

## Support model and cache

Use decoder.dec1 post-ReLU 16-D at stored384x384; float32 forward, float64 norms/geometry, eval/no_grad, AMP off, stochastic_classifier=false, batch_size=8 and registered forward seed set after model construction/load.

For finite registered z, active iff norm(z)>1e-12. Active direction is z/norm(z). Otherwise the observation is a fixed null atom with no direction. P_hat=pi_null*delta_null+(1-pi_null)*Q_directional; original per-class equal-case weights determine pi_null.

Every registered UID retains one row in directions[N,16] float64, active_mask[N] bool and raw_norms[N] float64. Null direction rows are zero placeholders, never normalized or passed to cosine. Store raw array SHA, shape/dtype, H(ordered UIDs) and SHA256(C-contiguous little-endian float64 original weights). active_mask=false iff raw_norm<=1e-12. Active norm error tolerance is1e-12. No eps, fake direction, deletion, replacement, resampling, feature/source/panel change or historical cache reuse.

Reuse only the exact44MB sampling-plan bytes, raw SHA `96277c841165d8cd84a63d3d4a9301b27c50dcdd87d0b0d215c162d95c345e24`, copy read-only, never rematerialize labels/coordinates. Full-map or registered NaN/Inf immediately gives BLOCKED_NONFINITE_FEATURE; finite null never gives an engineering block.

## Fit and degenerate support

Fit K=1 reference and candidates2,3,5 on active directions with original weights. The v1 CPU float64 weighted spherical K-means algorithm, five restarts,100 iterations,1e-6 angular tolerance, seeds, tie rules and inactive slots are unchanged. Keep original UID ranks. Solver normalization by active weight sum does not change centers. Never reduce K or reroll.

Zero active mass yields all K centers inactive, directional_support=NONE, no exception and conservative Q/R95=2. An all-null bootstrap retains its fixed draw, all slots inactive and matched cosine0. A claimed-active center must be finite and nonzero/unit norm; otherwise BLOCKED_NONFINITE_OR_INVALID_CENTER. Zero-weight bootstrap support also receives inactive slots, not rerolled data.

## Null-aware geometry

Active features with active centers use cosine distance1-maxcos and sphere distance sqrt(max(0,2-2maxcos)). Null features use **2 for both distances**. Metrics explicitly receive active_mask; they may not use a zero dot product to imply null handling.

Report null_mass, active_direction_mass, null_count, active_count, conditional directional Q/cosine-p95/R95 and Q_null_worst_case/cosine-p95_null_worst_case/R95_null_worst_case. Worst-case means and the registered weighted ECDF use **all original UIDs/weights**, with null distances2 included in the same distribution. Conditional directional statistics never replace all-observation metrics.

If no active center exists, assign conservative distance2 to every query, including active val directions; conditional metrics are undefined/null because there is no fitted directional model. This cannot manufacture K improvement. Empty boundary/interior strata are structural nulls; otherwise preserve every null UID and the existing equal-case-within-stratum weights and7x7 masks.

Verify Q_wc(K1)-Q_wc(K)=active_mass*(Q_cond(K1)-Q_cond(K)) for each train/val class-unit having directional support, with atol1e-12/rtol1e-10. All-null/no-center cases instead verify K-independent Q_wc=2. No-null geometry must reproduce v1 numerical geometry. Positive-norm census p01/median use float64 linear quantiles only for support diagnostics; geometry retains the exact noninterpolated weighted ECDF.

## Census before any clustering

Complete all72 unique panel/seed/stage/role units and all18 checkpoint before/after/disk immutability audits before starting any of432 geometry jobs. Every case/class records registered/active/null counts, fraction/weighted mass, positive norm summaries, full-map zeros/nonfinite, registered nonfinite, exact coordinate/null hashes and at most32 null coordinate examples. Unit/panel reports retain classwise null mass, maximum case-class null fraction, unsupported cases and finite status. Panel class null mass is the equal mean across its18 feature units, never a cross-panel average.

Validate cache N, dtype, mask/raw-norm equivalence, placeholder zeros, unit norms, every UID/weight/order/hash and full72 keys. There is no preview-based early primary verdict. Compute all four panels, all432 jobs, five fixed bootstraps and both strata before adjudication.

## Frozen admission, new radius definition

B0-EMA alone controls admission over18 foreground units; B0-student, C0-EMA and C0-student are separate controls and cannot select K or rescue primary.

| Rule | v2 statistic and unchanged threshold |
| --- | --- |
| A1 | count of strict val R95_null_worst_case decreases >=12/18 |
| A2 | median of9 seed-domain equal-foreground-macro relative R95_null_worst_case decreases >=0.10 |
| A3 | fraction of active directional clusters with occupancy>=0.05 >=0.90; occupancy normalized over active assignment mass; null is not a slot |
| A4 | foreground five-bootstrap/Hungarian matched cosine median>=0.85; inactive/all-null slots=0 |
| A5 | strict cross-seed equal-foreground-macro R95_null_worst_case decrease in>=2/3 domains |
| A6 | background excluded from all admission statistics |

If pooled active occupancy support is empty, A3 fraction=0, not an exception. Partially unsupported units stay in the18unit denominator; no extra null-fraction admission threshold is added. Select the smallest passing K; none passing means K1 EXPLICIT_DOWNSTREAM_FALLBACK_ONLY and FAIL_MULTI_MODALITY_NOT_SUPPORTED. If **all18 primary original foreground train class-units** lack active mass, label that no-direction special case FAIL_DIRECTIONAL_SUPPORT_NOT_SUPPORTED; this changes only the scientific failure label, not a passing threshold. Never use conditional R95 for A1/A2/A5 or hide missing/null rows.

## Run identity, tests and stop

Unique formal namespace: `/root/LCRSeg/runs/di_dmpa_gate1_v2/<V2_PREREGISTRATION_COMMIT>/gate1a_v2_<EXACT_CODE_SHA>_attempt1`. Use two GPU feature shards,16 CPU float64 geometry workers,1 BLAS thread each and the existing600s cooperative shutdown contract. All caches are newly produced by the same published exact code.

Synthetic tests cover every null/mask/weight/ECDF/all-null/no-null equivalence/center/nonfinite/provenance/history/coverage/adjudication/report requirement. Real integration must retain the known B0 seed2 stage0 EMA REFUGE train_labeled class1 zero at REFUGE_test_n0128 (125,212), mask=false, distance2, no row deletion or clustering and unchanged model/checkpoint.

Legal scientific outcomes: PASS_MULTI_MODALITY_SUPPORTED, FAIL_MULTI_MODALITY_NOT_SUPPORTED, FAIL_DIRECTIONAL_SUPPORT_NOT_SUPPORTED. Engineering failures: BLOCKED_PROTOCOL_OR_LEAKAGE, BLOCKED_NONFINITE_FEATURE, BLOCKED_MODEL_MUTATION, BLOCKED_INCOMPLETE_PANEL, and the explicitly required invalid-center block. Preserve every failure; no automatic second formal v2 attempt.

All method flags remain false; optimizer steps=0; hidden/test-GT usage=none. No Gate1B/C, transport, reliability, gradient-conflict, teacher-noise, theory final, training, Prostate/MnMS, Gate2 or main merge. Commit/push the report, verify remote SHA, then STOP_FOR_INDEPENDENT_REVIEW.
