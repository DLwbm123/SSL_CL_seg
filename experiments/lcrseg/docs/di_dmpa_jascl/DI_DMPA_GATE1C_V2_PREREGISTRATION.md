# DI-DMPA Gate 1C v2 preregistration

Registration: `DI_DMPA_GATE1C_V2_K2_IDENTITY_HISTORY_RELIABILITY`. Date: 2026-08-30 (Asia/Shanghai). This is an offline diagnostic, not a method/training registration.

## Frozen identities and publication barriers

Branch `codex/gate1c-v2-identity-history-reliability` starts at `4ea4d7723db9cd29295ab000707c7bbb0044d0dc`, not main. Gate 1B freeze/closure `cda045db7cf9e2fc01903c51c9aca04126494917` was independently pushed and remotely verified before this preregistration. Its JSON SHA-256 is `85be4cef01435f3908cd0f6bd9b338a67da95b6034d43f75e17bd35243a05ae3`.

The paired JSON is the machine-readable contract: it explicitly contains all 72 original gradient pair records (case IDs, checkpoint SHA, two student forward seeds, eight teacher seeds), all 495 current-validation case draw seeds, all original role plans and nine B0 checkpoint identities. Pair-list SHA-256: `a0eaea7e6b79ee8fe202dd7b40337812afb2c12bd4f200c2230a136e20d006de` using original compact JSON hashing. Original Gate 1 preregistration: `cfb62554f1e6a2a36850547485b1857dc9a28a20`, JSON SHA-256 `6f50bd9df404d987aa70e2035a5c3f3853aa59ce49d21ffface34172cf754cbf`. No pair or draw is reselected.

Gate 1A v2 report `9b2ffd04c7a8e9da73f08edb0760be3f269065d8`; freeze `58f19e968700bd7708ec00e44a11759b48ce756f`; preregistration `eaae37bbaa7546679d9e6893023afbeeef0ab5c6`; exact code `8ae5d7532f90aee5d53c0d966706ef64c18a19ac`. K=2 is frozen. Gate 1B v2 preregistration `b20f186deff287843f3c9f18bf4ab5633908f441`; authorization `c6f72b86fdfa3683a6e2c7dbf593f73cab74c592`; exact code `f2a3ed7476323119b1a4fa22481b44038bc4148c`; report `959e62df5608fe170f6702a7fd1a1f2a42eec8ad`; receipt `4ea4d7723db9cd29295ab000707c7bbb0044d0dc`; artifact manifest SHA-256 `26e69d13935133b1cfa4e3176ff5555ba8bef73755fd8fe3c0157505a92e0ea2`.

Gate 1B remains **FAIL_TRANSPORT_NOT_SUPPORTED** (B3/B4/B5 failed). T0 identity is the only downstream history path; R4 is unavailable, with no R4 rows. T1/T2 outputs, refits, alternative transport and a “drift-calibrated” contribution are forbidden.

Publish this preregistration alone, push and verify SHA; then independently publish execution authorization. Before both barriers: zero new checkpoint tensor reads, model forwards, reliability caches or gradients. Implement and synthetic-test only afterwards; publish and remotely verify exact diagnostic code before real integration/formal execution. Frozen prior files are append-only references, never modified.

## Primary and banks

B0 EMA probability and dec1 features; student is only the gradient receiver and the inherited R1 student-validity control. Primary = R3 identity-history, pixel normalized. Selected K remains 2; no feature-source selection.

| Stage/current | Historical operational bank |
| --- | --- |
| 0 REFUGE | empty |
| 1 RIM_ONE_r3 | REFUGE K2 |
| 2 Drishti_GS | REFUGE K2 + RIM_ONE_r3 K2 |

Banks are exactly the B0-EMA train_labeled original fits in `GATE1A_V2_FREEZE.json`, source-stage coordinates and original center order. No oracle, student/C0, pseudo-label, online or validation fit. Logmeanexp subtracts log(valid prototype count) separately for each class.

## Null-aware weights

Active iff the float64 norm of the float32 dec1 feature is >1e-12. Normalize only active rows, with no epsilon. Null UIDs remain in the full valid case-balanced denominator. R0 confidence remains defined and R1 executes the unmodified Gate0 v2 PAS. Null R2/R3 have zero weight and false prototype validity; all their scores, margins and history gate are null. PoE is unavailable with weight zero. Nonfinite model features/logits/probabilities/weights/gradients block; explicit structural nulls are not fabricated directions or numerical failures.

Let `p_t=softmax(teacher logits), y_hat=argmax p_t, q=max p_t`. Lowest class ID wins argmax ties.

```text
R0 = q
R1 = (student_confidence > .7 AND student_legacy_proto_cos > .7)
     AND (teacher_confidence > .7 AND teacher_legacy_proto_cos > .7)
a_cur(c) = logmeanexp_k(cos(u,p_cur[c,k])/.07)
m_cur = a_cur(y_hat) - max_other_class a_cur
R2 = q * sigmoid(m_cur/.10)
a_hist(c) = logmeanexp_{historical domains,k}(cos(u,p_hist[d,c,k])/.07)
s_hist = max_same_y_hat_historical_cosine
g_hist = sigmoid((s_hist-.30)/.10)
m_hist = a_hist(y_hat)-max_other_class a_hist
R3 = R2 * ((1-g_hist)+g_hist*sigmoid(m_hist/.10))
```

R1 uses the checkpoint's `prototypes`, not K2, and must match the Gate0 shared PAS path pixel-for-pixel. Missing usable history is neutral; stage0 R3 exactly equals R2. Missing current predicted-class center/competitor yields weight zero and invalidity. All targets/weights/banks are detached. R3 is bounded by R2.

## Validation and ranking

Exactly 9 current-domain val units, 495 case records, every non-ignore pixel. Both models eval, AMP off, stochastic classifiers on. Case-at-a-time forwards use explicit `S(['val-teacher-v1',seed,stage,case_id,0])` and `S(['val-student-v1',seed,stage,case_id,0])`, reseeded **after** constructor/load. The builder never receives GT. A separate evaluator reads val labels; unlabeled labels and final test roles are forbidden. Raw case-name substrings such as “test” do not override frozen manifest roles.

Within each predicted-class stratum every nonempty case has equal total mass. Positive weights only; descending weight with full GT-independent SHA tie-break `H(['reliability-tie-v1',seed,stage,case_id,y,x])`. Coverage = accepted mass / all non-ignore mass, including null features. Precision points are .05/.10/.20/.30/.40/.50; unsupported points are null with reason. Use the first ranked prefix reaching a supported point and publish achieved coverage. Common support is [0,min(candidate max,R1 max)]; integrate right-continuous prefix risks, clipping the last interval, and divide by that upper bound. Also report full available-support AURC. Never extrapolate.

The matched-point set is the intersection across all 18 required foreground units, not a selected best point. Publish both per-unit deltas and the global shared set. Report global/background/rim/cup/foreground macro curves, max support, ECE (15 equal-width bins), multiclass Brier, reliability-bin accuracy (including empty bins), accepted predicted/true composition and true-class recalls. Coverage guard uses each seed/domain's global min(.50,R1 max,candidate max), based only on scores/support.

## Gradients, noise and controls

The original 72 pairs remain byte-equivalent as JSON values: 8 pairs ×3 seeds ×3 stages; labeled/unlabeled batch size2, stored RGB/255, no augmentation. Reseed each registered forward. The same student graph/probabilities and teacher draw are shared across all candidate/normalization comparisons. Persist primary outputs; later phase recomputation for autograd must match them bitwise under the exact same student seed.

```text
L_sup = mean CE(student labeled logits, GT), ignore=255
loss_i = sum_c(p_student(i,c)-stopgrad(p_teacher(i,c)))^2
L_u(r) = sum(r_i*loss_i)/(sum(r_i)+1e-12)
L_u_cb(r) = mean over positive-weight predicted classes c:
            sum_class(r_i*loss_i)/(sum_class(r_i)+1e-12)
zero weight => p_student.sum()*0.0
g_sup=autograd.grad(L_sup); g_u=autograd.grad(L_u)
ratio=.5*||g_u||/||g_sup||
```

No backward, parameter.grad write, optimizer construction/step, EMA/GAS/prototype update. Inventory every student parameter; official sigma and grad_update must have None gradients, while encoder(enc1/2/3), bottleneck, decoder.dec3/2/1 and decoder.conv_logit.mu exactly partition all active parameters. Report global/six-block cosine, zero-gradient flag, norm ratio, negative fraction, median/p10/p90 (linear unweighted quantiles), each domain and draw0 class gradient vector decomposition. Undefined zero-norm cosine stays null and makes a required condition fail; never delete the row. Float32 autograd linearity audit uses predeclared atol1e-6/rtol1e-4 only; gate thresholds have zero tolerance.

For the same 72 pairs use exactly 8 registered teacher seeds (576 records), with draw0 primary and shared across candidates; no draw averaging for admission. Report population probability/weight/cosine/ratio variance, class change versus draw0, per domain/block, retaining all undefined draw values. Complete draw0 for all pairs before noise diagnostics. Then a posterior-mean teacher (`stochastic_classifier=false`) is a separate control with fixed primary student outputs; it cannot rescue R3 and would require a new baseline if adopted.

PoE is target-only control: `p_PoE ∝ p_t * p_current^.5 * p_history^(.25*g_hist)`; history factor1 at stage0, unavailable for null rows. It uses the **same detached R3 weights** to isolate target fusion, not an additional reliability rule. Its own argmax defines class strata/class-balanced grouping. Preserve zero probabilities in stable log-space without epsilon smoothing. Report changed predictions, own class composition, validation curves, gradient conflict and the same eight-draw variance. PoE and posterior mean never become primary or rescue a failure.

## Admission and stop

All gradient comparisons, including class-balanced R3, use **pixel-normalized R1**. Compute every value unrounded; no tolerance relaxation.

| Gate | Required condition |
| --- | --- |
| C1 | foreground macro common-support AURC relative reduction >=.10 OR global shared-point precision increase >=.01 |
| C2 | >=12/18 foreground units with strictly lower AURC OR strictly greater shared-point mean precision |
| C3 | each teacher-predicted foreground class retained fraction >=.8 × R1 at its matched global point |
| C4 | negative global cosine fraction over all72 pairs reduced by >=.20; reference zero cannot demonstrate improvement |
| C5 | global median cosine increase >=.05 |
| C6 | each domain median cosine worsening <=.05 |
| C7 | no teacher/bank/state gradients; complete model/checkpoint/legacy/K2/history state bitwise unchanged |
| C8 | hidden GT training usage none; test GT usage none |

Evaluate pixel R3 first. If all C1–C8 pass: `PASS_IDENTITY_HISTORY_WEIGHT_ONLY`, selected `R3_IDENTITY_HISTORY_WEIGHT_ONLY` / `PIXEL_NORMALIZED`. Only if pixel fails and class-balanced independently passes: `PASS_IDENTITY_HISTORY_CLASS_BALANCED_ONLY` / `CLASS_BALANCED`; disclose ordinary failure. Otherwise `FAIL_IDENTITY_HISTORY_RELIABILITY_NOT_SUPPORTED`, no selected reliability. R2 passing only means current-only promising control and independent redesign.

Regardless of C, overall Gate1 = `FAIL_TRANSPORT_NOT_SUPPORTED`. Reduced candidate is `ELIGIBLE_FOR_NEW_NON_TRANSPORT_METHOD_PREREGISTRATION` only if R3 passes one authorized normalization; else `NOT_ELIGIBLE`. Neither status authorizes training. Legal engineering blocks and all 56 requested test categories are listed in JSON; real integration must cover all3 stages, available null EMA coordinate, one complete fixed gradient pair, exact PAS parity and unchanged models. Any incomplete required unit fails closed.

Use the existing environment, both GPUs with stable unit assignment, no new dependency. Preserve all failures/warnings and create-only attempts. Public outputs exclude image/GT/checkpoint tensors; raw caches remain hashed on the execution host. Method switches, model/transport optimizer steps, training, Gate2, theory final, Prostate, MnMS, sweep and main merge remain false/zero. Publish the complete report and separate receipt, then **STOP_FOR_INDEPENDENT_REVIEW**.
