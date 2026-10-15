# MAIN_STYLE_SSL_V1 — frozen main experiment, 2026-10-14

Authority: user explicitly authorized implementing and launching the previously proposed single DC²T-inspired V27 upgrade for fundus, prostate, M&Ms. RL remains paused. Historical gates/results are immutable. This version is not a full DC²T reproduction.

## Frozen method

Same native UNet/linear readout, FP32, Adam/poly decay, EMA .99, target projected A/B rank8 and Q/R rank4. Target LR A/B .0005, R .001; source dense native LR .001, weight decay 4e-5. One StageTrainer engine. Dense source trains encoder/decoder/linear mu; stochastic sigma/grad_update are frozen. Source has no historical entry or target R, hence no entry projection/SWD there. Source uses L+U, hierarchy, and style consistency; target additionally uses R-only clean SWD and immutable entry gradient protection. Two teachers are current EMA plus preceding deployed entry only at target.

Student same-site different-patient style: channel spatial population mean/std, alpha .2; unchanged geometry and labels. Weak originals feed EMA/prototypes and clean SWD. Existing complementary two-forward LCTX collection and donor noise .02 retained. Fundus clips style to [0,1]; MRI clips to [-1,1]. No extra style forwards. L hierarchy weight .1, temperature .1, parent maps (0,1,1)/(0,1)/(0,1,1,1); no live alignment loss. The singleton prostate foreground has coarse contrast only.

Confidence .7 unchanged, strict >.7 admission. Foreground does not use cosine hard rejection. Background uses existing PAS cosine >.5 (existing confidence-only fallback when its L prototype is absent), and confident entry-conflict rejection. On foreground, confident entry conflicts multiply by .5. Confidence taper clip((qmax-.7)/.3,0,1); per-batch inverse sqrt accepted pseudo-class frequency, capped5. KL denominator is sum(mask*classweight), excluding taper/conflict factors. All masks/q/weights detached. Reference pixels additionally require entry confidence >=.7 and current/entry class agreement; average per-present-class KL, project merged raw gradient to entry halfspace. Adam effects are not a finite-step retention guarantee. Background-conflicting pixels remain excluded from KL but clean SWD keeps the existing pre-entry-admission mask, as approved.

Objective target: CE+Dice+.1 hierarchy + ramp*.25*(weighted styled KL+.2 clean SWD). Source same without SWD; no target R exists. No VAE, replay, third teacher, CPC, new head, calibration, threshold search, ablation, or independent-test accessor.

## Data, orders, budget

Frozen split seed0; optimizer seeds163/164. Fundus RGB384, C3 background/rim/cup; source REFUGE40L+160U; orders REFUGE→RIM_ONE_r3→Drishti_GS and REFUGE→Drishti_GS→RIM_ONE_r3. Train source once per seed, share across its two orders. Prostate single-channel256, C2 background/whole-prostate; RUNMC→BMC→I2CVB→HK→UCL→BIDMC. M&Ms single-channel384, C4 background/LV/MYO/RV; Siemens→Philips→GE→Canon; canonical320 only, auxiliary25 completely excluded.

All U label paths/hashes empty; no hidden-label loader. MRI floats are stored normalized [-1,1], cast FP32, never /255. Patient first, phase second, slice third; each pair distinct patients. Slice selection image-only/uniform; no U GT or content-based selection. Uniform patient sampling means these are nominal sample-budget cycles, not guaranteed complete passes over slices. Augmentation: existing paired flips/90-degree rotation, no spatial mismatch. Budget 100*ceil(max(L_samples,U_samples)/2), where MRI samples are frozen slice counts. Warmup20%, U ramp20%.

| Benchmark | Domain updates per seed | Total |
|---|---|---|
| Fundus | source8000; RIM3200; Drishti2100; two target orders | 37200 |
| Prostate | RUNMC13600; BMC22000; I2CVB28800; HK6000; UCL7200; BIDMC11100 | 177400 |
| M&Ms | Siemens43600; Philips53300; GE25100; Canon26500 | 297000 |

Exactly 30 DAG nodes, 511600 formal optimizer invocations. Engineering qualification separately capped330 invocations per recorded attempt; repairs retain previous failed receipts/calls, never reset scientific trials. Two-case qualification96 fixed L-only updates per dataset (engineering only, no validation selection). One formal attempt/node; failure blocks descendants, independent branches continue. No automatic retry or extension.

## Evaluation and retention

Only seen-domain frozen val, final fixed checkpoint each stage, no best-epoch selection. Fundus primary rim/cup macro Dice, auxiliary cup/disc union. MRI reconstructs full volume; patient 3D Dice, M&Ms averages ED/ES within patient first. HD95/ASSD use stored processed ZYX spacing in mm, confirmed via training headers; missing/invalid spacing fails rather than assumes1mm. Both-empty Dice1/distances0; one-empty Dice0/distances Infinity retained with counts. No dropping negative/empty/failing patients. Domain and class tables, both seeds/orders, Old/Incoming/Final/Forgetting and signed site changes; no post-hoc primary switch. Clinical benefit and improvement are not assumed. New source U/exposure and cross-dataset budgets prevent attribution to a single module or an unfair matched-source claim.

All runtime outputs private NAS, create-only study root; checkpoint latest overwritten only inside that new node, final state and physical ledger retained. Raw/patient outputs stay private. All completed aggregate results, failures and source/report will be publicly delivered to the project GitHub according to global completion policy; no images/checkpoints/IDs in Git. Background DAG does not depend on SSH. Startup health check only; no automatic monitoring.

Engineering amendment2026-10-14: the user authorized recovery of Prostate seed164 from its completed RUNMC checkpoint after BMC initialization failed before any optimizer invocation. The main method requires numerical rank at least the retained protection dimension k, rather than full source-kernel rank. The SVD tolerance, exact top-k singular vectors, k, adapter rank8, objective, options, data, orders and budgets are unchanged. Historical callers retain full-rank admission by default. Only this recovered five-target branch uses the amended immutable snapshot; existing workers continue on their original snapshot. Preserve the original FAILED/BLOCKED attempt and all engineering failures in the attempt history. No source rerun, independent test access or automatic additional retry is authorized by this amendment.

Engineering correction before formal launch: MRI first convolution has a 16x9 flattened weight, so numerical full rank is min(16,9)=9. Its allowed single-channel minimum input width is9; native rank8 is retained. Existing protection k=min(m//2,d-r) consequently gives first-layer k1 for MRI and k8 for RGB. All other native projection rules and RGB basis values are unchanged; golden checks verify this. Q uses contiguous layout in both normal and resume construction, retaining bitwise update checks. Validation phase/case rows and partial domain aggregates are flushed before completion, including failures; failed-run costs and every physical call remain retained.
