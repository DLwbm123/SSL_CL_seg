# L05_SSL factorial and fixed-split replication — final report

All admitted fixed tasks completed: L05_SSL optimization_seed0 stage1+2; optimization_seed1/2 each completed its own common stage0 and S/L05 stage1+2. Five incremental trajectories passed final single-student deployment. The conditional L05_SSL seed1/2 tasks were not admitted by the sealed P_GATE. No substitute recipe or extra seed was started.

| Question | Terminal | Evidence |
|---|---|---|
| Engineering execution | COMPLETE | 12 training stages; 29 successful child exits; 5 single-student deployments |
| L05 fixed-split optimization-seed replication | NOT_ESTABLISHED | New seed2 F is lower than S; paired new-seed mean F and H gains fail |
| L05_SSL seed0 screening | NOT_ADMITTED_BY_FROZEN_GATE | F and H are lower than L05 despite passing current-class guards |
| L05_SSL new-seed confirmation | NOT_ADMITTED_BY_FROZEN_GATE | P3 was not executed; no invented seed1/2 outcomes |

## L05 replication uses only the two new optimization seeds

| Optimization seed | S F | L05 F | Paired F gain | Paired H gain | Paired N gain |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.567808101951 | 0.574944906957 | +0.007136805006 | +0.003866040535 | +0.008376778147 |
| 2 | 0.604288048107 | 0.562749397388 | -0.041538650720 | -0.063888725728 | +0.010435974297 |

L05 preserves every current-domain rim/cup guard in both new seeds; in fact all eight class differences are positive. This does not rescue its final retention result: the paired F mean is -0.017200922857 (required >=+0.010), paired H mean is -0.030011342596 (required >=0), and seed2 F gain is -0.041538650720 (required >0). The observed adaptation gains do not establish reproducible consolidation under this fixed recipe.

The previously selected seed0 remains descriptive. Across optimization seeds0/1/2, S F mean/SD is 0.586017777168 / 0.018240048568; L05 is 0.586042223446 / 0.030400559893. Their nearly equal three-seed F means must not hide the negative result in the two new confirmation seeds. Mean/SD uses sample SD and training seeds as units, not pixels or cases.

## L05_SSL screening and the missing factorial cell

| Recipe, optimization_seed0 | F | H | N |
|---|---:|---:|---:|
| L05 | 0.620432365995 | 0.565903762749 | 0.722308723860 |
| L05_SSL | 0.551683447660 | 0.477544978416 | 0.700276869679 |
| U0 | 0.676784120114 | 0.698257900508 | 0.677446014749 |
| D | 0.718089435235 | 0.742590645209 | 0.698910788759 |
| S | 0.585957181445 | 0.525668818061 | 0.704849150073 |
| L10 | 0.571301650714 | 0.504197861897 | 0.709716209193 |
| E_R1 | 0.687037996675 | 0.704006347226 | 0.692652881345 |

L05_SSL minus L05 is F -0.068748918335, H -0.088358784333, N -0.022031854181. P_GATE fails its F and H conditions. Its current-domain macro and rim/cup guards relative to S all pass; the screening failure is not a current-class safety failure. The sealed gate is retained with its exact values and source in GATE_ACCOUNTING.json and on NAS as P_GATE.json. Consequently P3 cost is zero.

| Seed0 complete-recipe contrast | F difference | H difference | N difference |
|---|---:|---:|---:|
| CE_without_U_KD | -0.068748918335 | -0.088358784333 | -0.022031854181 |
| CE_with_U_KD | +0.041305315121 | +0.044332744701 | +0.021464774010 |
| U_KD_with_CE | +0.166405987575 | +0.265045666793 | -0.001366080921 |
| interaction | +0.110054233456 | +0.132691529034 | +0.043496628191 |

The positive D-U0 contrast observed with U-KD does not transfer to adding CE without U-KD. The interaction is a descriptive difference between full trained recipes; it is not a proven pixel-level causal mechanism. D remains a descriptive high-F comparator with its original stage2 cup 0.635104700870 versus S 0.745715297908. No historical judgment has been rewritten.

## Current class costs

| Seed | Recipe | Reference | Stage | Rim | Cup | Rim difference | Cup difference |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | L05 | S | 1 | 0.738359054 | 0.691896697 | +0.006318236 | +0.017608730 |
| 0 | L05 | S | 2 | 0.694255730 | 0.764723415 | +0.026903211 | +0.019008117 |
| 1 | L05 | S | 1 | 0.738680950 | 0.673531938 | +0.000138573 | +0.006011872 |
| 1 | L05 | S | 2 | 0.695311679 | 0.766788211 | +0.019052190 | +0.008304478 |
| 2 | L05 | S | 1 | 0.732510671 | 0.684169285 | +0.000118042 | +0.035302857 |
| 2 | L05 | S | 2 | 0.703952798 | 0.746948615 | +0.004464624 | +0.001858375 |
| 0 | L05_SSL | S | 1 | 0.739482921 | 0.661703786 | +0.007442103 | -0.012584181 |
| 0 | L05_SSL | S | 2 | 0.653478188 | 0.746442584 | -0.013874330 | +0.000727286 |
| 0 | L05_SSL | L05 | 1 | 0.739482921 | 0.661703786 | +0.001123867 | -0.030192911 |
| 0 | L05_SSL | L05 | 2 | 0.653478188 | 0.746442584 | -0.040777541 | -0.018280831 |

CURRENT_CLASS_DIFFERENCES.csv includes macro differences as well. WORST_CLASS_DIFFERENCES.csv distinguishes the worst current class from the worst class over all seen-domain stages; it does not discard historical-domain losses. STAGE_DOMAIN_MATRIX.csv and PER_CLASS_METRICS.csv contain all 84 stage-domain rows, and REPLICATION_BY_SEED.csv retains F/H/N, BWT and domain-specific forgetting at full precision.

## Execution and memory audit

New formal successful updates are exactly 42,500: P1 5,300; P2 37,200; P3 0. Historical formal attempts remain 57,208, so all-attempt formal total is 99,708. No failed formal trajectory occurred in this run. Qualification/development cost is separate: 252 synthetic updates in the passing exact-source local/server suites, plus 378 in prior development or failed local qualification attempts, for 630 actual synthetic updates. No real diagnostic optimizer updates were introduced. TEST_REPORT.json preserves every attempt and its actual exit status.

The formal child-process span was 40.326 minutes. Logged GPU admission waiting totaled 1.325 seconds. Training-process durations and task wall spans follow; they include actual shared-server contention and are not controlled method-speed comparisons.

| Seed | Task | Training process minutes | Task wall minutes |
|---:|---|---:|---:|
| 0 | L05_SSL | 12.611 | 12.890 |
| 1 | L05 | 19.965 | 20.250 |
| 1 | S | 10.096 | 10.389 |
| 1 | common | 9.606 | 9.679 |
| 2 | L05 | 13.296 | 13.670 |
| 2 | S | 8.471 | 8.725 |
| 2 | common | 9.471 | 9.553 |

All S/common stages use one full model; all L05/L05_SSL stages use two. Maximum measured CUDA allocation/reservation was 938.905 / 1050 MiB for L05_SSL, 531.399 / 596 MiB for L05, and 488.845 / 570 MiB for S/common. Complete per-stage tensor bytes, CPU RSS, optimizer/GAS/forward counters, epoch time sums and child commands are in TRAINING_AND_MEMORY_ACCOUNTING.json. The inherited 195-byte inert prototype/support placeholder in label-only schemas is explicitly counted; it performs no prototype estimation.

L05_SSL opened only the authorized current U images (63 in stage1, 41 in stage2), and recorded exactly zero teacher-U forwards. It made 10,600 student weak/strong U forwards across its two stages; teacher forwards were labeled-only. S/L05 opened zero U images. Own-seed common and own-arm stage1 parent hashes, frozen teacher hashes, and one-student final deployment were checked by the frozen executor.

MECHANISM_DIAGNOSTICS.json retains raw/weighted U CE, labeled KD, class PAS counts, conflict/deleted-mass diagnostics, and logit/probability summaries every fifth epoch including endpoints. These are per-batch aggregates, not pooled pixel quantiles. Pseudo-label precision remains NOT_EVALUATED; optional extra val PAS precision forwards were not run, and high coverage is not precision. No train-U hidden GT was read for these diagnostics.

## Provenance, scope and stop

All actual training, qualification and frozen gate execution used 53075d055cc78a10b03e6b1da4ca2b436c9a1cf7. The server execution checkout remained clean at that exact source. Report commits only add public evidence under experiments/lcrseg/docs/l05_ssl_replication/. Branch codex/l05-ssl-factorial-replication starts at f2c7bd366cf9bdf80ae52c35798ece6be2e3a60d. The original source-generated report, terminal and gate remain on NAS without replacement; this public report expands their explanation.

Initialization, order, geometry, strong augmentation and stochastic classifier/GAS streams use optimization_seed separately from fixed data_split_seed=0. Seed0 random mapping parity and seeds1/2 actual differences were qualified. This is a fixed-patient-split training-randomness check, not a new patient split, external validation, statistical-significance claim or full JASCL/UniMatch reproduction.

**Final decision: stop this backbone/recipe family’s current small-grid development.** Neither L05 confirmation nor the SSL addition met its preregistered criterion. The current-class gains remain reported, but do not authorize a new name, recipe, projection repair, class-specific rule, extra seed, threshold change or parameter search. Old terminal states, locks, code, archives and results remain unchanged; no test/hidden GT, historical train replay, EMA/third model or main merge occurred.

NAS root: /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/l05_ssl_replication_20260908_01. Public artifacts contain code, frozen protocol, aggregate metrics/diagnostics and operational receipts. Weights, images, GT, patient identifiers, raw case outputs and credentials stay private. PUBLICATION_VERIFICATION.json records the result commit, anonymous access and create-only NAS archive verification.
