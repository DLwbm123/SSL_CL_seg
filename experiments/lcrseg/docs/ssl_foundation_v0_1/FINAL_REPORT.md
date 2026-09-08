# Single-domain SSL foundation V0.1 — final report

## 1. Engineering and finite execution

**Engineering COMPLETE; SSL_FOUNDATION_NOT_ESTABLISHED.** All 12 prescribed domain-arm tasks completed 100 epochs: 31,800 formal Adam updates, 60 fixed diagnostic snapshots and 12 final-student deployments. All 24 train/deploy children exited zero. No failed formal trajectories occurred. No candidate passed screening; seeds12/13 were NOT_ADMITTED and were not run. Prior reported formal attempts 99,708 plus this round give 131,508. Separately, qualifications consumed 815 successful synthetic updates across all attempts and 12 real diagnostic updates (6/domain). Failed development-test evidence remains preserved.

Exact execution source: `d9542bd6f7b4acfe351bdaa8c8107bb875d40717`; branch `codex/single-domain-ssl-foundation-v0-1`; base `6eb48fd845f6192698c4d39893b2524447afb3df`. The server checkout remained clean at the locally/CUDA-qualified commit. NAS run: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/ssl_foundation_v0_1_20260908_01`. Existing weights, historical locks/results, hidden-U GT, test files and old formal_03 were not used or changed. Main was not merged.

Verified train/deploy wall span: 42.98 minutes on GPUs4/5. Task time including inline diagnostics summed to 72.51 minutes; diagnostic time summed to 6.31 minutes. Shared-server load prevents controlled speed claims. Subsequent GPU6/7 authorization was received after the queue reached its terminal state; no scientific task remained to place there.

RIM input counts were 16 labeled/63 U/40 val; Drishti 10/41/25, total 195 authorized records, split seed0 unchanged. All final val cases had evaluable GT, so evaluable-only sensitivity equals the main metrics. Initial hashes matched across all six arms within each domain; same-GAS warm-up states and full labeled presentation hashes matched. SUP and every warm-up opened zero U images. No smoke state initialized formal training.

## 2. Final student and fixed screening

Dice is the patient equal mean of rim/cup Dice; Q averages the two domains equally. Rounded table values are presentation only; decisions used full-precision CSV.

| Arm | RIM | Drishti | Q student | Q gain vs same-GAS SUP |
|---|---:|---:|---:|---:|
| SUP_G0 | 0.704764334 | 0.687317276 | 0.696040805 | +0.000000000 |
| MT_CONF_G0 | 0.706411344 | 0.677526168 | 0.691968756 | -0.004072049 |
| MT_PAS_G0 | 0.708373308 | 0.686671069 | 0.697522189 | +0.001481384 |
| SUP_G1 | 0.700038946 | 0.710690158 | 0.705364552 | +0.000000000 |
| MT_CONF_G1 | 0.696490978 | 0.702373105 | 0.699432042 | -0.005932510 |
| MT_PAS_G1 | 0.701293567 | 0.723369809 | 0.712331688 | +0.006967136 |

The strongest candidate **MT_PAS_G1** improved RIM by +0.001254621 and Drishti by +0.012679651. Its mean gain +0.006967136 failed the prespecified >=0.010 requirement; every other screen condition passed, including improvement over the stronger SUP. No threshold was lowered. MT_CONF_G0/G1 additionally failed the Drishti -0.005 domain guard. All four candidates passed the rim/cup -0.020 guard. SCREEN_DECISION.json contains every full-precision check.

CONFIRMATION_BY_SEED.csv records NOT_ADMITTED. There are no seed12/13 observations or replication claims.

## 3. EMA versus final student

| Arm | Q student | Q EMA | EMA minus student |
|---|---:|---:|---:|
| SUP_G0 | 0.696040805 | 0.696566166 | +0.000525362 |
| MT_CONF_G0 | 0.691968756 | 0.693727108 | +0.001758352 |
| MT_PAS_G0 | 0.697522189 | 0.698612586 | +0.001090397 |
| SUP_G1 | 0.705364552 | 0.710742584 | +0.005378032 |
| MT_CONF_G1 | 0.699432042 | 0.699613312 | +0.000181270 |
| MT_PAS_G1 | 0.712331688 | 0.712861926 | +0.000530238 |

EMA slightly increased every final Q; the largest difference was SUP_G1 +0.005378032. MT_PAS_G1 has a positive student development signal, so its direction is not solely an EMA effect, but it still failed the primary screen. Its EMA advantage over SUP_G1 EMA was only +0.002119342. No primary output was switched to EMA or a best intermediate epoch.

## 4. PAS quality, PAS Dice increment and GAS

At seed11, MT_PAS minus MT_CONF in Q was +0.005553433 under G0 and +0.012899646 under G1; each domain difference was positive. These are descriptive component signals. New-seed PAS increment remains NOT_CONFIRMED_NEW_SEEDS because no P2 candidate was admitted.

Mandatory precision, prediction coverage, image coverage and accepted-correct recall were evaluated for all three classes, both label sources, both modes, raw/confidence/PAS and all five snapshots. PSEUDO_QUALITY_BY_CLASS.csv and PSEUDO_QUALITY_AGGREGATE.csv expose the same complete class-level aggregate view (2,352 rows). Private per-case/per-draw rows remain on NAS. Four diagnostic draws are averaged within patient first; pooled counts are explicitly draw-pixels, not independent patients. Defined denominators and undefined support are retained.

At epoch100, MT_PAS_G1 teacher training-like diagnostics with the actual training library gave RIM rim precision/coverage/recall 0.751583/0.655663/0.476599 and cup 0.774874/0.489483/0.388145. Drishti rim gave 0.794709/0.644331/0.454369 and cup 0.724013/0.908889/0.867407. These measured tradeoffs are not safety guarantees or proof of a unique failure mechanism.

For the same MT_PAS_G1 network with fresh diagnostic prototypes, Drishti rim precision rose from confidence-only 0.7703 to PAS 0.7836, while coverage dropped from 0.8821 to 0.6032 and recall from 0.6092 to 0.4144. RIM rim precision instead fell from 0.7577 to 0.7516. PAS did not uniformly improve precision. Higher precision obtained by rejecting pixels does not substitute for Dice improvement. PAS_COVERAGE_PRECISION.md supplies all actual-library classes and detailed within-network filtering contrasts.

| G1 minus G0, final student | RIM | Drishti | Q |
|---|---:|---:|---:|
| SUP | -0.004725387 | +0.023372882 | +0.009323747 |
| MT_CONF | -0.009920366 | +0.024846937 | +0.007463286 |
| MT_PAS | -0.007079741 | +0.036698739 | +0.014809499 |

GAS improved Drishti and worsened RIM for all three objectives at this seed. GAS-by-SSL Q interaction was -0.001860462 for CONF and +0.005485752 for PAS. G0 still uses official normalization, temperature and convolution geometry. This is not a comparison with a truly linear head or a judgment of the full paper theory.

Training used two models; deployment one. Each model has 1,936,064 parameter bytes; the unused sigma has 1,728 bytes and no gradient. Peak allocated memory was approximately 459.10 MiB for SUP and 909.31 MiB for MT. MEMORY_RUNTIME_AND_ACCESS.json includes measured calls, GAS gradients, EMA updates, prototype/diagnostic costs and separate qualification accounting.

## 5. Evidence boundary and stop

This fixed adaptation did not establish the specified SSL foundation despite a positive MT_PAS_G1 development direction. It closes this recipe on the previously exposed fixed patient split, not general SSL. There is no independent-patient confirmation, CL sequence, forgetting estimate or SOTA claim. No loss, parameter, threshold, backbone, classifier temperature, label fraction or seed was changed after validation feedback.

Do not restore L05/SCD, append seeds, lower the gate or start frozen-layer/CL work. NEXT_STAGE_DRAFT.md requires a separately authorized prospective question. Public release includes source, protocol, aggregate metrics and audit receipts; model weights, images, GT, identities and private predictions remain on NAS. The exact report commit and anonymous-access/archive checks are recorded in PUBLICATION_VERIFICATION.json. Public HTTP availability does not constitute independent scientific review.
