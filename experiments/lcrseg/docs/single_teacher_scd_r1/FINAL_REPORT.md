# Single-teacher R1: completed fixed experiment

All four new arms completed both stages, final seen-domain validation, and single-student deployment. Each arm made exactly 3,200 + 2,100 = 5,300 formal optimizer updates. The frozen report executor independently reconciled contiguous step logs, final receipts, child exit codes, teacher hashes, own-arm stage2 lineage, and deployment student hashes. There is no combined scientific PASS.

| Question | Frozen outcome |
|---|---|
| Explicit pseudo-label CE, old D versus U0 | PSEUDO_CE_DEVELOPMENT_SIGNAL |
| Added U-KD, U0 versus both L05 and L10 | UNLABELED_KD_CONTRIBUTION_NOT_ESTABLISHED |
| SCD projection, E_R1 versus best fixed simple arm | SCD_INCREMENTAL_VALUE_NOT_ESTABLISHED |
| Prospective current-capability protection | Only L05: FIXED_RECIPE_CANDIDATE_FOR_REPLICATION |
| E numerical engineering | Qualified once; full fresh 5,300-step trajectory and deployment completed |
| Historical V0.1 / original B-S SSL support | INCOMPLETE_TRAINING_MATRIX preserved; old descriptive support conditions remain unmet |

## Scores and class costs

F is final three-domain mean; H is final two-history-domain mean; N is the mean score when entering the two incremental domains. Scores use the frozen per-case rim/cup macro Dice on val only. Tables below round for reading; CSV files preserve full precision and GATE_ACCOUNTING.csv records every threshold comparison.

| Arm | F | H | N | BWT | Forget REFUGE | Forget RIM |
|---|---:|---:|---:|---:|---:|---:|
| S | 0.585957181 | 0.525668818 | 0.704849150 | -0.247356877 | 0.215889510 | 0.278824244 |
| A | 0.573670633 | 0.503046191 | 0.709291739 | -0.270229288 | 0.197384175 | 0.343074402 |
| B | 0.503730021 | 0.417375901 | 0.692086118 | -0.357934587 | 0.281903295 | 0.433965878 |
| C | 0.684184175 | 0.747438542 | 0.630016095 | -0.025183331 | 0.042937733 | 0.007428929 |
| D | 0.718089435 | 0.742590645 | 0.698910789 | -0.043220135 | 0.029089197 | 0.057351073 |
| U0 | 0.676784120 | 0.698257901 | 0.677446015 | -0.083713334 | 0.107531423 | 0.059895244 |
| L05 | 0.620432366 | 0.565903763 | 0.722308724 | -0.213103674 | 0.198432948 | 0.227774400 |
| L10 | 0.571301651 | 0.504197862 | 0.709716209 | -0.274207232 | 0.257976153 | 0.290438311 |
| E_R1 | 0.687037997 | 0.704006347 | 0.692652881 | -0.083539385 | 0.053562962 | 0.113515809 |

| Arm | Stage | Current rim | Current cup | Rim minus S | Cup minus S |
|---|---:|---:|---:|---:|---:|
| S | 1 | 0.732040818 | 0.674287966 | 0.000000000 | 0.000000000 |
| S | 2 | 0.667352519 | 0.745715298 | 0.000000000 | 0.000000000 |
| D | 1 | 0.754808275 | 0.702660849 | 0.022767458 | 0.028372883 |
| D | 2 | 0.703069330 | 0.635104701 | 0.035716811 | -0.110610597 |
| U0 | 1 | 0.737606779 | 0.704504161 | 0.005565961 | 0.030216195 |
| U0 | 2 | 0.613244215 | 0.654428904 | -0.054108304 | -0.091286394 |
| L05 | 1 | 0.738359054 | 0.691896697 | 0.006318236 | 0.017608730 |
| L05 | 2 | 0.694255730 | 0.764723415 | 0.026903211 | 0.019008117 |
| L10 | 1 | 0.734227722 | 0.693618658 | 0.002186904 | 0.019330692 |
| L10 | 2 | 0.664641052 | 0.746377405 | -0.002711467 | 0.000662107 |
| E_R1 | 1 | 0.756034941 | 0.708373993 | 0.023994124 | 0.034086027 |
| E_R1 | 2 | 0.701622811 | 0.604579780 | 0.034270292 | -0.141135518 |

## Separate scientific judgments

**Pseudo-CE:** D minus U0 gives F +0.041305315121, H +0.044332744701, N +0.021464774010. All frozen class guards pass, but stage2 cup changes by -0.019324202899, only 0.000675797101 above the -0.02 guard. This is a marginal development signal under this recipe, not evidence that all semi-supervised mechanisms help. U0 still uses PAS references and U-KD. D itself fails the separate current-capability guard versus S.

**U-KD:** U0 exceeds the better label-only F by 0.056351754119 and preserves more historical performance, but N is lower than L05 by 0.044862709112 and L10 by 0.032270194444. Stage2 rim/cup are lower than both controls; versus L05 the losses are 0.081011514815 / 0.110294511506. The combined frozen contribution condition is unmet. This does not isolate every representation, regularization, or data-volume mechanism causally.

**E increment:** E_R1 F is 0.031051438560 below the best simple comparator D, against a required +0.01 margin. H exceeds B by 0.286630446713, but stage2 macro and cup versus B lose 0.023336965318 / 0.142187923741. Stage2 cup versus S loses 0.141135517601. A successfully repaired solver is not evidence that SCD is worth its extra complexity.

**Prospective candidate:** L05 alone passes all fixed F/current-macro/current-rim/current-cup guards versus S. Its F improvement is 0.034475184550; every current class score improves. L10 protects current classes but fails the F margin. This is a single-seed fixed-development-split screening result, not replication, training-seed uncertainty, or SOTA evidence. No seed1/2 is started.

**Old SSL support:** original B minus S remains F -0.082227160806 and N -0.012763031687. Neither the new ablations nor repaired E changes the old terminal or old B support finding.

## Numerical, mechanism, and memory evidence

The single solver revision uses an analytic feasible bracket and fixed 64 log1p-coordinate bisections, with the unchanged 1e-9 residual requirement. The old max-abs normalization was already present. Compensated shifted-moment evaluation preserves the original p construction; its observed difference from cancellation-prone evaluation is documented in NUMERIC_EQUIVALENCE.md. No failed-sample drop, r=p fallback, KD disabling, or second numerical repair occurred.

All 2,064,384 captured actual projector inputs were checked, including all 8,423 saved failed pixels and previously unaffected inputs. CPU/GPU maximum target difference was 5.662137425588298e-15; unaffected old-solver difference was 2.942091015256665e-14. Independent 100-digit references included the synthetic witness/boundaries and 24 deterministic private failed vectors. The unexecuted U call after the original failing L call was not claimed as captured.

- E_R1 stage1: 1,129,611,876 analytic pixel solves; 0 exact boundary solves; 72,295,160,064 actual pixel bisections; maximum residual 1.620925615952729e-14; logged solver/diagnostic time 1034.710 s.
- E_R1 stage2: 967,689,802 analytic pixel solves; 0 exact boundary solves; 61,932,147,328 actual pixel bisections; maximum residual 1.643130076445232e-14; logged solver/diagnostic time 2550.349 s.

Full-trajectory counts include repeated pixel occurrences across steps and are not unique patient or image counts. Boundary and unchanged targets perform zero bisections; the raw class iterations field is the configured 64. Synthetic qualification exercised the exact boundary even though neither formal E stage entered it.

EPOCH_CURVES.csv retains supervised/raw Lu/weighted Lu/L-KD/U-KD and actual/nominal coefficients. MECHANISM_EVERY_5_EPOCHS.csv has 720 class/branch aggregates; NUMERIC_TRAJECTORY.json retains per-batch logit and probability summaries every fifth epoch, including stage endpoints. These quantiles are not pooled pixel quantiles. First-step unweighted branch gradient probes and full-trajectory solver costs are in FULL_TRAJECTORY_SOLVER_AND_GRADIENT_AUDIT.json. Raw Lu is zero at the prescribed initial warm-up, so that first-step probe alone is not evidence about later SSL gradient size.

Original B/C/D aggregate PAS/KD/loss/gradient evidence is preserved in PRIOR_MECHANISM_REVIEW.json. Pseudo-label precision remains NOT_EVALUATED: coverage is not precision, and no hidden train-U GT was opened. No additional post-hoc val precision forward was performed.

L05/L10 receipts each confirm zero U images opened, no PAS calls or prototype estimation. The inherited checkpoint schema retains an inert 195-byte zero prototype/support placeholder; these bytes remain counted. U0 retains weak/strong U computation and raw Lu, with actual_lambda_u exactly zero. All stages report max_full_models=2; final deployment checks report one student with the teacher unavailable.

| Arm | Stage | Epoch time sum (s) | CUDA allocated peak (MiB) | CUDA reserved peak (MiB) | CPU peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| U0 | 1 | 489.874 | 946.248 | 1068.000 | 1477.676 |
| U0 | 2 | 324.807 | 946.248 | 1068.000 | 1486.312 |
| L05 | 1 | 273.106 | 531.118 | 596.000 | 1445.016 |
| L05 | 2 | 183.147 | 531.118 | 596.000 | 1447.391 |
| L10 | 1 | 273.548 | 531.118 | 596.000 | 1445.539 |
| L10 | 2 | 185.082 | 531.118 | 596.000 | 1454.168 |
| E_R1 | 1 | 1565.716 | 946.248 | 1086.000 | 1486.961 |
| E_R1 | 2 | 3480.342 | 946.248 | 1084.000 | 1481.328 |

Logical target tensor sizes may contain aliases; they are separate from measured CUDA peaks. Student and predecessor teacher each contain 1,936,064 parameter bytes. Full counters, optimizer/gradient/buffer bytes, teacher hashes, and deployment receipts are in MEMORY_AND_DEPLOYMENT.json and STAGE_LINEAGE.json.

## Execution cost and stop

New formal updates: U0/L05/L10 15,900; E_R1 5,300; total 21,200. Historical formal attempts remain 36,008, so all-attempt formal cost is 57,208. The reused common plus old five complete arms plus new E comparison is an effective 39,800-update matrix, not historical compute cost. All nine complete recipes with common counted once total 55,700 effective updates. The 1,508 old successful E updates remain separate failed-attempt evidence.

Engineering cost is separate: 224 exact-source synthetic updates locally plus 224 on the server; 376 earlier synthetic development updates (including one failed test fixture attempt); and 3 successful real diagnostic replay updates, within the cap of 8. No common or old completed arm was retrained. No new real two-case overfit run is claimed; the R1 qualification scope and reused V0.1 evidence are explicit in QUALIFICATION.md.

The four formal lanes occupied 84.706 minutes of overlapping wall time from 2026-09-07T13:42:34.041745+00:00 to 2026-09-07T15:07:16.383245+00:00. Per-arm train-process time / total parent span in minutes:

- U0: 13.741 / 13.979.
- L05: 7.786 / 8.042.
- L10: 7.821 / 8.071.
- E_R1: 84.252 / 84.536.

E encountered changing co-resident GPU load from late stage1 onward. Actual runtime is not a controlled method-speed benchmark; other processes were not terminated and the recipe/source was not changed. Qualification timings are separately recorded in LOCAL_QUALIFICATION.json and SERVER_QUALIFICATION.json.

**Stopped:** all fixed arms, val matrices, class costs, numerical and memory audits, and single-student deployments are complete. No further solver attempt, tuning, external data, seed1/2, test/hidden GT, history replay, EMA/third full model, or main merge is authorized or started. NEXT_ACTION.md contains only the L05 fixed-recipe replication candidate and requires separate future authorization.

## Provenance and public boundary

Execution and all exact-source qualifications: f94be07a8a3db1c062035a9bc2e542af9d4b156b. Branch: codex/single-teacher-r1-effectiveness-ablation, based on bd29494153aa0175b5b52d360f87a44255293d9f. Later commits add reporting evidence only; the remote execution checkout stayed at f94be07.

Old common/S/A/B/C/D source: 057ce07fa7bd02f8319e0c2bdd9d4adbceff23d8; old failed E: 270e23985c8d78d0508fe6c4d43150f6ebffcc92; old report: 3109e5a9d9dec2d75a4135b2f5c5b1293c99aa43. Old results and locks are unchanged. The original frozen-executor adjudication and raw generated report are retained on NAS.

NAS root: /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_r1_20260908_01. Public release contains source, fixed contracts, aggregate metrics/diagnostics and small operational receipts. Images, GT, patient identifiers, weights, and private pixel vectors remain excluded. PUBLICATION_VERIFICATION records public access and create-only NAS publication archive checks.
