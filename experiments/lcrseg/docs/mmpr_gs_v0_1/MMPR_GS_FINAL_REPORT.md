# MMPR-GS v0.1 final feasibility report

**Scientific status: `FAIL_MATCHED_MASS_RANKING_NOT_SUPPORTED`.** Gate results: `{"F1": true, "F2": false, "F3": false, "F4": true, "F5": true}`. All scientific failures: `FAIL_MATCHED_MASS_RANKING_NOT_SUPPORTED, FAIL_RAW_GRADIENT_COMPATIBILITY`. No scientific failure was used to shorten the registered diagnostics or change the rules. The prototype pseudo-label selection line remains stopped pending independent review. This is a feasibility diagnostic, not evidence of segmentation performance or authorization to implement/train a method.

Registration: `MMPR_GS_V0_1_MASS_MATCHED_R3_RANKING`. Method name: Mass-Matched Multi-Prototype Ranking with Gradient-Safe Consistency (MMPR-GS-JASCL). Branch: `codex/mmpr-gs-v0-1-feasibility`.

## Prospective authority and unchanged old line

The unique latest published base containing `GATE1C_V3_COMPLETION_AND_ARCHIVE_REPORT.md` was `ad0e1d47b76165beccd118db41242a3025044504`. Separate commits were pushed and remote-verified in order: old-line closure `ac746fe8f1d335036ee35bb2f2f041aac3aae118`; preregistration `13494e175a2f5cd9a262c03c22d3ca45bfda7619`; authorization `639bf974383b8d11d490902ea4a7d73e4a89ba25`; exact execution code `bda0af8e25db492785ff09315b2722042e0174e0`; tested source/call-graph publication `a7d53869cca9847da9c95f6282e4c14e24d2e69c`. The source publication preceded the first private tensor/cache-array read and every real forward. The server stayed at the exact detached execution commit. No historical tracked file, result, preregistration or input bundle was changed.

Frozen B0 remains three regenerated seeds, fixed domain order, foreground Dice mean `0.617832663222`, exact old public mean, direct PAS banks in all nine stage-best checkpoints. K2 replication remains PASS: 18/18 foreground units improved, median R95 reduction 14.4944%, occupancy 100%, bootstrap cosine 0.99875748, 3/3 domains improved. Gate1B remains `FAIL_TRANSPORT_NOT_SUPPORTED`; no learned transport, R4 or drift-calibrated claim. Gate1C v3 remains `FAIL_IDENTITY_HISTORY_RELIABILITY_NOT_SUPPORTED`: R3 pixel C3/C4/C6 failed; class-balanced C3/C4/C5/C6 failed; R2 admission false; reduced candidate NONE; historical-bank claim false. Descriptive old R3 evidence remains AURC relative reduction 0.20030462949635253, 12/18 improving, 17/18 retention failures, R1 43/72 versus R3 39/72 negative cosines, RIM_ONE_r3 median worsening 0.1900009499703864. This new registration does not repair, rescue or reopen that line.

## Execution, budget and tests

| Phase | Launch UTC | Actual exit UTC | Child PID | Exit | Real forwards |
| --- | --- | --- | --- | --- | --- |
| input_audit | 2026-08-31T13:13:52.836586+00:00 | 2026-08-31T13:15:58.902002+00:00 | 2460906 | 0 | 0 |
| validation | 2026-08-31T13:17:07.932102+00:00 | 2026-08-31T13:34:56.733983+00:00 | 2462406 | 0 | 0 |
| integration | 2026-08-31T13:37:47.642495+00:00 | 2026-08-31T13:38:04.122183+00:00 | 2468078 | 0 | 15 |
| formal | 2026-08-31T13:38:33.822908+00:00 | 2026-08-31T13:41:23.763220+00:00 | 2468596 | 0 | 360 |
| final_audit | 2026-08-31T13:42:37.350164+00:00 | 2026-08-31T13:43:45.709327+00:00 | 2472628 | 0 | 0 |

Exactly **495 existing validation caches**, **3 new integration pairs** and **72 new formal pairs** completed. Validation used zero model forwards and generated no replacement input cache. Integration used GPU 4/5/6; formal used GPU 4/5/6/7, 18 pairs per GPU, without stopping unrelated processes. All controller and worker actual exits were 0. SSH success was never substituted for child-exit evidence.

Per pair: 3 native FP32 forwards + 2 same-Gaussian FP64 forwards, 3 native autograd calls + 6 FP64 autograd calls. Integration: 15 forwards; formal: 360; total **375/375**, native autograd **225**, FP64 autograd **450**. Synthetic test/compiler computations are separate and consumed zero real-input forwards. The compiled output counts were enforced: per pair 6 mass rows, 2 alignment rows, 12 block rows, 21 class-component rows, 7 retention rows, 21 precision-comparison rows and 1 model guard.

The exact execution source passed **326 tests: 246 unchanged related Gate1C regressions and 80 new tests; 0 failures, 0 errors, 0 skips; actual child exit 0**. JUnit SHA256: `1c45187470bf1eac53af5fed53c96be935dfa7fb63656935661fcabcd25d5c33`. The earlier 325-test development snapshot is preserved separately and was replaced before private inputs by the 326-test exact-source freeze. No failed real attempt was retried. Prospectively excluded old real-input/extra-forward integration and B0 optimizer/backward/overfit tests are listed in the preregistration and warnings; their historical files were not modified.

## Selection and gradient contracts

Q0 is the original joint strict `>0.7` student/teacher confidence and PAS-similarity mask from the direct checkpoint bank, with native draw0 bitwise parity. Q1 uses the original unrounded R3 only as a rank. Within each image and teacher-predicted class, EMA-active pixels receive exactly the original R1 active count. Ties use the frozen full SHA256 key, then coordinates only after a hash tie. Null pixels retain R1. No GT, ignore mask, cross-image/class redistribution, threshold search or count selection enters the builder. All 495 Q1/Q2 masks were sealed before the first validation GT read; GT255 is ignored only by the independent evaluator.

The loss is the original detached-teacher probability MSE, `sum(w*sum_c((p_s-stopgrad(p_t))**2))/(sum(w)+1e-12)`, with graph-connected zero mass and pixel normalization only. Supervised CE uses the fixed labeled-cycle batch. Native FP32 logits/features/probabilities/PAS masks match the old draw0 hashes; R1 diagnostic cosines match within 1e-10. Every original R3/R2 validation score was reproduced exactly from the frozen raw cache.

The entire student trainable parameter list is retained: all75 pairs have51 parameter tensors /484,016 elements, with active/inactive metadata and explicit zero substitution for None gradients, including sigma/GAS. Exact active counts and inactive names for each pair are in `MMPR_GS_EVIDENCE_CROSSCHECK.json`; full inventories and raw/full/class-component vectors remain in private pair evidence. Projection is exactly `g_u - min(0,g_s.T@g_u)/(g_s.T@g_s+1e-12)*g_s`, with no extra clipping/correction. The six registered blocks and per-class components decompose the same global projection. Individual block/class dots can remain negative; the registered nonnegative-dot gate is global. No block-specific projection or class-balanced loss is introduced.

Native/FP64 global comparisons all passed the frozen comparability requirement; minimum cosine `0.999999966238`, maximum relative L2 `0.000274606517875`. Full comparisons are in `mmpr_native_fp64_precision.csv`.

## Fixed F1–F5 adjudication

| Gate | Frozen requirement | Observed | Pass |
| --- | --- | --- | --- |
| F1 | Validation/formal image-class mass difference = 0 | 1485 + 432 rows; all 0 | True |
| F2 | Foreground macro relative error reduction >= 0.10 | 0.0780337103884 | False |
| F2 | Strictly improving foreground units >= 12/18 | 11/18 | False |
| F2 | Worst precision drop <= 0.02 | 0.102838216357 | False |
| F3 | Raw negative cosines <= 43/72 | 41/72; R1 43/72 | True |
| F3 | Global median cosine increase >= 0.05 | -0.0866852483267 | False |
| F3 | Every stage median worsening <= 0.05 | -0.199847110915, 0.188452165448, -0.171314576631 | False |
| F3 | Undefined required comparisons = 0 | 0 | True |
| F4 | Projected dot >= -1e-10 in 72/72 | 72/72; minimum -1.23773560845e-12 | True |
| F4 | Projected zero gradients = 0 | 0 | True |
| F4 | Global median / p10 norm ratio >= 0.50 / 0.20 | 0.974192715556 / 0.552465032349 | True |
| F4 | Each stage median norm ratio >= 0.40 | 1, 0.922632965508, 0.910248613467 | True |
| F4 | Every parameter/block value finite | True | True |
| F5 | Frozen models/banks, GT isolation, no gradients in .grad, zero updates | All 75 guards; old input bundle reverified | True |

Foreground macro errors are R1 `0.34132867065` and Q1 `0.314693528018`. The macro is the equal mean of exactly 18 class/domain/seed unit errors; each unit uses the preregistered equal-case stratum weights. Relative reduction is taken after this macro, not averaged from unit percentages. No unit was discarded. `4/18` units had precision drops above 0.02. Q2/Q3 cannot rescue Q1. Scientific failure precedence is F2, then F3, then F4; every failed gate is retained.

| Seed | Domain | Predicted class | R1 error | MMPR error | Relative reduction | Precision drop |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | REFUGE | 1 | 0.348947423422 | 0.410882148552 | -0.177490134536 | 0.0619347251291 |
| 0 | REFUGE | 2 | 0.348184674876 | 0.364228884322 | -0.0460795968445 | 0.0160442094457 |
| 0 | RIM_ONE_r3 | 1 | 0.350500337322 | 0.453338553679 | -0.293404043894 | 0.102838216357 |
| 0 | RIM_ONE_r3 | 2 | 0.349014607546 | 0.314869087008 | 0.0978340728439 | -0.0341455205383 |
| 0 | Drishti_GS | 1 | 0.681071653597 | 0.524669646829 | 0.229641045757 | -0.156402006768 |
| 0 | Drishti_GS | 2 | 0.0069003478123 | 0.0090957213007 | -0.318154033408 | 0.0021953734884 |
| 1 | REFUGE | 1 | 0.298215204064 | 0.328846958677 | -0.102716944662 | 0.0306317546131 |
| 1 | REFUGE | 2 | 0.245890729388 | 0.227785691367 | 0.0736304213913 | -0.0181050380211 |
| 1 | RIM_ONE_r3 | 1 | 0.31994509856 | 0.422256469974 | -0.319777899003 | 0.102311371414 |
| 1 | RIM_ONE_r3 | 2 | 0.63021633446 | 0.576586276239 | 0.0850978549572 | -0.0536300582216 |
| 1 | Drishti_GS | 1 | 0.356924752502 | 0.300801451198 | 0.157241269794 | -0.0561233013042 |
| 1 | Drishti_GS | 2 | 0.114973465339 | 0.0986345331878 | 0.142110460903 | -0.0163389321509 |
| 2 | REFUGE | 1 | 0.361165620833 | 0.230762174696 | 0.361062733038 | -0.130403446137 |
| 2 | REFUGE | 2 | 0.294551983257 | 0.11048826665 | 0.624893828829 | -0.184063716607 |
| 2 | RIM_ONE_r3 | 1 | 0.535335016152 | 0.515613790075 | 0.0368390362725 | -0.019721226078 |
| 2 | RIM_ONE_r3 | 2 | 0.489609649082 | 0.447199700766 | 0.0866199193479 | -0.0424099483154 |
| 2 | Drishti_GS | 1 | 0.222241292565 | 0.226243518137 | -0.0180084696489 | 0.00400222557189 |
| 2 | Drishti_GS | 2 | 0.19022788093 | 0.102180631661 | 0.462851443427 | -0.0880472492683 |

## Controls and stage distributions

| Candidate | Foreground macro error | Relative reduction vs R1 | Improving /18 | Worst precision drop |
| --- | --- | --- | --- | --- |
| Q0 | 0.34132867065 | 0 | 0 | 0 |
| Q1 | 0.314693528018 | 0.0780337103884 | 11 | 0.102838216357 |
| Q2 | 0.315782777404 | 0.0748425064838 | 11 | 0.0996700728356 |
| Q3 | 0.374923982729 | -0.0984251103618 | 6 | 0.400094831293 |

Q2 is matched mass using R2 and is a control only. Q3 is the original uncalibrated soft R3, descriptive only; its mass need not equal R1. Its metrics cannot be used as Q1 evidence. Case-wise changes, newly selected/removed precision, nulls, score ties and boundary/interior strata are fully reported in the linked CSV files.

`mmpr_selection_changes_by_foreground_unit.csv` consolidates these fields for exactly18 foreground units per candidate (Q1 and Q2). New/removed precision uses the same fixed case-stratum weights; all per-case source rows are preserved. For Q1, 1,094,265 foreground pixels are newly selected and exactly1,094,265 are deselected across the validation units; the one foreground null row retains its original R1 mask. `MMPR_GS_UNIT_SELECTION_DERIVATION.json` records source hashes and confirms that this report-only aggregation changes no selection, gradient or admission result.

| Stage/domain | Pairs | Raw median worsening vs R1 | Median projected/raw norm |
| --- | --- | --- | --- |
| 0: REFUGE | 24 | -0.199847110915 | 1 |
| 1: RIM_ONE_r3 | 24 | 0.188452165448 | 0.922632965508 |
| 2: Drishti_GS | 24 | -0.171314576631 | 0.910248613467 |

Full formal raw/projection distributions: `mmpr_gradient_alignment.csv`, `mmpr_gradient_blockwise.csv`, `mmpr_gradient_class_components.csv`, `mmpr_projection_retention.csv`. `mmpr_class_projection_summary.csv` adds per-class projected/raw norm ratios and the shared global activation flag using only these immutable tables; it does not change a run or recompute gradients. Integration tables are separately retained under `integration_tables/` and never counted as extra formal pairs. Full image/class mass is distinct from evaluator non-ignore mass.

## Isolation and immutable evidence

All 75 real pair model/checkpoint/bank/RNG guards passed. Teacher/bank gradients and all parameter `.grad` fields stayed None. Models and checkpoints are bitwise unchanged. Validation GT was evaluator-only; hidden training GT and test GT were unused. `model_optimizer_steps=0`, `transport_optimizer_steps=0`, `method_registered=false`, `training_launched=false`. No optimizer, backward, parameter.grad write, EMA/GAS/prototype update, teacher-noise phase, posterior, PoE, transport, replay or prototype inference occurred.

The original private bundle was fully verified before private arrays/tensors and reverified after diagnostics: **14,470 logical files / 17,712,127,650 bytes**; content SHA256 `8a82c7b8f0c72eb4faf619f51d7c1eae67a5f81059bc7f283b6b8df22d563526`; manifest SHA256 `480b627e0f63839ff5430d980020ca026c45838cf5eeb345f2b4cf7c4d578bb2`. All 2,962 frozen data checksum entries and all nine direct PAS checkpoints passed the input audit. The 72-pair identity hash is `d537e0f0fb7d44febf0d861a0384ce43f8c6d326cb5656e481b8882d79723958`.

## NAS archive and publication

New runtime evidence bundle: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/evidence_bundle`. Verified content-addressed integrity copy: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/archives/9a7312fef30b0787bf4b8bd89bb2646eba7e5e542186471798d194eb6f24201c`. Logical payload: **22,224 files / 4,813,748,985 bytes**. Content SHA256 `9a7312fef30b0787bf4b8bd89bb2646eba7e5e542186471798d194eb6f24201c`; manifest SHA256 `be64176a22efc2d7576d205a90834d2db1f606e796f3fb39edb50bb38f9a018c`. Every file and byte was SHA-verified before atomic promotion and the source bundle was verified again and retained. All five phase trees, both completed test snapshots, exact source bundle and operational records are included. Original 17.7 GB inputs are bound by digest and remain untouched, not duplicated into the new bundle.

All new arrays, logs, scratch, source-transfer bundles and archives are on NAS. The separate integrity copy uses separate files on the **same NAS**, and is not an independent-device disaster-recovery backup. Public Git contains source and text diagnostics/receipts only; no private checkpoint, image, cache or gradient array. Archive-operation receipts and later publication documents are external to the runtime bundle manifest to avoid self-reference and are indexed by the public artifact manifest/publication receipt.

See `MMPR_GS_ARTIFACT_MANIFEST.json`, `MMPR_GS_PRIVATE_ARCHIVE_AUDIT.json`, `MMPR_GS_PUBLIC_EVIDENCE_INDEX.json`, `MMPR_GS_EVIDENCE_CROSSCHECK.json`, `MMPR_GS_EXACT_COMMANDS.md`, and `MMPR_GS_PUBLICATION_RECEIPT.json` for byte hashes, exact commands and remote publication proof. The publication receipt is added only after the report commit has been pushed and remote SHA verified.

## Hard stop

Stop for independent review after publishing this branch. No C0 regeneration, online refresh, MMPR-GS training-method implementation, performance training, Gate2, Prostate, MnMS, sweep or main merge is authorized. The prototype pseudo-label selection line stays stopped. Relation-consolidation or any other method needs another independent preregistration. No follow-on process or automation is scheduled.
