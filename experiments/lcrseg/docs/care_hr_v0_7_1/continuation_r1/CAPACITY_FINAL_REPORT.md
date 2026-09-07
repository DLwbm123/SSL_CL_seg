# CARe-HR V0.7.1 R1 final report

**Terminal state: INCOMPLETE_EVALUATION. Reason: FROZEN_EXECUTOR_ASSET_PATH_BINDING_ERROR.**

This continuation actually executed A1 and the real A2 blind phase, then launched the isolated evaluator. The evaluator stopped before successfully opening its first GT file. No baseline parity result or scientific oracle result exists. This is an R1 implementation failure, not a scoring dispute, scientific capacity failure, or demonstrated absence of the frozen dataset. No code was repaired or rerun after domain access.

## Version and authorization

- Branch: `codex/care-hr-v0-7-1-capacity-audit` in `DLwbm123/SSL_CL_seg`.
- Predecessor report: `2f28c63d6dedb057fce96f131038f1889804ac10`; its BLOCKED report and all evidence remain unchanged.
- New tested source: `9bbbacd25f3c3abf885205009eb14d698b36f32b`.
- A1 evidence published before data access: `13ccd8f46392a144862146b2423c1286e8d124cf`.
- Blind seal published before evaluator admission: `d4d8ca318a8eb0631b5cc3146b7e50f1b3ba1af3`.
- Source, qualification and seal were verified on the branch and anonymously on GitHub before evaluator admission. The final report commit is the commit containing this report; it is distinct from tested source.

## A0/A1 and the missed deployment boundary

The adopted scorer implements the resolved V0.6B `shor_v0_4_test.case_metrics` semantics. Both local and server exact-source runs returned **161 passed, 0 failed, 0 errors, 0 skipped**: 95 unchanged predecessor regressions and 66 new-namespace tests. All 13 requested A1 coverage items passed. The exhaustive two-pixel parity check covers 1296 combinations within one pytest test; it is not 1296 tests. There are no unresolved scoring semantics.

The suite includes complete-action int64 confusion parity, the nonadditive harm witness, all four oracle selectors, exact ties/SAFE0, budgets, patient cross-seed/own-seed guards, projected CSV access, empty-support sensitivity, and generated-HDF5 full evaluator success/failure rehearsals. It also verifies 99 protected files, including all 82 original protections and every predecessor document.

**The synthetic HDF5 fixtures used the same erroneous directory layout as the new executor. They therefore failed to test compatibility with the inherited `safe_asset` binding. Passing A1 did not establish that deployment boundary.** Existing `di_dmpa_gate1.binding.safe_asset` resolves assets beneath `DATA/h5/v1`; the R1 executor incorrectly used `DATA/relative`. The error belongs to this implementation.

## Completed real blind work

The original PPC-SHOR V0.6B reservation selected `formal_02`; its manifest, row order, checkpoints, nine probability caches and frozen controls/routes verified against registered hashes. The cohort contains **198 seed-case rows, 177 unique cases/patients, 66 rows per seed**, all own-seed `train_labeled`. C6 routes 155 rows to history and 43 to current. No old `formal_03` or forbidden own-seed GT was read.

| Blind quantity | Actual count |
| --- | ---: |
| Proposals retained | 949 |
| Strict O_CAP actions, including one no-op per row | 15,796 |
| O_NO_AREA actions | 26,374 |
| O_FREE_SUBSET actions | 159,232 |
| Largest per-row strict/free space | 597 / 8,191 |
| Strict spaces containing only no-op | 67 / 198 |
| Free spaces containing only no-op | 43 / 198 |
| Zero-current-foreground rows | 0 |

These are prediction-only enumeration counts, not scored oracle results. The seal binds probabilities by original cache hashes, hard/blended predictions, frozen routes, proposal masks, action indices, lambdas and budget flags. Seal SHA-256: `9b186f8c4e80236987700fcb74cf90a5b41071b63df128a5b1e28142df7141a4`.

## Evaluator failure and precise access accounting

The evaluator verified the seal and atomically created its reservation, then materialized only the 66 admitted seed-0 metadata rows. Its first label-file open raised `FileNotFoundError` because the executor omitted `h5/v1`. The file at the inherited binding was subsequently checked using **stat only**: it exists and is 12,489 bytes. Its contents were not opened or verified, and the remaining labels were not inspected. We do not claim that frozen GT is missing or that all GT is available.

| Access/result quantity | Actual value |
| --- | ---: |
| Failed GT file-open attempts | 1 |
| Successful real GT payload reads/decodes | 0 |
| True-domain records materialized in projected dataframe | 66 |
| True-domain rows processed before failure | 1 |
| Baseline case comparisons completed | 0 |
| Oracle rows scored | 0 |
| Actual evaluator child exit code | 1 |

Raw `RUNTIME_COUNTERS.json` is preserved: `GT_reads=1` was incremented before the failed open, and `domain_reads=1` counts loop-processed rows rather than all dataframe materialization. Raw status `rows=1, patients=1` is the first revealed bookkeeping entry, **not** a completed evaluated cohort. `EXECUTION_AUDIT.json` records these limitations explicitly; no successful-read claim is inferred from the raw counter names.

The separate parent receipts record actual child exits: A1=0, blind=0, evaluator=1. SSH status is not substituted for them. The exact case-bearing traceback and operational paths remain private on NAS; the public audit provides a hash and redacted cause.

## Scientific results

| Policy / diagnostic | R1 result |
| --- | --- |
| Current, frozen Ridge hard, frozen SHOR, frozen PPC C6 | NOT_EVALUATED; baseline reproduction incomplete |
| O_CAP at 0.50, 0.75 and envelope | NOT_EVALUATED |
| O_SAFE0 at 0.50, 0.75 and envelope | NOT_EVALUATED |
| O_NO_AREA at 0.50, 0.75 and envelope | NOT_EVALUATED |
| O_FREE_SUBSET at 0.50, 0.75 and envelope | NOT_EVALUATED |
| Gain/harm Pareto, selected change areas and selected no-op frequency | NOT_EVALUATED |
| Evaluable-only and patient-equal sensitivity | NOT_EVALUATED |
| Actual all-ignore rows/patients, valid/ignored pixel counts | UNKNOWN; no GT decoded |
| Bootstrap/refit stability and feasibility | NOT_EVALUATED; no real fits |

The historical control values 0.6016731401004847, 0.8156293208569301, 0.8052579174291238 and 0.802187012880861 remain **historical references**, not this run's measured values. All original draft gates are retained with NOT_EVALUATED status. No PASS/FAIL capacity decision is justified. Missing metrics are blank with explicit status, not fabricated zeros.

## Cost, preservation and stop

Nine probability caches totaling 1,051,067,520 bytes were reused, representing 594 frozen sample-expert outputs. **Actual new sample-expert forwards=0; batch forwards=0; real Ridge/PAV/temperature/risk/router/conformal fits=0; segmentation/optimizer/EMA/GAS/prototype updates=0.** One frozen checkpoint was loaded solely for tensor/parameter cost accounting. Each expert has 484,016 parameters; nine snapshots total 4,356,144.

Blind execution took 46.879 seconds with peak RSS 1,479,960 KiB; evaluator execution took 1.855 seconds with peak RSS 112,744 KiB. New private artifacts occupied 90,351,191 bytes at measurement, excluding the source bundle and A1 scratch. No GPU forward was performed. Outputs remain in the create-only NAS protocol directory `protocols/care_hr_v0_7_1_r1_20260907_01/`.

All 99 protected files remain unchanged locally and on the server. The tested server checkout remains clean at the source SHA above. The old V0.7 REVIEW_LOCK, old implementations/tests and historical states were not altered. Main remains `46e892960240543c946c570a9378d409b226384b`; it was not merged. Public delivery excludes GT, images, model weights, probabilities, private case identities and raw private per-case artifacts.

**Execution is stopped.** `NEXT_STAGE_DRAFT.md` describes a separately authorized engineering continuation only. No risk-model fitting, router fitting, new scoring experiment or automatic rerun was started.
