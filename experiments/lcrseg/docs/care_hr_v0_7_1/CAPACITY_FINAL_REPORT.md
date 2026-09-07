# CARe-HR V0.7.1 capacity audit: blocked preflight

**Status: BLOCKED_EVALUATOR_SEMANTICS_MISMATCH.** This is an engineering/protocol stop, not a scientific FAIL. A0 and A1 are partial; A2 and A3 were not started. No capacity conclusion is available.

Source commit: `c98a4e8e46d133d1b51360410c43873a2c283de0`. Report commit: the commit introducing this file, separately verified in the delivery response; the report HEAD is not the execution source. Branch: `codex/care-hr-v0-7-1-capacity-audit`. Immutable base: `61c1e302fae515fe51adf4e777887a1a489859be`.

## Blocking evidence

The adopted attachment section 2 explicitly requires a stop when historical scoring rules disagree. Frozen PPC V0.6B calls `shor_v0_4_test.case_metrics`, which excludes GT=255 and assigns Dice=1 to an empty class, including all-invalid images. Old CARe-HR V0.7's `targets._dice` does not exclude GT=255. A concrete synthetic example current=[1,0], revised=[1,1], truth=[1,255] gives frozen scorer gain=0 but V0.7 harm=1/3. With truth=[255,255], current=[1,1], revised=[0,0], V0.7 reports rim gain=1 while the frozen scorer reports 0. This is additional to the authorized union-versus-macro naming correction. `EVALUATOR_SEMANTICS_AUDIT.json` contains exact source hashes and witnesses. No resolution has been supplied, and no new scoring convention was silently selected.

Server authentication was resolved with the user's supplied login. Read-only checks confirmed the expected host, both authorized GPUs 4/5 idle, NAS mounted and the legacy runs symlink intact. The synthetic server run used the existing Python 3.10.6 / Torch 2.2.1 environment. Missing access is no longer the blocker. Private inputs were not inspected after the semantic gate failed; their presence or absence is not asserted.

## Completed and tested

New namespace contains exact patient conformal rank/infinite-bound handling, aligned finite vector guards, strict blind proposal validation, inherited deterministic proposals and probability blending, exact subset enumeration under the original and two relaxed budgets, no-op and no zero-foreground floor, and a reproducible semantic preflight audit. No new scoring evaluator, fitted router, risk model, or scored capacity oracle was installed.

Both local and server exact-source suites: **129 passed, 0 failed, 0 errors, 0 skipped** (34 new preflight tests plus 95 unchanged predecessor regressions). Server parent child exit: **0**. The separate synthetic semantic audit exited **2**, deliberately reporting the blocker. The 82 protected historical files match their SHA-256 inventory; inherited historical-diff checks and the old V0.7 lock remain intact.

**This is not complete A1 coverage.** Final combined-action scoring, non-additive harm examples, oracle versus brute-force scoring parity, metric-monotonicity checks, action-conditioned targets and real cross-seed lineage integration remain incomplete. Every requested A1 item is separately accounted for in `SYNTHETIC_TEST_REPORT.json`; no skipped test count is used to conceal unimplemented items.

The retained predecessor tests perform synthetic-only fits. Each final instrumented suite recorded 94 Ridge entry calls, 93 successful Ridge returns (one invalid-input rejection), plus two old calibrator calls and one wrapper call; no PAV fit was invoked on those particular fixtures. Nested API calls are not summed as independent fits. New conformal calculations are numerical residual-order tests, not real calibration. Development checks before the instrumented final runs also used synthetic-only inputs; their fitting calls were not instrumented and no total across every exploratory invocation is claimed.

## Real execution and oracle results

Real rows/patients read: **0/0**. Image/probability-cache reads, checkpoint loads, new or reused sample-expert forwards, batch forwards, segmentation GT reads, true-domain reads, real router/risk/conformal fits, segmentation updates and old formal_03 reads: **all 0**. Only source, protocol, storage metadata and synthetic test artifacts were accessed. No ACTION_SPACE_SEAL was created and GT admission stayed false.

| Strategy | This-round result |
|---|---|
| current B0 stage2 | NOT_RECOMPUTED |
| frozen Ridge hard | NOT_RECOMPUTED |
| frozen SHOR | NOT_RECOMPUTED |
| frozen PPC C6 | NOT_RECOMPUTED |
| O_CAP, separate lambdas and envelope | NOT_EVALUATED |
| O_SAFE0, separate lambdas and envelope | NOT_EVALUATED |
| O_NO_AREA | NOT_EVALUATED |
| O_FREE_SUBSET | NOT_EVALUATED |

Historical public references (not this-round results) are current 0.6016731401004847, Ridge hard 0.8156293208569301, SHOR 0.8052579174291238, PPC 0.802187012880861. They have not passed this round's per-case baseline-consistency gate. Their provenance and expected 198-row/177-patient population and 155/198 C6 binding are in `SOURCE_LINEAGE.json`; no private route hash has been reverified here.

`CAPACITY_METRICS.csv` and `BUDGET_ATTRIBUTION.csv` intentionally contain column headers only. Real proposal/action/rejection/no-op/change-area statistics, Pareto data, relaxed-space gains, parameter counts, probability-cache bytes, inference timings and peak inference memory are **NOT_EVALUATED**, not zeros. Synthetic action-set/budget tests cannot establish real capacity. No refit p90, feasibility or calibrated-safety claim is made. All original draft gates are retained in `DRAFT_GATE_ACCOUNTING.json` with explicit NOT_EVALUATED states.

## Errors, publication and stopping point

Two early local development tests failed because the old static audit refuses uncommitted new namespaces; clean exact-source regression resolved that condition without editing the old audit or lock. Direct server GitHub clone encountered a TLS transport failure; a standard bundle of the same published Git source was used. Neither issue is a scientific result. Raw logs, JUnit and the actual server parent exit receipt remain on NAS, with sanitized summaries public.

All new files are confined to the three prescribed namespaces. No old status, artifact or lock was edited, no main merge occurred, and no frozen/private data were published. The pending semantic decision is not assumed from elapsed time. This attempt stops at this report. `NEXT_STAGE_DRAFT.md` records prerequisite and methodological limits only; capacity screening and risk fitting are not admitted and do not auto-start.
