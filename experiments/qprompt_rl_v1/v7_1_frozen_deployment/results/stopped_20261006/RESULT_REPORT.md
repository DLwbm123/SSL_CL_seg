# V7.1 — stopped on qualification failure

## Status

**STOPPED_QUALIFICATION_FAILURE.** Provenance passed; completed formal endpoints **0/8**. No native qualification updates or final validation were started. The attempt started at 2026-10-06 13:52:02 Asia/Shanghai (05:52:02 UTC); both failed jobs had exited by 13:52:32 Asia/Shanghai (05:52:32 UTC). These timestamps come from actual process receipts.

## Cause and preserved evidence

The new `diagnosis.py` omitted `from pathlib import Path`. Both zero-update diagnosis jobs raised `NameError` at executed line 121 when beginning historical exposure aggregation. The executed source commit is `e22b71a65c644e97cffb163b04d6c9ab890b730f`. The missing import was subsequently corrected locally, but **no diagnostic, qualification or endpoint was retried**. The executed NAS code, raw logs and cost records remain intact.

Policy distribution and last-group branch-0 reward calculations had occurred before the exception, but their metric arrays were still in memory and were not written. They are therefore marked unavailable, not reconstructed from claims of progress. No full stage-A or C1 PASS is claimed. C2's 32 updates were never started.

## Provenance

All four V7 frozen policies and all 18 historical endpoints passed checks of actual training commit, source model hash, native source code, patient-role identity, native options and recorded evaluation definitions/results. The source is seed 168, 8,000 updates. Actual V7 training commit: `37ac930652dc2da6585342a437d014ae3b4b995d`; public report baseline: `03224a49cef09a3325abb7d32ffd8762e678e880`. Patient-role manifests were compared by identities and published only as hashes/counts. See PROVENANCE_AUDIT.json.

## Eight endpoints and four pairs

RESULTS.csv lists every registered endpoint as NOT_RUN, with blank metrics. PAIRED_COMPARISONS.csv lists all four pairs as NOT_EVALUABLE. **No performance comparison, deployment-recovery explanation, learned selection advantage, retention gate or best-non-RL gate can be concluded.** Missing results are not zeros and do not imply negative scientific evidence. V7's original results and decision are unchanged.

## Physical costs

- Student optimizer updates: **0 / 21,232** authorized maximum.
- Controller optimizer updates: **0**; virtual updates: **0**.
- Actual feature extraction: 16 calls / 832 image pairs.
- Read-only policy-gradient diagnostics: 8 autograd calls.
- Allowed reward-role reads: 24 sample accesses; saved-anchor forward images: 64.
- Sum of parallel diagnostic job wall times: 26.737 seconds. Feature CUDA intervals: 18.123 seconds. These are not exact GPU busy time or whole-task elapsed time.
- Evaluation calls and new old-domain raw-image accesses: **0**. No sealed test access.

COSTS.json includes failed-attempt operation counts and both process exit receipts. Model restoration did not erase these costs. The existing policies required V7 learning cost (42,400 student and 1,696 controller updates); this attempt does not make that cost disappear.

## Scope and limitations

The planned comparison uses one fixed REFUGE source, two separate source-to-target adaptations and two controller seeds. It is not sequential three-domain training, four independent source runs or an independent-patient clinical confirmation. The native reference is NATIVE_LR_SRC_A, not an exact original KI reproduction. No training speedup or performance claim is made.

## Stop and publication

All attempt processes have exited. The failure rule blocked the 32-step training qualification, all eight endpoints and final evaluation. No automatic retry, no additional experiment branch and no next-round experiment were launched. This delivery publishes implementation, frozen scope, verified provenance, consumed costs and the stopped-attempt report; it does **not** claim experiment completion. A fresh authorization is required to retry after this frozen-plan stop.
