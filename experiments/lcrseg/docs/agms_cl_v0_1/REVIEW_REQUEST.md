# AGMS_CL_V0_1 R1 repair: request for external re-review

**STOP_AWAITING_EXTERNAL_CODE_REVIEW.** Review the full SHA at the head of
`codex/agms-cl-v0-1`, following [REVIEW_RESPONSE_R1.md](REVIEW_RESPONSE_R1.md).
The original R1 decision remains CHANGES_REQUESTED with approved_phases=[]; no
APPROVED document, production permit or user launch receipt was created.

## Exact repair scope

- `p0.py`: both independent prefix acceptances finish before the single P0 cost
  session. Load-time identity/hash checks, read-only semantics, sealed idempotency
  and failure accounting remain intact.
- `reporting.py`: explicit invalid geometry / unselected valid partition in normalized
  coverage export. Raw diagnostic files and public raw rows retain legacy values.
- `authority.py`, `revalidation.py`: narrowly validate current-code composition of
  pinned old native evidence with new scoped evidence. Unknown changes fail.
- `review_r1_regression.py`: isolated generated regression, zero optimizer, two-call
  maximum. Actual P0 orchestration/counters; stub native model/provider/environment.

The core, model, trainer, state/restore, data, RNG paths, old methods, science PLAN,
EXECUTION_PLAN, options, prefixes/imports and environment bindings did not change.
See the exact five-file delta and 774 protected-file hashes in the composite report.
The complete baseline code manifest and CPU file inventory are retained and pinned.

## Evidence to inspect

1. [Original external review](REVIEW_R1/EXTERNAL_REVIEW_R1.md),
   [repair authorization scope](REVIEW_R1/REPAIR_PROMPT.md), decision and targeted checks.
2. Immutable [native CPU attempt4](CPU/ATTEMPT_4_REPORT.json) from baseline
   `ac01ab6de6250e11877fe9000a30665c2245292d`: 28 calls PASS; cumulative80/96,
   four/four amended attempts; old134 separate (214 cumulative CPU updates across studies).
3. [R1 regression](REVIEW_R1_REGRESSION/REPORT.json): attempt1 of at most2,
   11 checks PASS; new optimizer calls/attempts0. Original nested failure reproduced,
   sequential replacement passed, failures kept, sealed audit not repeated, load-time
   identity/hash refusal, report partition/export, read-only state/RNG, authority negatives.
4. [Current composite binding](REVIEW_R1_REGRESSION/COMPOSITE_REPORT.json):
   SCOPED_ZERO_UPDATE_REVALIDATION, not a full native rerun. Includes baseline provenance,
   current code/plan/execution hashes, exact delta and protected file proofs.
5. [Counter costs](REVIEW_R1_REGRESSION/ATTEMPT_1_COSTS.json),
   [coverage schema](COVERAGE_SCHEMA.md), [RUNBOOK](RUNBOOK.md).

Re-review should focus on orchestration sequencing, immutable-source binding and refusal
of unknown changes, scope of fixture stubs, and raw versus normalized diagnostic meanings.
The broad original implementation review entry remains archived in REVIEW_R1/BASE_REVIEW_REQUEST.md.

## Unexecuted production

Real prefix tensors0, patients0, real optimizer0, P0 production0, CUDA0, smoke0,
formal0, monitoring0. All production gates remain PENDING and require a genuine fresh
external approval of this commit plus separately forwarded user Prompt B.

Frozen future P1: seed163, O1/O2, A1–A5 second-stage only, ten new nodes/26500 updates;
A0 two historical imports. No source or first-stage training. CUDA36, smoke24 and
P0≤32 L images/0 updates unchanged. No automatic follow-up or scope expansion.
