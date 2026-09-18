# AGMS external R1 response

Status: **STOP_AWAITING_EXTERNAL_CODE_REVIEW**. This repair implements the forwarded
REPAIR_PROMPT only. It does not execute Prompt B or supersede CHANGES_REQUESTED.
Baseline: `ac01ab6de6250e11877fe9000a30665c2245292d`.

## AGMS-R1: nested P0 cost sessions

Fixed in `agms_cl_v0_1/p0.py:run`: accept O1/O2 separately and retain (path, receipt)
before entering the P0 session. The native accept_prefix implementation and its
cost/session/evidence handling are unchanged. The P0 load still compares actual payload
identity and tensor fingerprint to the accepted receipt. No counter restriction was removed.

Read-through of sibling call chains: qualification smoke accepts both prefixes before
its enclosing qualification session; formal execution accepts a prefix before its target
training session; target acceptance runs after training closes. Those paths needed no edit.
Prefix and P0 costs remain separately reported through the existing report aggregation.

Regression uses actual P0 run orchestration and actual cost_session/NativeOperations/
Operations, with CPU-only generated tensors/provider and stub native model, GPU/environment,
and positive preflight. The old nested structure fails exactly with nested counter scope;
both failed cost sessions remain recorded. The successful sequence is:
`prefix1_enter, prefix1_exit, prefix2_enter, prefix2_exit, P0_enter, P0_exit`.
A generated failure on prefix2 leaves prefix1 success and prefix2 failure intact, opens no
P0 session/report, and performs no generated current-L provider read. Real patient reads
are zero throughout. Sealed re-entry adds no images or sessions; partial P0 refuses retry.
Identity/hash mismatches fail inside the P0 cost session before provider construction.
This is not a native model verification, actual prefix acceptance, or production P0 run.

## AGMS-R2: coverage meaning

Fixed only in `reporting.py:normalize_coverage/export`, documented in COVERAGE_SCHEMA.md.
The normalized coverage artifact adds invalid_geometry and unselected_valid counts and
fractions, preserves legacy raw fields and labels their meaning. It checks disjoint count
partitions and returns null legal-domain fractions when geometry=0. Generated cases cover
all legal/unselected, all fine, all coarse, mixed and all invalid; actual report export
preserves its input records and raw PUBLIC_RESULTS/GRADIENT_DIAGNOSTICS values.
Core coverage/masks/loss/risk/trainer remain byte-identical to baseline.

## Qualification provenance and authorization

Original CPU/ files are byte-identical: attempts1–4, aggregate attempt log, physical ledger,
attempt4 PASS and all cost evidence remain pinned by BASE_CPU_FILES.json. Native training
qualification belongs to the baseline, not an independently rerun new code tree.

`revalidation.py` verifies that immutable evidence and complete baseline code manifest,
unchanged plan/execution digests, the exact five-file allowed delta, 774 protected Python
files, current code hash and current scoped regression digest. `authority.preflight` calls
this narrow composition while retaining clean HEAD, current manifest, fresh external
approval, separate Prompt B and environment restrictions. An old PASS alone cannot pass;
unknown code changes, missing repair files, altered baseline records or stale regression
fail. No approved template/permit/launch receipt was created. The unchanged tests.py and
28-call native suite were not invoked.

## Results and costs

REVIEW_R1_REGRESSION attempt1: **11/11 PASS**, zero optimizer calls/attempts.
Maximum authorized regression invocations2; one consumed, no second run needed.
Model sentinel state, gradients, risk, optimizer and Python/NumPy/Torch RNG stayed unchanged
across orchestration/report checks. Negative real preflight rejects missing authority,
old-study approval and the current CHANGES_REQUESTED decision before GPU/prefix access.

Old native CPU134 plus AGMS80 =214 total historical optimizer calls; original AGMS
budget80/96 and attempts4/4 unchanged. The repair consumed0 additional optimizer calls.
P0 production, CUDA36, real prefixes, smoke24, P1 ten nodes/26500 and monitoring remain
PENDING/unexecuted. Real updates0. The frozen scientific matrix and advancement gates
are unchanged. All archived external evidence retains original bytes; `.py` attachments
are stored as `.py.txt` to keep archived source distinct from executable code.
