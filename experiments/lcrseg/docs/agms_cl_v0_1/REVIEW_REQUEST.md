# External code review request: AGMS_CL_V0_1

**STOP_AWAITING_EXTERNAL_CODE_REVIEW — CPU_QUALIFICATION_BLOCKED_ATTEMPT_CAP.**
This is a code-preparation submission, not an experiment approval or claim of qualification.
The final commit is the head of `codex/agms-cl-v0-1`; use its full Git SHA for review bindings.

## Review outcome requested

Inspect the complete independent implementation and evidence. Do not authorize production
from the earlier passing CPU report: the final candidate has no complete passing suite.
No approval template is included and no old NKA/DOSE approval is reused.

## Changes and entry points

| File under the new package | Responsibility |
|---|---|
| protocol.py / anchors.py | Complete independent canonical matrix, frozen attachments, original public metadata and code manifest |
| core.py | Parent logsumexp loss; legal-geometry normalization; balanced Brier; detached lagged risk weights and masks |
| model.py | Native B2-compatible main network, two294-parameter L-only heads and captured dec3/dec2 readouts |
| trainer.py | Original B2 forwarding/RNG, same-mask auxiliary LCTX supervision, explicit optimizer/EMA, U whitelist, risk transaction and40-point diagnostics |
| authority.py | Fresh final-SHA external review and independent Prompt B capability; clean-tree/environment/current-CPU gates |
| state.py | Actual native main/aux/EMA/Adam/R/prototype/support/RNG resume and exact semantic refusal |
| execution.py | Prefix/target file acceptance, finite10-node controller, failure/tail budget stop and isolated evaluation |
| qualification.py / tests.py | Future36 CUDA/24 smoke plus capped current CPU-generated suite and durable attempts |
| p0.py | Future zero-update, fixed current-L confidence-only opportunity audit |
| reporting.py | Actual12/36/27/40 export, fixed nine contrasts and prespecified development gates |
| __main__.py | prepare / plan / test / p0 / qualify / run / report CLI |

The only tracked additions are this package and matching docs. Original code, method plans,
DOSE results, approvals and ledgers are untouched. New public-input copies retain original
commit provenance. Public row hashes and original private receipt hashes are separate fields.

## CPU evidence and remaining blocker

- Attempt1:2 calls, failed a newly written assertion on raw A. The native A-only method
  projects the effective delta, not raw A; native constraints passed. The test was corrected
  to check `delta @ V` with the original residual contract; no scientific change occurred.
- Attempt2:28 calls, complete PASS. Covered all six native arms, active U, auxiliary
  gradients/EMA/whitelist, exact A0/B2 main/EMA/optimizer/prototype/read/RNG equivalence,
  A1/A5 continuation, generated model verification, report coverage and two injected failures.
- Attempt3:22 calls. Final RNG/diagnostic hardening passed pure contracts, actual auxiliary
  geometry and all six warmup/active native arms; both continuation comparisons returned.
  The aggregate-test harness then attempted to create its previous tail-fixture directory
  and raised FileExistsError. Baseline/failure cases later in this attempt were not executed.
- Total new CPU52/96, attempts3/3, old CPU134 unchanged. Real optimizer=0; no real prefix
  tensor, patient data, P0, CUDA, smoke, training or monitoring was accessed.

The final harness repair creates per-attempt payload directories. A final scope tightening
also rejects a formal A0/smoke node before prefix IO. These repairs were syntax-checked only;
no fourth suite, hidden optimizer calls, budget reset, or false final PASS was produced.
`CPU/TEST_REPORT.json` intentionally remains FAIL with the actual tested code digest.
`PREPARATION_STATUS.json` gives the final manifest and records the qualification mismatch.
All attempts/calls/cost records are retained; public tracebacks redact local paths only.

## Particular review targets

1. Original fine PAS/KL and main valid-readout geometry remain unchanged. H never overlaps
   fine pixels; uncertain fine class (.04,.48,.48) may enter the parent branch.
2. LCTX auxiliary targets are collected from the SAME original complementary mask.
   Teacher clean L/U feature reuse adds readouts, not backbone forwards.
3. Heads are explicit Adam/EMA members and excluded from U. Check current-L balanced risk,
   one-step lag, no invalid-label update, commit boundary and resume semantics.
4. A0 historical identities are not rewritten; complete close-state equivalence and exact
   environment remain mandatory. No automatic baseline/source retraining fallback exists.
5. Canonical schema plus fresh capabilities guard real prefix/P0/CUDA/smoke/P1. Verify source
   isolation, per-node physical tails, actual model checks on skip, and full completion costs.
6. Reports distinguish historical unavailable diagnostics from newly measured values; all
   ten nodes precede gates. Two orders are one seed, and DeltaForget=-DeltaOld is not a
   second independent observation. No SOTA, original-KI or independent-patient claim.

No P1R/P2/P3 executable nodes, seed164, SCNP, distance-field training, RL, SWD search,
monitoring, source training or first-target training is added.
