# Response to external R2: R07 snapshot ownership

**STOP_AWAITING_EXTERNAL_CODE_REVIEW. CODE_ONLY. No experiment approval.**

Base commit: `bac21b6585ec284fb6e93f213e280c1492f6e57a`.
Branch: `codex/sslcl-five-frameworks-v1-review`.
This response and the new R3 qualification artifacts are an append-only revision after external review R2. The review commit is the commit containing this file; its SHA is supplied in the handoff. The earlier [R1 response](REVIEW_RESPONSE_R1.md), [162-test report](TEST_REPORT_R2.json), [R2 cost](CPU_COST_R2.json), and original decisions remain available. The old current log, code manifest and unapproved lock are preserved under `history/R2_*`.

## Disposition

R01–R06: **CLOSED_PRESERVED**, as accepted in the supplied [external R2 decision](../external_review_R2/REVIEW_DECISION.json). Their implementation and existing 162 tests were not edited. The five framework definitions, loss combinations, gradient permissions, grids and study files are unchanged. The plan remains **212 trajectories / 424 target stages upper bound**, with no efficacy gate added.

R07: **FIXED_CPU_REGRESSION_PASSED_AWAITING_EXTERNAL_REVIEW**. Only `integration.py` changed in execution code.

- `CurrentDomainDataAdapter` first takes a private deep snapshot. Validation, digest and all reads use that same snapshot. Its public manifest property returns a deep copy and cannot be assigned. Each reader receives a copy of its row, so even a mutable nested geometry object returned or modified by a callback cannot alter later reads. There is no per-read full-manifest hash.
- `ExecutionPermit` recursively copies mappings into read-only mapping proxies and lists/tuples into tuples. Bindings, nested allowed digests, phases and budget cannot change through either constructor inputs or public attributes. Non-JSON-like mutable leaf objects are rejected. Only these metadata fields are frozen; `_seal` retains object identity and is never passed through `deepcopy`.
- `NativeParentBridge` privately owns a deep metadata copy. Validation and named-parameter lookup use that copy. Both `metadata` and `semantic_metadata()` return independent, JSON-compatible deep copies; returned parameter-group lists are also independent. Native parameter identities and module deep-copy behavior are retained.

This is ownership protection against accidental mutable aliases, not a sandbox against arbitrary code in the same Python process. Reader path resolution, symlinks and actual native loader permissions remain future binding work. No new security framework, registry entry or parent substitute was added.

## Actual CPU evidence and cost

All tests imported actual repository modules. `SSLCL_R2_SNAPSHOT` was removed from each test environment. The attached eleven-case test was copied unchanged; `audit_loader.py` and source snapshots are preserved solely as external input evidence and were not executed.

| Invocation | Actual result | Measured pytest time | Adam.step | autograd.grad | autograd.backward |
|---|---:|---:|---:|---:|---:|
| External R2 tests against unchanged base execution code | 7 passed / 4 failed | 0.293 s | 0 | 1 | 0 |
| Fixed external R2 plus five ownership checks | 16 passed | 0.320 s | 0 | 1 | 0 |
| Final full CPU qualification | 178 passed | 32.377 s | 250 | 937 | 13 |
| Whole R07 revision, all three invocations | 205 case executions, including baseline failures | 32.990 s | 250 | 939 | 13 |

The final suite preserves all 162 prior tests, adds the external 11, and adds five cases for L/U list replacement and public records, nested geometry/callback aliasing, permit phase/budget/public-state freezing with seal identity, and native metadata/parameter-whitelist ownership. Existing tests still verify an empty real-runner registry and refusal of hand-written parent/approval JSON before payload or approval reads. No tests were removed, skipped, xfailed or weakened. Three existing optimizer fault-injection warnings are unchanged; zero failures/errors/skips in final qualification. The existing Python 3.12.9 / Torch 2.6.0 environment and locked author source dependencies were reused.

Counts are instrumented API calls, not FLOPs or a scalar VJP count per differentiated input. Physical synthetic fault-injection work can be uncommitted. The final qualification count is explicitly separate from the whole revision, and neither claims to include previous tasks' debugging. No optimizer/autograd work outside these three invocations was performed in this revision.

Evidence: [local base reproduction](REGRESSION_BASELINE_R2_LOCAL.txt), [focused log](TEST_LOG_R3_FOCUSED.txt), [final log](TEST_LOG_R3_FINAL.txt), [per-case report](TEST_REPORT_R3.json), [cost by invocation](CPU_COST_R3.json), [ownership regressions](../../../tests/five_frameworks_v1/test_revision_r3.py).

## Remaining status

CUDA = NOT_RUN. Real data reads = 0; real checkpoint tensor reads = 0; real smoke updates = 0; formal optimizer updates = 0. Toy readers only recorded symbolic values. No GPU or SSH jobs, source training or B/C/D/E real execution were started.

Parent binding remains **PARENT_BINDING_REQUIRED**. Original KI identity and its actual parameter/constraint/stage policy, readout/geometry, native loss/optimizer, source provenance, real manifests and step budgets remain unverified, as recorded in the existing parent binding document. No new historical search or F_CONV/SVD substitution was performed. A user-recognized source/configuration/entry can resolve the missing identity; this revision does not require recovery of a particular old command before future binding work can proceed.

`REAL_RUNNERS` remains empty. The updated review lock has `approved=false`, no user launch confirmation and no approved phases. Passing CPU tests is not external approval. Push this append-only commit to the same branch and stop for external code review.
