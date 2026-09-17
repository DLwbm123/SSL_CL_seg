# NATIVE_KEY_ALIGNMENT_V0_1 — external code review R1

## Decision

Reviewed commit: `9915168cc3707fb70afdc2c719c8781cdf76e3fb`.

**CHANGES_REQUESTED — no production execution approval.**

Method-component assessment: the reviewed patch-coordinate alignment and B2 loss integration are consistent with the submitted design. Retain them. One reproducible frozen-plan validation defect must be fixed before production integration. The absence of a production scheduler, checkpoint adapter, and real-prefix gate was explicitly disclosed and is not treated as a concealed defect or a failure of the requested code-only deliverable.

This is independent AI code review, not human signoff or a claim that native CUDA, source/prefix tensors, or real-data smoke have passed.

## Evidence and scope

Inspected, pinned to the reviewed commit:
- `experiments/lcrseg/native_key_alignment_v0_1/{alignment,trainer,protocol,tests,__main__}.py`.
- Its docs: `CODE_DIFF.txt`, `REVIEW_CHECKLIST.md`, `LIFECYCLE.md`, `TEST_REPORT.json`, `FROZEN_OPTIONS.json`.
- The referenced `five_frameworks_v1/kernels.py` SWD kernel.
- The conversation's original method specification and code-preparation prompt.

The repository reports 7/7 CPU groups, 30 cumulative synthetic optimizer calls, 2 invocations, capped at 32 calls and 2 invocations. This review did not rerun that native-model suite or modify those ledgers.

Independent checks: 24 passed, 0 failed, 0 optimizer calls, 8 gradient VJPs. Only generated CPU tensors and generated metadata were used. Five separate negative examples demonstrate the metadata validator defect; they are not counted as five passed safeguards.

The complete local copies of `alignment.py` and `protocol.py` have Git blob hashes matching the connector-returned hashes:
- alignment.py: `5db0a92c7ef46b2dfc55cf8f8fabd425585e55ef`
- protocol.py: `461a8705cc89f66926c023f1fa4135d7f17b6dcf`

Checks execute AST-extracted exact functions. Generated metadata substitutes for historical receipts; no complete repository checkout, upstream native model, or production permit is used. Local Python/PyTorch differ from the server; see TARGETED_CHECKS.json. The 384x384 check exercises feature-patch indexing and gradients, not a native 384x384 model forward or smoke.

No private patient payload, real checkpoint tensor, CUDA optimizer operation, remote write, experiment, or monitoring was performed. The full transitive repository code tree was not independently cloned/rehash-verified.

## Components that should be retained

1. The selected input is `decoder.dec1.merge.block.3`; binding requires LR_SRC_A, no left projection, [16,16,3,3] weight, V[144,8], and the actual stride/padding/dilation/group values. C3 uses the existing student-entry V. C2 changes only auxiliary coordinates, not native protection.
2. `patches` uses channel/kernel-row/kernel-column order consistent with flattened Conv2d weights. Small-grid values and gradients match unfold; a generated 384x384 feature grid matches explicit patch extraction.
3. Teacher reference and V are detached; student feature gradients remain. No-support loss is graph-connected zero. Sampling positions are independent of coordinate dimension. The SWD kernel remains empirical equal-cardinality projected/sorted squared distance.
4. Loss construction preserves B2 supervised/KL paths, adds one clean-U feature forward, and scales only the auxiliary term by ramp*0.05. C0 runs its no-op forward without an unnecessary graph. B2 lambda_U remains 1.0 in the delivered frozen options; the inherited lambda_SWD options field is not the new module's semantic binding.
5. CPU evidence reports exact two-step B2/C0/C3-zero-weight state and RNG equality, and nonzero gradients on reachable upstream B. These are useful CPU evidence, not CUDA equivalence claims.
6. The planned D1 diagnostic is 16 second-target nodes, 42,400 formal updates, with four explicitly shared B2 first-target prefixes. It is not a complete new-method trajectory.

## NKA-R1: required frozen-plan semantic validation repair

Location: `protocol.py::validate_plan`, lines 71–86.

The validator checks a self-supplied digest, aggregate node count/update count, the arm/seed/order/stage set, non-executable state, and internally consistent prefix/options hashes. It does not compare every field against a canonical expansion and independently anchored B2/prefix evidence.

Rehashing changed metadata is enough for all these generated counterexamples to be accepted:

| Counterexample | Observed |
|---|---|
| Change B2 lambda_U from 1.0 to 0.25, update options hashes and plan digest | Accepted |
| Swap O1/O2 final target domains and their matching steps/options, preserving total 42,400 | Accepted |
| Move one update from one arm to another, preserving total 42,400 | Accepted |
| Change C3's declared coordinate to random and its auxiliary weight to 0.2 | Accepted |
| Bind a node to another seed's prefix at plan-validation level | Accepted |

The dedicated `validate_prefix` does reject a cross-seed prefix when invoked with the corresponding receipt. This is a useful existing guard. The finding is that the plan-wide validator does not itself establish all frozen scientific semantics.

**Qualification:** this is not evidence that the delivered plan was altered, or that a live training authorization was bypassed. Production is disabled. A separately implemented exact approved-plan hash gate would add another defense. The defect is the current mismatch between 'frozen semantic validation' and 'internal hash consistency'; close it before enabling production.

Required repair:
- Separate pure canonical construction from writes (`freeze` must not be the only source of expected semantics).
- Bind source B2 options and four prefix evidence records to pinned content digests; reject altered source evidence as well as altered derived plan fields.
- Validate every node's ID, arm, seed, order, stage, domain, update count, prefix identity/hash, and complete effective options.
- Require O1 final Drishti_GS/2100 and O2 final RIM_ONE_r3/3200; require updates == total_steps; do not accept compensating cross-node changes.
- Validate arm definitions, layer and patch dimensions, coordinate choice, .05 auxiliary weight, D/8 scaling, masks, diagnostic steps, analysis gates, parent, zero source/first-target budgets, and empty D2/D3 execution scope.
- Validate runtime objects against this semantic record before any production payload read. Do not merely compare a JSON object's newly computed hash with its newly supplied hash.
- Add the five negative cases above, plus wrong PAS/ramp/projection and altered evidence, as zero-optimizer regression checks.

## Next integration scope — pending by design

### A. New study identity and controlled prefix reuse

Keep native recipe/model family B2 and study/arm identity separate. The D1 common-prefix permission applies only to the four named B2 first-stage records, four specified new arms, and stage 2. Do not globally weaken old predecessor checks or rewrite historical receipt identities. Verify actual prefix student/F/file hashes and schema behind new authorization; all four arms start from the same prefix tensors, but independent fresh adapters/EMA/prototypes/optimizer/warmup.

### B. Correct trainer and checkpoint lifecycle

The current subclass is intentionally CPU-generated-provider-only. Production must be enabled through an explicit new capability, not by deleting the guard or claiming SyntheticCurrentDomain is real current-domain data.

Native resume currently reconstructs the old StageTrainer. A new protocol adapter must reconstruct KeyAlignmentTrainer, preserve actual arm/loss semantics, and restore the exact auxiliary basis and diagnostic counters. Bind study, arm, complete configuration, prefix, layer, V/basis hash, diagnostics, and RNG namespaces. Reject C2↔C3 or changed-prefix restoration. Do not relabel an old B2 checkpoint as an auxiliary-arm checkpoint. Save/restore optimizer, scheduler, EMA, prototypes, step/cursor and RNG in addition to weights.

Continuous/resumed C2/C3 comparisons and zero-weight equivalence must be tested on finite generated data, including nonzero active auxiliary support. Saving weights alone is not a training-resume test.

### C. Fixed lifecycle, budget and reporting

D1 remains 16/42,400, no new source/first-target training. Production qualification and real smoke must have separate finite proposed budgets; this review does not authorize their execution. Reuse existing durable accounting, integrity acceptance and separate-cost/reporting utilities where appropriate, not the old F5 scheduler or approval.

Persist the fixed diagnostic steps instead of relying on a transient `last` field that is overwritten. No extra diagnostic forward or online retuning. Qualification feature exposure and VJPs are separately counted. Report C3-C0, C3-C2 and C3-C1, per seed/order, then within seed; distinguish early first-target values from final values. Verify common-prefix DeltaForget=-DeltaOld, without calling them independent evidence. Finish all 16 before scientific gates; engineering stops remain explicit. D2/D3 are not auto-launched.

### D. CPU budget provenance

The original suite has consumed both of its allowed invocations (30/32 updates, 2/2 attempts). Do not blindly rerun or erase it. Pure review regression checks can be separately recorded with zero optimizer calls. Any new optimizer-based integration tests require a separately authorized finite supplement and clear cumulative accounting. The supplied next-task prompt proposes that supplement, but no such calls have occurred in this review.

## Recommended next action

Send the supplied code-only repair/integration prompt to Codex. Retain the method, all four arms, main coefficient, layer, PAS and D1 budget. Return a new code commit and updated independent evidence for review. Do not run CUDA, real smoke, D1 or monitoring from this review decision.
