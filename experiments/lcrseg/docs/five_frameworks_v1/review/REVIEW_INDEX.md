# Five frameworks V1 — external code review

**STOP_AWAITING_EXTERNAL_CODE_REVIEW. CODE_ONLY. PARENT_BINDING_REQUIRED.**

Repository: https://github.com/DLwbm123/SSL_CL_seg

Branch: `codex/sslcl-five-frameworks-v1-review`

R3 revision base / externally reviewed R2 commit: `bac21b6585ec284fb6e93f213e280c1492f6e57a`

Original study base: `8a6e93a8bc39554c878cffe0dedf7fbfefe7d16d`

The review commit is the commit containing this index (`git rev-parse HEAD`); the final handoff gives its complete SHA. No self-referential commit hash is placed inside the hashed tree. PR: NOT_CREATED.

The five forward/backward implementations and five baselines share one stage trainer. All run on CPU synthetic tensors, including the locked complex steerable pyramid and real second-order D-Convexity equation. This is **not** a claim that the original KI parent was found or integrated. A user-recognized original KI source/configuration/entry is still required; the existing binding record lists the unresolved native details.

Start with [response to external R2 / R07 fix](REVIEW_RESPONSE_R2.md) and [current R3 test report](TEST_REPORT_R3.json): **178 passed**, 0 failed/skipped, CUDA NOT_RUN. R01–R06 are CLOSED_PRESERVED; R07 is fixed in CPU/interface scope and awaits external review. The eleven external R2 regressions first reproduced 7 passed / 4 failed on the base execution code. See [current cost by invocation](CPU_COST_R3.json). The earlier [R1 response](REVIEW_RESPONSE_R1.md) and [162-test R2 report](TEST_REPORT_R2.json) remain historical evidence. Real parent binding is pending.

Read in this order:

1. [Parent binding](PARENT_BINDING.json), [completeness](FRAMEWORK_COMPLETENESS.csv), [deviations and pending work](IMPLEMENTATION_DEVIATIONS.md).
2. [Coordinates](ARCHITECTURE_AND_COORDINATES.md), [gradient contract](LOSS_AND_GRADIENT_CONTRACT.md), [state lifecycle](STATE_LIFECYCLE.md), [data access](DATA_ACCESS_CONTRACT.md).
3. [Source map](SOURCE_MAP.md), [dependency lock](DEPENDENCY_LOCK.json), [parameter manifest](TRAINABLE_PARAMETER_MANIFEST.json).
4. [Actual CPU test report](TEST_REPORT.json), [test log](TEST_LOG_SUMMARY.txt), [test coverage map](TEST_COVERAGE.md).
5. [Resolved protocol / complete DAG](RESOLVED_PROTOCOL.json), [target matrix](TASK_MATRIX_PREVIEW.csv), [framework candidates](SEARCH_CANDIDATES.json), [baseline candidates](BASELINE_CANDIDATES.json), [budget](BUDGET_PLAN.json), [future ablations](ABLATIONS_PLAN_ONLY.json).
6. [Unapproved lock](REVIEW_LOCK.json), [code fingerprint](CODE_MANIFEST.json).

Executable code: `experiments/lcrseg/five_frameworks_v1/`; tests: `experiments/lcrseg/tests/five_frameworks_v1/`. The delivery directory is the preserved user input. In particular its `LOCAL_REFERENCE_TEST_REPORT.json` records reference-package tests, **not this integration's results**.

Reproduce from repository root with the existing Python environment (torch 2.6.0, CPU):

```sh
# Obtain only the small author source files listed in DEPENDENCY_LOCK.json into
# a locally licensed dependency directory. Do not download images/checkpoints.
export SSLCL5_DEP_ROOT=/your/local/source_dependencies
python -m experiments.lcrseg.five_frameworks_v1.cli inspect --metadata-only
python -m experiments.lcrseg.five_frameworks_v1.cli plan --emit /tmp/five-framework-plan
python -m experiments.lcrseg.five_frameworks_v1.cli qualify --synthetic --device cpu
```

The required local layout is `CWMI/model/CWMI_loss/{CWMI_loss.py,ComplexSteerablePyramid.py}`, `D-Convexity/loss.py`, `JDTLosses/losses/jdt_loss.py`. Exact commit URLs and digests are locked; missing sources fail tests, never silently skip. CWMI is not redistributed because no license was declared. See [dependency retrieval](DEPENDENCY_SETUP.md).

Every `cli run` invocation rejects before opening an approval file or real payload while the parent is unbound. The separate guard tests all approval bindings and five independent budget caps. Green tests are not external approval. No genuine user launch receipt has been issued.

The planned comparison includes all five families in replication, regardless of development scores. No positive-effect threshold, CI exclusion, or all-seed/all-class-win rule is used. Nothing in this package reports new real performance.
