# Native integration and actual launch

**Completed 2026-09-16:** see [FINAL_REPORT_REDUCED.md](FINAL_REPORT_REDUCED.md) for the verified single-seed completion, all method results and costs. The following launch and amendment snapshots are historical.

**2026-09-15 schedule amendment:** the user reduced unstarted work to one
replication seed and a smaller matrix: 312 target stages instead of 424.
See [the amendment](SCHEDULE_REDUCTION_20260915.md). The launch snapshot below
is historical; its original PID, queue size and monitoring statement are not
current execution status.

**RUNNING — source pretraining.** Observation: 2026-09-14T02:53:42.638676+00:00.

- Reference: **NATIVE_LR_SRC_A_3DOMAIN_V1**, explicitly selected by the user; not original KI recovery.
- Execution commit: `cc21871a43e3cd105cddaf831f5c3ea2fef59b9e`. All current qualification receipts and workers bind this clean checkout.
- Run ID: `native_lr_src_a_3domain_v1_20260914_01`.
- Executor PID: `3450182`. The finite background executor is independent of this chat/SSH session.

## Qualification

| Scope | Actual result | Optimizer cost |
|---|---|---|
| Final server CPU | 192 passed; 0 failures/errors/skips | All attempts in COST_SUMMARY.json |
| CUDA first attempt | 13 passed / 1 source-CE deterministic-kernel incompatibility | 29 calls |
| CUDA corrected native suite | 14 passed | 31 calls; cumulative 60/256 |
| Current RIM L smoke | F1/F2/F3/F4/F5: 4/8/4/4/4 successful updates; U batches = 0 | 24/24; no formal inheritance |

The source CE compatibility repair reuses the designated CE reduction without disabling deterministic algorithms. Smoke used disposable random native initialization and the actual warmup schedule, without selecting settings by scores; its weights do not become source or target predecessors. CUDA final peak allocated bytes: 1,899,587,072. The three active source workers each used 924 MiB at the recorded observation.

## Actual training evidence

| Source seed | Physical GPU | PID | Validated successful updates observed |
|---|---:|---:|---:|
| 161 | 5 | 3450335 | 400 |
| 162 | 6 | 3450336 | 3280 |
| 163 | 7 | 3450337 | 3280 |

The snapshot records 6,960 source updates and zero target updates. Each worker has a first completed optimizer event with optimizer_steps=1 and later validated scientific progress; those original records are preserved in RUNNING_RECEIPT.json. Their command lines and GPU process listings were checked. Seed 164 is queued. Parent/baseline development, five-framework development and all-family replication await their declared source/selection/stage predecessors. This is not a claim that all five target frameworks are already training or that any experiment is complete.

The fixed plan remains four source nodes (32,000 updates maximum), 212 target trajectories / 424 stages (1,123,600 updates maximum), and 1,155,600 combined scientific updates. No extra candidate, efficacy threshold, cross-trajectory checkpoint, old training replay, hidden U labels or Phase E was added. Source and target physical ledgers and extended native operation records retain attempted work and failures.

## Published and private evidence

Published: implementation, designated parent/authority/plan, CPU evidence, both CUDA attempt summaries, smoke and current source-launch/cost aggregates. See [RUNNING_RECEIPT.json](RUNNING_RECEIPT.json), [COST_SUMMARY.json](COST_SUMMARY.json), [USER_AUTHORIZATION.json](USER_AUTHORIZATION.json) and [README.md](README.md).

Images, labels, patient/case identities, per-case validation, checkpoints, raw server logs and private configuration remain on the authorized NAS. No credentials or private paths are published. Historical KI recovery and external review records remain unchanged; the new authority is USER_DELEGATED_EXECUTION, not a fabricated external approval.

## Resume and reporting

The private NAS run root holds private_config.json, status.json, executor.log, per-node launch/progress/physical/operations records and latest.pt checkpoints. Resume uses the same pinned execution checkout and EXEC_CONFIG with the neutral module entry documented in the code. A failed node stops dispatch and retains evidence; diagnose it before any recovery. Per-node physical caps cannot be reset to obtain extra attempts. No monitoring automation was created. The existing executor continues the finite matrix and writes aggregate_results.json / replication_summary.json on completion; a later live status check is needed before claiming completion or publishing final scientific results.
