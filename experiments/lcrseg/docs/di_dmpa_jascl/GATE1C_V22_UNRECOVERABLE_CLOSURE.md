# Gate1C v2.2 unrecoverable closure

Recorded 2026-08-31 (Asia/Shanghai) under the user's explicit clean-regeneration
instruction. This is a closure of the evidence-recovery path, not a scientific
success or failure judgment about the old full run.

The new branch `codex/gate1c-v3-clean-regeneration` starts directly from handoff
commit `70ba3dfb4fc989a5149a6343d857e7d10fd2017d` on
`codex/sslcl-long-running-reproduction`. It does not start from or merge `main`.
This MD/JSON pair is committed and pushed separately before v3 work is published.

| Required field | Frozen value |
| --- | --- |
| `old_full_status` | `UNKNOWN_AFTER_SSH_OBSERVATION_INTERRUPTION` |
| `private_reference_recovery` | `PERMANENTLY_UNAVAILABLE` |
| `full_C1_C8_conclusion` | `null` |
| `same_attempt_retry_authorized` | `false` |
| `old_outputs_preserved` | `true` |
| `next_protocol` | `GATE1C_V3_CLEAN_REGEN_K2_IDENTITY_HISTORY` |

No historical file or available private output is modified or deleted by this
closure. Preservation does not assert that the inaccessible remote files still
exist: their current state cannot be independently inspected.

The historical scientific state remains:

- Gate0 v2: repaired baseline correctness PASS; B0 did not outperform C0 on the
  three-seed mean.
- Gate1A v2: `PASS_MULTI_MODALITY_SUPPORTED`; selected K is 2.
- Gate1B v2: `FAIL_TRANSPORT_NOT_SUPPORTED`; B3/B4/B5 failed. Learned transport
  is forbidden; identity is the only selected transform. R4 is unavailable;
  drift-calibrated naming is forbidden.
- Gate1C v2.1: `BLOCKED_INCOMPLETE_EVIDENCE`.
- Gate1C v2.2 integration: exact code
  `1cfd8235293e157afd6b40f0f091ce6bc6df9f9f`; 210/210 tests and 75 new real
  integration forwards PASS. These supply no C1-C8 conclusion.
- Gate1C v2.2 full: launched, but final status, actual controller exit,
  completed forward count, and C1-C8 are unknown. The final manifest was not
  independently verified; the private references are permanently unavailable.

SSH observation exit 255 is not a child-process exit receipt and implies neither
success nor failure. The old 1,800-forward budget is planned work, never a
completed count. The old v2.2 attempt will not be retried or renamed as v3.

The next line is a new prospective protocol:
`DI_DMPA_GATE1C_V3_CLEAN_REGEN_K2_IDENTITY_HISTORY`. It requires a read-only
destination/data/runtime audit, newly regenerated B0 checkpoints containing exact
historically hashed PAS banks, a K2 replication check, new validation caches,
one new 75-forward integration, and one new 1,800-forward formal diagnostic.
Each downstream phase remains conditional on its predecessors and its separately
published registration/authorization; this closure is not a launch receipt.

Even a future Gate1C pass leaves the original overall Gate1 status at
`FAIL_TRANSPORT_NOT_SUPPORTED`. It cannot establish DI-DMPA reproduction or
`PASS_CORE_ADMISSION`. No C0, method training, Gate2, Prostate, MnMS, full sweep,
or main merge is launched by this closure.

Historical evidence: [migration handoff](GATE1C_V22_MIGRATION_HANDOFF_20260831.md),
[migration state](gate1c_v22_results/9593908/MIGRATION_STATE.json), and
[original integration status](gate1c_v22_results/9593908/integration_attempt1/GATE1C_V22_STATUS.json).
Machine-readable closure: [JSON](GATE1C_V22_UNRECOVERABLE_CLOSURE.json).
