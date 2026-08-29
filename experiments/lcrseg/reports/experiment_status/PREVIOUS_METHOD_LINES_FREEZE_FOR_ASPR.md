# Previous method lines freeze for ASPR-Seg V0.1

**Status:** `PREVIOUS_METHOD_LINES_FROZEN_FOR_ASPR`  
**Protocol:** `asprseg_v0_1`  
**Created:** `2026-08-28T20:33:40+08:00`

## Required declarations

- LCR routing/SRA branch = frozen.
- SR-GAS branch = frozen.
- RAGM = not executed.
- ASPR = new prototype-memory hypothesis.
- R0 losses/network/optimizer remain frozen.
- No architecture change is authorized.
- No third training network is authorized.

## Authoritative prior outcomes

- LCR-Seg V0.3: `FUNDUS_V0_3_INTERNAL_GATE_FAILED`.
- LCR-Seg V0.4a: `FUNDUS_V0_4A_INTERNAL_GATE_FAILED`; engineering and mechanism gates passed, while the research gate failed.
- SR-GAS V0.2: `SRGAS_V0_2_SEED0_PILOT_FAILED`; contract tests and full regression passed, but the frozen worst-point safety gate failed.
- RAGM was not executed and supplies no result or implementation to ASPR.

These outcomes are immutable evidence. ASPR does not tune, reinterpret, resume,
or overwrite any prior branch or formal run.

## ASPR boundary

ASPR-Seg V0.1 begins as an independent post-hoc prototype-memory feasibility
hypothesis. Until `ASPR_FEASIBILITY_SUPPORTED` is produced, only relation-space
auditing, deterministic memory reconstruction, and post-hoc feasibility analysis
are permitted. Training-method implementation and new optimizer steps are
prohibited before that gate.

The Fundus data, manifests, splits, checksum records, R0 checkpoints, model
architecture, relation feature source, relation temperature, R0 loss formula,
optimizer, scheduler, batch semantics, and random trajectories remain frozen.

## Source evidence and SHA-256

| Evidence | SHA-256 |
|---|---|
| `reports/experiment_status/V0_3_FINAL_REPORT.md` | `fe4ca0d36506b76fd203781f8f573d5660a61070912c91933b1b87b185de6c8d` |
| `reports/experiment_status/V0_4A_FREEZE_FOR_SRGAS.md` | `6a8bd0b8e416139671179bf1a4452bc7d37fe5154c781c8980f862fd6c406ae2` |
| remote `reports/experiment_status/V0_4A_FUNDUS_COMPLETION.json` | authoritative live source, read-only |
| `reports/experiment_status/SRGAS_V0_2_HARD_STOP_REPORT.md` | `e78b348834bcb561bb11178d89702fe2b4e5fae18d0a835833a45af0a4d7af04` |
| `reports/experiment_status/SRGAS_V0_2_TEST_REPORT.md` | `6cc3578532661356cc7d7c9e34d2be00cfd81f5affbde22f7715308d6b42479a` |
| ASPR Codex prompt | `bdd61ef9cef333f7a76973e68e29cadd2003baacb1652c0d53d5a7788a90e16e` |
| ASPR implementation plan | `1a5f5e98af7a16efd5c2cb42f8c1ee3972295c97e369ece0694fbbbe0e2337a2` |
| ASPR experiment plan | `7ecf4d6ceb327593421123fe5fb666821442abe0b2dfee0a92218747c6fca6f4` |

No prior run, report, checkpoint, manifest, split, or frozen dataset was modified
to create this record.
