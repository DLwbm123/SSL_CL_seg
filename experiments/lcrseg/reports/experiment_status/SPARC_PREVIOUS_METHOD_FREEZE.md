# SPARC-Seg V0.1 previous-method freeze

**Freeze status:** `SPARC_PREVIOUS_METHODS_FROZEN`  
**Generated:** 2026-08-28 23:18:39 +08:00  
**Scope:** immutable preregistered input to SPARC-Seg V0.1

## Required declarations

- `STOP_NEW_RELATION_METHODS` remains binding for relation variants.
- SPARC is a representation-level new protocol.
- Uniform historical relation KD is unchanged.
- No old-method artifact will be modified.
- RAGM is not executed.
- No EMA teacher is used in the proposed method.
- No cross-site prototype memory is used.

## Frozen upstream outcomes

| Line | Frozen status |
|---|---|
| LCR-Seg V0.3 | `FUNDUS_V0_3_INTERNAL_GATE_FAILED` |
| SR-GAS V0.2 | `SRGAS_V0_2_SEED0_PILOT_FAILED` |
| ASPR-Seg V0.1 | `ASPR_UNLABELED_MEMORY_NOT_SUPPORTED` |
| TARC-Seg V0.1 | `TARC_RELATION_FIDELITY_NOT_SUPPORTED` |
| BPRC-Seg V0.1 | `BPRC_GRADIENT_SCALE_NOT_SUPPORTED` |
| Relation-method policy | `STOP_NEW_RELATION_METHODS` |

No SPARC method/configuration or training optimizer step is authorized unless the compiled feasibility state is exactly `SPARC_FEASIBILITY_SUPPORTED`. Part A may add only audit code, audit-only primitives, tests, and append-only reports.

## Frozen evidence

| Artifact | SHA-256 |
|---|---|
| `BPRCSEG_V0_1_FINAL_REPORT.md` | `6ef2f248f441cd1295ad12636e73e25bc0f1d53898694c3905cf432fc99fcf8c` |
| `BPRC_FEASIBILITY_AUDIT.md` | `6e747f72b64a27049bdca2c6ba32f9f16003ddbf869608917399936a1153a01f` |
| `STOP_NEW_RELATION_METHODS.md` | `e7ed033774d28cff1d9ebae7dd8137ab4e810f4a4eeceb9d8beee2cd732b4454` |
| `TARCSEG_V0_1_FINAL_REPORT.md` | `4365f09236fe90a51edd09a592450a0625120a3b99be098064b99336934c3f6e` |
| `ASPRSEG_V0_1_FINAL_REPORT.md` | `1f1b50e650acb30c20aa6c80058adaba57421dbbab4f0164a9dddf57abb7b121` |
| `SRGAS_V0_2_HARD_STOP_REPORT.md` | `e78b348834bcb561bb11178d89702fe2b4e5fae18d0a835833a45af0a4d7af04` |
| `V0_3_FINAL_REPORT.md` | `fe4ca0d36506b76fd203781f8f573d5660a61070912c91933b1b87b185de6c8d` |
| SPARC execution prompt | `931f2f53da1ed7e154aedadda9821e89ea03fce553b31d2ab4f2f593d845e34d` |
| SPARC implementation plan | `d16f8347faeb3d871603f35637b9dfde1c0cbc1677308fe4d3ac421ff0c800d6` |
| SPARC experiment plan | `878610c60a5a6ea850cfbbded177136ec8a18aa83e695c53cee2bb7426bb42e0` |

At freeze time, remote GPUs 5, 6, and 7 matched the registered UUIDs and were idle apart from 268 MiB system allocations. `/home` had about 115 GiB free and the NAS had about 30 TiB free. Training/high-frequency I/O remains local; accepted stages are archived append-only to NAS.

This freeze is declarative and modifies no historical artifact or frozen input.
