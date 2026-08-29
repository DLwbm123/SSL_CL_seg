# TARC V0.1 freeze for BPRC V0.1

**Freeze status:** `TARC_V0_1_FROZEN_FOR_BPRC`  
**Generated:** 2026-08-28 22:00 +08:00  
**Scope:** immutable preregistered input to BPRC-Seg V0.1

## Frozen upstream result

- TARC final status is `TARC_RELATION_FIDELITY_NOT_SUPPORTED`.
- TARC Gate A (all-class/background anchor transport) passed.
- TARC Gates B (relation fidelity), C (current-site safety), and D (functional virtual step) failed.
- No TARC training method, training config, bridge, pilot, full run, external run, or cross-dataset run exists.
- TARC optimizer steps were zero and its complete regression suite passed 183/183 tests.

## BPRC boundary

- BPRC does not transport anchors or features.
- BPRC does not modify or repair TARC transport.
- BPRC changes only the representation and reduction of the historical relation-consolidation term.
- R0 supervised loss, SSL assimilation, native relation field, native anchor lifecycle, model, optimizer, scheduler, temperature, relation weight, manifests, splits, and checkpoints remain frozen.
- Old relation scores/targets are detached; the existing current native relation score path retains its existing gradients.
- No hidden GT, boundary GT, GT-class balancing, compatibility/rejection, replay, prototype memory, site modes, new temperature, or new loss weight is allowed.
- No BPRC method/config or optimizer execution is authorized unless the compiled state is exactly `BPRC_FEASIBILITY_SUPPORTED`.

## Frozen evidence

| Artifact | SHA-256 |
|---|---|
| `TARCSEG_V0_1_FINAL_REPORT.md` | `4365f09236fe90a51edd09a592450a0625120a3b99be098064b99336934c3f6e` |
| `TARC_FEASIBILITY_AUDIT.json` | `af6c12dd706299c7b51c721487a9ad6ebba0a93903cc243d82ea4aec4d4141c1` |
| `ASPRSEG_V0_1_FINAL_REPORT.md` | `1f1b50e650acb30c20aa6c80058adaba57421dbbab4f0164a9dddf57abb7b121` |
| `V0_3_FINAL_REPORT.md` | `fe4ca0d36506b76fd203781f8f573d5660a61070912c91933b1b87b185de6c8d` |
| BPRC execution prompt | `ae677805f1242d027e7c18e1c6e0c77387add569d04296e9b197e2d8af6b4095` |
| BPRC experiment plan | `90f0e1ec239895b9d45be3ce740b1ee7052514c2523a63ee17765e4c7dee9c81` |

This freeze is declarative and modifies no historical artifact or frozen input.
