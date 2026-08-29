# ASPR V0.1 freeze for TARC V0.1

**Freeze status:** `ASPR_V0_1_FROZEN_FOR_TARC`  
**Generated:** 2026-08-28 21:22 +08:00  
**Scope:** immutable preregistered input to the TARC-Seg V0.1 feasibility audit

## Frozen findings

- ASPR final status is `ASPR_UNLABELED_MEMORY_NOT_SUPPORTED`.
- Gate A (reliable unlabeled memory) failed.
- Gate D (site-mode utility) failed.
- Gate B (relation-coordinate/prototype drift) passed.
- Gate C (labeled-evidence adaptive transport) passed.
- H3 is not promoted or renamed as a proposed method.
- TARC creates no new unlabeled prototype memory and no background memory bank.
- TARC creates no site-mode bank and does not use ASPR site modes.
- Transport estimation may use only the current site's visible `train_labeled` cases.
- Hidden GT, validation/test data, previous-site images, and unlabeled cases are forbidden for transport fitting.
- The R0 architecture, optimizer, scheduler, loss weights, relation temperature, data manifests, splits, and checkpoint lineage remain frozen.

## TARC consequence

Only the ASPR-supported observations—relation-coordinate drift and labeled-evidence adaptive transport—may be tested. TARC must first complete its all-class Part-A feasibility gate. No TARC method, loss, config, bridge, pilot, or full training is authorized unless the compiled state is exactly `TARC_FEASIBILITY_SUPPORTED`.

TARC transport must cover class IDs `0, 1, 2`, including background class `0`; a foreground-only fallback is prohibited.

## Frozen evidence

| Artifact | SHA-256 |
|---|---|
| `ASPRSEG_V0_1_FINAL_REPORT.md` | `1f1b50e650acb30c20aa6c80058adaba57421dbbab4f0164a9dddf57abb7b121` |
| `ASPR_FEASIBILITY_AUDIT.json` | `69e6842158c6c9e8bec74eaeae074969c2d703c1b7bc39d4ac5ca896713792a9` |
| `SRGAS_V0_2_HARD_STOP_REPORT.md` | `e78b348834bcb561bb11178d89702fe2b4e5fae18d0a835833a45af0a4d7af04` |
| `V0_3_FINAL_REPORT.md` | `fe4ca0d36506b76fd203781f8f573d5660a61070912c91933b1b87b185de6c8d` |
| TARC execution prompt | `e328e8990f5e6ca14bf3e9e46a2eecfdaed27c3af30a8c979c191f521ec37e68` |
| TARC experiment plan | `e060c65a4b7094ab036b578077fcd68966cb746994cccb920825700aabe7cc77` |

This freeze is declarative and does not modify any historical report, checkpoint, manifest, split, or run.
