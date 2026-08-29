# SPARC V0.1 freeze for CRISP-Seg V0.1

Generated: `2026-08-29T09:55:01+08:00`

Status: `SPARC_V0_1_FROZEN_FOR_CRISP`

## Binding declarations

- SPARC status is `SPARC_PAS_NOT_SUPPORTED`.
- SPARC optimizer steps are `0`; no SPARC optimizer run exists.
- PAS/prototype spatial gating is not continued in CRISP.
- SPARC feature/path audit artifacts remain immutable.
- `STOP_NEW_RELATION_METHODS` remains binding.
- Uniform historical relation KD remains unchanged.
- CRISP is a channel-role representation-allocation protocol.
- CRISP feasibility work may not register a method or training config.
- Conditional method implementation requires the exact status `CRISP_FEASIBILITY_SUPPORTED`.

## Frozen evidence

| Artifact | SHA256 |
|---|---|
| `SPARCSEG_V0_1_FINAL_REPORT.md` | `eada7f8e857eab53ea580147cf4cf3f014c6b9359e1db6de7bd0313f7e3fca22` |
| `SPARCSEG_V0_1_FINAL_REPORT.json` | `ba4c105f0cb4fd81e4bbc078f488244f8d3c42530ce899cf58c48745e317d73b` |
| `SPARC_FEASIBILITY_AUDIT.md` | `c534a0eca77c57687966baa6617b300590495ea5eae97119fe9dd871b88d1ca8` |
| `BPRCSEG_V0_1_FINAL_REPORT.md` | `6ef2f248f441cd1295ad12636e73e25bc0f1d53898694c3905cf432fc99fcf8c` |
| `STOP_NEW_RELATION_METHODS.md` | `e7ed033774d28cff1d9ebae7dd8137ab4e810f4a4eeceb9d8beee2cd732b4454` |
| `TARCSEG_V0_1_FINAL_REPORT.md` | `4365f09236fe90a51edd09a592450a0625120a3b99be098064b99336934c3f6e` |
| `ASPRSEG_V0_1_FINAL_REPORT.md` | `1f1b50e650acb30c20aa6c80058adaba57421dbbab4f0164a9dddf57abb7b121` |
| `SRGAS_V0_2_HARD_STOP_REPORT.md` | `e78b348834bcb561bb11178d89702fe2b4e5fae18d0a835833a45af0a4d7af04` |

The supplied CRISP protocol documents were frozen at SHA256 `9c6099fc2650055e555986e4d01f17864e7f49139187a56c6ea2bbeb16ac6f15`, `af41f0eb8c2dc2f9ef954d566a4f639734c06ee2386ae2126beb8d5215f8fcae`, and `0843cee19f9235a4f3aa8319c4de4cf856b3544c941eab2b3cec0b0ac603a40b` for the prompt, experiment plan, and implementation plan respectively.

## Frozen data and checkpoint boundary

All three Fundus training manifests and splits were hashed before CRISP work. The six seed-transition checkpoint pairs are recorded in the JSON companion. The local training root remains `/home/jiangsuiyang/SSL_CL`; accepted artifacts are archived without overwrite under `/data_nas/jiangsuiyang/LCR-Seg`.

No dataset, manifest, split, historical report, historical checkpoint, method registry, or training config was modified while creating this freeze.
