# R0 inventory and budgets

Read-only old-server inventory on 2026-09-25, before exact M1 manifest. Source HDF5 files were checked for existence, bytes, and top-level keys; source payload SHA checks and transfer are pending.

| Domain / role / kind | Files | Bytes |
|---|---:|---:|
| Drishti_GS / L / image | 10 | 2,331,470 |
| Drishti_GS / L / label | 10 | 119,658 |
| Drishti_GS / val / image | 25 | 5,774,838 |
| Drishti_GS / val / label | 25 | 299,755 |
| RIM_ONE_r3 / L / image | 16 | 3,506,623 |
| RIM_ONE_r3 / L / label | 16 | 183,970 |
| RIM_ONE_r3 / val / image | 40 | 8,887,106 |
| RIM_ONE_r3 / val / label | 40 | 463,057 |
| **HDF5 total** | **182** | **21,566,477** |
| Frozen manifest + split | 2 | 759,948 |
| **Data minimum** | **184** | **22,326,425** |

Old-server exact roots have a split layout: metadata under the old canonical metadata root, HDF5 `images/` and `labels/` under its `h5/v1` child. `migration/minimal_manifest.py` accepts `metadata_root` separately and maps both sources into target `data/`. No source files were changed.

`CODE_MANIFEST.json` lists 42 source/document/import files totaling 189,257 bytes, excluding the manifest itself and `.gitignore`; no code has been transferred. Single-weight count/bytes remain pending asset binding. Excluded from M1: U payload and U labels, REFUGE payload, test, MRI, old checkpoints/runs, full caches/environments, secrets. Official weight not found at the three expected old cache locations; broader search was not performed.

Scientific optimizer updates actually executed: **0**. R0 CPU synthetic optimizer calls: **pending**, cap64 across at most two suite attempts. R1 plan remains 28 tasks, 24 final students, 40,000 physical formal updates; qualification≤32 and L-only smoke16 are future review-gated costs.
