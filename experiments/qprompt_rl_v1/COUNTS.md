# R0 inventory and budgets

M1 inventory and completed byte migration on 2026-09-25. Source HDF5 files were checked for existence, bytes, SHA256, and top-level keys. The target checked every allowlisted file's size and SHA256 before and after no-overwrite promotion. This is a byte/key audit, not medical semantic validation.

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

The initial exact M1 manifest contained 43 code files (including `CODE_MANIFEST.json`), 182 HDF5 files, and two frozen metadata files: **227 files, 22,526,059 bytes**. The final code manifest and migration receipt bind the review commit after the R0 report update; private path/case/hash lists are held only in private receipts. A separate, single official DINOv2-S weight on target storage is **88,283,115 bytes** and has its own SHA256 receipt. Excluded from M1: U payload and U labels, REFUGE payload, test, MRI, old checkpoints/runs, full caches/environments, secrets.

Scientific optimizer updates actually executed: **0**. R0 CPU synthetic optimizer calls: **5**, across **2/2** successful suite attempts (cap64 calls). R1 plan remains 28 tasks, 24 final students, 40,000 physical formal updates; qualification≤32 and L-only smoke16 are future review-gated costs.
