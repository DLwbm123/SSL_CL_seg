# LCTX Weight Memory V0.1 — running, not completed

Verified at 2026-09-10T21:42:07.388203+08:00: four initial target tasks are training on GPUs4/5/6/7, with 2024 committed formal updates and no immediate failure. The detached parent PID1207073 has PPID1; this run does not depend on the Codex session or SSH connection. Run identifier: `lctx_weight_memory_v0_1_20260910_02`.

Execution source: `50e094dd188c90c0117b62bb63bf961c096ea37d`; branch `codex/lctx-weight-memory-v0-1`. The six existing SRC_CE students were verified by complete receipt and content hash; source recovery updates0. The fixed36-target/95400-update matrix uses current L only, all six arms and all three seeds/two orders. Source identity,84 layers, bases and initialization hashes are in SOURCE_BINDING_AND_LAYER_MANIFEST.json.

Qualification total:118 synthetic optimizer updates and24 discarded real-L smoke updates. Full numerical/parent/resume tests and measured smoke belong to the original c097633 source; the recovery source adds CUDA initialization before peak accounting and passed separate cold-process two-update engine tests on CPU and CUDA. The smoke admission explicitly reuses the unchanged mathematical implementation's measured946MiB reserved peak, with a2048MiB free-memory threshold; no second real smoke was run. See ENGINEERING_RECOVERY.md and QUALIFICATION_AND_LAUNCH.json for both source identities and all actual counts.

Attempt01 failed in cold CUDA instrumentation before any formal optimizer update or image/GT read. Its four failure counters, original reservation and logs remain preserved. No mathematical candidate, threshold or hyperparameter changed. The current attempt has fresh output and reservation directories; previous source states and historical experiments are unchanged.

After all36 target models are complete, the runner seals their student hashes, evaluates six zero-update STATIC_SOURCE references, writes the fixed paired analyses and terminates. Final scientific results and public closeout remain pending. No automatic monitor or additional experiment has been created. Private weights, patient data and raw logs remain on NAS.
