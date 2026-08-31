# Prospective runtime-only amendment

The original registration at `91d4533` remains unchanged. The 90b4599 CUDA test run has 54 passes and one failure: Python 3.10 raises PermissionError while the duplicate old-host test probes `/root/LCRSeg`, before its conditional skip. All four production-model resume trajectories and the v3 full three-stage adapter/independent audit pass.

Execute with source `4e3e2740e1ecd3285390a5f051c439a044fe3d9a`. Only the test launcher changes: apply the previously disclosed duplicate-test skip at collection, and record/remove pytest-created internal `*current` scratch aliases before manifest sealing. No checkpoint, dataset, report or other payload file is deleted; no permission is changed. The first failed output and its parent exit=1 remain preserved. The first phase manifest was also correctly refused because it contained temporary symlinks; no verified archive is claimed for that failed phase.

The baseline adapter, training entry, supervisor, archive verifier and baseline tests remain byte-identical to registered code 90b4599; their SHA values are in JSON. New exact-code CUDA tests and a complete verified phase manifest are required before separate execution authorization. Training equations, fixed parameters, data roles, GPU scope and downstream boundaries remain unchanged. No formal B0 or Gate1C forward has started.
