# Classifier head control V0.1 — running

Verified 2026-09-08T17:59:50.322557+08:00. This is a launch report, not an experiment-completion claim.

Execution source: `f478cc204a87a807858c76a95fc4246d301eeff2`. Public base: `405337e71d22aef011a041664fbc1b4b72b60408`. Branch: `codex/ssl-classifier-head-control-v0-1`.

CPU and CUDA exact-source qualifications passed all8 tests each. Separate real smoke completed8 optimizer updates, below the12 cap; no extra real diagnostic backward. Development and qualification steps remain separate from formal training. The selected195 metadata rows and12 old paired references passed preflight.

P0 completed all4 fixed old model pairs:1300 sample-model forwards,0 optimizer updates, actual epoch96 prototypes retained. Quantitative P0 interpretation awaits the full report and cannot alter the queue.

P1 is running on GPUs4–7. First startup snapshot observed1751 formal updates across four active tasks, with no failure records. The two LIN_MT_PAS tasks follow in their fixed lanes. P1 has6 tasks and15900 updates. The frozen gate then selects either P2_SSL (42400 extra), P2_SUP (21200 extra), or no further training. Only seeds21/22 are permitted for P2; no old seed12/13.

Detached parent PID `20637` has its own session and PPID1. Its logs and outputs are under:

`/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/ssl_head_control_v0_1_20260908_01`

Parent log: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/ssl_head_control_v0_1_20260908_01/runner.log`. Per-task logs live at `seed11/<domain>/<arm>_train.log`; per-step records and checkpoints stay under each arm directory. The parent writes final tables and TERMINAL.json and stops. The session can close without stopping training. No recurring monitor was created.

Code, protocol, qualification and startup receipts are public. Private data, predictions, checkpoints and raw evaluator rows remain on NAS. Completion validation and public final-results delivery remain pending until the fixed queue finishes.
