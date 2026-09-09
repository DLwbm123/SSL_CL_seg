# AMS Sequential Transfer V0.1 — launched, not completed

Status at 2026-09-09 21:43:33 Asia/Shanghai: **RUNNING_P1**. This is an initial launch receipt, not a result report.

Frozen execution source: `f174cf2a68212cf04ad19b18d8203a6455c94aee`; base: `77af53864fa116030c832d69e7a404c79de2456e`. Later receipt-only commits do not change the server execution checkout. Run: `ams_seq_transfer_v0_1_20260909_01`.

Exact-source local CPU and server CUDA qualification each passed five tests, including math equivalence, data/source isolation, source-student initialization, and resume consistency. Each used 313 synthetic optimizer updates. The earlier development suite used 305; total synthetic qualification updates are 931. The separate real-resolution smoke passed all eight updates; it used only authorized labeled samples with labels stripped from pseudo-U inputs and discarded all weights. No actual U image was read by smoke. Peak CUDA reserved memory was 912 MiB, and formal admission requires at least 2,048 MiB free per GPU. Qualification and smoke updates are excluded from the 95,400 formal updates.

The detached parent PID is 658819 (PPID 1 verified). Four source tasks have started on GPUs 4, 5, 6 and 7; the startup snapshot recorded 272 committed updates, no immediate failure and available logs. GPU sharing is allowed; existing processes are preserved. Each subsequent task checks current free memory before launch.

The fixed DAG completes six fresh source tasks (15,900 updates), then all thirty target tasks (79,500 updates), including all seeds/orders and five target arms. Source validation scores cannot remove target tasks. Isolated final evaluators and aggregate reporting follow the frozen protocol. Patient-level data, checkpoints, raw logs and runtime paths remain private on NAS.

This run does not depend on the Codex session or SSH connection. No recurring monitor was created. The runner stops after the admitted matrix and terminal report; no additional experiment is enabled. Final result verification and public result delivery remain pending. Historical KI and Anchored Mix evidence is unchanged.

See `QUALIFICATION_RECEIPTS.json`, `LAUNCH_RECEIPT.json`, `TASK_MATRIX.csv`, `BUDGET.json` and the verbatim `PROTOCOL.md`.
