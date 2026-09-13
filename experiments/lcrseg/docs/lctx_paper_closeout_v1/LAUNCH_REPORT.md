# LCTX paper closeout V1 — launched, not complete

Run `lctx_paper_closeout_v1_20260913_01` started at **2026-09-13 19:03:48 +08:00** on `jiangsuiyang`, controller PID 3126483. Runtime source is `3d9dade9ab254b4e3cabb488f3034e3198a67087`; public documentation commits do not alter the deployed checkout.

The startup check at **2026-09-13 19:04:23 +08:00** observed four active trainers on shared GPUs 4/5/6/7, with 116 completed scientific updates in their durable logs. Training receipts: 0/30; evaluation receipts: 0/30. No task failure receipt was present. Each active worker used 900 MiB according to nvidia-smi. All four actual boundaries enumerated fourteen trainable convolution weight tensors and 438,192 scalars. Full parent/worker ps arguments and GPU process names were neutral; no method or project arguments were exposed.

| Task | GPU | PID | Startup updates |
|---|---:|---:|---:|
| O1_s71_C_CE | 4 | 3126715 | 27 |
| O1_s72_C_CE | 5 | 3126721 | 31 |
| O2_s71_C_CE | 7 | 3126719 | 28 |
| O2_s72_C_CE | 6 | 3126725 | 30 |

The finite queue contains exactly 30 new targets and 79,500 optimizer/backward/EMA updates. It reuses six SRC_CE and six standard-L LCTX targets; none are retrained or rescored. New target training has zero U-image accesses and zero response VJPs. All new final targets seal before the 30 fixed evaluation processes and shared-patient paired analysis. The controller generates reports and stops; no performance threshold selects which results are written.

Qualification used **44 synthetic updates** across two successful 22-update passes and **10 discarded real-L smoke updates**. Between these, the first smoke attempt failed before update zero because CUDA was not initialized before resetting peak memory. That zero-update attempt and the initial successful qualification are preserved under `qualification_attempts`; the entry was repaired before any formal training. Actual operation ledgers are separate from the scientific budget. The final smoke's peak reserved memory was 945,815,552 bytes (902 MiB), with 3,072 MiB free-memory admission including headroom. No unrelated process was stopped.

Standard target L counts are RIM16 and Drishti10; nested low counts are 8 and 5. The prescribed metadata-only selection and public digests are in `SUBSET_MANIFEST.json`, with patient identities retained on NAS. The low odd-batch rule is fixed in `PROTOCOL.json`; actual repeated exposure is accounted for. `DATA_SCOPE.json` records a bounded role-metadata check but does not establish independent-patient evaluation. No hidden/test GT was opened.

## Artifacts and access

- Local active worktree: `/Users/bominwang/Desktop/codes/SSL_CL_seg-paper-closeout`, branch `codex/lctx-paper-closeout-v1`.
- NAS run root: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/lctx_paper_closeout_v1_20260913_01`.
- Runtime checkout: run root `/r0`; controller state `/CONTROLLER_STATUS.json`, parent log `/parent.log`, task logs `/logs`, individual states and private scores `/tasks`.
- Editable draft: `paper/PAPER_DRAFT.md` and `paper/PAPER_DRAFT.tex`; claim map, tables, 42-row experiment index and submission scope accompany them. New scientific results are explicitly PENDING.
- Completion outputs will be `/public_results`, including all five contrasts, per-domain/class and seed effects, conditional patient intervals, resource/exposure accounting and an NAS archive receipt. The Markdown manuscript and completed table artifact are populated by the queue; an unavailable server-side LaTeX converter is documented rather than blocking the scientific report.

The detached queue survives closing this session. No recurring monitoring was created. At the next requested status check, use current receipts and controller state. After actual completion, verify and publicly publish aggregate results and updated manuscript; keep patients, weights and raw records private. Do not append seeds, variants, label fractions, third-domain training or merge main.
